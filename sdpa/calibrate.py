"""Per-component calibration for DP-aware semantic communication.

For every semantic component h (h = 0..H-1) of the disentangled latent
code Z = [Z_0, ..., Z_{H-1}], Z_h in R^{d_h}, we need four scalars:

    g_h      block source power  Tr Sigma_h  (sum of coordinate variances)
    df_h     L2 sensitivity      sqrt(d_h) * (b_h - a_h)      (clipped, 99% mass)
    w_h      utility weight      d LPIPS / d sigma_h^2        (fitted on a sigma grid)
    kappa_h  residual fraction   Var[z_h - zhat_h] / sigma_h^2 after deprotection

Vector-release convention: each block h is released as a d_h-dimensional
vector. The worst-case AWGN mutual information uses the AM-GM (Jensen) bound
I(z_h; z_h+N_h) <= (d_h/2) log(1 + Tr Sigma_h / (d_h sigma_h^2)), so g_h is
the block trace Tr Sigma_h; d_h enters both this leak bound and the block L2
sensitivity Delta f_h = sqrt(d_h)(b-a). (The old 1/2 log(1+Tr Sigma/sigma^2)
"trace proxy" was a lower bound for d_h>1 and has been corrected.)

The pair (w_h, kappa_h) enters the objective of the budget-allocation
program; g_h and df_h enter the privacy constraints.

Layout convention (matches the Semantic StyleGAN latent used in the paper):
    h = 0, 1        : shared latent codes,   d_h = 512
    h = 2..14       : local groups (shape+texture), d_h = 1024
    total elements  : 2*512 + 13*1024 = 14336 = 28 * 512

Usage
-----
    cal = calibrate_from_latents(Z)                       # real pipeline
    cal = synthetic_calibration()                         # demo / unit tests
    cal.save("outputs/calibration.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Clipped dynamic range implied by the reference paper (arXiv:2510.23274):
# Delta_f_global = sqrt(14336) * (b - a) = 351.88  =>  b - a = 2.940
REFERENCE_GLOBAL_SENSITIVITY = 351.88
REFERENCE_TOTAL_ELEMENTS = 28 * 512  # 14336
REFERENCE_CLIP_RANGE = REFERENCE_GLOBAL_SENSITIVITY / np.sqrt(REFERENCE_TOTAL_ELEMENTS)

N_SHARED = 2
N_LOCAL_GROUPS = 13
D_SHARED = 512
D_LOCAL = 1024


def component_dims(n_shared: int = N_SHARED, n_local: int = N_LOCAL_GROUPS) -> np.ndarray:
    """Vector of per-component element counts d_h."""
    return np.array([D_SHARED] * n_shared + [D_LOCAL] * n_local, dtype=float)


def clipped_sensitivity(Z: np.ndarray, quantile: float = 0.995) -> np.ndarray:
    """Per-component L2 sensitivity of the clipped latent codes.

        df_h = sqrt(d_h) * (b_h - a_h),  a_h = 0.5% quantile, b_h = 99.5% quantile

    Sweeping all image pairs is O(N^2) and outliers distort the supremum, so the
    clipping surrogate of the reference paper is used (same rationale, but applied
    per component instead of globally).
    """
    if Z.ndim != 3:
        raise ValueError("Z must have shape (N, H, d)")
    lo = np.quantile(Z, 1.0 - quantile, axis=(0, 2))
    hi = np.quantile(Z, quantile, axis=(0, 2))
    d = Z.shape[2]
    return np.sqrt(d) * (hi - lo)


def source_power(Z: np.ndarray) -> np.ndarray:
    """Per-block source power g_h = Tr Sigma_h = sum over coordinates of Var."""
    if Z.ndim != 3:
        raise ValueError("Z must have shape (N, H, d)")
    return Z.var(axis=0).sum(axis=1)


def fit_utility_weight(sigma_grid: np.ndarray, lpips_curve: np.ndarray) -> float:
    """Fit w_h from a one-component noise sweep: LPIPS(sigma) ~ lpips0 + w * sigma^2."""
    x = np.asarray(sigma_grid, dtype=float) ** 2
    y = np.asarray(lpips_curve, dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(max(coef[1], 0.0))


def fit_kappa(sigma_grid: np.ndarray, residual_mse: np.ndarray) -> float:
    """Residual fraction after deprotection: MSE_res(sigma) ~ kappa * sigma^2."""
    x = np.asarray(sigma_grid, dtype=float) ** 2
    y = np.asarray(residual_mse, dtype=float)
    denom = float(np.sum(x * x))
    if denom <= 0.0:
        return 1.0
    return float(np.clip(np.sum(x * y) / denom, 1e-3, 1.0))


@dataclass
class Calibration:
    """Container for the four per-component scalars plus bookkeeping."""

    g: np.ndarray
    df: np.ndarray
    w: np.ndarray
    kappa: np.ndarray
    d: np.ndarray = field(default_factory=component_dims)
    private_mask: np.ndarray | None = None
    n_samples: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.g = np.asarray(self.g, dtype=float)
        self.df = np.asarray(self.df, dtype=float)
        self.w = np.asarray(self.w, dtype=float)
        self.kappa = np.asarray(self.kappa, dtype=float)
        self.d = np.asarray(self.d, dtype=float)
        H = len(self.g)
        for name, arr in (("df", self.df), ("w", self.w), ("kappa", self.kappa), ("d", self.d)):
            if len(arr) != H:
                raise ValueError(f"{name} must have length H={H}, got {len(arr)}")
        if np.any(self.g <= 0) or np.any(self.df <= 0) or np.any(self.kappa <= 0):
            raise ValueError("g, df and kappa must be strictly positive")
        if self.private_mask is None:
            self.private_mask = np.ones(H, dtype=bool)
        else:
            self.private_mask = np.asarray(self.private_mask, dtype=bool)
            if len(self.private_mask) != H:
                raise ValueError("private_mask must have the same length as g")
            if not self.private_mask.any():
                raise ValueError("private_mask must select at least one component")

    # ---- derived quantities -------------------------------------------------
    @property
    def H(self) -> int:
        return len(self.g)

    @property
    def a(self) -> np.ndarray:
        """Utility cost coefficient a_h = w_h * kappa_h."""
        return self.w * self.kappa

    @property
    def global_sensitivity(self) -> float:
        """Sensitivity under the reference paper's global accounting."""
        return float(np.sqrt(self.d.sum()) * (self.df / np.sqrt(self.d)).max())

    @property
    def sensitivity_inflation(self) -> float:
        """Delta_f_global / Delta_f_h for the worst component (= sqrt(H_eff))."""
        return self.global_sensitivity / float(self.df.min())

    # ---- io ----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "g": self.g.tolist(),
            "df": self.df.tolist(),
            "w": self.w.tolist(),
            "kappa": self.kappa.tolist(),
            "d": self.d.tolist(),
            "private_mask": np.asarray(self.private_mask, dtype=bool).tolist(),
            "n_samples": self.n_samples,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            g=np.array(obj["g"]),
            df=np.array(obj["df"]),
            w=np.array(obj["w"]),
            kappa=np.array(obj["kappa"]),
            d=np.array(obj["d"]),
            private_mask=np.array(obj.get("private_mask", np.ones(len(obj["g"]), dtype=bool))),
            n_samples=obj.get("n_samples", 0),
            meta=obj.get("meta", {}),
        )

    @property
    def private(self) -> np.ndarray:
        return np.asarray(self.private_mask, dtype=bool)

    @property
    def task_overlap(self) -> float:
        """Fraction of the utility weight that sits inside the private subspace.

        1.0 means the private semantics *are* the task (identity reconstruction:
        a certificate is expensive). ~0 means the task lives outside the private
        subspace (task-oriented semantics: a certificate is nearly free).
        """
        wk = self.w * self.kappa
        return float(wk[self.private].sum() / wk.sum())


def calibrate_from_latents(
    Z: np.ndarray,
    sigma_grid: Sequence[float] | None = None,
    lpips_curves: Iterable[np.ndarray] | None = None,
    residual_curves: Iterable[np.ndarray] | None = None,
) -> Calibration:
    """Full calibration from latent samples + optional per-component sweeps.

    Parameters
    ----------
    Z : (N, H, d)
        Latent codes produced by GAN inversion on the calibration split.
    sigma_grid : (S,) optional
        Noise standard deviations used in the per-component sweep.
    lpips_curves : (H, S) optional
        LPIPS measured when ONLY component h is perturbed at sigma_grid[s].
    residual_curves : (H, S) optional
        De-protected latent MSE for the same sweeps.
    """
    Z = np.asarray(Z, dtype=float)
    H = Z.shape[1]
    g = source_power(Z)
    df = clipped_sensitivity(Z)
    if lpips_curves is not None:
        curves = np.asarray(list(lpips_curves), dtype=float).reshape(H, -1)
        w = np.array([fit_utility_weight(sigma_grid, curves[h]) for h in range(H)])
    else:
        w = _default_weights(H)
    if residual_curves is not None:
        res = np.asarray(list(residual_curves), dtype=float).reshape(H, -1)
        kappa = np.array([fit_kappa(sigma_grid, res[h]) for h in range(H)])
    else:
        kappa = _default_kappa(H)
    return Calibration(g=g, df=df, w=w, kappa=kappa, d=Z.shape[2] * np.ones(H), n_samples=Z.shape[0])


def _default_weights(H: int) -> np.ndarray:
    """Shared codes dominate global appearance; local groups decay."""
    base = np.array([1.00, 0.95] + [0.60 * 0.86 ** i for i in range(H - 2)])
    return base[:H]


def _default_kappa(H: int) -> np.ndarray:
    """Secret shaper + deprotector remove most of the shared-code noise."""
    base = np.array([0.92, 0.88] + [0.62, 0.55, 0.48, 0.58, 0.66, 0.44, 0.52,
                                    0.61, 0.57, 0.40, 0.50, 0.63, 0.47])
    return base[:H]


def synthetic_calibration(H: int = N_SHARED + N_LOCAL_GROUPS, seed: int = 0) -> Calibration:
    """Plausible stand-in calibration so the allocator can be exercised offline.

    Values are anchored to the reference paper (clip range 2.940, global
    sensitivity 351.88) so that the reported numbers stay comparable.
    """
    rng = np.random.default_rng(seed)
    d = component_dims()[:H]
    df = np.sqrt(d) * REFERENCE_CLIP_RANGE
    df = df * (1.0 + 0.05 * rng.standard_normal(H)).clip(0.85, 1.15)
    g = np.where(d == D_SHARED, 1.0, 0.55) * (1.0 + 0.10 * rng.standard_normal(H)).clip(0.7, 1.3)
    w = _default_weights(H) * (1.0 + 0.08 * rng.standard_normal(H)).clip(0.8, 1.2)
    kappa = _default_kappa(H) * (1.0 + 0.05 * rng.standard_normal(H)).clip(0.85, 1.10)
    kappa = np.clip(kappa, 0.2, 0.98)
    return Calibration(g=g, df=df, w=w, kappa=kappa, d=d, n_samples=0,
                       meta={"source": "synthetic", "note": "anchored to arXiv:2510.23274 clip range"})


IDENTITY_COMPONENTS = (0, 1, 3, 4, 5, 6)


def reconstruction_calibration(H: int = N_SHARED + N_LOCAL_GROUPS, seed: int = 0) -> Calibration:
    """Identity *is* the task: the private subspace covers the whole latent.

    Every component is declared private and every component carries utility
    weight, so ``task_overlap = 1`` and a certificate is maximally expensive.
    This is the regime of the reference paper (face reconstruction).
    """
    cal = synthetic_calibration(H=H, seed=seed)
    return Calibration(g=cal.g, df=cal.df, w=cal.w, kappa=cal.kappa, d=cal.d,
                       private_mask=np.ones(cal.H, dtype=bool), n_samples=0,
                       meta={"source": "synthetic-reconstruction", "task_overlap": 1.0})


def task_decoupled_calibration(
    H: int = N_SHARED + N_LOCAL_GROUPS,
    seed: int = 0,
    identity_idx: Sequence[int] = IDENTITY_COMPONENTS,
    identity_weight: float = 4e-3,
) -> Calibration:
    """Identity-bearing components are private but (almost) task-null.

    The private subspace is the identity code set; a task-oriented receiver
    (attribute / expression / retrieval) puts almost no weight there, hence
    ``task_overlap`` is small and the certificate becomes affordable. This is the
    regime the allocation scheme is designed for.
    """
    cal = synthetic_calibration(H=H, seed=seed)
    w = cal.w.copy()
    idx = [i for i in identity_idx if i < H]
    w[idx] = identity_weight
    mask = np.zeros(cal.H, dtype=bool)
    mask[idx] = True
    wk = w * cal.kappa
    return Calibration(g=cal.g, df=cal.df, w=w, kappa=cal.kappa, d=cal.d,
                       private_mask=mask, n_samples=0,
                       meta={"source": "synthetic-task-decoupled",
                             "identity_components": list(idx),
                             "task_overlap": float(wk[mask].sum() / wk.sum())})
