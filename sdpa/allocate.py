"""Privacy accounting + closed-form dynamic budget allocation.

Three accounting schemes are implemented so that their *structural* effect on the
optimal allocation can be compared (Theorem 4 of the accompanying section):

    basic composition (ell_1)   sum_h c_h df_h / sigma_h  <= eps
    RDP (Gaussian)              sum_h alpha df_h^2 / (2 sigma_h^2) <= eps_RDP(alpha)
    GDP                         mu_tot = sqrt(sum_h (df_h/sigma_h)^2)

The allocator minimises the predicted legitimate-user distortion

    J(sigma) = J0 + sum_h w_h kappa_h sigma_h^2                      (utility)

subject to an information-leakage budget

    I(sigma) = sum_{h in P} 0.5 * ln(1 + g_h / sigma_h^2)  <=  I_max  (privacy)

where g_h is the block source power Tr Sigma_h (trace bound, see the
vector-release convention in ``calibrate.py``); the coordinate count d_h
enters the block sensitivity and the trace proxy. The program is convex: the
leakage term is convex in x_h = sigma_h^2, the objective is linear. Its KKT
point is available in closed form (the paper denotes the dual multiplier by
nu; the code field is still named ``mu``):

    x_h^*(nu) = (g_h / 2) * ( sqrt(1 + 2 nu / (w_h kappa_h g_h)) - 1 )

with nu >= 0 fixed by the budget through bisection. The dual function is
strictly decreasing in nu, from +inf (nu -> 0) to 0 (nu -> inf), so the root
is unique.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from math import log, log2, sqrt
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .calibrate import Calibration

LN2 = log(2.0)
BENCHMARK_LPIPS = 0.112  # "Direct transmission without protection" (reference paper)


# --------------------------------------------------------------------------- #
# privacy machinery
# --------------------------------------------------------------------------- #
def tau_ceiling(n_identities: int = 10_000) -> float:
    """Largest certifiable floor tau for a gallery of K identities.

        tau_max(K) = (log2 K - 1) / log2(K - 1)

    With K = 10^4 this is 0.925: Fano cannot certify a stronger claim than that
    no matter how much noise is added, because the entropy of the identity label
    itself is finite. Enlarging the gallery raises the ceiling slowly.
    """
    K = int(n_identities)
    if K < 2:
        raise ValueError("need K >= 2")
    return (log2(K) - 1.0) / log2(K - 1.0)


def i_max_bits(tau: float, n_identities: int = 10_000) -> float:
    """Leakage budget (bits) that certifies an identification error floor tau.

        I_max(tau) = log2 K - 1 - tau * log2(K - 1)

    Derivation: Fano's inequality gives P_e >= (H(B|Y) - 1) / log2(K - 1) and
    H(B|Y) = log2 K - I(B;Y) >= log2 K - I_leak. The budget is non-negative only
    for tau below the certification ceiling ``tau_ceiling(K)``.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie in (0,1)")
    K = int(n_identities)
    if K < 2:
        raise ValueError("need K >= 2")
    cap = tau_ceiling(K)
    if tau >= cap:
        raise ValueError(
            f"tau={tau:.4f} exceeds the Fano certification ceiling {cap:.4f} "
            f"for K={K}; enlarge the identity gallery or lower tau"
        )
    return log2(K) - 1.0 - tau * log2(K - 1.0)


def fano_floor(i_bits: float, n_identities: int = 10_000) -> float:
    """Certified lower bound on the eavesdropper's identification error."""
    K = int(n_identities)
    val = (log2(K) - float(i_bits) - 1.0) / log2(K - 1.0)
    return float(np.clip(val, 0.0, 1.0))


def leakage_nats(sigma: np.ndarray, g: np.ndarray, mask: np.ndarray | None = None) -> float:
    """sum_{h in P} 0.5 ln(1 + g_h / sigma_h^2): relaxed trace-bound AWGN MI.

    Vector-release convention: block h is a d_h-dimensional vector. The Gaussian
    worst-case MI is bounded by the relaxed trace bound
    I(z_h; z_h+N_h) <= 1/2 ln(1 + Tr Sigma_h / sigma_h^2); g_h is the block
    trace Tr Sigma_h (see Sec. 3 of the section). The sum runs over the
    *declared private subspace* P only (``mask``).
    """
    x = np.maximum(np.asarray(sigma, dtype=float) ** 2, 1e-300)
    g = np.asarray(g, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        x, g = x[m], g[m]
    return float(np.sum(0.5 * np.log1p(g / x)))


def rdp_gaussian(sigma: np.ndarray, df: np.ndarray, alpha: float,
                 mask: np.ndarray | None = None) -> float:
    """RDP of the Gaussian mechanism on the private subspace:
    eps_h(alpha) = alpha * df_h^2 / (2 sigma_h^2), summed over h in mask.
    """
    x = np.maximum(np.asarray(sigma, dtype=float) ** 2, 1e-300)
    df = np.asarray(df, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        x, df = x[m], df[m]
    return float(alpha * np.sum(df ** 2 / (2.0 * x)))


def rdp_epsilon_delta(
    sigma: np.ndarray,
    df: np.ndarray,
    delta: float,
    alpha_grid: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Convert RDP to (eps, delta): eps(delta) = min_alpha [eps_RDP(alpha) + ln(1/delta)/(alpha-1)].

    Returns (eps, best_alpha).
    """
    if alpha_grid is None:
        alpha_grid = np.concatenate([np.linspace(1.05, 4.0, 60), np.geomspace(4.0, 256.0, 80)])
    best_eps, best_alpha = np.inf, np.nan
    for a in alpha_grid:
        e = rdp_gaussian(sigma, df, float(a), mask) + log(1.0 / delta) / (float(a) - 1.0)
        if e < best_eps:
            best_eps, best_alpha = e, float(a)
    return float(best_eps), best_alpha


def gdp_mu(sigma: np.ndarray, df: np.ndarray, mask: np.ndarray | None = None) -> float:
    """GDP composition in quadrature over the private subspace:
    mu_tot = sqrt(sum_{h in mask} (df_h/sigma_h)^2)."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-300)
    df = np.asarray(df, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        sigma, df = sigma[m], df[m]
    return float(sqrt(np.sum((df / sigma) ** 2)))


def gdp_epsilon(mu: float, delta: float) -> float:
    """Invert the GDP privacy profile delta(eps) = Phi(-eps/mu + mu/2)."""
    from scipy.stats import norm

    if mu <= 0:
        return float(np.inf)
    z = norm.ppf(delta)  # Phi^{-1}(delta) < 0
    return float(mu * (mu / 2.0 - z))


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #
def waterfill_sigma(mu: float, a: np.ndarray, g: np.ndarray,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Closed-form KKT solution x_h*(mu) = (g_h/2)(sqrt(1 + 2 mu/(a_h g_h)) - 1).

    Components outside the private subspace carry no privacy obligation and no
    benefit from noise, so their allocated noise is exactly zero.
    """
    a = np.asarray(a, dtype=float)
    g = np.asarray(g, dtype=float)
    if mu < 0:
        raise ValueError("mu must be non-negative")
    x = 0.5 * g * (np.sqrt(1.0 + 2.0 * mu / (a * g)) - 1.0)
    sigma = np.sqrt(np.maximum(x, 0.0))
    if mask is not None:
        sigma = sigma * np.asarray(mask, dtype=float)
    return sigma


def _solve_mu(
    cal: Calibration,
    i_max_nats: float,
    depth_lo: float = -18.0,
    depth_hi: float = 18.0,
) -> float:
    """Bisection on mu so that leakage equals the budget (dual function is monotone)."""
    mask = cal.private

    def excess(log_mu: float) -> float:
        mu = 10.0 ** log_mu
        sigma = waterfill_sigma(mu, cal.a, cal.g, mask)
        return leakage_nats(sigma, cal.g, mask) - i_max_nats

    lo, hi = depth_lo, depth_hi
    if excess(lo) <= 0.0:  # budget larger than the no-noise leakage: no noise needed
        return 0.0
    if excess(hi) >= 0.0:
        raise ValueError("privacy budget too tight for the configured sigma range")
    return 10.0 ** brentq(excess, lo, hi, maxiter=200, xtol=1e-12)


def solve_mi_budget(
    cal: Calibration,
    i_max: float | None = None,
    tau: float | None = None,
    n_identities: int = 10_000,
) -> "Allocation":
    """Allocate under the MI leakage budget (optionally specified as a Fano floor tau)."""
    if i_max is None:
        if tau is None:
            tau = 0.90
        i_max = i_max_bits(tau, n_identities) * LN2  # bits -> nats
    mu = _solve_mu(cal, i_max)
    sigma = waterfill_sigma(mu, cal.a, cal.g, cal.private)
    return Allocation(scheme="mi", sigma=sigma, mu=mu, budget=i_max,
                      cal=cal, n_identities=n_identities, delta=None)


def _scale_to_epsilon(cal: Calibration, eps_target: float, eps_of_sigma: Callable[[np.ndarray], float]) -> np.ndarray:
    """Generic scaling search: eps(sigma) is decreasing in the noise scale."""
    base = waterfill_sigma(1.0, cal.a, cal.g, cal.private)
    base = np.where(base > 0, base, 1e-6) * cal.private

    def excess(log_c: float) -> float:
        c = 10.0 ** log_c
        return eps_of_sigma(c * base) - eps_target

    lo, hi = -12.0, 12.0
    if excess(lo) <= 0.0:
        return 10.0 ** lo * base
    if excess(hi) >= 0.0:
        raise ValueError("epsilon target not reachable in the searched scale window")
    c = 10.0 ** brentq(excess, lo, hi, maxiter=200, xtol=1e-12)
    return c * base


def solve_rdp_budget(cal: Calibration, eps_target: float, delta: float = 1e-5) -> "Allocation":
    """Allocate so that the converted (eps, delta)-DP guarantee meets eps_target."""
    sigma = _scale_to_epsilon(
        cal, eps_target, lambda s: rdp_epsilon_delta(s, cal.df, delta, mask=cal.private)[0]
    )
    return Allocation(scheme="rdp", sigma=sigma, mu=np.nan, budget=eps_target,
                      cal=cal, n_identities=10_000, delta=delta)


def solve_gdp_budget(cal: Calibration, eps_target: float, delta: float = 1e-5) -> "Allocation":
    """Allocate so that the GDP profile meets (eps_target, delta)."""
    sigma = _scale_to_epsilon(
        cal, eps_target, lambda s: gdp_epsilon(gdp_mu(s, cal.df, mask=cal.private), delta)
    )
    return Allocation(scheme="gdp", sigma=sigma, mu=np.nan, budget=eps_target,
                      cal=cal, n_identities=10_000, delta=delta)


def eps_l1_stylized(sigma: np.ndarray, df: np.ndarray, delta: float,
                    mask: np.ndarray | None = None) -> float:
    """Linear-composition law, amplitude term only: eps = sum c * df_h / sigma_h.

    This is the widely quoted "basic composition" form. It is sound only in the
    *small-information* regime mu_h = df_h/sigma_h <= c (Dwork-Roth Thm 3.20
    keeps the mu_h^2/2 term, see ``eps_l1_exact``). We keep it because it yields
    a closed-form allocation shape and because it is the form that
    Laplace-style accounting effectively assumes.
    """
    c = sqrt(2.0 * log(1.25 / delta))
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-300)
    df = np.asarray(df, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool); sigma, df = sigma[m], df[m]
    return float(c * np.sum(df / sigma))


def eps_l1_exact(sigma: np.ndarray, df: np.ndarray, delta: float,
                 mask: np.ndarray | None = None) -> float:
    """Rigorous Gaussian tail bound composed linearly (Dwork-Roth, Thm 3.20):

        eps = sum_h [ mu_h^2/2 + c mu_h ],   mu_h = df_h/sigma_h,
        c   = sqrt(2 log(1.25/delta)).

    The mu_h^2/2 term is what every "looks like Laplace" argument silently
    drops; at the operating point of the reference scheme (mu ~ 10..10^2) it
    dominates and the stylized law under-reports eps by orders of magnitude.
    """
    c = sqrt(2.0 * log(1.25 / delta))
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-300)
    df = np.asarray(df, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool); sigma, df = sigma[m], df[m]
    mu = df / sigma
    return float(np.sum(0.5 * mu ** 2 + c * mu))


def solve_basic_budget(cal: Calibration, eps_target: float, delta: float = 1e-5) -> "Allocation":
    """ell_1 (basic composition) accounting: closed form x_h ∝ (df_h / (w_h kappa_h))^{2/3}.

    Constraint: sum_h c_h df_h / sigma_h <= eps,  c_h = sqrt(2 ln(1.25/delta)).
    KKT:        a_h - lambda * c_h df_h / (2 x_h^{3/2}) = 0
                =>  x_h = (lambda c_h df_h / (2 a_h))^{2/3}
    """
    c = sqrt(2.0 * log(1.25 / delta))
    b = c * cal.df
    a = cal.a

    def sigma_for(s: float) -> np.ndarray:
        x = (s * b / (2.0 * a)) ** (2.0 / 3.0)
        return np.sqrt(x)

    def eps_of(s: float) -> float:
        sig = sigma_for(s)
        return float(np.sum(b[cal.private] / np.maximum(sig[cal.private], 1e-300)))

    def excess(log_s: float) -> float:
        return eps_of(10.0 ** log_s) - eps_target

    lo, hi = -12.0, 12.0
    if excess(lo) <= 0.0:
        s = 10.0 ** lo
    elif excess(hi) >= 0.0:
        raise ValueError("epsilon target not reachable under ell_1 accounting")
    else:
        s = 10.0 ** brentq(excess, lo, hi, maxiter=200, xtol=1e-12)
    return Allocation(scheme="basic", sigma=sigma_for(s), mu=np.nan, budget=eps_target,
                      cal=cal, n_identities=10_000, delta=delta)


def static_index_allocation(
    cal: Calibration,
    n_protected: int = 4,
    sigma_floor: float = 1e-3,
    sigma_max: float | None = None,
) -> "Allocation":
    """Baseline of the reference paper: hard-select the k 'cheapest' components.

    Selection score df_h / (w_h kappa_h) - high sensitivity at low utility cost is
    'cheap' to protect. Everything else is left (almost) untouched.
    """
    if sigma_max is None:
        sigma_max = float(np.sqrt(cal.g.max() * 4.0))
    score = cal.df / cal.a
    idx = np.argsort(-score)[: max(1, int(n_protected))]
    sigma = np.full(cal.H, sigma_floor, dtype=float)   # untouched components
    sigma[idx] = sigma_max                              # hard-selected components
    return Allocation(scheme=f"static-{n_protected}", sigma=sigma, mu=np.nan,
                      budget=np.nan, cal=cal, n_identities=10_000, delta=None)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
@dataclass
class Allocation:
    scheme: str
    sigma: np.ndarray
    mu: float
    budget: float
    cal: Calibration
    n_identities: int = 10_000
    delta: float | None = None
    lpips0: float = BENCHMARK_LPIPS
    extra: dict = field(default_factory=dict)

    # ---- metrics ---------------------------------------------------------
    @property
    def l2_power(self) -> np.ndarray:
        return self.sigma ** 2

    def predicted_lpips(self, cap: float = 0.75) -> float:
        """Globally saturating Jacobian model.

            J0 + cap * S / (cap + S),   S = sum_h a_h sigma_h^2

        Bounded in [J0, J0 + cap], so it stays physical once the optimal sigma
        leaves the small-noise regime; for small sigma it reduces to the linear
        model J0 + S that the optimiser minimises.
        """
        S = float(np.sum(self.cal.a * self.sigma ** 2))
        return float(self.lpips0 + cap * S / (cap + S))

    def predicted_lpips_linear(self) -> float:
        """First-order model the allocator actually minimises."""
        return float(self.lpips0 + np.sum(self.cal.a * self.sigma ** 2))

    def release_snr_db(self, mask: np.ndarray | None = None) -> float:
        """Mean per-component release SNR 10 log10(g_h / sigma_h^2) over the mask."""
        m = self.cal.private if mask is None else np.asarray(mask, dtype=bool)
        g = self.cal.g[m]
        x = np.maximum(self.sigma[m] ** 2, 1e-300)
        return float(np.mean(10.0 * np.log10(g / x)))

    def leakage_bits(self) -> float:
        return leakage_nats(self.sigma, self.cal.g, self.cal.private) / LN2

    def certified_error_floor(self) -> float:
        return fano_floor(self.leakage_bits(), self.n_identities)

    def audit_l1(self, delta: float | None = None) -> dict:
        """What the two linear-composition laws would certify for this release."""
        d = self.delta if (self.delta is not None) else (delta or 1e-5)
        d = delta if delta is not None else d
        return {
            "eps_stylized": eps_l1_stylized(self.sigma, self.cal.df, d, mask=self.cal.private),
            "eps_exact": eps_l1_exact(self.sigma, self.cal.df, d, mask=self.cal.private),
        }

    def rdp(self) -> tuple[float, float]:
        d = self.delta if self.delta is not None else 1e-5
        return rdp_epsilon_delta(self.sigma, self.cal.df, d, mask=self.cal.private)

    def gdp(self) -> tuple[float, float]:
        d = self.delta if self.delta is not None else 1e-5
        mu = gdp_mu(self.sigma, self.cal.df, mask=self.cal.private)
        return mu, gdp_epsilon(mu, d)

    def sparsity(self, thresh: float = 0.05) -> float:
        """Fraction of total noise power carried by the single heaviest component."""
        p = self.l2_power
        return float(p.max() / p.sum()) if p.sum() > 0 else float("nan")

    def kkt_residual(self) -> float:
        """Max |a_h - mu d_h g_h / (2 x_h (x_h + g_h))| ; zero at the KKT point."""
        if not np.isfinite(self.mu) or self.mu <= 0:
            return float("nan")
        x = self.sigma ** 2
        stat = np.abs(self.cal.a - self.mu * self.cal.g /
                      (2.0 * x * (x + self.cal.g)))
        return float(np.max(stat))

    def summary(self) -> dict:
        rdp_eps, rdp_alpha = self.rdp()
        gdp_mu_v, gdp_eps = self.gdp()
        return {
            "scheme": self.scheme,
            "mu": self.mu,
            "budget": self.budget,
            "delta": self.delta,
            "lpips_bob": self.predicted_lpips(),
            "leakage_bits": self.leakage_bits(),
            "fano_floor": self.certified_error_floor(),
            "rdp_eps": rdp_eps,
            "rdp_alpha": rdp_alpha,
            "gdp_mu": gdp_mu_v,
            "gdp_eps": gdp_eps,
            "noise_power": float(np.sum(self.l2_power)),
            "noise_power_frac": float(np.sum(self.l2_power) / np.sum(self.cal.g)),
            "sparsity": self.sparsity(),
            "sigma": self.sigma.tolist(),
        }

    def to_row(self) -> dict:
        d = self.summary()
        d.pop("sigma")
        return d


def evaluate(cal: Calibration, sigma: np.ndarray, delta: float | None = 1e-5) -> dict:
    """Free-standing evaluation of an arbitrary sigma vector."""
    alloc = Allocation(scheme="custom", sigma=np.asarray(sigma, dtype=float), mu=np.nan,
                       budget=np.nan, cal=cal, delta=delta)
    return alloc.summary()


def monotone_check(cal: Calibration, mus: np.ndarray) -> np.ndarray:
    """Leakage as a function of mu (strictly decreasing, used by the bisection)."""
    return np.array([leakage_nats(waterfill_sigma(float(m), cal.a, cal.g, cal.private),
                                  cal.g, cal.private) for m in mus])
