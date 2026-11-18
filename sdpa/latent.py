"""Latent-level measurement on the real SemanticStyleGAN generator.

This module implements the "图像 → 分解潜码 → 四个标量" measurement chain
against the official pretrained SemanticStyleGAN (CelebAMask-HQ, 512x512,
seasonSH/SemanticStyleGAN release 1.0.0, vendored under ``third_party/``).

Latent layout of the checkpoint (verified against ``ckpt["args"]``):
    n_local = 13 local generators, base_layers = 2
    n_latent = 2 + 13*2 = 28 style codes of dim 512  (= 14,336 elements)

Block convention of the paper (matches sdpa.calibrate):
    block 0, 1      : shared base codes          (d = 512 each)
    block 2 + j     : local class j shape+texture codes concatenated
                      (codes 2+2j and 2+2j+1, d = 1024), j = 0..12

What is measured here, and how:
    g_h, df_h : source statistics of the *released code distribution*.
        Real-image inversion (visualize/invert.py) is optimisation-based and
        needs a GPU to run at scale; on this CPU-only machine we instead push
        the exact generator prior z ~ N(0, I) through the real 8-layer mapping
        network and measure on the resulting w-code distribution.  This is the
        exact source distribution under the paper's Gaussian-latent model;
        inversion-based statistics on real identities remain the GPU follow-up
        (see README, "latent-level measurement").
    w_h       : REAL utility weight.  w codes are decoded by the real
        generator with block h perturbed by AWGN sigma; Bob's distortion is
        the LPIPS (VGG) distance between the clean and the perturbed render.
    kappa_h   : REAL residual fraction of deprotection.  Reported curves use
        the exact per-block linear-MMSE (Wiener) filter whose covariance is
        measured on the same w codes (``wiener_kappa_curves``); for the
        independent near-Gaussian blocks this is essentially Bayes-optimal.
        The neural sigma-conditioned deprotector of sdpa.train is retained
        (``train_deprotector``) as the reference implementation for GPU
        reruns; within the CPU budget it converges to the identity map and
        is not used for the reported numbers.

Everything is resumable: the utility sweep checkpoints after every
(block, sigma) pair so it can advance in bounded wall-clock slices.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party" / "SemanticStyleGAN"
DEFAULT_CKPT = ROOT / "data" / "pretrained" / "CelebAMask-HQ-512x512.pt"

N_SHARED = 2
N_LOCAL = 13
D_SHARED = 512
D_LOCAL = 1024
N_CODES = N_SHARED + 2 * N_LOCAL  # 28 codes of 512
CODE_DIM = 512
H = N_SHARED + N_LOCAL  # 15 blocks

# local semantic class order of the CelebAMask-HQ checkpoint (seg_dim = 13),
# VERIFIED against third_party/SemanticStyleGAN/data/preprocess_celeba.py:
# 0 background, 1 skin, 2 eyes(l+r), 3 brows(l+r), 4 mouth(lips), 5 nose,
# 6 ears(l+r), 7 hair, 8 neck(+neck_l), 9 cloth, 10 eye_g, 11 hat, 12 ear_r.
# transparent_dims in the checkpoint args is (10, 12) = eye_g and ear_r.
LOCAL_CLASS_NAMES = [
    "background", "skin", "eyes", "brows", "mouth", "nose", "ears",
    "hair", "neck", "cloth", "eye_g", "hat", "ear_r",
]
BLOCK_NAMES = ["shared_0", "shared_1"] + [
    f"{n}(shape+texture)" for n in LOCAL_CLASS_NAMES
]

# Identity-bearing blocks in the LATENT layout: the two shared codes (global
# appearance) plus the facial-feature local classes (skin, eyes, brows,
# mouth, nose). Background, ears, hair, neck, cloth and the two accessory
# classes are treated as non-identity for the decoupled scenario.
LATENT_IDENTITY = (0, 1, 3, 4, 5, 6, 7)


def block_dims() -> np.ndarray:
    return np.array([D_SHARED] * N_SHARED + [D_LOCAL] * N_LOCAL, dtype=float)


def code_slice(block: int) -> slice:
    """Columns of the (N, 28, 512) w-code tensor that form paper block h."""
    if block < N_SHARED:
        return slice(block, block + 1)
    j = block - N_SHARED
    return slice(N_SHARED + 2 * j, N_SHARED + 2 * j + 2)


# --------------------------------------------------------------------------- #
# generator io
# --------------------------------------------------------------------------- #
def load_generator(ckpt_path: str | Path = DEFAULT_CKPT, threads: int | None = None):
    """Load the vendored SemanticStyleGAN generator (g_ema) on CPU."""
    import torch

    if threads:
        torch.set_num_threads(int(threads))
    sys.path.insert(0, str(THIRD_PARTY))
    from models.semantic_stylegan import SemanticGenerator

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    a = vars(ckpt["args"])
    g = SemanticGenerator(
        size=a["size"], style_dim=a["latent"], n_mlp=a["n_mlp"],
        channel_multiplier=a["channel_multiplier"], seg_dim=a["seg_dim"],
        local_layers=a["local_layers"], base_layers=a["base_layers"],
        depth_layers=a["depth_layers"], local_channel=a["local_channel"],
        coarse_channel=a["coarse_channel"], coarse_size=a["coarse_size"],
        min_feat_size=a["min_feat_size"], residual_refine=a["residual_refine"],
        detach_texture=a["detach_texture"], transparent_dims=a["transparent_dims"],
    )
    g.load_state_dict(ckpt["g_ema"])
    g.eval()
    assert g.n_latent == N_CODES, f"unexpected latent layout {g.n_latent}"
    return g


def sample_w_codes(g, n: int, seed: int = 0, batch: int = 1024) -> np.ndarray:
    """Push the exact prior through the real mapping network.

    Returns float32 array (n, 28, 512) of w codes.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    out = np.empty((n, N_CODES, CODE_DIM), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, n, batch):
            hi = min(lo + batch, n)
            z = [torch.randn(hi - lo, CODE_DIM, generator=gen) for _ in range(N_CODES)]
            w = torch.stack([g.style(s) for s in z], dim=1)
            out[lo:hi] = w.cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# g_h / Delta f_h
# --------------------------------------------------------------------------- #
def source_stats_blocks(W: np.ndarray, quantile: float = 0.995):
    """Exact per-block source power and clipped sensitivity (heterogeneous d).

    Same estimators as sdpa.calibrate.source_power / clipped_sensitivity but
    applied per block so that local pairs use d = 1024 and the union quantile.
    Blocks are cast to float64 one at a time to bound peak memory.
    """
    g = np.empty(H)
    df = np.empty(H)
    for h in range(H):
        cols = np.asarray(W[:, code_slice(h), :], dtype=np.float64)
        cols = cols.reshape(len(cols), -1)  # (N, d_h)
        g[h] = cols.var(axis=0).sum()
        lo, hi = np.quantile(cols, [1.0 - quantile, quantile])
        df[h] = math.sqrt(cols.shape[1]) * (hi - lo)
    return g, df


# --------------------------------------------------------------------------- #
# w_h : generator-decode noise sweep with LPIPS (resumable)
# --------------------------------------------------------------------------- #
class UtilitySweep:
    """Resumable LPIPS sweep over (block, sigma) pairs.

    Checkpoint layout (.npz):
        clean      uint8 (N, 3, 512, 512)  clean renders (clipped to [-1, 1])
        lpips      float32 (H, S, N)       per-sample LPIPS, NaN = pending
        sigma_grid float32 (S,)
        W          float32 (N, 28, 512)    sweep latents
        meta       json string
    """

    def __init__(self, path: str | Path, sigma_grid: np.ndarray, n_samples: int,
                 seed: int = 0):
        self.path = Path(path)
        self.sigma_grid = np.asarray(sigma_grid, dtype=np.float32)
        self.n = n_samples
        self.seed = seed
        self.lpips = np.full((H, len(self.sigma_grid), n_samples), np.nan,
                             dtype=np.float32)
        self.clean = None
        self.W = None

    # -- checkpoint io -------------------------------------------------------
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, clean=self.clean, lpips=self.lpips,
                 sigma_grid=self.sigma_grid, W=self.W,
                 meta=json.dumps({"seed": self.seed, "n": self.n}))

    @classmethod
    def load(cls, path, sigma_grid, n_samples, seed=0):
        obj = cls(path, sigma_grid, n_samples, seed)
        if obj.path.exists():
            d = np.load(obj.path, allow_pickle=False)
            if (d["lpips"].shape == obj.lpips.shape
                    and np.allclose(d["sigma_grid"], obj.sigma_grid)):
                obj.lpips = d["lpips"]
                obj.clean = d["clean"]
                obj.W = d["W"]
        return obj

    @property
    def pending(self):
        return [(h, s) for h in range(H) for s in range(len(self.sigma_grid))
                if np.isnan(self.lpips[h, s]).any()]

    @property
    def done(self) -> bool:
        return not self.pending


def _decode(g, W_block: np.ndarray, batch: int = 8) -> np.ndarray:
    """Decode (N, 28, 512) w codes to uint8 images clipped to [-1, 1]."""
    import torch

    outs = []
    with torch.no_grad():
        for lo in range(0, len(W_block), batch):
            wt = torch.from_numpy(W_block[lo:lo + batch].astype(np.float32))
            img, _ = g([wt], input_is_latent=True, randomize_noise=False)
            img = img.clamp(-1, 1)
            outs.append(((img + 1) * 127.5).round().to(torch.uint8).cpu().numpy())
    return np.concatenate(outs, 0)


def _lpips_pairs(metric, clean_u8: np.ndarray, noisy_u8: np.ndarray,
                 batch: int = 8) -> np.ndarray:
    """LPIPS(clean, noisy) per sample; inputs uint8 (N, 3, H, W)."""
    import torch

    vals = []
    with torch.no_grad():
        for lo in range(0, len(clean_u8), batch):
            a = torch.from_numpy(clean_u8[lo:lo + batch]).float() / 127.5 - 1
            b = torch.from_numpy(noisy_u8[lo:lo + batch]).float() / 127.5 - 1
            vals.append(metric(a, b, normalize=False).cpu().numpy().ravel())
    return np.concatenate(vals)


def advance_utility_sweep(g, sweep: UtilitySweep, metric, seconds: float,
                          batch: int = 8, blocks=None, log=print) -> bool:
    """Advance the sweep for at most ``seconds``; returns True when complete.

    ``blocks`` optionally restricts this call to a subset of block indices so
    several processes can advance disjoint checkpoint files in parallel.
    """
    t0 = time.time()
    if sweep.W is None:
        sweep.W = sample_w_codes(g, sweep.n, seed=sweep.seed, batch=256)
        sweep.clean = _decode(g, sweep.W, batch=batch)
        sweep.save()
        log(f"[sweep] latents sampled + clean decodes cached ({sweep.n} imgs)")

    todo = sweep.pending
    if blocks is not None:
        todo = [(h, s) for h, s in todo if h in set(blocks)]
    for h, s in todo:
        if time.time() - t0 > seconds:
            sweep.save()
            log(f"[sweep] time budget reached, {len(sweep.pending)} pairs pending")
            return False
        sigma = float(sweep.sigma_grid[s])
        rng = np.random.default_rng((sweep.seed, h, s))
        noise = np.zeros_like(sweep.W)
        cols = code_slice(h)
        noise[:, cols, :] = rng.standard_normal(
            (sweep.n, cols.stop - cols.start, CODE_DIM), dtype=np.float32) * sigma
        noisy = _decode(g, sweep.W + noise, batch=batch)
        sweep.lpips[h, s] = _lpips_pairs(metric, sweep.clean, noisy, batch=batch)
        sweep.save()
        log(f"[sweep] h={h:2d} sigma={sigma:.3f} "
            f"LPIPS={np.nanmean(sweep.lpips[h, s]):.4f} "
            f"({len(sweep.pending)} pending)")
    return True


# --------------------------------------------------------------------------- #
# kappa_h : linear-MMSE (Wiener) deprotector on measured latents
# --------------------------------------------------------------------------- #
def wiener_kappa_curves(W_train: np.ndarray, W_eval: np.ndarray,
                        sigma_grid: np.ndarray):
    """Per-block residual curves of the linear-MMSE (Wiener) deprotector.

    The 28 codes are sampled independently by the generator, so blocks carry
    no cross-block information and the Bayes denoiser factorises per block.
    Within a block the w coordinates are correlated; the exact linear-MMSE
    filter uses the covariance measured on ``W_train``:

        z_hat = mu + S (S + sigma^2 I)^{-1} (y - mu),   S = Cov[z_h]
        residual MSE / dim = (1/d) tr( sigma^2 S (S + sigma^2 I)^{-1} )

    Returns (empirical_curves, analytic_curves), each (H, S).  The empirical
    curves apply the filter to held-out codes; the analytic curves use the
    eigen formula.  Both are returned so the paper can show they agree.
    """
    S = len(sigma_grid)
    emp = np.zeros((H, S))
    ana = np.zeros((H, S))
    for h in range(H):
        cols = code_slice(h)
        Ztr = np.asarray(W_train[:, cols, :], dtype=np.float64)
        Ztr = Ztr.reshape(len(Ztr), -1)
        Zev = np.asarray(W_eval[:, cols, :], dtype=np.float64)
        Zev = Zev.reshape(len(Zev), -1)
        d = Ztr.shape[1]
        mu = Ztr.mean(axis=0)
        Zc = Ztr - mu
        Sigma = (Zc.T @ Zc) / len(Zc)
        evals, V = np.linalg.eigh(Sigma)
        evals = np.clip(evals, 0.0, None)
        Zev_c = Zev - mu
        C = Zev_c @ V  # coordinates in the eigenbasis
        for s, sigma in enumerate(sigma_grid):
            s2 = float(sigma) ** 2
            gain = evals / (evals + s2)           # Wiener gain per eigen-mode
            ana[h, s] = float(np.mean(s2 * evals / (evals + s2)))
            # empirical: add noise, filter, measure residual
            rng = np.random.default_rng((7, h, s))
            noise = rng.standard_normal(Zev.shape) * float(sigma)
            Y = Zev_c + noise
            z_hat = (Y @ V) * gain @ V.T
            res = Zev_c - z_hat
            emp[h, s] = float(np.mean(res ** 2))
    return emp.astype(np.float32), ana.astype(np.float32)


def train_deprotector(W_train: np.ndarray, sigma_lo: float, sigma_hi: float,
                      steps: int = 2000, batch: int = 128, hidden: int = 1024,
                      lr: float = 3e-4, seed: int = 0, log_every: int = 250,
                      seconds: float | None = None,
                      ckpt_path: str | Path | None = None,
                      log=print):
    """Train the sigma-conditioned deprotector on real w codes.

    Mirrors sdpa.train.train but with measured latents and per-sample sigma
    drawn log-uniformly in [sigma_lo, sigma_hi] (single network serving the
    whole budget curve).  Resumable: with ``ckpt_path`` the optimiser state
    and step counter are checkpointed after every slice; ``seconds`` bounds
    the wall clock of one call.  Returns (shaper, deprotector, history, done).
    The shaper is rebuilt deterministically from ``seed`` every call.

    NOTE: retained as the neural reference for GPU reruns.  The reported
    kappa curves come from ``wiener_kappa_curves`` (exact linear MMSE);
    on this CPU budget the neural net does not beat the identity map.
    """
    import torch

    from .train import TrainConfig, build_modules

    cfg = TrainConfig(n_components=H, d_shared=D_SHARED, d_local=D_LOCAL,
                      hidden=hidden, lr=lr, steps=steps, batch=batch, seed=seed)
    shaper, deprotector = build_modules(cfg, sigma=np.full(H, 0.3))
    opt = torch.optim.AdamW(deprotector.parameters(), lr=lr, weight_decay=0.01)

    start = 0
    history: list[dict] = []
    ckpt = Path(ckpt_path) if ckpt_path else None
    if ckpt and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        deprotector.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"]
        history = state.get("history", [])
        log(f"[kappa] resumed at step {start}/{steps}")

    Z = torch.from_numpy(W_train.reshape(len(W_train), -1).astype(np.float32))
    gen = torch.Generator().manual_seed(seed + 10_000)
    log_lo, log_hi = math.log(sigma_lo), math.log(sigma_hi)

    t0 = time.time()
    done = start >= steps
    last = start - 1
    for step in range(start, steps):
        last = step
        idx = torch.randint(0, len(Z), (batch,), generator=gen)
        z = Z[idx]
        u = torch.rand(batch, 1, generator=gen)
        log_sigma = log_lo + (log_hi - log_lo) * u
        sigma = torch.exp(log_sigma)
        n = torch.randn(z.shape, generator=gen) * sigma
        y = shaper.inverse(shaper(z + n))
        z_hat = deprotector(y, log_sigma)
        loss = torch.mean((z - z_hat) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss)})
            log(f"[kappa] step {step:04d} mse={float(loss):.5f}")
        if seconds is not None and time.time() - t0 > seconds:
            break
    else:
        done = True

    if ckpt:
        torch.save({"step": last + 1, "model": deprotector.state_dict(),
                    "opt": opt.state_dict(), "history": history}, ckpt)
    return shaper, deprotector, history, done


def kappa_sweep(shaper, deprotector, W_eval: np.ndarray, sigma_grid: np.ndarray,
                n_eval: int = 1024, seed: int = 0) -> np.ndarray:
    """Per-block residual curves: MSE(z_h - zhat_h) with noise only on block h.

    Returns float32 array (H, S).
    """
    import torch

    dims = [D_SHARED, D_SHARED] + [D_LOCAL] * N_LOCAL
    offsets = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    Z = torch.from_numpy(
        W_eval[:n_eval].reshape(n_eval, -1).astype(np.float32))
    curves = np.zeros((H, len(sigma_grid)), dtype=np.float32)
    with torch.no_grad():
        for h in range(H):
            lo, hi = offsets[h], offsets[h + 1]
            for s, sigma in enumerate(sigma_grid):
                gen = torch.Generator().manual_seed(
                    (seed * 1_000_003 + h * 10_007 + int(s)) % (2 ** 63))
                n = torch.zeros_like(Z)
                n[:, lo:hi] = torch.randn(n_eval, hi - lo, generator=gen) * float(sigma)
                # shaper.inverse(shaper(x)) == x exactly (orthogonal, unit
                # scale); the 14336^2 round-trip is skipped, not approximated.
                y = Z + n
                log_sigma = torch.full((n_eval, 1), math.log(float(sigma)))
                z_hat = deprotector(y, log_sigma)
                res = (Z - z_hat)[:, lo:hi]
                curves[h, s] = float(torch.mean(res ** 2))
    return curves


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_calibration(W: np.ndarray, sigma_grid: np.ndarray,
                      lpips_curves: np.ndarray, residual_curves: np.ndarray,
                      scenario: str, meta_extra: dict | None = None):
    """Calibration with measured g/df/w/kappa in the paper's 15-block layout."""
    from .calibrate import (
        Calibration, fit_kappa, fit_utility_weight,
    )

    g, df = source_stats_blocks(W)
    w = np.array([fit_utility_weight(sigma_grid, lpips_curves[h]) for h in range(H)])
    kappa = np.array([fit_kappa(sigma_grid, residual_curves[h]) for h in range(H)])

    mask = np.ones(H, dtype=bool)
    if scenario == "task_decoupled":
        mask = np.zeros(H, dtype=bool)
        mask[list(LATENT_IDENTITY)] = True
    elif scenario != "reconstruction":
        raise ValueError("scenario must be reconstruction or task_decoupled")

    meta = {
        "source": f"semanticstylegan-latent-measured-{scenario}",
        "generator": "SemanticStyleGAN CelebAMask-HQ-512x512 (release 1.0.0)",
        "latent": "w codes, prior pushed through the real mapping network",
        "n_samples": int(len(W)),
        "block_names": BLOCK_NAMES,
        "measured": ["g", "df", "w", "kappa"],
        "pending": ["attack P_e (needs GPU-scale inversion of real identities)"],
        "note": "g/df from the exact w-code prior; w from generator-decode "
                "LPIPS sweep; kappa from the trained reference deprotector.",
    }
    if meta_extra:
        meta.update(meta_extra)
    return Calibration(g=g, df=df, w=w, kappa=kappa, d=block_dims(),
                       private_mask=mask, n_samples=len(W), meta=meta)
