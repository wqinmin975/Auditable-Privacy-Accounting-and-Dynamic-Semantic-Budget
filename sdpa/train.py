"""Reference PyTorch skeleton: true-DP noise + secret invertible shaper + deprotector.

Design contract (Theorem 1 of the accompanying section)
------------------------------------------------------
* The *randomness* is genuine Gaussian DP noise sampled per component with the
  allocated scale sigma_h from ``allocate.waterfill_sigma`` / ``solve_*``.
* The *sha**per* T is deterministic, data independent, invertible (block-diagonal
  orthogonal + per-component scaling) and kept as a per-session secret.  Because
  T never touches the raw latent, the release T(Z + n) is post-processing of a
  DP output and therefore inherits the same (eps, delta) / RDP / GDP guarantee,
  while remaining invertible for the legitimate receiver.
* The deprotector is conditioned on log(sigma) so that a *single* network serves
  the whole privacy-budget curve (no retraining per eps).

Nothing here needs a GPU to smoke-test: ``python -m sdpa.train --smoke`` runs
two steps on random latents.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainConfig:
    n_components: int = 15
    d_shared: int = 512
    d_local: int = 1024
    hidden: int = 2048
    lr: float = 3e-4
    steps: int = 200
    batch: int = 64
    latent_mse_weight: float = 1.0
    leakage_weight: float = 1e-3
    seed: int = 0
    device: str = "cpu"


def _dims(cfg: TrainConfig) -> list[int]:
    return [cfg.d_shared, cfg.d_shared] + [cfg.d_local] * (cfg.n_components - 2)


def build_modules(cfg: TrainConfig, sigma: np.ndarray, seed: int | None = None):
    """Return (shaper, deprotector) torch modules; torch is imported lazily."""
    import torch
    import torch.nn as nn

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed if seed is None else seed)

    class SecretShaper(nn.Module):
        """Block-diagonal orthogonal rotation + diagonal scaling, per component.

        Deterministic and data independent. Held as a *session secret*; it is a
        buffer (not a parameter) so gradients never leak it into the optimiser.
        """

        def __init__(self, dims: list[int]):
            super().__init__()
            blocks = []
            for d in dims:
                q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
                blocks.append(q)
            self.register_buffer("blocks", torch.block_diag(*blocks))
            scale = torch.cat([torch.ones(d) for d in dims])
            self.register_buffer("scale", scale)

        @property
        def dim(self) -> int:
            return int(self.blocks.shape[0])

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            y = z @ self.blocks.t()
            return y * self.scale

        def inverse(self, y: torch.Tensor) -> torch.Tensor:
            return (y / self.scale) @ self.blocks

    class Deprotector(nn.Module):
        """Conditioned residual denoiser: input [y_private, log(sigma)]."""

        def __init__(self, in_dim: int, hidden: int):
            super().__init__()
            self.fc1 = nn.Linear(in_dim + 1, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, in_dim)
            self.film = nn.Linear(1, 2 * hidden)
            self.act = nn.SiLU()

        def forward(self, y: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
            h = self.act(self.fc1(torch.cat([y, log_sigma], dim=-1)))
            g, b = self.film(log_sigma).chunk(2, dim=-1)
            h = self.act(self.fc2(h) * (1.0 + g) + b)
            return y - self.fc3(self.act(h))  # residual correction

    dims = _dims(cfg)
    shaper = SecretShaper(dims)
    deprotector = Deprotector(shaper.dim, cfg.hidden)
    return shaper, deprotector


def sample_noise(sigma: np.ndarray, dims: list[int], n: int, seed: int = 0):
    """Per-component Gaussian DP noise, shape (n, sum(dims))."""
    import torch

    g = torch.Generator().manual_seed(seed)
    parts = []
    for d, s in zip(dims, sigma):
        parts.append(torch.randn(n, int(d), generator=g) * float(s))
    return torch.cat(parts, dim=1)


def expected_release_snr(sigma: np.ndarray, g: np.ndarray) -> float:
    """Per-component release SNR in dB, g_h / sigma_h^2 (channel noise excluded)."""
    return float(np.mean(10.0 * np.log10(np.asarray(g) / np.asarray(sigma) ** 2)))


def train(cfg: TrainConfig | None = None, sigma: np.ndarray | None = None, log_every: int = 50):
    """Two-stage reference loop (legitimate network only; Eve never cooperates)."""
    import torch

    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    dims = _dims(cfg)
    if sigma is None:
        sigma = np.full(cfg.n_components, 0.35)  # placeholder, replace with allocator output
    sigma = np.asarray(sigma, dtype=float)
    assert len(sigma) == cfg.n_components, "sigma length must equal n_components"

    shaper, deprotector = build_modules(cfg, sigma)
    shaper.to(cfg.device)
    deprotector.to(cfg.device)
    opt = torch.optim.AdamW(deprotector.parameters(), lr=cfg.lr, weight_decay=0.01)

    # Synthetic latents stand in for GAN inversion output (shape (n, sum(dims)).
    base = torch.randn(cfg.batch, shaper.dim) * torch.cat(
        [torch.full((d,), 1.0 if d == cfg.d_shared else 0.74, dtype=torch.float) for d in dims]
    )
    log_sigma = torch.full((cfg.batch, 1), float(np.log(np.mean(sigma))))

    history = []
    for step in range(cfg.steps):
        z = base + 0.02 * torch.randn_like(base)
        n = sample_noise(sigma, dims, cfg.batch, seed=cfg.seed + step)
        released = shaper(z + n)
        # legitimate receiver: undo the secret shaper, then de-noise
        y = shaper.inverse(released)
        z_hat = deprotector(y, log_sigma)

        loss_util = torch.mean((z - z_hat) ** 2) * cfg.latent_mse_weight
        # leakage proxy: the network must not undo more than the allocated noise
        loss_leak = torch.mean((z_hat - y) ** 2) / (torch.mean(n ** 2) + 1e-12)
        loss = loss_util + cfg.leakage_weight * loss_leak

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == cfg.steps - 1:
            rec = {"step": step, "loss": float(loss), "util": float(loss_util), "leak": float(loss_leak)}
            history.append(rec)
            print(f"[step {step:04d}] loss={rec['loss']:.4e} util={rec['util']:.4e} leak={rec['leak']:.4e}")
    return {"history": history, "sigma": sigma.tolist(),
            "release_snr_db": expected_release_snr(sigma, np.array([0.74] * cfg.n_components))}


def _cli() -> None:
    p = argparse.ArgumentParser(description="Reference training skeleton (synthetic latents).")
    p.add_argument("--smoke", action="store_true", help="2 steps, tiny model")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--calibration", type=str, default=None, help="calibration json for sigma")
    args = p.parse_args()
    sigma = None
    if args.calibration:
        from .calibrate import Calibration
        from .allocate import solve_mi_budget

        cal = Calibration.load(args.calibration)
        sigma = solve_mi_budget(cal).sigma
    cfg = TrainConfig(steps=(2 if args.smoke else args.steps), hidden=(256 if args.smoke else 2048),
                       batch=(8 if args.smoke else 64))
    train(cfg, sigma=sigma, log_every=1 if args.smoke else 50)


if __name__ == "__main__":
    _cli()
