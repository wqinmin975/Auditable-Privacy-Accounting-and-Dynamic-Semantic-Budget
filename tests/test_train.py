"""Smoke tests for the PyTorch skeleton (kept tiny so CI stays fast)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sdpa.train import TrainConfig, build_modules, train  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> TrainConfig:
    return TrainConfig(n_components=4, d_shared=16, d_local=32, hidden=64, batch=6, steps=3)


def test_shaper_is_orthogonal_and_invertible(cfg):
    sigma = np.full(cfg.n_components, 0.3)
    shaper, _ = build_modules(cfg, sigma, seed=7)
    B = shaper.blocks
    eye = torch.eye(B.shape[0])
    assert torch.allclose(B @ B.t(), eye, atol=1e-6), "shaper must be orthogonal"

    y = torch.randn(5, B.shape[0])
    assert torch.allclose(shaper.inverse(shaper(y)), y, atol=1e-6), "inverse must recover"


def test_deprotector_conditioning_changes_output(cfg):
    sigma = np.full(cfg.n_components, 0.3)
    _, dep = build_modules(cfg, sigma, seed=7)
    y = torch.zeros(4, dep.fc1.in_features - 1)
    out_small = dep(y, torch.full((4, 1), -2.0))
    out_large = dep(y, torch.full((4, 1), 2.0))
    assert not torch.allclose(out_small, out_large), "log sigma must modulate the deprotector"


def test_training_decreases_loss(cfg):
    cfg.steps = 60
    cfg.lr = 3e-3
    res = train(cfg, sigma=np.full(cfg.n_components, 0.3), log_every=1000)
    first, last = res["history"][0]["loss"], res["history"][-1]["loss"]
    assert last < first, "optimisation must make progress on synthetic latents"
