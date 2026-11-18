"""Plumbing tests for the latent-level measurement chain (no generator needed)."""

from __future__ import annotations

import numpy as np

from sdpa.latent import (
    H,
    N_CODES,
    CODE_DIM,
    UtilitySweep,
    block_dims,
    build_calibration,
    code_slice,
    source_stats_blocks,
)


def _fake_W(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, N_CODES, CODE_DIM), dtype=np.float32)
    W *= 1.1  # unit-ish coordinate scale like the mapped w codes
    return W


def test_block_layout():
    dims = block_dims()
    assert len(dims) == H == 15
    assert dims[0] == dims[1] == 512
    assert (dims[2:] == 1024).all()
    assert sum(code_slice(h).stop - code_slice(h).start for h in range(H)) == 28
    # blocks partition the 28 codes without overlap
    cols = np.concatenate([np.arange(code_slice(h).start, code_slice(h).stop)
                           for h in range(H)])
    assert (np.sort(cols) == np.arange(28)).all()


def test_source_stats_matches_homogeneous_formula():
    W = _fake_W(2000)
    g, df = source_stats_blocks(W)
    # Gaussian codes: trace g_h ~= d_h, clipped range ~= 2 * 2.807 * std
    assert np.all(g > 0) and np.all(df > 0)
    assert abs(g[0] / 512 - 1.1 ** 2) < 0.15
    assert abs(g[5] / 1024 - 1.1 ** 2) < 0.15
    z995 = 2.575829  # standard-normal 0.995 quantile
    assert abs(df[0] / (np.sqrt(512) * 2 * z995 * 1.1) - 1) < 0.05


def test_utility_sweep_checkpoint_roundtrip(tmp_path):
    grid = np.array([0.1, 0.3, 0.6], dtype=np.float32)
    sw = UtilitySweep(tmp_path / "sw.npz", grid, n_samples=4, seed=1)
    sw.W = _fake_W(4)
    sw.clean = np.zeros((4, 3, 8, 8), dtype=np.uint8)
    sw.lpips[0, 0] = np.arange(4, dtype=np.float32)
    sw.save()

    sw2 = UtilitySweep.load(tmp_path / "sw.npz", grid, n_samples=4, seed=1)
    assert np.allclose(sw2.lpips[0, 0], np.arange(4))
    assert (0, 0) not in sw2.pending
    assert (0, 1) in sw2.pending
    assert not sw2.done
    sw2.lpips[:] = 1.0
    assert sw2.done


def test_build_calibration_end_to_end():
    W = _fake_W(1500)
    grid = np.array([0.05, 0.1, 0.2, 0.4])
    # synthetic but well-posed curves: quadratic in sigma^2
    lp = 0.08 + 0.9 * grid[None, :] ** 2 + 0.001 * np.arange(H)[:, None]
    res = np.tile(0.6 * grid ** 2, (H, 1))
    for scenario in ("reconstruction", "task_decoupled"):
        cal = build_calibration(W, grid, lp, res, scenario)
        assert cal.g.shape == (H,)
        assert np.all(cal.w > 0)
        assert np.all((cal.kappa > 0) & (cal.kappa <= 1))
        if scenario == "task_decoupled":
            assert 0 < cal.private.sum() < H
        else:
            assert cal.private.all()
