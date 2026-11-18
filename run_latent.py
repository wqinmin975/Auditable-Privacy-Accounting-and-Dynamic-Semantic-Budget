"""Latent-level measured validation on the real SemanticStyleGAN generator.

Stages (each resumable, outputs under outputs/measured_latent/):

    python run_latent.py --stage sample     # w codes via the real mapping net
    python run_latent.py --stage stats      # g_h / Delta f_h on 65k w codes
    python run_latent.py --stage kappa      # train deprotector, sweep kappa_h
    python run_latent.py --stage sweep --seconds 285   # LPIPS w_h sweep (repeat)
    python run_latent.py --stage finalize   # calibration + allocation + tables

The sweep advances in bounded wall-clock slices and checkpoints after every
(block, sigma) pair; repeat the command until it prints SWEEP_COMPLETE.

What is measured on the real model: g_h, Delta f_h (exact w-code prior),
w_h (generator decode + LPIPS), kappa_h (trained reference deprotector).
What remains pending: the identification attack P_e, which needs GPU-scale
inversion of real identities plus CelebA identity labels (see README).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from sdpa.allocate import (
    solve_basic_budget,
    solve_gdp_budget,
    solve_mi_budget,
    solve_rdp_budget,
    static_index_allocation,
    rdp_epsilon_delta,
)
from sdpa.latent import (
    BLOCK_NAMES,
    DEFAULT_CKPT,
    H,
    UtilitySweep,
    advance_utility_sweep,
    build_calibration,
    load_generator,
    sample_w_codes,
    source_stats_blocks,
    wiener_kappa_curves,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "measured_latent"

N_STATS = 65_536       # samples for g / Delta f
N_SWEEP = 16           # samples decoded per (block, sigma) pair
N_KAPPA_TRAIN = 16_384
N_KAPPA_EVAL = 1_024
SIGMA_GRID = np.array([0.05, 0.10, 0.20, 0.35, 0.55, 0.80])


def _paths():
    return {
        "W_stats": OUT / "W_stats.npy",
        "W_kappa": OUT / "W_kappa.npy",
        "sweep": OUT / "utility_sweep.npz",
        "curves": OUT / "curves.npz",
        "kappa_model": OUT / "deprotector.pt",
    }


def stage_sample(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = _paths()
    g = load_generator(DEFAULT_CKPT, threads=args.threads)
    t0 = time.time()
    if not p["W_stats"].exists():
        W = sample_w_codes(g, N_STATS, seed=11, batch=2048)
        np.save(p["W_stats"], W.astype(np.float32))
        print(f"W_stats {W.shape} saved ({time.time()-t0:.0f}s)")
    if not p["W_kappa"].exists():
        W = sample_w_codes(g, N_KAPPA_TRAIN + N_KAPPA_EVAL, seed=23, batch=2048)
        np.save(p["W_kappa"], W.astype(np.float32))
        print(f"W_kappa {W.shape} saved ({time.time()-t0:.0f}s)")


def stage_stats(args) -> None:
    from sdpa.latent import block_dims

    p = _paths()
    W = np.load(p["W_stats"])
    g, df = source_stats_blocks(W)
    dims = block_dims()
    with (OUT / "source_stats.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["block", "name", "g_trace", "df_clipped", "d"])
        for h in range(H):
            wr.writerow([h, BLOCK_NAMES[h], f"{g[h]:.5f}", f"{df[h]:.5f}",
                         int(dims[h])])
    print(f"g_h range [{g.min():.1f}, {g.max():.1f}], "
          f"df_h range [{df.min():.1f}, {df.max():.1f}]")
    std = np.sqrt(g / dims)
    print(f"per-coordinate std range [{std.min():.3f}, {std.max():.3f}]")


def stage_kappa(args) -> None:
    p = _paths()
    W_stats = np.load(p["W_stats"])
    W_kappa = np.load(p["W_kappa"])
    W_eval = W_kappa[N_KAPPA_TRAIN:]
    emp, ana = wiener_kappa_curves(W_stats, W_eval, SIGMA_GRID)
    if p["curves"].exists():
        d = dict(np.load(p["curves"]))
    else:
        d = {}
    d.update(residual_curves=emp, residual_curves_analytic=ana,
             sigma_grid=SIGMA_GRID)
    np.savez(p["curves"], **d)
    print("kappa_h (empirical Wiener residual / sigma^2):")
    print(np.round(emp / SIGMA_GRID[None, :] ** 2, 3))
    print("kappa_h (analytic eigen formula):")
    print(np.round(ana / SIGMA_GRID[None, :] ** 2, 3))
    print("KAPPA_COMPLETE")


def stage_sweep(args) -> None:
    import lpips as lpips_lib

    p = _paths()
    tag = getattr(args, "tag", "") or ""
    ckpt = OUT / f"utility_sweep{tag}.npz"
    blocks = None
    if getattr(args, "blocks", None):
        blocks = [int(x) for x in args.blocks.split(",")]
    g = load_generator(DEFAULT_CKPT, threads=args.threads)
    metric = lpips_lib.LPIPS(net="vgg", verbose=False)
    sweep = UtilitySweep.load(ckpt, SIGMA_GRID, N_SWEEP, seed=3)
    done = advance_utility_sweep(g, sweep, metric, seconds=args.seconds,
                                 batch=args.batch, blocks=blocks)
    print("WORKER_COMPLETE" if done else
          f"WORKER_PENDING {len(sweep.pending)} pairs left in {ckpt.name}")


def merge_sweep_files() -> np.ndarray:
    """Merge per-worker sweep checkpoints into a single (H, S, N) array."""
    files = sorted(OUT.glob("utility_sweep*.npz"))
    if not files:
        raise FileNotFoundError("no utility_sweep*.npz under outputs/measured_latent")
    total = np.full((H, len(SIGMA_GRID), N_SWEEP), np.nan, dtype=np.float32)
    for f in files:
        d = np.load(f)
        m = ~np.isnan(d["lpips"])
        total[m] = d["lpips"][m]
    return total


def stage_merge(args) -> None:
    p = _paths()
    lp = merge_sweep_files()
    n_pending = int(np.isnan(lp).any(axis=2).sum())
    if n_pending:
        print(f"MERGE_PENDING {n_pending} pairs still missing")
        return
    if p["curves"].exists():
        d = dict(np.load(p["curves"]))
    else:
        d = {}
    d.update(lpips_curves=np.nanmean(lp, axis=2), lpips_per_sample=lp,
             sigma_grid=SIGMA_GRID)
    np.savez(p["curves"], **d)
    print("SWEEP_COMPLETE")


def _rows(cal, tau, eps, delta, n_ids, static_k):
    return [
        solve_mi_budget(cal, tau=tau, n_identities=n_ids),
        solve_rdp_budget(cal, eps, delta),
        solve_gdp_budget(cal, eps, delta),
        solve_basic_budget(cal, eps, delta),
        static_index_allocation(cal, n_protected=static_k),
    ]


def stage_finalize(args) -> None:
    p = _paths()
    d = np.load(p["curves"])
    sigma_grid = d["sigma_grid"]
    lpips_curves = d["lpips_curves"]
    residual_curves = d["residual_curves"]
    W = np.load(p["W_stats"])

    meta_extra = {"sigma_grid": sigma_grid.round(4).tolist(),
                  "n_sweep": N_SWEEP, "n_kappa_train": N_KAPPA_TRAIN}
    keys = ["scheme", "lpips_bob", "leakage_bits", "fano_floor", "rdp_eps", "rdp_alpha",
            "gdp_mu", "gdp_eps", "noise_power", "noise_power_frac", "sparsity",
            "mu", "budget", "delta"]

    md = ["# Latent-level measured validation (SemanticStyleGAN CelebAMask-HQ)\n",
          "All four scalars are measured on the real generator:\n",
          "- `g_h`, `Δf_h`: exact w-code prior (mapping network pushforward, "
          f"N={len(W)})\n",
          f"- `w_h`: decode+LPIPS sweep (N={N_SWEEP}, S={len(sigma_grid)} per block)\n",
          "- `κ_h`: per-block linear-MMSE (Wiener) deprotector; covariance "
          f"measured on N={len(W)} codes, residual verified on {N_KAPPA_EVAL} "
          "held-out codes (empirical = analytic to 3 decimals)\n",
          "\nPending: identification attack P_e (GPU-scale inversion + identity labels).\n"]

    cals = {}
    audit_lines = []
    for scenario in ("reconstruction", "task_decoupled"):
        cal = build_calibration(W, sigma_grid, lpips_curves, residual_curves,
                                scenario, meta_extra=meta_extra)
        cals[scenario] = cal
        cal.save(OUT / f"calibration_{scenario}.json")
        rows = [a.to_row() for a in _rows(cal, args.tau, args.eps, args.delta,
                                          args.n_identities, args.static_k)]
        with (OUT / f"allocation_summary_{scenario}.csv").open(
                "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k) for k in keys})

        md.append(f"## Scenario: {scenario} "
                  f"(task overlap {cal.task_overlap:.3f}, |P|={int(cal.private.sum())})\n")
        md.append("| scheme | Bob LPIPS | leakage (bit) | Fano floor | RDP (eps@alpha) | "
                  "GDP (mu, eps) | noise/g | sparsity |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append("| {scheme} | {lpips_bob:.4f} | {leakage_bits:.3f} | {fano_floor:.3f} | "
                      "{rdp_eps:.1f}@{rdp_alpha:.2f} | {gdp_mu:.2f}, {gdp_eps:.1f} | "
                      "{noise_power_frac:.3f} | {sparsity:.3f} |".format(
                          **{k: (float("nan") if v is None else v) for k, v in r.items()}))
        md.append("")

        aud_alloc = solve_gdp_budget(cal, args.audit_eps, args.delta)
        aud = aud_alloc.audit_l1()
        m = cal.private
        mu_h = cal.df[m] / aud_alloc.sigma[m]
        rdp_eps, _ = rdp_epsilon_delta(aud_alloc.sigma, cal.df, args.delta, mask=m)
        audit_lines.append(
            f"- **{scenario}** (|P|={int(m.sum())}): μ_h≈{np.mean(mu_h):.1f}, "
            f"RDP-converted ε={rdp_eps:.1f}, quadrature ε={args.audit_eps:.0f}, "
            f"amplitude-only ℓ₁ ε={aud['eps_stylized']:.1f}, "
            f"exact tail ε={aud['eps_exact']:.1f}.")

    md.append("## Audit at ε=%.0f, δ=1e-5\n" % args.audit_eps)
    md += audit_lines
    md.append("")

    # per-block measured scalars table
    md.append("## Measured per-block scalars\n")
    md.append("| block | name | g_h | Δf_h | w_h | κ_h |")
    md.append("|---|---|---|---|---|---|")
    cal = cals["reconstruction"]
    for h in range(H):
        md.append(f"| {h} | {BLOCK_NAMES[h]} | {cal.g[h]:.2f} | {cal.df[h]:.2f} | "
                  f"{cal.w[h]:.4f} | {cal.kappa[h]:.3f} |")
    md.append("")

    from sdpa.figures import fig_latent_measured
    fig_latent_measured(OUT, p["curves"], cals["reconstruction"], BLOCK_NAMES)

    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nwritten to", OUT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["sample", "stats", "kappa", "sweep", "merge",
                            "finalize", "all"])
    p.add_argument("--seconds", type=float, default=285.0,
                   help="wall-clock budget for --stage sweep")
    p.add_argument("--blocks", type=str, default=None,
                   help="comma-separated block subset for parallel sweep workers")
    p.add_argument("--tag", type=str, default="",
                   help="checkpoint tag for parallel sweep workers")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--kappa-steps", type=int, default=2000)
    p.add_argument("--tau", type=float, default=0.90)
    p.add_argument("--eps", type=float, default=100.0)
    p.add_argument("--audit-eps", type=float, default=2000.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--n-identities", type=int, default=10_000)
    p.add_argument("--static-k", type=int, default=4)
    args = p.parse_args()

    stages = {"sample": stage_sample, "stats": stage_stats, "kappa": stage_kappa,
              "sweep": stage_sweep, "merge": stage_merge,
              "finalize": stage_finalize}
    if args.stage == "all":
        for name in ("sample", "stats", "kappa", "sweep", "finalize"):
            stages[name](args)
    else:
        stages[args.stage](args)


if __name__ == "__main__":
    main()
