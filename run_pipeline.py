"""One-click pipeline: calibrate -> allocate -> report -> figures.

    python run_pipeline.py                          # synthetic calibration, tau = 0.9
    python run_pipeline.py --calibration outputs/calibration.json
    python run_pipeline.py --tau 0.95 --eps 100 --no-figures

Outputs
-------
outputs/allocation_summary.csv   one row per accounting scheme
outputs/allocation_sigma.csv     per-component noise std, one column per scheme
outputs/summary.md               human-readable table (paste into the paper)
figures/dat/*.dat, figures/pdf/*.pdf, figures/png/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdpa.allocate import (
    solve_basic_budget,
    solve_gdp_budget,
    solve_mi_budget,
    solve_rdp_budget,
    static_index_allocation,
)
from sdpa.calibrate import (
    Calibration,
    synthetic_calibration,
    reconstruction_calibration,
    task_decoupled_calibration,
)
from sdpa.figures import build_all

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", type=str, default=None,
                   help="path to a calibration json (overrides --scenario)")
    p.add_argument("--scenario", type=str, default="synthetic",
                   choices=["synthetic", "reconstruction", "task_decoupled"],
                   help="which built-in calibration to use when no json is given")
    p.add_argument("--tau", type=float, default=0.90, help="certified identification error floor")
    p.add_argument("--eps", type=float, default=100.0, help="epsilon used for the DP-accounting comparison")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--n-identities", type=int, default=10_000)
    p.add_argument("--static-k", type=int, default=4)
    p.add_argument("--outdir", type=str, default=str(ROOT / "outputs"))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. calibration -------------------------------------
    if args.calibration and Path(args.calibration).exists():
        cal = Calibration.load(args.calibration)
        print(f"[calibrate] loaded {args.calibration}  (H={cal.H})")
    else:
        factory = {"synthetic": synthetic_calibration,
                   "reconstruction": reconstruction_calibration,
                   "task_decoupled": task_decoupled_calibration}[args.scenario]
        cal = factory()
        cal.save(out / "calibration.json")
        print(f"[calibrate] synthetic anchor written to {out / 'calibration.json'}")
    print(f"[calibrate] global sensitivity = {cal.global_sensitivity:.2f}  "
          f"(per-component min = {cal.df.min():.2f}, inflation x{cal.sensitivity_inflation:.2f})")

    # ---------------- 2. allocation --------------------------------------
    allocs = [
        solve_mi_budget(cal, tau=args.tau, n_identities=args.n_identities),
        solve_rdp_budget(cal, args.eps, args.delta),
        solve_gdp_budget(cal, args.eps, args.delta),
        solve_basic_budget(cal, args.eps, args.delta),
        static_index_allocation(cal, n_protected=args.static_k),
    ]

    rows = [a.to_row() for a in allocs]
    import csv

    keys = ["scheme", "lpips_bob", "leakage_bits", "fano_floor", "rdp_eps", "rdp_alpha",
            "gdp_mu", "gdp_eps", "noise_power", "noise_power_frac", "sparsity", "mu", "budget", "delta"]
    with (out / "allocation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    with (out / "allocation_sigma.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["h", *[a.scheme for a in allocs]])
        for h in range(cal.H):
            w.writerow([h, *[f"{a.sigma[h]:.6g}" for a in allocs]])

    md = ["| scheme | Bob LPIPS | leakage (bit) | Fano floor | RDP (eps@alpha) | GDP (mu, eps) | noise/g | sparsity |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {scheme} | {lpips_bob:.4f} | {leakage_bits:.3f} | {fano_floor:.3f} | "
                  "{rdp_eps:.1f}@{rdp_alpha:.2f} | {gdp_mu:.2f}, {gdp_eps:.1f} | "
                  "{noise_power_frac:.3f} | {sparsity:.3f} |".format(**{k: (float('nan') if v is None else v)
                                                                       for k, v in r.items()}))
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---------------- 3. figures -----------------------------------------
    if not args.no_figures:
        build_all(eps=float(args.eps))
        print("[figures] figures/dat, figures/pdf, figures/png updated")

    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
