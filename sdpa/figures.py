"""IEEE column-width figures + PGFPlots data sources.

Each figure is produced twice:
  * ``figures/pdf/<name>.pdf`` and ``figures/png/<name>.png``  (matplotlib preview)
  * ``figures/dat/<name>.dat``                                  (PGFPlots source)

``figures/figures_pgfplots.tex`` consumes the .dat files, so the camera-ready
version is vector-native and inherits the LaTeX fonts.

All numbers are *analytical*: closed-form allocator plus the surrogate decision
model documented in ``eve_identification_error``. Replace with measured curves by
calling ``write_dat`` with your own arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .allocate import (
    tau_ceiling,
    i_max_bits,
    solve_basic_budget,
    solve_gdp_budget,
    solve_mi_budget,
    solve_rdp_budget,
    static_index_allocation,
    waterfill_sigma,
)
from .calibrate import Calibration, reconstruction_calibration, task_decoupled_calibration

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
COL = 3.5
DBL = 7.16


def ieee_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6.3,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.3,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.22,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.45,
        "grid.linestyle": ":",
    })


# --------------------------------------------------------------------------- #
def eve_identification_error(sigma: np.ndarray, cal: Calibration, s_clean: float = 0.86,
                             thr: float = 0.31, sd: float = 0.055, beta: float = 2.0) -> float:
    """Surrogate ArcFace-style decision.

        s(sigma) = s_clean * exp(-beta * sum_h sigma_h^2 / g_h),  P_e = Phi((thr - s)/sd)

    Stand-in for the real attack table (see README): monotone, bounded, and never
    presented as measured data.
    """
    from scipy.stats import norm

    rel = float(np.sum(np.asarray(sigma, dtype=float) ** 2 / cal.g))
    s_mean = s_clean * np.exp(-beta * rel)
    return float(norm.cdf((thr - s_mean) / sd))


def write_dat(path, header, columns) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = [np.asarray(c, dtype=float) for c in columns]
    with path.open("w", encoding="utf-8") as f:
        f.write(" ".join(header) + "\n")
        for i in range(len(arrs[0])):
            f.write(" ".join(f"{a[i]:.6g}" for a in arrs) + "\n")
    return path


def _save(fig, name: str):
    (FIG_DIR / "pdf").mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "png").mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "pdf" / f"{name}.pdf"
    png = FIG_DIR / "png" / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return pdf, png


# --------------------------------------------------------------------------- #
# Fig. 1 : accounting law determines the budget shape
# --------------------------------------------------------------------------- #
def fig_budget_split(cal: Calibration, eps: float = 100.0, n_static: int = 4, delta: float = 1e-5):
    import matplotlib.pyplot as plt

    ieee_style()
    basic = solve_basic_budget(cal, eps, delta)
    rdp = solve_rdp_budget(cal, eps, delta)
    gdp = solve_gdp_budget(cal, eps, delta)
    static = static_index_allocation(cal, n_protected=n_static)

    frac = lambda a: a.l2_power / a.l2_power.sum()
    H, idx = cal.H, np.arange(cal.H)

    fig, ax = plt.subplots(figsize=(COL, 2.25))
    w = 0.21
    ax.bar(idx - 1.5 * w, frac(gdp), w, label="GDP", color="#1f4e79")
    ax.bar(idx - 0.5 * w, frac(rdp), w, label="RDP", color="#2e75b6")
    ax.bar(idx + 0.5 * w, frac(basic), w, label=r"basic ($\ell_1$)", color="#c55a11")
    ax.bar(idx + 1.5 * w, frac(static), w, label=f"static ($k={n_static}$)",
           color="#7f7f7f", alpha=0.9)

    ax2 = ax.twinx()
    wk = cal.w * cal.kappa
    ax2.plot(idx, wk / wk.sum(), "k-", lw=0.7, marker="o", ms=2.0,
             label=r"$w_h\kappa_h$ (norm.)")
    ax2.set_ylabel(r"$w_h\kappa_h$")
    ax2.set_ylim(0, 1.25 * float((wk / wk.sum()).max()))
    ax2.grid(False)
    ax2.tick_params(labelsize=6.2)

    ax.set_xlabel(r"semantic component $h$")
    ax.set_ylabel("share of noise power")
    ax.set_xticks(idx)
    ax.set_xlim(-0.8, H - 0.2)
    ax.set_ylim(0, 0.9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=5.8)
    fig.tight_layout(pad=0.2)

    write_dat(FIG_DIR / "dat" / "budget_split.dat",
              ["h", "gdp", "rdp", "basic", "static", "wkappa_norm"],
              [idx, frac(gdp), frac(rdp), frac(basic), frac(static), wk / wk.sum()])
    return _save(fig, "fig1_budget_split")


# --------------------------------------------------------------------------- #
# Fig. 2 : the price of a certificate
# --------------------------------------------------------------------------- #
def fig_certificate_price(taus: np.ndarray | None = None, n_ids: int = 10_000):
    """Bob's distortion vs the certified floor, with / without a null private subspace."""
    import matplotlib.pyplot as plt

    ieee_style()
    aligned = reconstruction_calibration(seed=1)
    dec = task_decoupled_calibration(seed=1)
    cap = tau_ceiling(n_ids)
    if taus is None:
        taus = np.linspace(0.35, 0.975 * cap, 24)

    curves = {}
    for tag, cal in (("aligned", aligned), ("decoupled", dec)):
        lp, pe, floor = [], [], []
        for t in taus:
            a = solve_mi_budget(cal, tau=float(t), n_identities=n_ids)
            lp.append(a.predicted_lpips())
            pe.append(eve_identification_error(a.sigma, cal))
            floor.append(a.certified_error_floor())
        curves[tag] = (np.asarray(lp), np.asarray(pe), np.asarray(floor))

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.plot(taus, curves["aligned"][0], "-s", color="#c00000", label=r"Bob, aligned $w_h$")
    ax.plot(taus, curves["decoupled"][0], "-o", color="#2e75b6", label=r"Bob, task-decoupled $w_h$")
    ax.axhline(0.112, color="k", ls=":", lw=0.6)
    ax.annotate("no-protection 0.112", xy=(taus[0], 0.112), xytext=(1.5, 5),
                textcoords="offset points", fontsize=5.5)

    ax2 = ax.twinx()
    ax2.plot(taus, curves["decoupled"][1], "--^", color="#1f4e79", lw=0.9,
             label=r"Eve $P_e$ (surrogate)")
    ax2.plot(taus, curves["decoupled"][2], ":", color="#1f4e79", lw=1.0,
             label=r"certified floor $\tau$")
    ax2.set_ylabel(r"identification error $P_e$")
    ax2.set_ylim(0.25, 1.15)
    ax2.grid(False)

    ax.set_xlabel(r"certified error floor $\tau$")
    ax.set_ylabel("Bob LPIPS")
    ax.set_ylim(0.105, 1.02)
    ax.axvline(cap, color="k", ls="-.", lw=0.6)
    ax.annotate(rf"$\tau_{{\max}}={cap:.3f}$", xy=(cap - 0.008, 0.955), ha="right",
                fontsize=5.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, fontsize=5.6)
    fig.tight_layout(pad=0.2)

    write_dat(FIG_DIR / "dat" / "certificate_price.dat",
              ["tau", "lpips_aligned", "lpips_decoupled", "p_e_surrogate",
               "certified_floor", "tau_ceiling"],
              [taus, curves["aligned"][0], curves["decoupled"][0],
               curves["decoupled"][1], curves["decoupled"][2], np.full_like(taus, cap)])
    return _save(fig, "fig2_certificate_price")


# --------------------------------------------------------------------------- #
# Fig. 3 : double-column composite
# --------------------------------------------------------------------------- #
def fig_composite(eps_list: np.ndarray | None = None, delta: float = 1e-5):
    import matplotlib.pyplot as plt

    ieee_style()
    aligned = reconstruction_calibration(seed=1)
    dec = task_decoupled_calibration(seed=1)
    if eps_list is None:
        eps_list = np.array([3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 2000.0])

    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.05))

    # (a) water-filling: one direction, three task weights
    ax = axes[0]
    mus = np.geomspace(1e-3, 1e5, 80)
    h0 = 6
    base_w = float(aligned.w[h0])
    w_list = [base_w, base_w / 8.0, base_w / 125.0]
    labels = [r"$w_h$ (task-aligned)", r"$w_h/8$", r"$w_h/125$ (task-null)"]
    colors = ["#1f4e79", "#2e75b6", "#c55a11"]
    for wv, lab, col, ls in zip(w_list, labels, colors, ["-", "--", ":"]):
        a_single = np.full(aligned.H, wv * aligned.kappa[h0])
        yy = [waterfill_sigma(m, a_single, aligned.g, np.ones(aligned.H, bool))[h0] ** 2 / aligned.g[h0]
              for m in mus]
        ax.loglog(mus, yy, color=col, lw=1.0, ls=ls, label=lab)
    ax.set_xlabel(r"dual variable $\nu$")
    ax.set_ylabel(r"$\sigma_h^{2}/g_h$")
    ax.set_title("(a) water-filling: cost of a direction")
    ax.legend(loc="lower right", fontsize=5.2)
    ax.set_ylim(1e-4, 1e8)

    # (b) audit: what each accounting law would certify at the same release
    ax = axes[1]
    eps_list = np.array([3.0, 30.0, 100.0, 300.0, 1000.0, 2000.0, 4000.0])
    snr, e_gdp, e_ex, e_sty = [], [], [], []
    for e in eps_list:
        aq = solve_gdp_budget(dec, float(e), delta)
        snr.append(-aq.release_snr_db())
        e_gdp.append(e)
        au = aq.audit_l1()
        e_ex.append(au["eps_exact"])
        e_sty.append(au["eps_stylized"])
    ax.semilogy(snr, e_gdp, "-^", color="#1f4e79", label="quadrature (GDP)")
    ax.semilogy(snr, e_ex, "-s", color="#548235", label=r"$\ell_1$ exact tail bound")
    ax.semilogy(snr, e_sty, "--o", color="#c55a11", label=r"$\ell_1$ amplitude-only")
    ax.set_xlabel("noise-to-signal ratio (dB)")
    ax.set_ylabel(r"certified $\epsilon$")
    ax.set_title(r"(b) audited $\epsilon$ at matched release")
    ax.legend(loc="upper right", fontsize=5.2)

    # (c) certificate gap
    ax = axes[2]
    taus = np.linspace(0.45, 0.985 * tau_ceiling(10_000), 18)
    gaps = []
    for t in taus:
        a = solve_mi_budget(dec, tau=float(t))
        gaps.append(eve_identification_error(a.sigma, dec) - a.certified_error_floor())
    ax.plot(taus, gaps, "-o", color="#7030a0", lw=0.9)
    ax.axhline(0.0, color="k", ls=":", lw=0.6)
    ax.set_xlabel(r"certified floor $\tau$")
    ax.set_ylabel(r"$P_e^{\rm sim}-\tau$")
    ax.set_title("(c) certificate tightness")

    fig.tight_layout(pad=0.2, w_pad=0.35)

    write_dat(FIG_DIR / "dat" / "accounting_audit.dat",
              ["nsr_db", "eps_quadrature", "eps_l1_exact", "eps_l1_stylized"],
              [np.array(snr), np.array(e_gdp), np.array(e_ex), np.array(e_sty)])
    write_dat(FIG_DIR / "dat" / "waterfill_sweep.dat",
              ["mu", "w_1", "w_8", "w_125"],
              [mus]
              + [[waterfill_sigma(m, np.full(aligned.H, wv * aligned.kappa[h0]), aligned.g,
                                  np.ones(aligned.H, bool))[h0] ** 2 / aligned.g[h0]
                 for m in mus] for wv in w_list])
    write_dat(FIG_DIR / "dat" / "certificate_gap.dat", ["tau", "gap"], [taus, np.array(gaps)])
    return _save(fig, "fig3_composite")


def fig_latent_measured(outdir, curves_npz, cal, block_names):
    """Latent-level measured evidence (SemanticStyleGAN backbone).

    Panel (a): measured utility weight w_h per block (the utility landscape).
    Panel (b): LPIPS sweep curves LPIPS(sigma^2) with the fitted quadratic
    model for representative blocks (shared, skin, hair, eyes).
    Panel (c): residual fraction kappa(sigma^2) of the linear-MMSE
    deprotector, empirical vs analytic.
    Writes PNG + PDF + .dat sources under ``outdir``.
    """
    import matplotlib.pyplot as plt

    ieee_style()
    outdir = Path(outdir)
    d = np.load(curves_npz)
    sigma_grid = d["sigma_grid"].astype(float)
    lp = d["lpips_curves"].astype(float)          # (H, S)
    res = d["residual_curves"].astype(float)      # (H, S)
    res_ana = d["residual_curves_analytic"].astype(float)
    H = len(cal.g)
    x = sigma_grid ** 2

    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.05))

    # (a) utility landscape
    ax = axes[0]
    idx = np.arange(H)
    short = [n.replace("(shape+texture)", "") for n in block_names]
    colors = ["#c00000" if cal.private[h] else "#7f7f7f" for h in range(H)]
    ax.bar(idx, cal.w, 0.75, color=colors)
    ax.set_xticks(idx)
    ax.set_xticklabels(short, rotation=60, ha="right", fontsize=5.0)
    ax.set_ylabel(r"measured $w_h$")
    ax.set_title("(a) utility landscape", fontsize=7.5)

    # (b) LPIPS sweep + quadratic fits
    ax = axes[1]
    show = [(0, "shared_0", "#c00000"), (3, "skin", "#c55a11"),
            (9, "hair", "#548235"), (4, "eyes", "#2e75b6")]
    for h, name, col in show:
        ax.plot(x, lp[h], "o", ms=2.2, color=col, label=name)
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, lp[h], rcond=None)
        xs = np.linspace(0, x.max(), 50)
        ax.plot(xs, coef[0] + coef[1] * xs, "-", lw=0.8, color=col)
    ax.set_xlabel(r"$\sigma_h^2$")
    ax.set_ylabel("LPIPS (clean vs perturbed)")
    ax.set_title("(b) block noise sweep", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=5.2)

    # (c) kappa
    ax = axes[2]
    ax.plot(x, (res / x[None, :]).mean(axis=0), "-o", ms=2.4, color="#1f4e79",
            label="empirical (held-out)")
    ax.plot(x, (res_ana / x[None, :]).mean(axis=0), "--", lw=0.9,
            color="#c55a11", label="analytic (eigen)")
    ax.set_xlabel(r"$\sigma_h^2$")
    ax.set_ylabel(r"$\kappa_h(\sigma_h^2)$")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) deprotection residual", fontsize=7.5)
    ax.legend(loc="upper right", fontsize=5.2)

    fig.tight_layout(pad=0.2, w_pad=0.35)
    (outdir / "png").mkdir(parents=True, exist_ok=True)
    (outdir / "pdf").mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "png" / "latent_measured.png")
    fig.savefig(outdir / "pdf" / "latent_measured.pdf")
    plt.close(fig)

    write_dat(outdir / "latent_utility_landscape.dat",
              ["h", "w_h", "kappa_h", "private"],
              [idx, cal.w, cal.kappa, cal.private.astype(float)])
    write_dat(outdir / "latent_sweep_examples.dat",
              ["sigma2", "shared0", "skin", "hair", "eyes"],
              [x, lp[0], lp[3], lp[9], lp[4]])
    write_dat(outdir / "latent_kappa.dat",
              ["sigma2", "kappa_empirical", "kappa_analytic"],
              [x, (res / x[None, :]).mean(axis=0),
               (res_ana / x[None, :]).mean(axis=0)])
    return outdir / "png" / "latent_measured.png"


def fig_measured(cal_rec, cal_dec, outdir, n_ids: int = 10_000, delta: float = 1e-5):
    """Measured-calibration analogue of Fig. 2 + Fig. 3(b).

    Panel (a): Bob LPIPS vs certified floor on the two measured calibrations.
    Panel (b): audit of the three accounting laws on the measured decoupled
    release. Writes .dat sources and a PNG next to ``outdir``.
    """
    import matplotlib.pyplot as plt

    ieee_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cap = tau_ceiling(n_ids)
    taus = np.linspace(0.35, 0.975 * cap, 24)

    rec_lp, dec_lp, floor, pe = [], [], [], []
    for t in taus:
        ar = solve_mi_budget(cal_rec, tau=float(t), n_identities=n_ids)
        ad = solve_mi_budget(cal_dec, tau=float(t), n_identities=n_ids)
        rec_lp.append(ar.predicted_lpips())
        dec_lp.append(ad.predicted_lpips())
        floor.append(ad.certified_error_floor())
        pe.append(eve_identification_error(ad.sigma, cal_dec))

    eps_list = np.array([3.0, 30.0, 100.0, 300.0, 1000.0, 2000.0, 4000.0])
    snr, e_gdp, e_ex, e_sty = [], [], [], []
    for e in eps_list:
        aq = solve_gdp_budget(cal_dec, float(e), delta)
        snr.append(-aq.release_snr_db())
        e_gdp.append(e)
        au = aq.audit_l1()
        e_ex.append(au["eps_exact"])
        e_sty.append(au["eps_stylized"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL, 2.05))
    ax1.plot(taus, rec_lp, "-s", color="#c00000", label=r"Bob, aligned $w_h$")
    ax1.plot(taus, dec_lp, "-o", color="#2e75b6", label=r"Bob, decoupled $w_h$")
    ax1.axhline(0.112, color="k", ls=":", lw=0.6)
    ax1.set_xlabel(r"certified floor $\tau$")
    ax1.set_ylabel("Bob LPIPS")
    ax1.set_title("(a) measured certificate price", fontsize=7.5)
    ax1.legend(loc="upper center", fontsize=5.2)

    ax2.semilogy(snr, e_gdp, "-^", color="#1f4e79", label="quadrature (GDP)")
    ax2.semilogy(snr, e_ex, "-s", color="#548235", label=r"$\ell_1$ exact tail")
    ax2.semilogy(snr, e_sty, "--o", color="#c55a11", label=r"$\ell_1$ amplitude-only")
    ax2.set_xlabel("noise-to-signal ratio (dB)")
    ax2.set_ylabel(r"certified $\epsilon$")
    ax2.set_title("(b) measured audit", fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=5.2)
    fig.tight_layout(pad=0.2, w_pad=0.35)

    (outdir / "png").mkdir(exist_ok=True)
    fig.savefig(outdir / "png" / "measured_validation.png")
    (outdir / "pdf").mkdir(exist_ok=True)
    fig.savefig(outdir / "pdf" / "measured_validation.pdf")
    plt.close(fig)

    write_dat(outdir / "measured_certificate_price.dat",
              ["tau", "lpips_aligned", "lpips_decoupled", "p_e_surrogate", "certified_floor"],
              [taus, rec_lp, dec_lp, pe, floor])
    write_dat(outdir / "measured_audit.dat",
              ["nsr_db", "eps_quadrature", "eps_l1_exact", "eps_l1_stylized"],
              [np.array(snr), np.array(e_gdp), np.array(e_ex), np.array(e_sty)])
    return outdir / "png" / "measured_validation.png"


def fig_imax_gallery() -> Path:
    """Leakage budget I_max(tau, K) versus the floor, for three gallery sizes.

    Illustrates the reviewer-requested sensitivity to K: the feasibility ceiling
    tau_max(K) and the whole budget curve grow only logarithmically in K.
    """
    import matplotlib.pyplot as plt

    ieee_style()
    Ks = [1_000, 10_000, 1_000_000]
    taus = np.linspace(0.30, 0.90, 28)
    curves = []
    for K in Ks:
        cap = tau_ceiling(K)
        curves.append([max(i_max_bits(float(t), K), 0.0) if t < cap else 0.0
                       for t in taus])

    fig, ax = plt.subplots(figsize=(COL, 2.1))
    styles = ["-", "--", ":"]
    for K, c, ls in zip(Ks, curves, styles):
        ax.plot(taus, c, ls, lw=1.0, label=rf"$K=10^{{{int(round(np.log10(K)))}}}$")
    for K in Ks:
        ax.axvline(tau_ceiling(K), color="k", ls="-.", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"certified floor $\tau$")
    ax.set_ylabel(r"$I_{\max}$ (bit)")
    ax.legend(loc="upper right", fontsize=5.6)
    fig.tight_layout(pad=0.2)

    write_dat(FIG_DIR / "dat" / "imax_gallery.dat",
              ["tau", "k1e3", "k1e4", "k1e6"],
              [taus, *curves])
    return _save(fig, "fig4_imax_gallery")


def build_all(eps: float = 100.0) -> dict:
    aligned = reconstruction_calibration(seed=1)
    dec = task_decoupled_calibration(seed=1)
    out = {}
    out["fig1"] = fig_budget_split(dec, eps=eps)
    out["fig2"] = fig_certificate_price()
    out["fig3"] = fig_composite()
    out["fig4"] = fig_imax_gallery()
    return out


if __name__ == "__main__":
    build_all()
    print("figures written to", FIG_DIR)
