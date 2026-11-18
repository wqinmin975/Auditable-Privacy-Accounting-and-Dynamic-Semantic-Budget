# Semantic DP Accounting — code, theory, and figures

Repository: https://github.com/wqinmin975/Auditable-Privacy-Accounting-and-Dynamic-Semantic-Budget

A self-contained package that turns the informal "the noise looks like Laplace, so it is
differentially private" argument of the reference scheme (arXiv:2510.23274) into a
**book-keepable** privacy budget, together with the allocation law that the budget induces and a
proof-level write-up of why certain claims cannot hold.

Three deliverables live here:

| Deliverable | File |
|---|---|
| Formal section with Theorems 1–4 and proofs | `theory/section_privacy_accounting.tex` |
| Two-column preview (compiles to 6 pages) | `theory/preview.pdf` (`pdflatex theory/preview.tex`) |
| Reproducible code skeleton + tests + one-click run | `sdpa/`, `tests/`, `run_pipeline.py` |
| IEEE-width figures + PGFPlots data sources | `figures/pdf/`, `figures/png/`, `figures/dat/`, `figures/figures_pgfplots.tex` |

---

## 1. The mathematics in one page

**Latent decomposition.** A semantic latent `z = (z_1..z_H)`, `z_h ∈ R^{d_h}`,
`n = Σ d_h`; the reference backbone uses 28 codes of `d=512` (`n = 14336`). A declared
**private subspace** `P ⊆ {1..H}` carries the semantics we promise to protect. Everything
else (`Q`) is public by construction and never noised.

**Sensitivity (Proposition 1).** With a common coordinate range `b−a =: R`,

```
Δf_glob = R √(Σ d_h)          Δf_h = R √(d_h)
```

so the whole latent is `√(Σd_h / d_h)` times more sensitive than one component
(`√28 = 5.29` for the 28-code decomposition, `3.85` for our 2+13 group structure).
Two consequences, both in the section: (i) copying the reported global
sensitivity (`Δf = 351.88` in the reference anchor) into every code's accounting
inflates the required **noise power by 28× (14.5 dB)**; (ii) that
gain is a *book-keeping correction*, not a mechanism — a correct monolithic vector
accounting already matches correct per-component quadrature accounting.

**Shaping (Theorem 1 + Remark 1).** Any `T` of the release is post-processing, hence free.
If `T` is invertible and public, the guarantee is *exactly* unchanged — the protection comes
from the noise, never from the shaping. Therefore **DP noise cannot be "learned away" for Bob
only**: a deprotection net that recovers `z` is available to Eve too. Legitimate advantage can
only come from (a) a *secret* `T` (a keyed scheme — must be stated as such, not as pure DP), or
(b) noise whose projection onto the *task* subspace is zero. This is the claim most likely to be
challenged in the reference paper and it is handled head-on in `Remark 1`.

**Certificate (Theorem 2).** For the private template `B = f(z_P)`, uniform over `K`
identities,

```
I_leak(σ) = ½ Σ_{h∈P} log(1 + g_h/σ_h²)  ≤  I_max(τ) = log2 K − 1 − τ log2(K−1)   [bit]
⇒ P_e(Êve) ≥ τ,      τ < τ_max(K) = (log2 K − 1)/log2(K−1)
```

via Gaussian max-entropy + data processing + Fano. The **ceiling is real**:
`τ_max(10⁴) = 0.9248`, `τ_max(10⁶) = 0.9498`. There is no free lunch: an arbitrarily strong
*certified* identification guarantee is impossible for a finite template bank.

**Allocation (Theorem 3).** `x_h = σ_h²`, `a_h = w_h κ_h`:

```
x_h*(ν) = (g_h/2)( √(1 + 2ν/(a_h g_h)) − 1 )
x_h ≈ ν/(2a_h)                (task-driven, x ≪ g)
x_h ≈ √(ν g_h/(2a_h))         (geometry-driven, x ≫ g)
```

Convex, unique optimum, multiplier pinned by bisection on `log ν` (the paper
denotes the KKT dual by `ν`; the privacy-loss parameter `μ_h = Δf_h/σ_h` is a
different quantity).

**Accounting decides the shape (Theorem 4).**

| law | constraint term | profile exponent | verdict |
|---|---|---|---|
| quadrature (RDP/GDP) | `Σ Δ²/σ²`, i.e. `Σ g/x` after absorbing `Δ` | `x ∝ (·/a)^1` | smooth, water-filling |
| amplitude-only `ℓ₁` | `Σ cΔ/σ` | `x ∝ (Δ/a)^{2/3}` | geometric; **only** shape a "`ε=Δf/s`" accounting can give |
| rigorous tail `ℓ₁` | `Σ(μ²/2 + cμ)` | `x ∝ Δ/√a` at `μ ≫ c` | same `μ²/2` term as quadrature |

**The `μ²/2` matters.** At the reference scheme's own operating point
(`σ ≈ 3.6`, `Δf ≈ 91`, hence `μ_h ≈ 24 ≫ c = 4.84`, `|P| = 6`), dropping that term is not cosmetic:
at a release quadrature certifies as `ε = 2000`, the amplitude-only form reports
`ε = 693` — **unsound**, i.e. *smaller* than the scheme it is being compared with — while
the rigorous bound reports `2441`. This is exactly the failure mode of `ε' = Δf/s`.

---

## 2. Layout

```
semantic_dp_accounting/
├── sdpa/
│   ├── calibrate.py     # Calibration (g, Δf, w, κ, private_mask) + 3 built-in scenarios
│   ├── allocate.py      # water-filling, MI/RDP/GDP/ℓ₁ budget solve, audits, metrics
│   ├── train.py         # reference two-stage loop: secret shaping T + deprotector
│   ├── latent.py        # SemanticStyleGAN loader, w-code stats, w/κ sweeps, block layout
│   └── figures.py       # IEEE-width figures + .dat writers
├── tests/
│   ├── test_allocate.py # KKT, duality, asymptotics, Fano, ceiling, audits, roundtrip
│   ├── test_train.py    # orthogonality, invertibility, FiLM conditioning, loss decrease
│   └── test_latent.py   # block layout, code slices, Wiener κ, sweep bookkeeping
├── theory/
│   ├── section_privacy_accounting.tex   # IEEEtran-includable section (Theorems 1–4)
│   └── preview.tex / preview.pdf        # two-column layout proof
├── figures/
│   ├── figures_pgfplots.tex             # camera-ready vector figures (reads .dat)
│   ├── dat/  pdf/  png/                 # data sources and rendered figures
├── third_party/SemanticStyleGAN/        # vendored generator (CPU fallback patches)
├── outputs/                             # summary.md, allocation_*.csv, calibration.json
│   └── measured_latent/                 # latent-level measured calibration + summary
├── run_pipeline.py                      # one-click: calibrate → allocate → report → figures
├── run_latent.py                        # latent measurement: sample/stats/kappa/sweep/finalize
└── pyproject.toml
```

### Data availability — files intentionally not in this repository

To respect GitHub's 100 MB per-file / 5 GB per-repository limits, the following
large binaries are **not** tracked (see `.gitignore`); they are either downloaded
from their original source or regenerable from the committed code and `.dat` tables:

| Not tracked | Size | How to obtain |
|---|---|---|
| `data/celebamask_hq` parquet shard | ~476 MB | `python download_data.py --shard 0` (Hugging Face `v-xchen-v/celebamask_hq`) |
| `data/pretrained/CelebAMask-HQ-512x512.pt` | ~374 MB | authors' SemanticStyleGAN GitHub release (generator code vendored in `third_party/`) |
| `outputs/measured_latent/W_kappa.npy`, `W_stats.npy` | ~1–3.75 GB | regenerated by `python run_latent.py --stage kappa` / `--stage stats` on the checkpoint above |
| `outputs/measured_latent/*.npz`, `*.png`, other `*.npy` | up to ~13 MB each | regenerated by the corresponding `run_latent.py --stage …` steps |
| raster `*.png` figures | — | vector PDFs and their `.dat` sources are tracked; regenerate via `sdpa/figures.py` |

All scalars, allocation tables, calibration JSONs, audit `.dat` files, vector PDFs,
the manuscript, and the full source tree **are** included, so the reported numbers
are reproducible from this repository plus the original public data sources.

## 3. Run it

```bash
pip install -e ".[dev]"            # numpy, scipy, matplotlib, pytest (torch optional)
python -m pytest -q                # 27 tests, all green

python run_pipeline.py                       # synthetic ≈ reconstruction scenario
python run_pipeline.py --scenario task_decoupled --tau 0.90 --eps 100
python run_pipeline.py --calibration my/calibration.json   # real, measured latents
python run_pipeline.py --no-figures          # numbers only

python run_measured.py --n 500               # CelebAMask-HQ measured calibration

pdflatex theory/preview.tex      # compile the accounting-section preview (twice)
pdflatex theory/main_paper.tex   # compile the full two-column manuscript (twice)
```

`outputs/summary.md` is the headline table; `outputs/allocation_sigma.csv` has per-component
noise; `outputs/calibration.json` round-trips the calibration.

### Expected numbers (synthetic anchor, task-decoupled, `τ=0.90`, `ε=100`, `δ=1e-5`)

| scheme | Bob LPIPS | leakage (bit) | Fano floor |
|---|---|---|---|
| MI certificate | 0.236 | 0.329 (budget met) | **0.900** |
| RDP | 0.793 | 0.007 | 0.924 |
| GDP | 0.788 | 0.007 | 0.924 |
| ℓ₁ (amplitude) | 0.853 | 0.006 | 0.924 |
| static `k=4` | 0.146 | 20.25 (**no guarantee**) | 0.000 |

The last row is the point of the whole comparison: the static-index scheme looks cheapest
precisely because it protects nothing that a certificate can see.

### Measured validation (CelebAMask-HQ, 500 faces)

`run_measured.py` decodes real images + part masks from the HF parquet copy
(`v-xchen-v/celebamask_hq`), measures per-part variance and clipped range, and
builds two calibrations. Outputs in `outputs/measured/`:

```bash
python download_data.py --shard 0   # 476 MB parquet -> data/celebamaskhq/
python run_measured.py --n 500
```

| scenario | MI Bob LPIPS | leakage (bit) | Fano floor | task overlap |
|---|---|---|---|---|
| reconstruction (identity = task) | 0.733 | 0.329 | 0.900 | 1.000 |
| task-decoupled | 0.118 | 0.329 | 0.900 | 0.013 |

Audit at quadrature ε=2000, δ=1e-5 (decoupled, |P|=6): amplitude-only ℓ₁ reports
**699**, exact tail bound **2447** — the same ordering as the synthetic anchor
(693 / 2441), reproduced on measured part statistics.

**What is measured vs surrogate.** `g_h` (part pixel variance) and `Δf_h`
(part clipped range) are measured on real masks. `w_h`, `κ_h` and the attack
`P_e` still use the documented surrogates. The 498 MB parquet shard is downloaded
to `data/celebamaskhq/` and is **not** shipped in the code archive. This
image-space proxy is superseded by the latent-level measurement below.

### Latent-level measurement (SemanticStyleGAN generator, all four scalars measured)

`run_latent.py` loads the official SemanticStyleGAN CelebAMask-HQ 512×512
checkpoint (`data/pretrained/CelebAMask-HQ-512x512.pt`, from the authors'
GitHub release; generator code vendored in `third_party/SemanticStyleGAN/`
with CPU fallbacks for the fused CUDA ops) and measures `g_h`, `Δf_h`, `w_h`,
`κ_h` on the real decomposed latent layout (2 shared codes of d=512 + 13
local part codes of d=1024):

```bash
python run_latent.py --stage sample    # 65,536 w-codes through the real mapping network
python run_latent.py --stage stats     # per-block g_h, clipped Δf_h
python run_latent.py --stage kappa     # per-block Wiener deprotector residual curves
python run_latent.py --stage sweep     # decode+LPIPS per-block noise sweeps (N=16, 6 sigmas)
python run_latent.py --stage finalize  # calibration + allocations + tables + figures
```

The sweep stage supports `--blocks`, `--tag` and `--seconds` for resumable
parallel shards. Outputs in `outputs/measured_latent/`: `calibration.json`
(feed with `run_pipeline.py --calibration outputs/measured_latent/calibration.json`),
`summary.md`, and the measured figure `png/latent_measured.png` plus PGFPlots
sources in `figures/dat/latent_*.dat`. All stages run on CPU.

Headline measured readings: per-coordinate std ≈ 0.697 with a clipped range
≈ 4.32 (heavier-tailed than the Gaussian 3.59); `w_h` concentrates on the
shared/hair/skin/background blocks while the facial-feature blocks are nearly
free to protect (`w_eyes = 0.0097`); the Wiener residual matches its analytic
prediction to three decimals (`κ_h ≈ 0.24`); the identity-relevant private set
keeps a task overlap of 0.590 under the LPIPS task; and the audit ordering
(amplitude-only 952.3 < quadrature 2000 < RDP 2037.8 < exact tail 2700.2 at
ε=2000, reconstruction) reproduces on the measured scalars.

**Still pending:** the identification attack `P_e` — requires GPU-scale
inversion of a labelled identity corpus (CelebA identity join) and a trained
recognition head; the certificate floor and surrogate `P_e` are reported
until then. Generator-sampled codes carry no real identity, so the released
calibration data raises no subject-privacy concern.

## 4. Plugging in real calibration

```python
from sdpa import calibrate_from_latents, Calibration, solve_mi_budget

cal = calibrate_from_latents(latents, private_mask=mask)   # (N, H, d_h) or list of blocks
cal.save("my/calibration.json")
alloc = solve_mi_budget(cal, tau=0.90, n_identities=10_000)
print(alloc.summary(), alloc.audit_l1())
```

`w_h` is the calibrated sensitivity of the task loss to component `h` (finite differences of
LPIPS or of the task head), `κ_h` maps latent power to distortion units, `g_h` is the measured
per-component variance, and `private_mask` declares what you promise to protect.

## 5. Caveats you must carry into the paper

1. **No free lunch** — `τ_max(K)` in Theorem 2 is a hard ceiling; report it next to every floor.
2. **The certificate is attack-specific** (identification). It is strictly weaker than DP and
   strictly stronger than an empirically measured drop in attack accuracy.
3. **The price of a certificate is the task–private overlap** — Fig. 2: with `task overlap = 1`
   the floor `τ=0.9` forces Bob's LPIPS to `0.85` (the certificate destroys the legitimate
   receiver); with `task overlap ≈ 0` the same floor costs `0.24`. This, not the ε value, is what
   decides whether the scheme is publishable as *SemCom*.
4. **Amplitude-only `ℓ₁` is unsound at high SNR** — Fig. 3(b), and the numbers in §1.
5. `predicted_lpips` is now driven by the measured per-block LPIPS sweeps (a saturating
   scale on measured `w_h`), and `g_h/Δf_h/w_h/κ_h` are measured on the released
   SemanticStyleGAN generator; `eve_identification_error` remains a documented
   **surrogate** (ArcFace-style decision margin) until the inverted-identity
   attack in §3 is run on GPU.
6. Assumption A1 (block-diagonal second order) only loosens the bound; correlations may be
   charged to `g_h` (Remark 3).
