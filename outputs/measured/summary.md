# Measured validation (CelebAMask-HQ, 500 images)

`g_h` and `Δf_h` are measured on real part masks; `w_h/κ_h` remain documented surrogates pending the trained semantic encoder.

## Scenario: reconstruction (task overlap 1.000, |P|=15)

| scheme | Bob LPIPS | leakage (bit) | Fano floor | RDP (eps@alpha) | GDP (mu, eps) | noise/g | sparsity |
|---|---|---|---|---|---|---|---|
| mi | 0.8618 | 0.329 | 0.900 | 16.6@2.80 | 2.69, 15.1 | 37.698 | 0.116 |
| rdp | 0.8597 | 3.738 | 0.643 | 100.0@1.45 | 10.13, 94.5 | 3.655 | 0.165 |
| gdp | 0.8595 | 3.963 | 0.626 | 105.6@1.45 | 10.51, 100.0 | 3.399 | 0.165 |
| basic | 0.8613 | 1.303 | 0.827 | 41.4@1.85 | 5.49, 38.5 | 9.968 | 0.137 |
| static-4 | 0.8527 | 137.756 | 0.000 | 3112261464.3@1.05 | 76994.26, 2964386690.3 | 1.939 | 0.250 |

## Scenario: task_decoupled (task overlap 0.013, |P|=6)

| scheme | Bob LPIPS | leakage (bit) | Fano floor | RDP (eps@alpha) | GDP (mu, eps) | noise/g | sparsity |
|---|---|---|---|---|---|---|---|
| mi | 0.7674 | 0.329 | 0.900 | 17.5@2.70 | 2.83, 16.0 | 3.793 | 0.247 |
| rdp | 0.3742 | 3.020 | 0.697 | 100.0@1.45 | 10.13, 94.5 | 0.295 | 0.247 |
| gdp | 0.3619 | 3.180 | 0.685 | 105.6@1.45 | 10.51, 100.0 | 0.274 | 0.247 |
| basic | 0.7437 | 2.308 | 0.751 | 76.3@1.55 | 8.45, 71.8 | 0.486 | 0.204 |
| static-4 | 0.6826 | 24.315 | 0.000 | 327319169.1@1.05 | 24969.27, 311838814.0 | 1.939 | 0.250 |

## Audit at ε=2000, δ=1e-5

- **reconstruction** (|P|=15): μ_h≈14.4, RDP-converted ε=2037.8, quadrature ε=2000, amplitude-only ℓ₁ ε=1049.9, exact tail ε=2797.8.
- **task_decoupled** (|P|=6): μ_h≈24.1, RDP-converted ε=2037.8, quadrature ε=2000, amplitude-only ℓ₁ ε=700.7, exact tail ε=2448.5.

The amplitude-only form under-reports ε once μ_h≫c; the exact tail bound stays above quadrature. Ordering reproduced on measured part statistics.
