# Measured validation (CelebAMask-HQ, 500 images)

`g_h` and `Δf_h` are measured on real part masks; `w_h/κ_h` remain documented surrogates pending the trained semantic encoder.

## Scenario: reconstruction (task overlap 1.000, |P|=15)

| scheme | Bob LPIPS | leakage (bit) | Fano floor | RDP (eps@alpha) | GDP (mu, eps) | noise/g | sparsity |
|---|---|---|---|---|---|---|---|
| mi | 0.8618 | 0.329 | 0.900 | 16.4@2.80 | 2.67, 15.0 | 38.196 | 0.115 |
| rdp | 0.8595 | 4.727 | 0.569 | 100.0@1.45 | 10.13, 94.5 | 2.659 | 0.116 |
| gdp | 0.8593 | 5.084 | 0.542 | 105.6@1.45 | 10.51, 100.0 | 2.473 | 0.116 |
| basic | 0.8613 | 1.405 | 0.819 | 41.4@1.85 | 5.49, 38.5 | 9.968 | 0.137 |
| static-4 | 0.8527 | 77687.505 | 0.000 | 3112261464.3@1.05 | 76994.26, 2964386690.3 | 1.939 | 0.250 |

## Scenario: task_decoupled (task overlap 0.013, |P|=6)

| scheme | Bob LPIPS | leakage (bit) | Fano floor | RDP (eps@alpha) | GDP (mu, eps) | noise/g | sparsity |
|---|---|---|---|---|---|---|---|
| mi | 0.7706 | 0.329 | 0.900 | 17.1@2.75 | 2.77, 15.7 | 3.944 | 0.247 |
| rdp | 0.3746 | 4.394 | 0.594 | 100.0@1.45 | 10.13, 94.5 | 0.295 | 0.247 |
| gdp | 0.3623 | 4.725 | 0.569 | 105.6@1.45 | 10.51, 100.0 | 0.274 | 0.247 |
| basic | 0.7437 | 3.092 | 0.692 | 76.3@1.55 | 8.45, 71.8 | 0.486 | 0.204 |
| static-4 | 0.6826 | 7666.195 | 0.000 | 327319169.1@1.05 | 24969.27, 311838814.0 | 1.939 | 0.250 |

## Audit at ε=2000, δ=1e-5

- **reconstruction** (|P|=15): μ_h≈15.0, RDP-converted ε=2037.8, quadrature ε=2000, amplitude-only ℓ₁ ε=1089.2, exact tail ε=2837.0.
- **task_decoupled** (|P|=6): μ_h≈24.1, RDP-converted ε=2037.8, quadrature ε=2000, amplitude-only ℓ₁ ε=701.1, exact tail ε=2448.9.

The amplitude-only form under-reports ε once μ_h≫c; the exact tail bound stays above quadrature. Ordering reproduced on measured part statistics.
