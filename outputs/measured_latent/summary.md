# Latent-level measured validation (SemanticStyleGAN CelebAMask-HQ)

All four scalars are measured on the real generator:

- `g_h`, `Δf_h`: exact w-code prior (mapping network pushforward, N=65536)

- `w_h`: decode+LPIPS sweep (N=16, S=6 per block)

- `κ_h`: per-block linear-MMSE (Wiener) deprotector; covariance measured on N=65536 codes, residual verified on 1024 held-out codes (empirical = analytic to 3 decimals)


Pending: identification attack P_e (GPU-scale inversion + identity labels).

## Scenario: reconstruction (task overlap 1.000, |P|=15)

| scheme | Bob LPIPS | leakage (bit) | Fano floor | RDP (eps@alpha) | GDP (mu, eps) | noise/g | sparsity |
|---|---|---|---|---|---|---|---|
| mi | 0.8618 | 0.329 | 0.900 | 28.8@2.15 | 4.18, 26.6 | 49.681 | 0.123 |
| rdp | 0.8609 | 1.931 | 0.779 | 100.0@1.45 | 10.13, 94.5 | 8.476 | 0.123 |
| gdp | 0.8609 | 2.077 | 0.768 | 105.6@1.45 | 10.51, 100.0 | 7.882 | 0.123 |
| basic | 0.8617 | 0.634 | 0.877 | 44.7@1.85 | 5.81, 41.6 | 33.926 | 0.139 |
| static-4 | 0.8321 | 96725.149 | 0.000 | 100208468905.6@1.05 | 436890.46, 95438500123.7 | 1.145 | 0.250 |

## Scenario: task_decoupled (task overlap 0.590, |P|=7)

| scheme | Bob LPIPS | leakage (bit) | Fano floor | RDP (eps@alpha) | GDP (mu, eps) | noise/g | sparsity |
|---|---|---|---|---|---|---|---|
| mi | 0.8612 | 0.329 | 0.900 | 28.8@2.15 | 4.18, 26.6 | 10.972 | 0.247 |
| rdp | 0.8573 | 1.931 | 0.779 | 100.0@1.45 | 10.13, 94.5 | 1.873 | 0.248 |
| gdp | 0.8569 | 2.076 | 0.769 | 105.6@1.45 | 10.51, 100.0 | 1.742 | 0.248 |
| basic | 0.8605 | 1.367 | 0.822 | 77.3@1.55 | 8.52, 72.7 | 7.503 | 0.139 |
| static-4 | 0.8321 | 29018.371 | 0.000 | 30071630747.1@1.05 | 239330.93, 28640668831.6 | 1.145 | 0.250 |

## Audit at ε=2000, δ=1e-5

- **reconstruction** (|P|=15): μ_h≈14.5, RDP-converted ε=2037.8, quadrature ε=2000, amplitude-only ℓ₁ ε=1055.2, exact tail ε=2803.0.
- **task_decoupled** (|P|=7): μ_h≈21.2, RDP-converted ε=2037.8, quadrature ε=2000, amplitude-only ℓ₁ ε=718.4, exact tail ε=2466.3.

## Measured per-block scalars

| block | name | g_h | Δf_h | w_h | κ_h |
|---|---|---|---|---|---|
| 0 | shared_0 | 248.65 | 97.63 | 0.2713 | 0.242 |
| 1 | shared_1 | 249.35 | 97.77 | 0.3420 | 0.241 |
| 2 | background(shape+texture) | 497.58 | 138.08 | 0.1702 | 0.242 |
| 3 | skin(shape+texture) | 497.69 | 138.18 | 0.1741 | 0.241 |
| 4 | eyes(shape+texture) | 497.63 | 138.08 | 0.0097 | 0.242 |
| 5 | brows(shape+texture) | 498.56 | 138.19 | 0.0159 | 0.243 |
| 6 | mouth(shape+texture) | 498.76 | 138.25 | 0.0112 | 0.241 |
| 7 | nose(shape+texture) | 498.31 | 138.13 | 0.0088 | 0.242 |
| 8 | ears(shape+texture) | 497.78 | 138.12 | 0.0170 | 0.242 |
| 9 | hair(shape+texture) | 497.64 | 138.16 | 0.2753 | 0.241 |
| 10 | neck(shape+texture) | 497.57 | 138.01 | 0.0281 | 0.241 |
| 11 | cloth(shape+texture) | 498.50 | 138.27 | 0.0424 | 0.241 |
| 12 | eye_g(shape+texture) | 498.63 | 138.25 | 0.0117 | 0.243 |
| 13 | hat(shape+texture) | 497.75 | 138.13 | 0.0276 | 0.241 |
| 14 | ear_r(shape+texture) | 497.79 | 138.15 | 0.0078 | 0.240 |

