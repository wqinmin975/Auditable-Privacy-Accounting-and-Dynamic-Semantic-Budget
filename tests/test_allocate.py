"""Unit tests for the allocator and its accounting machinery.

Run:  python -m pytest -q
Every test is cheap (< 1 s) and asserts a *property* from the theory section,
not a hard-coded number, so it keeps working when the calibration changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdpa.allocate import (
    fano_floor,
    gdp_epsilon,
    gdp_mu,
    i_max_bits,
    leakage_nats,
    monotone_check,
    rdp_epsilon_delta,
    rdp_gaussian,
    solve_basic_budget,
    solve_gdp_budget,
    solve_mi_budget,
    solve_rdp_budget,
    static_index_allocation,
    waterfill_sigma,
)
from sdpa.calibrate import Calibration, synthetic_calibration


@pytest.fixture(scope="module")
def cal() -> Calibration:
    return synthetic_calibration(seed=1)


# ----------------------------------------------------------------- Theorem 3
def test_kkt_stationarity(cal):
    """At the solution, a_h = mu g_h / (2 x_h (x_h + g_h)) for every component."""
    alloc = solve_mi_budget(cal, tau=0.90)
    x = alloc.sigma ** 2
    stat = alloc.mu * cal.g / (2 * x * (x + cal.g))
    assert np.allclose(stat, cal.a, rtol=1e-6, atol=1e-9), "KKT stationarity violated"
    assert alloc.mu > 0


def test_budget_is_binding(cal):
    """Complementary slackness: the leakage constraint is active at the optimum."""
    tau = 0.90
    alloc = solve_mi_budget(cal, tau=tau)
    target_bits = i_max_bits(tau, alloc.n_identities)
    assert alloc.leakage_bits() == pytest.approx(target_bits, rel=2e-6, abs=1e-6)


def test_dual_function_is_monotone(cal):
    mus = np.geomspace(1e-4, 1e4, 40)
    leak = monotone_check(cal, mus)
    assert np.all(np.diff(leak) < 0), "leakage must decrease in mu (bisection precondition)"


def test_closed_form_matches_numeric_minimisation(cal):
    """Closed form vs a brute-force constrained solve on a 2-component slice."""
    from scipy.optimize import minimize

    H = 3
    a, g = cal.a[:H], cal.g[:H]
    i_target = 0.9 * leakage_nats(np.sqrt(g) * 0.3, g)  # feasible target

    sigma_cf = solve_mi_budget(_slice(cal, H), i_max=i_target).sigma

    def obj(x):
        return float(np.sum(a * x))

    cons = [{"type": "ineq", "fun": lambda x: i_target - leakage_nats(np.sqrt(np.maximum(x, 1e-12)), g)}]
    res = minimize(obj, np.full(H, g.mean()), method="SLSQP", constraints=cons,
                   bounds=[(1e-9, None)] * H, options={"ftol": 1e-12, "maxiter": 500})
    assert res.success
    assert np.allclose(res.x, sigma_cf ** 2, rtol=2e-3, atol=1e-6)


def _slice(cal: Calibration, H: int) -> Calibration:
    return Calibration(g=cal.g[:H], df=cal.df[:H], w=cal.w[:H], kappa=cal.kappa[:H], d=cal.d[:H])


# ----------------------------------------------------------------- Theorem 2
def test_fano_and_budget_are_inverse(cal):
    """fano_floor(i_max_bits(tau)) == tau for a range of tau."""
    for tau in (0.5, 0.70, 0.80, 0.88, 0.92):
        assert fano_floor(i_max_bits(tau), 10_000) == pytest.approx(tau, abs=1e-9)


def test_certificate_never_exceeds_one(cal):
    assert 0.0 <= solve_mi_budget(cal, tau=0.9).certified_error_floor() <= 1.0


def test_stricter_floor_costs_more_noise(cal):
    """A stricter certificate (higher tau) leaves a smaller leakage budget, hence more noise."""
    powers = []
    for tau in (0.60, 0.70, 0.80, 0.90):
        powers.append(solve_mi_budget(cal, tau=tau).summary()["noise_power"])
    assert np.all(np.diff(powers) > 0)


def test_certification_ceiling(cal):
    """tau above the Fano ceiling is infeasible and must be rejected."""
    from sdpa.allocate import tau_ceiling

    cap = tau_ceiling(10_000)
    assert 0.92 < cap < 0.93
    with pytest.raises(ValueError):
        i_max_bits(cap + 1e-3, 10_000)


# ----------------------------------------------------------------- Theorem 4
def test_ell1_allocation_law(cal):
    """Basic composition: x_h ∝ (df_h / (w_h kappa_h))^{2/3}."""
    alloc = solve_basic_budget(cal, eps_target=50.0)
    x = alloc.sigma ** 2
    pred = (cal.df / cal.a) ** (2.0 / 3.0)
    ratio = x / pred
    assert np.std(ratio / ratio.mean()) < 1e-9, "ell_1 allocation must follow the 2/3 power law"


def test_waterfill_asymptotics(cal):
    """Small-noise regime x_h ~ mu/(2 a_h); large-noise regime sqrt(mu g_h/(2 a_h))."""
    a, g = cal.a, cal.g
    x_small = waterfill_sigma(1e-6, a, g) ** 2
    assert np.allclose(x_small / (1e-6 / (2 * a)), 1.0, rtol=0.01)
    mu = 1e6
    x_large = waterfill_sigma(mu, a, g) ** 2
    pred = np.sqrt(mu * g / (2 * a))
    assert np.allclose(x_large / pred, 1.0, rtol=0.02)


def test_task_overlap_controls_the_price_of_privacy():
    """Same certificate: identity-is-the-task is expensive, task-decoupled is cheap."""
    from sdpa.calibrate import reconstruction_calibration, task_decoupled_calibration

    recon = reconstruction_calibration(seed=1)
    task = task_decoupled_calibration(seed=1)
    assert recon.task_overlap > 0.99 > task.task_overlap

    a_recon = solve_mi_budget(recon, tau=0.90)
    a_task = solve_mi_budget(task, tau=0.90)
    assert a_task.predicted_lpips() < a_recon.predicted_lpips() - 0.2
    # noise is confined to the declared private subspace
    assert np.all(a_task.sigma[~task.private] == 0.0)
    assert a_task.leakage_bits() == pytest.approx(i_max_bits(0.90), rel=1e-6)


def test_static_selection_is_a_corner_case(cal):
    alloc = static_index_allocation(cal, n_protected=3, sigma_floor=1e-4, sigma_max=2.0)
    active = int(np.count_nonzero(alloc.l2_power > 1e-6))
    assert active == 3, "static selection protects exactly k components"
    assert alloc.sparsity() > 0.3, "and concentrates the noise budget on them"


# ----------------------------------------------------------------- accounting
def test_gdp_composes_in_quadrature(cal):
    sigma = np.linspace(0.2, 0.6, cal.H)
    parts = (cal.df / sigma)
    assert gdp_mu(sigma, cal.df) == pytest.approx(float(np.sqrt(np.sum(parts ** 2))))


def test_rdp_conversion_is_decreasing_in_noise(cal):
    coarse = rdp_epsilon_delta(np.full(cal.H, 0.10), cal.df, 1e-5)[0]
    fine = rdp_epsilon_delta(np.full(cal.H, 0.40), cal.df, 1e-5)[0]
    assert coarse > fine > 0


def test_rdp_budget_is_met(cal):
    target = 50.0
    alloc = solve_rdp_budget(cal, eps_target=target, delta=1e-5)
    eps, _ = alloc.rdp()
    assert eps == pytest.approx(target, rel=1e-3)


def test_gdp_budget_is_met(cal):
    target = 50.0
    alloc = solve_gdp_budget(cal, eps_target=target, delta=1e-5)
    _, eps = alloc.gdp()
    assert eps == pytest.approx(target, rel=1e-3)


def test_stylized_ell1_underreports_at_high_snr(cal):
    """The amplitude-only linear-composition law is unsound once mu_h >> c.

    At a release that quadrature accounting certifies as eps=2000, the stylized
    law claims a *smaller* eps while the rigorous Gaussian tail bound gives a
    larger one: the mu^2/2 term, which Laplace-style accounting drops, is the
    whole story in this regime.
    """
    from sdpa.allocate import eps_l1_exact, eps_l1_stylized

    alloc = solve_gdp_budget(cal, eps_target=2000.0, delta=1e-5)
    audit = alloc.audit_l1()
    assert audit["eps_stylized"] < 2000.0 < audit["eps_exact"]
    # and the two linear laws never cross the other way
    assert eps_l1_exact(alloc.sigma, cal.df, 1e-5, cal.private) >= \
        eps_l1_stylized(alloc.sigma, cal.df, 1e-5, cal.private)


# ----------------------------------------------------------------- calibration
def test_per_component_sensitivity_factor():
    """Global accounting inflates sensitivity by sqrt(n_total / n_component) = sqrt(28)."""
    from sdpa.calibrate import REFERENCE_CLIP_RANGE, component_dims

    d = component_dims()
    df = np.sqrt(d) * REFERENCE_CLIP_RANGE
    glob = np.sqrt(d.sum()) * REFERENCE_CLIP_RANGE
    assert df[0] == pytest.approx(glob / np.sqrt(28), rel=1e-6)


def test_calibration_roundtrip(tmp_path, cal):
    p = cal.save(tmp_path / "cal.json")
    back = Calibration.load(p)
    assert np.allclose(back.g, cal.g) and np.allclose(back.df, cal.df)


def test_calibration_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Calibration(g=np.ones(3), df=np.ones(4), w=np.ones(3), kappa=np.ones(3), d=np.ones(3))
