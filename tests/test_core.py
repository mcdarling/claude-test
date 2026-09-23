import numpy as np

from uq_certification import simulator as sim
from uq_certification.conformal import ConformalDetector, coverage
from uq_certification.ensemble import DeepEnsemble, binary_entropy
from uq_certification.maturity import Assessment, Criterion
from uq_certification.measures import expected_calibration_error, wilson_lower_bound
from uq_certification.tradeoffs import Policy, choose_policy, pareto_mask, policy_rates


def test_simulator_is_reproducible_and_cells_in_range():
    s1, X1 = sim.generate(500, np.random.default_rng(0))
    s2, X2 = sim.generate(500, np.random.default_rng(0))
    np.testing.assert_array_equal(X1, X2)
    cells = sim.condition_cell(s1)
    assert cells.min() >= 0 and cells.max() < sim.N_CELLS
    assert X1.shape == (500, len(sim.FEATURE_NAMES))


def test_perturbed_scenes_stay_in_bounds_and_keep_terrain():
    seeds = sim.sample_scenes(10, np.random.default_rng(1))
    new = sim.perturb_scenes(seeds, 5, np.random.default_rng(2))
    assert len(new) == 50
    np.testing.assert_array_equal(new.terrain, np.repeat(seeds.terrain, 5))
    for name, (lo, hi) in sim.PARAM_BOUNDS.items():
        assert getattr(new, name).min() >= lo and getattr(new, name).max() <= hi


def test_ece_is_small_for_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(p, y) < 0.01
    # A confidently wrong model is badly calibrated.
    assert expected_calibration_error(np.full(1000, 0.99), np.zeros(1000, dtype=int)) > 0.9


def test_uncertainty_decomposition():
    s, X = sim.generate(600, np.random.default_rng(0))
    out = DeepEnsemble(n_members=3).fit(X, s.drone).predict(X[:100])
    assert np.all(out["epistemic"] >= 0)
    np.testing.assert_allclose(out["aleatoric"] + out["epistemic"], out["total"], atol=1e-9)
    assert binary_entropy(np.array([0.5]))[0] == 1.0


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    groups = rng.integers(0, 4, p.size)
    cal, test = slice(0, 10_000), slice(10_000, None)
    for mondrian in (False, True):
        cp = ConformalDetector(0.1, mondrian=mondrian).fit(p[cal], y[cal], groups[cal])
        sets = cp.predict_sets(p[test], groups[test])
        assert coverage(sets, y[test]).mean() >= 0.89


def test_wilson_bound_is_below_point_estimate():
    assert wilson_lower_bound(90, 100) < 0.9
    assert wilson_lower_bound(0, 0) == 0.0


def test_pareto_and_constrained_choice():
    rates = np.array([[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [0.0, 0.0, 0.9]])
    np.testing.assert_array_equal(pareto_mask(rates), [True, False, True])
    policies = [Policy(0.5, np.inf), Policy(0.5, np.inf), Policy(0.5, 0.1)]
    # Unconstrained, deferring everything is cheapest; the capacity cap rules it out.
    _, best, _ = choose_policy(policies, rates, weights=(10, 10, 0.1), max_deferral=1.0)
    assert best == 2
    _, best, _ = choose_policy(policies, rates, weights=(10, 10, 0.1), max_deferral=0.2)
    assert best == 0


def test_policy_rates_count_deferred_cases_as_handled():
    p = np.array([0.9, 0.1, 0.9, 0.1])
    y = np.array([1, 1, 0, 0])
    unc = np.array([0.0, 1.0, 0.0, 0.0])
    r = policy_rates(p, unc, y, Policy(t=0.5, tau=0.5))
    assert r == {"miss": 0.0, "false_alarm": 0.5, "deferral": 0.25}


def test_maturity_levels_are_cumulative():
    a = Assessment(
        "x",
        [
            Criterion(1, "", True, ""),
            Criterion(2, "", False, ""),
            Criterion(3, "", True, ""),
        ],
    )
    assert a.level == 1
