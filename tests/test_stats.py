"""Check the hand-written code against SciPy. If these pass, the maths is right."""
import math
import numpy as np
from scipy import stats as sp
from experimentlab.stats import (normal_cdf, normal_ppf, t_ppf, t_two_sided_p,
                                 two_prop_ztest, welch_ttest, sample_size_two_prop, power_two_prop)
from experimentlab.multiple import holm, benjamini_hochberg, bonferroni


def test_normal():
    for x in [-2.5, -1, 0, 0.7, 1.96, 3]:
        assert abs(normal_cdf(x) - sp.norm.cdf(x)) < 1e-12
    for p in [0.025, 0.5, 0.975, 0.995]:
        assert abs(normal_ppf(p) - sp.norm.ppf(p)) < 1e-9


def test_t_distribution():
    for df in [3, 10, 57.3, 400]:
        for t in [0.3, 1.5, 2.8]:
            assert abs(t_two_sided_p(t, df) - 2 * sp.t.sf(t, df)) < 1e-9
        assert abs(t_ppf(0.975, df) - sp.t.ppf(0.975, df)) < 1e-6


def test_two_prop_ztest_matches_scipy_formula():
    x1, n1, x2, n2 = 1000, 10000, 1120, 10000
    r = two_prop_ztest(x1, n1, x2, n2)
    pooled = (x1 + x2) / (n1 + n2)
    z = (x2 / n2 - x1 / n1) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    assert abs(r.p_value - 2 * sp.norm.sf(abs(z))) < 1e-12
    # also matches the chi-square test without continuity correction
    chi = sp.chi2_contingency([[x1, n1 - x1], [x2, n2 - x2]], correction=False)
    assert abs(r.p_value - chi[1]) < 1e-9


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 3, 200), rng.normal(10.6, 5, 250)
    r = welch_ttest(a.mean(), a.std(ddof=1), len(a), b.mean(), b.std(ddof=1), len(b))
    ref = sp.ttest_ind(b, a, equal_var=False)
    assert abs(r.stat - ref.statistic) < 1e-9
    assert abs(r.p_value - ref.pvalue) < 1e-9
    ci = ref.confidence_interval(0.95)
    assert abs(r.ci_low - ci.low) < 1e-6 and abs(r.ci_high - ci.high) < 1e-6


def test_sample_size_and_power():
    n = sample_size_two_prop(0.10, 0.12)
    assert 3800 < n < 3900                      # known textbook value ~3841
    assert power_two_prop(0.10, 0.12, n) >= 0.80
    assert power_two_prop(0.10, 0.12, n - 50) < 0.80


def test_multiple_testing():
    p = [0.001, 0.01, 0.02, 0.04, 0.5]
    assert bonferroni(p)[0] == 0.005
    assert all(h >= b for h, b in zip(holm(p), p))
    assert all(b <= h + 1e-12 for b, h in zip(benjamini_hochberg(p), holm(p)))


def test_randomized_two_prop_matches_scipy():
    rng = np.random.default_rng(20261018)
    for _ in range(250):
        n1, n2 = (int(v) for v in rng.integers(100, 100_000, size=2))
        p1, p2 = rng.uniform(0.01, 0.99, size=2)
        x1, x2 = int(rng.binomial(n1, p1)), int(rng.binomial(n2, p2))
        r = two_prop_ztest(x1, n1, x2, n2)
        pooled = (x1 + x2) / (n1 + n2)
        z = (x2/n2 - x1/n1) / math.sqrt(pooled*(1-pooled)*(1/n1+1/n2))
        assert abs(r.p_value - 2 * sp.norm.sf(abs(z))) < 1e-10


def test_invalid_statistical_inputs_raise():
    import pytest
    for args in [(1, 0, 2, 10), (11, 10, 2, 10), (-1, 10, 2, 10), (1, 10, 2, 10, 1.0)]:
        with pytest.raises(ValueError):
            two_prop_ztest(*args)
    with pytest.raises(ValueError):
        sample_size_two_prop(0.1, 0.1)
    with pytest.raises(ValueError):
        power_two_prop(0.1, 0.2, 0)
    with pytest.raises(ValueError):
        welch_ttest(1, 1, 1, 2, 1, 10)


def test_decision_never_ships_if_ci_does_not_clear_mde():
    from types import SimpleNamespace
    from experimentlab.decision import recommend
    # Significant positive estimate, but CI lower bound is below MDE.
    result = SimpleNamespace(effect=0.004, ci_low=0.001, ci_high=0.007,
                             significant=True, control_rate=0.10)
    decision = recommend(result, min_effect=0.005, n_per_group=10000)
    assert decision["decision"] == "continue"
    # Entire interval is below MDE: reject for the stated business objective.
    result.ci_low, result.ci_high = 0.001, 0.004
    assert recommend(result, 0.005)["decision"] == "reject"
    # Entire interval clears MDE: ship.
    result.ci_low, result.ci_high = 0.006, 0.009
    assert recommend(result, 0.005)["decision"] == "ship"


def test_simulation_reproducible_and_reports_uncertainty():
    from experimentlab.simulator import error_rate_study
    a = error_rate_study(n_per_group=1000, n_sims=200, seed=123)
    b = error_rate_study(n_per_group=1000, n_sims=200, seed=123)
    assert a == b
    assert "ci95" in a["false_positive_rate"]
    lo, hi = a["false_positive_rate"]["ci95"]
    assert 0 <= lo <= hi <= 1
