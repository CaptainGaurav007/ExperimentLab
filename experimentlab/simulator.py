"""
Simulations where WE choose the truth, so we can check whether our tools
give the right answer. This is the proof that the engine works.
"""
import numpy as np
import math
from .stats import two_prop_ztest
from .multiple import holm, benjamini_hochberg


def simulate_binary(p_control, abs_lift, n_control, n_treatment, rng):
    """One fake experiment. Returns (x_control, x_treatment) conversion counts."""
    return (int(rng.binomial(n_control, p_control)),
            int(rng.binomial(n_treatment, p_control + abs_lift)))


def _wilson_interval(successes, trials, z=1.959963984540054):
    """95% Wilson interval for a simulated proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    phat = successes / trials
    denom = 1 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _rate_summary(successes, trials):
    return {"estimate": successes / trials,
            "ci95": _wilson_interval(successes, trials)}


def error_rate_study(p_control=0.10, abs_lift=0.01, n_per_group=5000,
                     n_sims=1000, alpha=0.05, seed=42):
    """
    A/A runs (no real effect): share flagged significant = false positive rate. Should be ~alpha.
    A/B runs (real effect = abs_lift): share flagged significant = power.
    """
    rng = np.random.default_rng(seed)
    fp = pw = 0
    for _ in range(n_sims):
        x1, x2 = simulate_binary(p_control, 0.0, n_per_group, n_per_group, rng)
        fp += two_prop_ztest(x1, n_per_group, x2, n_per_group, alpha).significant
        x1, x2 = simulate_binary(p_control, abs_lift, n_per_group, n_per_group, rng)
        pw += two_prop_ztest(x1, n_per_group, x2, n_per_group, alpha).significant
    return {"n_sims": n_sims, "seed": seed, "alpha": alpha,
            "false_positive_rate": _rate_summary(fp, n_sims),
            "power": _rate_summary(pw, n_sims),
            "note": "Intervals quantify Monte Carlo uncertainty across simulations, not uncertainty within an individual experiment."}


def peeking_study(p_control=0.10, users_per_day=500, days=14,
                  n_sims=1000, alpha=0.05, seed=42):
    """
    No real effect. Compare:
      - look ONCE at the end (the correct plan)
      - look EVERY day and stop the first time p < alpha (peeking)
      - peek daily but use alpha / days (Bonferroni fix)
    """
    rng = np.random.default_rng(seed)
    once = peek = fixed = 0
    for _ in range(n_sims):
        c = np.cumsum(rng.binomial(users_per_day, p_control, size=days))
        t = np.cumsum(rng.binomial(users_per_day, p_control, size=days))
        pvals = [two_prop_ztest(int(c[d]), users_per_day * (d + 1),
                                int(t[d]), users_per_day * (d + 1), alpha).p_value
                 for d in range(days)]
        once += pvals[-1] < alpha
        peek += any(p < alpha for p in pvals)
        fixed += any(p < alpha / days for p in pvals)
    return {"n_sims": n_sims, "seed": seed, "days": days, "alpha": alpha,
            "false_win_look_once": _rate_summary(once, n_sims),
            "false_win_peeking_daily": _rate_summary(peek, n_sims),
            "false_win_peeking_with_bonferroni": _rate_summary(fixed, n_sims),
            "sequential_method": "Bonferroni correction across the pre-specified daily looks (conservative; assumes the number of looks is fixed in advance)."}


def segments_study(n_segments=20, p_control=0.10, n_per_segment=2000,
                   n_sims=500, alpha=0.05, seed=42):
    """No real effect in ANY segment. How often do we still "find" a winning segment?"""
    rng = np.random.default_rng(seed)
    raw = hol = bh = 0
    for _ in range(n_sims):
        pv = []
        for _ in range(n_segments):
            x1, x2 = simulate_binary(p_control, 0.0, n_per_segment, n_per_segment, rng)
            pv.append(two_prop_ztest(x1, n_per_segment, x2, n_per_segment, alpha).p_value)
        raw += min(pv) < alpha
        hol += min(holm(pv)) < alpha
        bh += min(benjamini_hochberg(pv)) < alpha
    return {"n_sims": n_sims, "seed": seed, "n_segments": n_segments, "alpha": alpha,
            "false_find_raw": _rate_summary(raw, n_sims),
            "false_find_holm": _rate_summary(hol, n_sims),
            "false_find_bh": _rate_summary(bh, n_sims),
            "note": "Holm controls family-wise error under arbitrary dependence; Benjamini-Hochberg controls FDR under standard independence/positive-dependence conditions."}
