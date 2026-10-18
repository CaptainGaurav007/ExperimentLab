"""
Core statistics, written by hand (only the `math` module is used).
scipy is used ONLY in tests/test_stats.py to check these answers.

Convention: group 1 = control, group 2 = treatment.
Effect = treatment - control, so a positive effect means treatment is better.
"""
import math
from dataclasses import dataclass, asdict


# ---------- Normal distribution ----------
def normal_cdf(x: float) -> float:
    """P(Z <= x) for a standard normal. erf is in Python's math module."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse of normal_cdf, found by bisection (slow but easy to explain)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- Student t distribution (via the regularized incomplete beta) ----------
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for a t distribution with df degrees of freedom."""
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF of the t distribution, for p in (0.5, 1), by bisection."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        upper_tail = 0.5 * t_two_sided_p(mid, df)   # P(T > mid)
        if 1.0 - upper_tail < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- Result container ----------
@dataclass
class TestResult:
    test: str
    control_rate: float      # control mean / conversion rate
    treatment_rate: float
    effect: float            # treatment - control (absolute)
    relative_lift: float     # effect / control (0 if control is 0)
    stat: float              # z or t
    p_value: float
    ci_low: float            # confidence interval for the effect
    ci_high: float
    alpha: float
    effect_size: float       # Cohen's h (proportions) or Cohen's d (means)
    significant: bool

    def to_dict(self):
        return asdict(self)


# ---------- Effect sizes ----------
def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions (arcsine transform makes sizes comparable)."""
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError("proportions must be between 0 and 1")
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


# ---------- Test 1: two-proportion z-test (conversion rates) ----------
def two_prop_ztest(x1, n1, x2, n2, alpha=0.05) -> TestResult:
    """
    x = number of conversions, n = number of users.
    The p-value uses the POOLED standard error, because H0 says both groups share
    one true rate. The confidence interval uses the UNPOOLED standard error,
    because we are estimating the real difference and do not assume H0.
    """
    _validate_alpha(alpha)
    _validate_binomial(x1, n1, "control")
    _validate_binomial(x2, n2, "treatment")
    p1, p2 = x1 / n1, x2 / n2
    if (x1 + x2) == 0 or (x1 + x2) == (n1 + n2):
        # The pooled standard error is zero: the z statistic is undefined.
        # Returning p=1 is conservative, while the interval remains informative.
        pass
    diff = p2 - p1
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = min(1.0, 2.0 * normal_cdf(-abs(z)))

    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = normal_ppf(1 - alpha / 2)
    return TestResult(
        test="two-proportion z-test",
        control_rate=p1, treatment_rate=p2, effect=diff,
        relative_lift=diff / p1 if p1 > 0 else 0.0,
        stat=z, p_value=p_value,
        ci_low=diff - z_crit * se_unpooled, ci_high=diff + z_crit * se_unpooled,
        alpha=alpha, effect_size=cohens_h(p1, p2), significant=p_value < alpha,
    )


# ---------- Test 2: Welch t-test (means, e.g. revenue per user) ----------
def welch_ttest(m1, s1, n1, m2, s2, n2, alpha=0.05) -> TestResult:
    """Compares two means without assuming equal variances. s = standard deviation."""
    _validate_alpha(alpha)
    for name, value in (("m1", m1), ("m2", m2)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name, value in (("s1", s1), ("s2", s2)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not isinstance(n1, int) or not isinstance(n2, int) or n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must be integers >= 2")
    diff = m2 - m1
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = math.sqrt(v1 + v2)
    t = diff / se if se > 0 else 0.0
    denom = v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1)
    if se == 0 or denom == 0:
        raise ValueError("Welch t-test is undefined when both groups have zero variance")
    df = (v1 + v2) ** 2 / denom   # Welch-Satterthwaite
    p_value = t_two_sided_p(t, df)
    t_crit = t_ppf(1 - alpha / 2, df)
    pooled_sd = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    return TestResult(
        test="Welch t-test",
        control_rate=m1, treatment_rate=m2, effect=diff,
        relative_lift=diff / m1 if m1 != 0 else 0.0,
        stat=t, p_value=p_value,
        ci_low=diff - t_crit * se, ci_high=diff + t_crit * se,
        alpha=alpha, effect_size=diff / pooled_sd if pooled_sd > 0 else 0.0,
        significant=p_value < alpha,
    )


# ---------- Power and sample size (two proportions) ----------
def _validate_alpha(alpha):
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")


def _validate_binomial(x, n, label):
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"{label} sample size must be a positive integer")
    if not isinstance(x, int) or isinstance(x, bool) or not 0 <= x <= n:
        raise ValueError(f"{label} conversions must be an integer between 0 and n")


def _validate_proportion(p, name):
    if not math.isfinite(p) or not 0 < p < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1")


def power_two_prop(p1, p2, n_per_group, alpha=0.05) -> float:
    """Chance of detecting a true difference p2 - p1 with n users per group."""
    _validate_proportion(p1, "p1")
    _validate_proportion(p2, "p2")
    _validate_alpha(alpha)
    if not isinstance(n_per_group, int) or n_per_group <= 0:
        raise ValueError("n_per_group must be a positive integer")
    if p1 == p2:
        return alpha
    d = abs(p2 - p1)
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n_per_group)                  # under H0
    se1 = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_group)        # under H1
    z_a = normal_ppf(1 - alpha / 2)
    return (1 - normal_cdf((z_a * se0 - d) / se1)) + normal_cdf((-z_a * se0 - d) / se1)


def sample_size_two_prop(p1, p2, alpha=0.05, power=0.8) -> int:
    """Users needed PER GROUP to detect p2 - p1 with the given power."""
    _validate_proportion(p1, "p1")
    _validate_proportion(p2, "p2")
    _validate_alpha(alpha)
    if not math.isfinite(power) or not 0 < power < 1:
        raise ValueError("power must be strictly between 0 and 1")
    if p1 == p2:
        raise ValueError("sample size is undefined for a zero effect")
    d = abs(p2 - p1)
    pbar = (p1 + p2) / 2
    z_a, z_b = normal_ppf(1 - alpha / 2), normal_ppf(power)
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) +
           z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / d ** 2)
