"""
Multiple-testing corrections. If you test many segments, some will look
"significant" by luck. These functions adjust the p-values.
"""
from typing import List


def bonferroni(pvals: List[float]) -> List[float]:
    """Multiply every p-value by the number of tests. Simple, very strict."""
    m = len(pvals)
    return [min(1.0, p * m) for p in pvals]


def holm(pvals: List[float]) -> List[float]:
    """Holm step-down. Controls the chance of ANY false win, less strict than Bonferroni."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted, running_max = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """BH. Controls the share of false wins among the wins (False Discovery Rate)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i], reverse=True)
    adjusted, running_min = [0.0] * m, 1.0
    for k, i in enumerate(order):
        rank = m - k
        running_min = min(running_min, pvals[i] * m / rank)
        adjusted[i] = min(1.0, running_min)
    return adjusted
