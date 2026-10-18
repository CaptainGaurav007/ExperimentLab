"""Turns statistics into a plain-English recommendation: ship, reject or continue."""
from .stats import sample_size_two_prop


def recommend(result, min_effect, guardrail_breaches=None, n_per_group=None):
    """Recommend ship/reject/continue using statistical AND practical significance.

    min_effect is the smallest absolute improvement worth shipping (e.g. 0.005
    means 0.5 percentage points). This is a decision aid, not an automatic
    causal claim. The caller is responsible for a valid, fixed-sample design.
    """
    if not 0 <= min_effect < 1:
        raise ValueError("min_effect must be in [0, 1)")
    r = result
    pct = lambda v: f"{v * 100:+.2f} percentage points"
    interval = f"[{pct(r.ci_low)}, {pct(r.ci_high)}]"
    guardrail_breaches = guardrail_breaches or []

    if guardrail_breaches:
        return {"decision": "reject", "reason":
                "A guardrail significantly worsened: " + ", ".join(guardrail_breaches) +
                ". Do not ship without resolving the guardrail risk."}

    # Ship only when the entire interval clears the practical-effect threshold.
    if r.ci_low >= min_effect:
        return {"decision": "ship", "reason":
                f"The full confidence interval {interval} exceeds the minimum worthwhile "
                f"gain ({pct(min_effect)}). This supports a practically meaningful gain, "
                "assuming the experiment design and guardrails are valid."}

    # Clear evidence of harm: even the optimistic end is below zero.
    if r.ci_high < 0:
        return {"decision": "reject", "reason":
                f"The full confidence interval {interval} is below zero, indicating harm "
                "to the primary metric."}

    # Evidence rules out the minimum worthwhile gain (but may allow a tiny gain/loss).
    if r.ci_high < min_effect:
        return {"decision": "reject", "reason":
                f"The upper confidence bound ({pct(r.ci_high)}) is below the minimum "
                f"worthwhile gain ({pct(min_effect)}). The data do not support shipping "
                "for this stated business objective; this does not prove the effect is zero."}

    need = sample_size_two_prop(r.control_rate, min(r.control_rate + min_effect, 0.999999))
    more = f" As a rough planning estimate, about {need:,} users per group are needed to detect the stated gain."
    if n_per_group:
        more += f" Current smaller group: {n_per_group:,} users."
    return {"decision": "continue", "reason":
            f"Evidence is inconclusive for the stated decision threshold. Estimated effect "
            f"{pct(r.effect)}, confidence interval {interval}; it does not yet establish a "
            f"gain of at least {pct(min_effect)}.{more}"}
