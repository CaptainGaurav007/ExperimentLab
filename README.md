# ExperimentLab

A small product-experimentation engine that analyzes fixed-sample A/B test results and returns a **ship / reject / continue** recommendation with uncertainty and explicit decision thresholds.

The core statistical calculations are implemented by hand using Python's `math` module. SciPy is used in tests as an independent reference. Simulations use known ground truth to estimate false-positive rates and power, with Wilson intervals to show Monte Carlo uncertainty.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. API docs are at `/docs`.

## What's inside

| File | Purpose |
|---|---|
| `experimentlab/stats.py` | Two-proportion z-test, Welch t-test, confidence intervals, effect sizes, power and sample size |
| `experimentlab/multiple.py` | Bonferroni, Holm and Benjamini-Hochberg adjustments |
| `experimentlab/simulator.py` | Reproducible simulations for false positives, power, repeated looks and many segments |
| `experimentlab/decision.py` | Decision aid separating statistical evidence from practical significance |
| `app.py` | FastAPI endpoints: `/analyze`, `/sample-size`, `/simulate/{study}` |
| `static/index.html` | Dashboard with experiment entry, interval chart, segment results and simulation summaries |
| `tests/test_stats.py` | Reference comparisons, randomized checks, validation tests, decision rules and reproducibility checks |
| `LEARN.md` | Interview study guide for the key choices and limitations |

## Decision logic

1. Reject if a configured guardrail has statistically significant harm.
2. Ship only when the **entire confidence interval** is at or above the minimum worthwhile effect (MDE).
3. Reject for the stated business objective when the upper confidence bound is below the MDE. This does **not** prove the effect is exactly zero; it says the data do not support the required gain.
4. Otherwise, continue collecting data or investigate the design, because the evidence is inconclusive for the stated threshold.

This is a decision aid, not an automatic causal claim. Business costs, guardrail definitions, randomization, data quality and the experiment protocol still matter.

## Simulation interpretation

Simulation outputs include a fixed seed, number of simulated experiments, estimated rate and a 95% Wilson interval. These intervals quantify **Monte Carlo uncertainty across simulated runs**; they are not the confidence interval for any individual A/B experiment. Small simulation counts can produce noisy estimates, so increase `n_sims` when reporting results.

- `error-rates`: compares false positives under no effect and power under a specified effect.
- `peeking`: compares one planned final look, uncorrected daily looks, and daily looks with Bonferroni correction across a pre-specified fixed number of days. Bonferroni is conservative but controls family-wise error for these planned looks.
- `segments`: compares raw segment findings with Holm and Benjamini-Hochberg adjusted findings. Holm controls family-wise error; Benjamini-Hochberg controls false discovery rate under its usual assumptions.

## Important limitations

- The API's primary analysis is for **fixed-sample binary conversion metrics**. Do not repeatedly inspect and stop an ordinary `/analyze` experiment early; the nominal p-values and intervals are not sequentially valid.
- The peeking simulation illustrates the problem and a Bonferroni correction for a fixed number of planned looks; it is not a general-purpose sequential-testing implementation.
- The normal-approximation z-test can be unreliable with sparse counts or very small expected cell counts. Check assumptions; use an exact or other suitable method when needed.
- Guardrails are checked individually at the selected alpha; multiple guardrails may need their own pre-specified multiplicity strategy.
- Segment analyses are exploratory. Adjusted p-values do not remove selection bias, confounding or the need to validate a discovered segment in a new experiment.
- Welch's t-test is implemented and tested for summary statistics but is not yet exposed in the API.
- The system assumes valid randomization, independent observations and trustworthy counts; it cannot diagnose every experiment-design failure.

## What is tested

The tests compare core calculations against SciPy, exercise randomized valid cases, check invalid inputs, test practical-significance decision boundaries, and verify that simulation outputs are reproducible and report Monte Carlo intervals. Passing tests increases confidence in the implementation; it does not prove the method is appropriate for every dataset or experiment design.
