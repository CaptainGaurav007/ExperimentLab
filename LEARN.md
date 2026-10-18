# Learn this before any interview

If you cannot explain a line, you cannot defend it. Read each answer, then close the file
and say it out loud. Then run the matching code and change the numbers to see what happens.

## 1. What a p-value means
Assume the change did nothing (H0). The p-value is the chance of seeing a difference
at least this large just from random noise.
It is NOT the chance that H0 is true, and NOT the chance that you are right.
p = 0.03 means: "if nothing changed, data like this would show up 3% of the time".
Try it: `simulate/error-rates`. With no real effect, about 5% of tests still hit p < 0.05.

## 2. Why the p-value alone is not enough
- A huge test can make a tiny, useless effect "significant".
- A small test can miss a big effect.
So we also report the effect, the confidence interval, effect size, and power.
The interval is the useful part: it shows the range of effects that fit the data.

## 3. Pooled vs unpooled standard error (`two_prop_ztest`)
- Test (p-value): H0 says both groups share one rate, so we pool them.
- Interval: we estimate the real difference, so we use each group's own rate.

## 4. Power and sample size
Power = chance of catching a real effect of a given size.
Smaller effects need many more users (needed users grow with 1 / effect squared).
Try it: `/sample-size` with baseline 0.10 and min_effect 0.01 vs 0.02.

## 5. Why peeking creates false wins
A p-value is only valid if you decide when to look BEFORE the test.
Each daily look is another chance for noise to cross p < 0.05.
Try it: `simulate/peeking`. Daily looks push false wins from 5% to about 20%.
Fixes: look once, use a stricter alpha (alpha / looks), or use a sequential method.

## 6. Many segments, many false winners
20 segments at alpha 0.05, no real effect: chance of at least one "win" = 1 - 0.95^20 = 64%.
- Bonferroni: multiply p by the number of tests. Simple, strict.
- Holm: same guarantee, slightly less strict.
- Benjamini-Hochberg: controls the share of false wins among your wins (FDR).
Try it: `simulate/segments`.

## 7. Simpson's paradox (not coded, but expect the question)
A treatment can win inside every segment and still lose overall if the groups have
different segment mixes. Example: treatment gets more traffic from a low-converting channel.
Fix: check segments, and make sure users were randomly assigned.

## 8. Questions to practice
- Why not just compare the two conversion rates?
- Explain the interval in plain words to a product manager.
- Your test is not significant. Does that prove there is no effect?
- How many users do you need to detect a 1 point lift on a 10% baseline?
- Why test the engine on simulated data?
- What would you change to support revenue per user? (Welch t-test, skewed data, outliers.)
- What would you add next? (CUPED variance reduction, sequential testing.)


## 8. Practical significance is not statistical significance
A tiny positive effect can have a small p-value in a very large experiment. ExperimentLab only recommends `ship` when the entire confidence interval clears the pre-declared minimum worthwhile effect (MDE). If the interval's upper bound is below the MDE, it recommends rejecting the change for that business objective; that does not prove the effect is exactly zero. If the interval overlaps the MDE, the result is inconclusive for the decision and may justify more data.

## 9. What simulation intervals mean
The simulator's Wilson intervals quantify Monte Carlo uncertainty in the estimated rate across repeated simulated experiments. They do not describe uncertainty for one experiment. Fixed random seeds make results reproducible; a larger number of simulations generally narrows the Monte Carlo interval.

## 10. Be honest about the design
The main endpoint is for fixed-sample binary conversion experiments. The ordinary p-value is not safe to repeatedly check while deciding when to stop. The peeking demonstration compares daily peeking with Bonferroni correction for a fixed number of planned looks; it is not a general sequential-testing engine. Segment analysis is exploratory, and guardrail multiplicity must be considered separately.
