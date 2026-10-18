"""FastAPI backend. Run:  uvicorn app:app --reload   then open http://127.0.0.1:8000"""
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict

from experimentlab.stats import two_prop_ztest, power_two_prop, sample_size_two_prop
from experimentlab.multiple import holm, benjamini_hochberg
from experimentlab.decision import recommend
from experimentlab import simulator

app = FastAPI(title="ExperimentLab")


class Arm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    users: int = Field(gt=1)
    conversions: int = Field(ge=0)


class Guardrail(BaseModel):
    name: str
    control: Arm
    treatment: Arm
    higher_is_worse: bool = True      # e.g. error rate: higher is worse


class Segment(BaseModel):
    name: str
    control: Arm
    treatment: Arm


class AnalyzeRequest(BaseModel):
    control: Arm
    treatment: Arm
    alpha: float = Field(0.05, gt=0, lt=0.5)
    min_effect: float = Field(0.005, gt=0, lt=1)     # smallest absolute gain worth shipping
    guardrails: List[Guardrail] = Field(default_factory=list, max_length=20)
    segments: List[Segment] = Field(default_factory=list, max_length=50)


def _check(arm: Arm):
    if arm.conversions > arm.users:
        raise HTTPException(422, "conversions cannot be larger than users")


def _test(c: Arm, t: Arm, alpha: float):
    _check(c); _check(t)
    return two_prop_ztest(c.conversions, c.users, t.conversions, t.users, alpha)


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    main = _test(req.control, req.treatment, req.alpha)

    # Guardrails: a metric that must not get worse (error rate, refunds, ...)
    guardrails, breaches = [], []
    for g in req.guardrails:
        res = _test(g.control, g.treatment, req.alpha)
        worse = res.significant and ((res.effect > 0) == g.higher_is_worse)
        guardrails.append({"name": g.name, "breached": worse, **res.to_dict()})
        if worse:
            breaches.append(g.name)

    # Segments: raw p-values are misleading when we test many, so we adjust them.
    segs = []
    if req.segments:
        results = [_test(s.control, s.treatment, req.alpha) for s in req.segments]
        raw = [r.p_value for r in results]
        hol, bh = holm(raw), benjamini_hochberg(raw)
        for s, r, ph, pb in zip(req.segments, results, hol, bh):
            segs.append({"name": s.name, "effect": r.effect, "ci_low": r.ci_low,
                         "ci_high": r.ci_high, "p_raw": r.p_value, "p_holm": ph,
                         "p_bh": pb, "significant_raw": r.p_value < req.alpha,
                         "significant_after_correction": ph < req.alpha})

    n_per_group = min(req.control.users, req.treatment.users)
    rec = recommend(main, req.min_effect, breaches, n_per_group)
    power_at_mde = power_two_prop(main.control_rate, main.control_rate + req.min_effect,
                                  n_per_group, req.alpha)
    return {"main": main.to_dict(), "recommendation": rec, "guardrails": guardrails,
            "segments": segs, "power_at_min_effect": power_at_mde,
            "min_effect": req.min_effect}


class PowerRequest(BaseModel):
    baseline_rate: float = Field(gt=0, lt=1)
    min_effect: float = Field(gt=0, lt=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    power: float = Field(default=0.8, gt=0, lt=1)


@app.post("/sample-size")
def sample_size(req: PowerRequest):
    if req.baseline_rate + req.min_effect >= 1:
        raise HTTPException(422, "baseline + effect must be below 1")
    n = sample_size_two_prop(req.baseline_rate, req.baseline_rate + req.min_effect,
                             req.alpha, req.power)
    return {"users_per_group": n, "total_users": 2 * n}


@app.get("/simulate/{study}")
def simulate(study: str, n_sims: int = 500):
    """Prove the engine works: studies where we know the true answer."""
    n_sims = max(50, min(n_sims, 2000))
    if study == "error-rates":
        return simulator.error_rate_study(n_per_group=15000, abs_lift=0.01, n_sims=n_sims)
    if study == "peeking":
        return simulator.peeking_study(n_sims=n_sims)
    if study == "segments":
        return simulator.segments_study(n_sims=max(50, n_sims // 2))
    raise HTTPException(404, "study must be error-rates, peeking or segments")


@app.get("/")
def home():
    return FileResponse("static/index.html")
