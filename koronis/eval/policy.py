"""Evaluate response policies, not just detectors.

A detector is scored on precision and recall. A response policy has to be
scored on what it costs the merchant and what it puts on an analyst's desk.

Two policies are reported side by side, and the distinction matters:

* `oracle_policy` is told the true number of attempts still to come, and
  whether the incident is genuine. Neither is knowable live. It is an **upper
  bound**, not a product.
* `causal_policy` sees only a prefix of each incident plus a forecast with an
  interval. The gap between them is the price of not knowing the future.

All costs are ASSUMPTIONS declared in `koronis/incident.py` and
`koronis/eval/cost.py`. Nothing here executes an action; the policies are
simulated over a recorded stream.
"""
import numpy as np
import pandas as pd

from ..forecast import ExposureForecaster
from ..incident import (
    ACTION_BY_NAME, ANALYST_COST_PER_MIN_INR, TRIAGE_MINUTES_PER_EVENT,
    Incident, IncidentRisk, build_incidents, choose_action,
)
from .cost import COST_PER_ATTEMPT_INR

# How far into an incident the causal policy commits. A live system decides on
# a prefix; deciding at the end is hindsight wearing a timestamp.
DECIDE_AFTER = 12
# When the upper forecast exceeds the median by this factor, the forecaster is
# effectively saying it does not know. Automating on that is worse than asking
# a person, so the policy escalates to review instead of acting confidently.
UNCERTAINTY_RATIO = 4.0


def _remaining_after(events: pd.DataFrame, campaign_id: str, t: float) -> int:
    """Campaign attempts still to come at `t`. Oracle input — offline only."""
    camp = events[events["campaign_id"] == campaign_id]
    return int((camp["ts"] > t).sum())


def _realised_cost(action_name: str, inc: Incident, remaining: int) -> dict:
    """What the action actually cost, given ground truth.

    `expected_cost` is what a policy believed before acting; this is the bill
    afterwards. Reporting both is what separates a decision system from a
    scoring function.
    """
    a = ACTION_BY_NAME[action_name]
    cost = a.friction_inr + a.analyst_minutes * ANALYST_COST_PER_MIN_INR
    if inc.is_true():
        cost += remaining * COST_PER_ATTEMPT_INR * (1.0 - a.stops)
        stopped = remaining * a.stops
    else:
        cost += a.false_harm_inr
        stopped = 0.0
    return {"cost": cost, "analyst_minutes": a.analyst_minutes, "stopped": stopped}


def evaluate_policies(events: pd.DataFrame, scores: np.ndarray, threshold: float,
                      risk_model: IncidentRisk,
                      forecaster: ExposureForecaster | None = None,
                      link_window_s: float = 900.0) -> tuple[pd.DataFrame, list[dict]]:
    """Compare five ways of responding to the same scored stream."""
    incidents = build_incidents(events, scores, threshold, link_window_s)
    risks = risk_model.predict(incidents)
    for inc, r in zip(incidents, risks):
        inc.risk = float(r)

    campaigns = list(events["campaign_id"].dropna().unique())
    total_exposure = sum(
        int((events["campaign_id"] == c).sum()) * COST_PER_ATTEMPT_INR
        for c in campaigns)

    detail = []
    for inc in incidents:
        # ── oracle input: the true future ──
        oracle_remaining = max(
            (_remaining_after(events, c, inc.t_start) for c in campaigns),
            default=0) if inc.is_true() else 0
        oracle_action, oracle_costs = choose_action(inc.risk, oracle_remaining)

        # ── causal input: a prefix and a forecast ──
        m = min(DECIDE_AFTER, len(inc.rows))
        p50, hi = (forecaster.predict_one(events, inc.rows, scores, m)
                   if forecaster is not None else (0.0, 0.0))
        # The forecast is the remaining count IF this is a campaign, which is
        # what the oracle branch above also passes. "Does it matter" is applied
        # once, inside expected_cost, as `risk * exposure`. This line used to
        # multiply by risk as well - "will there be more" TIMES "does it
        # matter" - which made the causal arm risk^2 * p50 while the oracle arm
        # stayed risk * true_remaining, so the two sides of the published regret
        # were not computed the same way. It also contradicted the formula
        # stated in forecast.py and in docs/evaluation.md.
        exp_remaining = int(round(p50))
        causal_action, causal_costs = choose_action(inc.risk, exp_remaining)
        uncertain = hi > max(p50, 1.0) * UNCERTAINTY_RATIO
        if uncertain and inc.risk > 0.5 and causal_action.name == "monitor":
            causal_action = ACTION_BY_NAME["hold_review"]

        detail.append({
            "incident_id": inc.incident_id, "risk": round(inc.risk, 4),
            "n_attempts": inc.n_attempts, "n_devices": inc.n_devices,
            "n_ips": inc.n_ips, "n_bins": inc.n_bins,
            "observed_at_decision": m,
            "forecast_remaining_p50": round(p50, 1),
            "forecast_remaining_p90": round(hi, 1),
            "forecast_exposure_p50_inr": round(p50 * COST_PER_ATTEMPT_INR, 2),
            "forecast_exposure_p90_inr": round(hi * COST_PER_ATTEMPT_INR, 2),
            "forecast_uncertain": bool(uncertain),
            "true_remaining_attempts": oracle_remaining,
            "true_remaining_exposure_inr": round(oracle_remaining * COST_PER_ATTEMPT_INR, 2),
            "genuine": inc.is_true(),
            "action": causal_action.name,
            "oracle_action": oracle_action.name,
            "option_costs": {k: round(v, 2) for k, v in causal_costs},
            "oracle_option_costs": {k: round(v, 2) for k, v in oracle_costs},
            "t_start": round(inc.t_start, 2),
        })

    n_events_fired = int((np.asarray(scores) >= threshold).sum())
    rows = []

    def summarise(name, actions, analyst_min_override=None):
        cost = stopped = analyst = 0.0
        sent = false_inc = 0
        for inc, d, act in zip(incidents, detail, actions):
            r = _realised_cost(act, inc, d["true_remaining_attempts"])
            cost += r["cost"]; stopped += r["stopped"]; analyst += r["analyst_minutes"]
            if act != "monitor":
                sent += 1
                if not inc.is_true():
                    false_inc += 1
        if analyst_min_override is not None:
            analyst = analyst_min_override
            cost += analyst * ANALYST_COST_PER_MIN_INR
        rows.append({
            "policy": name,
            "incidents_actioned": sent,
            "false_incidents": false_inc,
            "campaign_attempts_stopped": int(round(stopped)),
            "analyst_minutes": round(analyst, 1),
            "merchant_cost_inr": round(cost, 2),
        })

    summarise("always_allow", ["monitor"] * len(incidents))
    summarise("always_hold", ["hold_review"] * len(incidents))
    # Event thresholding: no consolidation, so every alerted event is its own
    # ticket. The action matches the causal policy; the workload does not.
    summarise("event_thresholding", [d["action"] for d in detail],
              analyst_min_override=n_events_fired * TRIAGE_MINUTES_PER_EVENT)
    summarise("causal_policy", [d["action"] for d in detail])
    summarise("oracle_policy", [d["oracle_action"] for d in detail])

    df = pd.DataFrame(rows)
    cost = dict(zip(df["policy"], df["merchant_cost_inr"]))
    df["regret_vs_oracle_inr"] = round(
        cost.get("causal_policy", 0.0) - cost.get("oracle_policy", 0.0), 2)
    df["actions_matching_oracle"] = sum(
        1 for d in detail if d["action"] == d["oracle_action"])
    df["exposure_if_unstopped_inr"] = round(total_exposure, 2)
    df["events_alerted"] = n_events_fired
    df["incidents_formed"] = len(incidents)
    return df, detail


def incident_reliability(incidents: list[Incident], risks: np.ndarray,
                         bins: int = 5) -> pd.DataFrame:
    """Reliability of INCIDENT risk, reported separately from the event model.

    Low event-level calibration error does not carry over: the events inside an
    incident are strongly dependent, so nothing about their individual
    reliability guarantees the aggregate is a probability. Measured, not
    inherited.
    """
    risks = np.asarray(risks, dtype=float)
    y = np.array([1.0 if i.is_true() else 0.0 for i in incidents])
    if risks.size == 0:
        return pd.DataFrame(columns=["bin_mid", "predicted", "observed", "count"])
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(risks, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append({
            "bin_mid": (edges[b] + edges[b + 1]) / 2,
            "predicted": float(risks[m].mean()) if m.any() else np.nan,
            "observed": float(y[m].mean()) if m.any() else np.nan,
            "count": int(m.sum()),
        })
    return pd.DataFrame(out)
