"""Evaluate response policies, not just detectors.

A detector is scored on precision and recall. A response policy has to be
scored on what it costs the merchant and what it puts on an analyst's desk.
These are different questions and the second is the one a fraud team asks.

All costs are ASSUMPTIONS declared in `koronis/incident.py` and
`koronis/eval/cost.py`. Nothing here executes an action; the policies are
simulated over a recorded stream.
"""
import numpy as np
import pandas as pd

from ..incident import (
    ACTION_BY_NAME, ANALYST_COST_PER_MIN_INR, TRIAGE_MINUTES_PER_EVENT,
    Incident, IncidentRisk, build_incidents, choose_action, expected_cost,
)
from .cost import COST_PER_ATTEMPT_INR


def _remaining_after(events: pd.DataFrame, campaign_id: str, t: float) -> int:
    """Campaign attempts still to come when a decision is taken at `t`."""
    camp = events[events["campaign_id"] == campaign_id]
    return int((camp["ts"] > t).sum())


def _realised_cost(action_name: str, inc: Incident, remaining: int) -> dict:
    """What the action actually costs, given ground truth.

    `expected_cost` is what the policy believed before acting; this is the bill
    afterwards. Reporting both is what separates a decision system from a
    scoring function.
    """
    a = ACTION_BY_NAME[action_name]
    analyst_min = a.analyst_minutes
    cost = a.friction_inr + analyst_min * ANALYST_COST_PER_MIN_INR
    if inc.is_true():
        cost += remaining * COST_PER_ATTEMPT_INR * (1.0 - a.stops)
        stopped = remaining * a.stops
    else:
        cost += a.false_harm_inr
        stopped = 0.0
    return {"cost": cost, "analyst_minutes": analyst_min, "stopped": stopped}


def evaluate_policies(events: pd.DataFrame, scores: np.ndarray, threshold: float,
                      risk_model: IncidentRisk,
                      link_window_s: float = 900.0) -> tuple[pd.DataFrame, list[dict]]:
    """Compare four ways of responding to the same scored stream."""
    incidents = build_incidents(events, scores, threshold, link_window_s)
    risks = risk_model.predict(incidents)
    for inc, r in zip(incidents, risks):
        inc.risk = float(r)

    campaigns = [c for c in events["campaign_id"].dropna().unique()]
    total_exposure = sum(
        int((events["campaign_id"] == c).sum()) * COST_PER_ATTEMPT_INR
        for c in campaigns)

    detail = []
    for inc in incidents:
        remaining = max(
            (_remaining_after(events, c, inc.t_start) for c in campaigns),
            default=0) if inc.is_true() else 0
        action, costs = choose_action(inc.risk, remaining)
        detail.append({
            "incident_id": inc.incident_id, "risk": round(inc.risk, 4),
            "n_attempts": inc.n_attempts, "n_devices": inc.n_devices,
            "n_ips": inc.n_ips, "n_bins": inc.n_bins,
            "remaining_attempts": remaining,
            "remaining_exposure_inr": round(remaining * COST_PER_ATTEMPT_INR, 2),
            "genuine": inc.is_true(), "action": action.name,
            "option_costs": {k: round(v, 2) for k, v in costs},
            "t_start": round(inc.t_start, 2),
        })

    n_events_fired = int((np.asarray(scores) >= threshold).sum())
    rows = []

    def summarise(name, per_incident_actions, analyst_min_override=None):
        cost = stopped = analyst = 0.0
        sent = 0
        false_inc = 0
        for inc, d, act in zip(incidents, detail, per_incident_actions):
            r = _realised_cost(act, inc, d["remaining_attempts"])
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
    # Event-by-event thresholding: no consolidation, so every alerted event is
    # its own ticket. The action is the same as ours; the workload is not.
    summarise("event_thresholding", [d["action"] for d in detail],
              analyst_min_override=n_events_fired * TRIAGE_MINUTES_PER_EVENT)
    summarise("incident_policy", [d["action"] for d in detail])

    df = pd.DataFrame(rows)
    df["exposure_if_unstopped_inr"] = round(total_exposure, 2)
    df["events_alerted"] = n_events_fired
    df["incidents_formed"] = len(incidents)
    return df, detail


def incident_reliability(incidents: list[Incident], risks: np.ndarray,
                         bins: int = 5) -> pd.DataFrame:
    """Reliability of INCIDENT risk, reported separately from the event model.

    Low event-level calibration error does not carry over to incidents: the
    events inside one are strongly dependent, so nothing about their individual
    reliability guarantees the aggregate is a probability. This is measured, not
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
