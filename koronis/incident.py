"""Consolidate event alerts into incidents, and choose an action for each.

An alert is not a decision. Four hundred event-level alerts are not four
hundred things for a fraud team to do; they are one campaign. This module
turns a stream of scored events into time-bounded incidents, assigns each a
*recalibrated* incident-level risk, and recommends the intervention with the
lowest expected cost under explicitly stated assumptions.

Defense-only: every action here is a recommendation in a simulated merchant
workflow. Nothing blocks a payment, calls a gateway, or touches live
infrastructure.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.schema import RELATIONS
from .eval.cost import COST_PER_ATTEMPT_INR

# ── action assumptions ──────────────────────────────────────────────────────
# Every number here is an ASSUMPTION about a merchant workflow, not a
# measurement. They are declared in one place so a reviewer can substitute
# their own and re-run. `stops` is the share of an ongoing campaign's remaining
# attempts the action prevents; `friction_inr` is what the action costs when
# the incident is genuine; `false_harm_inr` is the cost when it is not.
@dataclass(frozen=True)
class Action:
    name: str
    friction_inr: float
    false_harm_inr: float
    stops: float
    analyst_minutes: float
    blurb: str


ACTIONS = [
    Action("monitor", 0.0, 0.0, 0.00, 0.0,
           "no action; keep scoring"),
    Action("rate_limit", 120.0, 400.0, 0.55, 0.0,
           "throttle the linked entities"),
    Action("step_up", 350.0, 1800.0, 0.85, 0.0,
           "step-up verification on matching attempts"),
    Action("hold_review", 900.0, 6000.0, 0.97, 12.0,
           "hold matching attempts and queue for analyst review"),
    # The guardrail action. A human looks; nothing is blocked automatically.
    # It stops no abuse on its own, which is the honest cost of standing down:
    # when the traffic no longer resembles what the thresholds were fitted on,
    # a confident automated action is worth less than a person's judgement.
    Action("review_only", 0.0, 0.0, 0.0, 12.0,
           "route to analyst review; no automated action"),
]
ACTION_BY_NAME = {a.name: a for a in ACTIONS}

# What an analyst minute costs the merchant, used to price review workload.
ANALYST_COST_PER_MIN_INR = 9.0
# Triaging a single event-level alert, for the event-thresholding baseline.
TRIAGE_MINUTES_PER_EVENT = 0.5


@dataclass
class Incident:
    incident_id: str
    rows: list[int] = field(default_factory=list)
    t_start: float = 0.0
    t_end: float = 0.0
    n_attempts: int = 0
    n_devices: int = 0
    n_ips: int = 0
    n_bins: int = 0
    mean_score: float = 0.0
    max_score: float = 0.0
    campaign_share: float = 0.0        # ground truth, evaluation only
    risk: float = float("nan")         # filled by IncidentRisk

    def is_true(self) -> bool:
        """An incident counts as genuine if most of its attempts are campaign."""
        return self.campaign_share >= 0.5


class _Union:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_incidents(events: pd.DataFrame, scores: np.ndarray, threshold: float,
                    link_window_s: float = 900.0) -> list[Incident]:
    """Group alerted events that share an entity within `link_window_s`.

    Only alerted events are grouped. Linking every event that shares a gmail
    address would merge the entire stream into one incident; the point is to
    consolidate what an analyst would otherwise see as separate alerts.
    """
    scores = np.asarray(scores, dtype=float)
    fired = np.flatnonzero(scores >= threshold)
    if fired.size == 0:
        return []

    ts = events["ts"].to_numpy()
    labels = events["label"].to_numpy()
    pos = {int(r): i for i, r in enumerate(fired)}
    uf = _Union(fired.size)

    sub = events.iloc[fired]
    for rel in RELATIONS:
        for idx in sub.groupby(rel, sort=False).indices.values():
            rows = fired[np.sort(idx)]
            for a, b in zip(rows, rows[1:]):
                if ts[b] - ts[a] <= link_window_s:
                    uf.union(pos[int(a)], pos[int(b)])

    groups: dict[int, list[int]] = {}
    for r in fired:
        groups.setdefault(uf.find(pos[int(r)]), []).append(int(r))

    out = []
    for n, (_, rows) in enumerate(sorted(groups.items(),
                                         key=lambda kv: ts[kv[1][0]])):
        rows = sorted(rows)
        blk = events.iloc[rows]
        out.append(Incident(
            incident_id=f"INC-{n:03d}",
            rows=rows,
            t_start=float(ts[rows[0]]), t_end=float(ts[rows[-1]]),
            n_attempts=len(rows),
            n_devices=int(blk["device_id"].nunique()),
            n_ips=int(blk["ip_id"].nunique()),
            n_bins=int(blk["bin_id"].nunique()),
            mean_score=float(scores[rows].mean()),
            max_score=float(scores[rows].max()),
            campaign_share=float((labels[rows] == 1).mean()),
        ))
    return out


def incident_features(incs: list[Incident]) -> np.ndarray:
    """Features for incident-level risk. Deliberately structural."""
    return np.array([[
        np.log1p(i.n_attempts), np.log1p(i.n_devices), np.log1p(i.n_ips),
        np.log1p(i.n_bins), i.mean_score, i.max_score,
        np.log1p(i.t_end - i.t_start), 1.0,
    ] for i in incs], dtype=float)


class IncidentRisk:
    """Recalibrates risk at the INCIDENT level.

    A well-calibrated event model does not give a calibrated incident
    probability for free. Events inside an incident are strongly dependent —
    that dependence is the whole signal — so combining their scores by any
    independence-flavoured rule is wrong, and an average is not a probability
    of anything in particular. This fits a small logistic model on incident
    features using the calibration split, and its reliability is reported
    separately from the event model's.
    """

    def __init__(self, l2: float = 1.0, iters: int = 400, lr: float = 0.25):
        self.w: np.ndarray | None = None
        self.l2, self.iters, self.lr = l2, iters, lr
        self.mu = self.sd = None

    def _norm(self, x, fit=False):
        if fit:
            self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        return (x - self.mu) / self.sd

    def fit(self, incs: list[Incident]) -> "IncidentRisk":
        if not incs:
            self.w = None
            return self
        x = self._norm(incident_features(incs), fit=True)
        y = np.array([1.0 if i.is_true() else 0.0 for i in incs])
        w = np.zeros(x.shape[1])
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-x @ w))
            g = x.T @ (p - y) / len(y) + self.l2 * w / len(y)
            w -= self.lr * g
        self.w = w
        return self

    def predict(self, incs: list[Incident]) -> np.ndarray:
        if not incs:
            return np.zeros(0)
        if self.w is None:                       # nothing to learn from
            return np.array([min(i.mean_score, 1.0) for i in incs])
        x = self._norm(incident_features(incs))
        return 1.0 / (1.0 + np.exp(-x @ self.w))


def expected_cost(action: Action, risk: float, remaining_attempts: int) -> float:
    """Expected rupee cost of taking `action` on an incident.

    friction is paid whichever way the incident turns out; unstopped abuse is
    paid only if it is genuine; false harm only if it is not.
    """
    exposure = remaining_attempts * COST_PER_ATTEMPT_INR
    return (action.friction_inr
            + action.analyst_minutes * ANALYST_COST_PER_MIN_INR
            + risk * exposure * (1.0 - action.stops)
            + (1.0 - risk) * action.false_harm_inr)


# Actions the policy may select on its own. `review_only` is reachable only via
# the drift guardrail: it is a decision to stop automating, not an option to be
# weighed on expected cost, and including it in the argmin would let the policy
# pick "do nothing but bill an analyst" whenever that looked cheap.
AUTONOMOUS_ACTIONS = [a for a in ACTIONS if a.name != "review_only"]


def choose_action(risk: float, remaining_attempts: int) -> tuple[Action, list[tuple[str, float]]]:
    """Lowest expected cost, not highest risk score.

    Returns the chosen action and every option's cost, so the demo can show
    *why* the alternatives were rejected rather than asserting a verdict.
    """
    costs = [(a.name, expected_cost(a, risk, remaining_attempts))
             for a in AUTONOMOUS_ACTIONS]
    best = min(costs, key=lambda kv: kv[1])[0]
    return ACTION_BY_NAME[best], costs
