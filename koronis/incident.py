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

from collections import defaultdict, deque

from .data.schema import RELATIONS
from .eval.cost import COST_PER_ATTEMPT_INR
from .sketch import SlidingCountMin

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


# An entity value covering more than this share of the WHOLE stream is not
# evidence that two alerts belong together. gmail.com covers half of all
# traffic and says nothing; a device fingerprint on 0.03% of it says a great
# deal. Measured against the full stream rather than against the alerts,
# because that is what makes a value common — an earlier version used the
# alerted subset and the email domains landed just under the cap, so the merge
# survived.
#
# Applied per VALUE rather than per relation, so a popular issuer BIN is
# excluded while a campaign's minted BIN still links, and so the rule keeps
# working on a future dataset with a low-cardinality field this code has never
# seen.
#
# Without it, two concurrent campaigns with entirely separate infrastructure
# merged into one 597-attempt blob bridged solely by email domain: a merchant
# under attack from two rings would get one incident and one action for both.
MAX_LINK_SHARE = 0.02


def build_incidents(events: pd.DataFrame, scores: np.ndarray, threshold: float,
                    link_window_s: float = 900.0,
                    max_link_share: float = MAX_LINK_SHARE) -> list[Incident]:
    """Group alerted events that share a DISCRIMINATIVE entity within a window.

    Only alerted events are grouped, and only through entity values specific
    enough to be evidence. The relation set used for grouping is therefore
    narrower than the one used for scoring: a shared email domain is weak
    evidence that composes usefully inside the model, but it cannot support the
    claim that two alerts are the same incident.
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
    cap = max(len(events) * max_link_share, 1.0)
    for rel in RELATIONS:
        overall = events[rel].value_counts()
        for val, idx in sub.groupby(rel, sort=False).indices.items():
            if overall.get(val, 0) > cap:
                continue          # too common in the stream to be evidence
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


# The binding per-entity velocity threshold on this dataset, from the frontier
# sweep (`results/frontier.csv`): the device counter at tau = 8 is the most
# sensitive, so it is the one a distributed campaign must stay under. Used only
# to annotate the dossier; nothing here re-derives it.
_BINDING_VELOCITY_TAU = 8


def dossier(d: dict) -> str:
    """A human-auditable summary of one consolidated incident.

    `d` is a detail record as written to `results/incidents.csv` /
    `policy.json` — this only reformats fields already computed elsewhere, so
    it introduces no new modelling and no schema change.
    """
    dev = max(int(d["n_devices"]), 1)
    ip = max(int(d["n_ips"]), 1)
    bn = max(int(d["n_bins"]), 1)
    n = int(d["n_attempts"])
    act = ACTION_BY_NAME[d["action"]]
    costs = dict(d["option_costs"])
    chosen = costs[d["action"]]
    monitor = costs["monitor"]
    oracle = d["oracle_action"]
    agree = "matches" if oracle == d["action"] else f"differs - oracle: {oracle}"
    uncertain = "  · wide interval -> escalate to review" if d.get("forecast_uncertain") else ""
    bar = "-" * 70
    return "\n".join([
        f"-- Incident Dossier - {d['incident_id']} " + "-" * 40,
        f"  Spread          {n} alerted attempts - {dev} devices - {ip} IPs - {bn} BINs",
        f"                  per-entity load {n/dev:.1f}/device, {n/ip:.1f}/IP, "
        f"{n/bn:.1f}/BIN  (binding velocity tau = {_BINDING_VELOCITY_TAU})",
        f"  Consolidation   {n} event alerts -> 1 incident - link window 900 s",
        f"  Incident risk   {d['risk']:.3f}  (recalibrated logistic on calibration incidents)",
        f"  Forecast        decided after {int(d['observed_at_decision'])} events - "
        f"remaining P50 {d['forecast_remaining_p50']:.0f} "
        f"[P90 {d['forecast_remaining_p90']:.0f}]{uncertain}",
        f"                  exposure P50 INR {d['forecast_exposure_p50_inr']:,.0f} "
        f"[P90 INR {d['forecast_exposure_p90_inr']:,.0f}]",
        f"  Recommendation  {act.name} - {act.blurb}",
        f"                  expected INR {chosen:,.0f}  vs INR {monitor:,.0f} to keep monitoring",
        f"                  oracle action: {oracle}  ({agree})",
        bar,
    ])


def choose_action(risk: float, remaining_attempts: int) -> tuple[Action, list[tuple[str, float]]]:
    """Lowest expected cost, not highest risk score.

    Returns the chosen action and every option's cost, so the demo can show
    *why* the alternatives were rejected rather than asserting a verdict.
    """
    costs = [(a.name, expected_cost(a, risk, remaining_attempts))
             for a in AUTONOMOUS_ACTIONS]
    best = min(costs, key=lambda kv: kv[1])[0]
    return ACTION_BY_NAME[best], costs


# ── online consolidation ────────────────────────────────────────────────────
class StreamingIncidents:
    """Consolidate alerts into incidents as the stream arrives.

    `build_incidents` is a batch function, and one step of it is not causal:
    the link-share cap is computed with `value_counts()` over the WHOLE frame,
    so whether two alerts may link depends on traffic that had not happened
    yet. The scorer never had that problem - its edges point backwards in time -
    so the pipeline was causal in its first half and not its second.

    This closes that. Frequencies come from a sliding count-min sketch fed by
    every event as it passes, so the cap at time t reflects only what had been
    seen by time t, in memory that does not grow with the number of distinct
    entity values.

    It will not reproduce the batch grouping exactly, and it should not be
    expected to: the batch version is using information from the future. Where
    they differ, this one is the defensible answer. `koronis.cli online`
    measures how far apart they land.
    """

    def __init__(self, threshold: float, link_window_s: float = 900.0,
                 max_link_share: float = MAX_LINK_SHARE,
                 freq_window_s: float = 3600.0,
                 sketch_width: int = 4096):
        self.threshold = float(threshold)
        self.link_window_s = float(link_window_s)
        self.max_link_share = float(max_link_share)
        # ONE SKETCH PER RELATION, for two reasons. The denominator of a share
        # must be the number of EVENTS, and a single shared sketch counts one
        # add per relation per event - which silently divided every share by
        # the relation count and let a domain covering 6% of the stream slip
        # under a 2% cap and bridge two unrelated rings. Separate sketches also
        # stop a device id and an email domain colliding in the same counter,
        # where the collision is pure noise between incomparable namespaces.
        self.sketch = {rel: SlidingCountMin(window_s=freq_window_s,
                                            width=sketch_width)
                       for rel in RELATIONS}
        # alerted events only, per relation value, inside the link window
        self._recent: dict[str, dict[str, deque]] = {
            rel: defaultdict(deque) for rel in RELATIONS
        }
        self._parent: list[int] = []
        self._rows: list[int] = []
        self._n_seen = 0
        self._skipped_common = 0

    # -------------------------------------------------------------- union-find
    def _find(self, a: int) -> int:
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def _union(self, a: int, b: int) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    # -------------------------------------------------------------------- api
    def push(self, event, score: float, row: int | None = None) -> int | None:
        """Feed one scored event. Returns its incident key, or None if it did
        not alert. Every event updates the frequency sketch; only alerts link."""
        if isinstance(event, pd.Series):
            event = event.to_dict()
        ts = float(event["ts"])
        self._n_seen += 1
        for rel in RELATIONS:                     # frequency sees ALL traffic
            self.sketch[rel].add(str(event[rel]), ts)

        if score < self.threshold:
            return None

        me = len(self._parent)
        self._parent.append(me)
        self._rows.append(self._n_seen - 1 if row is None else row)

        cutoff = ts - self.link_window_s
        for rel in RELATIONS:
            val = str(event[rel])
            bucket = self._recent[rel][val]
            while bucket and bucket[0][0] < cutoff:   # expire, bounding memory
                bucket.popleft()
            # a value common enough to cover much of the stream is not evidence
            if self.sketch[rel].share(val, ts) > self.max_link_share:
                self._skipped_common += 1
                bucket.append((ts, me))
                continue
            for _, other in bucket:
                self._union(me, other)
            bucket.append((ts, me))
        return self._find(me)

    # ------------------------------------------------------------------ output
    def groups(self) -> dict[int, list[int]]:
        """Current components, as {incident key: [row indices]}."""
        out: dict[int, list[int]] = {}
        for i, row in enumerate(self._rows):
            out.setdefault(self._find(i), []).append(row)
        return out

    def stats(self) -> dict:
        live = sum(len(d) for rel in self._recent.values() for d in rel.values())
        return {
            "events_seen": self._n_seen,
            "alerts": len(self._parent),
            "incidents": len(self.groups()),
            "buffered_alert_refs": live,
            "sketch_kb": round(sum(s.memory_bytes() for s in self.sketch.values())
                               / 1024, 1),
            "links_skipped_as_common": self._skipped_common,
        }
