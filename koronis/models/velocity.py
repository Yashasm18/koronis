import numpy as np
import pandas as pd

from ..data.schema import RELATIONS

# Entity types a real rules engine would run counters on.
#
# `card_id` is excluded because a card-testing attempt uses a fresh card each
# time, so a per-card counter is definitionally useless against this attack.
#
# `email_domain` is excluded because it is far too coarse to gate on: every
# merchant has thousands of legitimate gmail.com customers, so the threshold
# that stays inside any sane false-positive budget is enormous. It remains a
# useful *graph relation* — sharing a domain is weak evidence that composes
# with other evidence — but it is not something a rules engine can count on
# alone. Keeping it here would flatter the baseline on our synthetic
# background, where only five domains exist, and would not survive real traffic.
VELOCITY_ENTITIES = ["device_id", "ip_id", "bin_id"]


class VelocityDetector:
    """Fires when one entity exceeds `tau` attempts inside `window_s`.

    This is the industry default, and the detector Claim 1 of the spec proves
    blind above a spread. It is implemented faithfully rather than as a
    strawman: a weak baseline would invalidate the whole comparison.

    The score is the entity's rolling count *above* threshold, so it is
    monotone and usable in a PR curve rather than only as a binary rule.
    """

    def __init__(self, tau: int, window_s: float, entity: str = "ip_id"):
        self.tau = tau
        self.window_s = window_s
        self.entity = entity

    def score_events(self, events: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(events), dtype=float)
        ts = events["ts"].to_numpy()
        for idx in events.groupby(self.entity, sort=False).indices.values():
            idx = np.sort(idx)
            t = ts[idx]
            # count of same-entity events in the preceding window
            start = np.searchsorted(t, t - self.window_s, side="left")
            counts = np.arange(len(t)) - start + 1
            out[idx] = np.maximum(counts - self.tau, 0)
        return out


def false_positive_rate(scores: np.ndarray) -> float:
    """Fraction of legitimate events a rule would have flagged."""
    return float((np.asarray(scores) > 0).mean())


class MultiEntityVelocityDetector:
    """A velocity rules engine as a real team would actually deploy one.

    Runs an independent counter per entity type and takes the strongest
    signal, rather than betting on a single hand-picked entity. This is the
    baseline the graph model has to beat; a weaker one would rig the result.
    """

    def __init__(self, taus: dict[str, int], window_s: float):
        self.taus = taus
        self.window_s = window_s

    def score_events(self, events: pd.DataFrame) -> np.ndarray:
        per_entity = [
            VelocityDetector(tau=tau, window_s=self.window_s,
                             entity=entity).score_events(events)
            for entity, tau in self.taus.items()
        ]
        return np.max(np.stack(per_entity), axis=0)


def tune_velocity(background: pd.DataFrame, window_s: float,
                  fp_budget: float,
                  entities: list[str] | None = None) -> dict[str, int]:
    """Pick the most sensitive threshold per entity that stays within budget.

    This is the empirical form of the tau-floor in Claim 1. Lowering tau makes
    the rule more sensitive, but legitimate heavy users - offices, shared CGNAT
    addresses, a busy device - start tripping it. The false-positive budget
    therefore puts a hard floor under tau, and the floor is measured here on
    clean traffic rather than assumed.

    Returns the smallest tau per entity whose false-positive rate on
    `background` (which is all label=0) is within its share of `fp_budget`.

    The budget is split across entities by the union bound: the combined
    detector fires when ANY counter fires, so tuning each one to the full
    budget independently would let the engine overshoot it several times over.
    """
    entities = entities or VELOCITY_ENTITIES
    per_entity_budget = fp_budget / max(len(entities), 1)
    taus: dict[str, int] = {}
    for entity in entities:
        # Search the full feasible range rather than a fixed cap. On dense
        # traffic the busiest legitimate entity can carry hundreds of events,
        # so a fixed ceiling would declare the baseline unusable when a valid
        # threshold exists just above it - unfair to the baseline, and it would
        # flatter this project.
        ceiling = int(background[entity].value_counts().max()) + 2
        chosen = None
        for tau in range(2, ceiling + 1):
            scores = VelocityDetector(tau=tau, window_s=window_s,
                                      entity=entity).score_events(background)
            if false_positive_rate(scores) <= per_entity_budget:
                chosen = tau
                break
        # If no threshold in range meets the budget, this entity cannot be
        # gated on at all at this false-positive budget. Park it above every
        # observable count so the counter never fires - which is the honest
        # reading, not "fires on everything".
        taus[entity] = chosen if chosen is not None else ceiling + 1
    return taus
