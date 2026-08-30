import numpy as np
import pandas as pd


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
