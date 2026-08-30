import numpy as np
import pandas as pd

from ..data.schema import RELATIONS
from ..graph.build import build_edges


class SharedEntityDetector:
    """A graph baseline with no learning at all.

    Scores each event by how much of its recent neighbourhood it shares
    entities with, weighted by how rare that entity is in the stream. A busy
    office IP is common and contributes little; a device seen only inside one
    burst contributes a lot.

    This exists to answer the sharpest question anyone can ask about the graph
    model: does a plain co-occurrence count do the same job without a neural
    network? If it does, the network is decorative and should be dropped. That
    is a question worth being able to answer with a number.
    """

    def __init__(self, window_s: float = 3600.0, max_degree: int = 32):
        self.window_s = window_s
        self.max_degree = max_degree

    def score_events(self, events: pd.DataFrame) -> np.ndarray:
        n = len(events)
        edges = build_edges(events, window_s=self.window_s,
                            max_degree=self.max_degree)
        score = np.zeros(n, dtype=float)

        for rel in RELATIONS:
            ei = edges[rel]
            if ei.shape[1] == 0:
                continue
            # Inverse-frequency weight: sharing a rare entity is informative,
            # sharing gmail.com is not.
            counts = events[rel].map(events[rel].value_counts()).to_numpy()
            w = 1.0 / np.log1p(counts)
            src, dst = ei[0], ei[1]
            np.add.at(score, dst, w[src])

        return score


class DeclineBurstDetector:
    """A rate baseline with no graph and no learning.

    Scores each event by the local decline rate in its time window. Card
    testing produces a burst of declines; this is the cheapest possible way to
    notice that, and any model claiming to detect card testing should be held
    against it.
    """

    def __init__(self, window_s: float = 600.0):
        self.window_s = window_s

    def score_events(self, events: pd.DataFrame) -> np.ndarray:
        ts = events["ts"].to_numpy()
        declined = (~events["approved"].to_numpy().astype(bool)).astype(float)
        lo = np.searchsorted(ts, ts - self.window_s, side="left")
        hi = np.arange(len(ts)) + 1
        cum = np.concatenate([[0.0], np.cumsum(declined)])
        n_in = hi - lo
        return np.where(n_in > 0, (cum[hi] - cum[lo]) / np.maximum(n_in, 1), 0.0)
