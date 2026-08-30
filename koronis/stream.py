"""Strictly causal streaming replay.

Consumes one payment event at a time and emits a score, an alert decision
against a frozen calibration threshold, and the evidence behind it.

The design rests on one property of the batch model: `build_edges` points every
edge backwards in time, so a node's layer-1 representation depends only on
events that preceded it. That makes layer-1 outputs cacheable as the stream
advances, and layer-2 for a new event computable from its neighbours' cached
layer-1 values. Streaming therefore reproduces batch scores exactly rather than
approximately, which `tests/test_stream.py` holds it to.

Defense-only: this consumes a synthetic in-memory event stream. It makes no
network calls, touches no payment gateway, and handles no real card data.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch

from .data.schema import RELATIONS
from .models.koronis import KoronisDetector, node_features


class StreamingKoronis:
    """Stateful, strictly causal scorer.

    No future event is reachable from `push`: the only state is a bounded
    window of events already seen.
    """

    def __init__(self, detector: KoronisDetector, threshold: float,
                 window_s: float | None = None, max_degree: int = 32):
        if detector.net is None:
            raise RuntimeError("detector must be fitted before streaming")
        self.net = detector.net
        self.net.eval()
        self.threshold = float(threshold)
        self.window_s = float(window_s if window_s is not None else detector.window_s)
        self.max_degree = max_degree

        self._x: list[torch.Tensor] = []          # per-event features
        self._h1: list[torch.Tensor] = []         # cached layer-1 outputs
        self._ts: list[float] = []
        self._alerted: list[bool] = []
        self._index: dict[str, dict[str, deque]] = {
            rel: defaultdict(deque) for rel in RELATIONS
        }
        self._n_seen = 0

    # ------------------------------------------------------------------ core

    def push(self, event) -> dict:
        """Score one event using only events already seen."""
        if isinstance(event, pd.Series):
            event = event.to_dict()

        ts = float(event["ts"])
        idx = len(self._x)
        x = torch.from_numpy(node_features(pd.DataFrame([event]))[0])

        neighbours, evidence = self._neighbours(event, ts)

        with torch.no_grad():
            l1, l2 = self.net.layers[0], self.net.layers[1]
            # Layer 1 aggregates neighbours' raw features.
            h1 = l1.forward_single(x, [self._stack(n, self._x) for n in neighbours])
            # Layer 2 aggregates neighbours' cached layer-1 outputs.
            h2 = l2.forward_single(h1, [self._stack(n, self._h1) for n in neighbours])
            score = float(torch.sigmoid(self.net.head(h2)).item())

        alert = score >= self.threshold
        self._x.append(x)
        self._h1.append(h1)
        self._ts.append(ts)
        self._alerted.append(alert)
        self._remember(event, idx, ts)
        self._n_seen += 1

        linked = sum(len(n) for n in neighbours)
        return {
            "ts": ts,
            "event_id": event.get("event_id"),
            "score": round(score, 6),
            "threshold": round(self.threshold, 6),
            "alert": bool(alert),
            "linked_prior_events": int(linked),
            "evidence": evidence,
            "ring": self._ring_summary(ts),
        }

    # ------------------------------------------------------------- internals

    def _neighbours(self, event, ts: float):
        """Prior events inside the window sharing an entity, per relation."""
        cutoff = ts - self.window_s
        out, evidence = [], {}
        for rel in RELATIONS:
            bucket = self._index[rel].get(str(event[rel]))
            picked: list[int] = []
            if bucket:
                while bucket and self._ts[bucket[0]] < cutoff:
                    bucket.popleft()               # expire, bounding memory
                picked = list(bucket)[-self.max_degree:]
            out.append(picked)
            evidence[rel] = len(picked)
        return out, evidence

    def _remember(self, event, idx: int, ts: float) -> None:
        for rel in RELATIONS:
            self._index[rel][str(event[rel])].append(idx)

    @staticmethod
    def _stack(indices: list[int], store: list[torch.Tensor]):
        if not indices:
            return None
        return torch.stack([store[i] for i in indices])

    def _ring_summary(self, ts: float) -> dict:
        """Rolling view of what has alerted inside the current window."""
        cutoff = ts - self.window_s
        alerts = [i for i, (t, a) in enumerate(zip(self._ts, self._alerted))
                  if a and t >= cutoff]
        if not alerts:
            return {"alerts_in_window": 0, "first_alert_ts": None,
                    "seconds_since_first_alert": None}
        first = self._ts[alerts[0]]
        return {
            "alerts_in_window": len(alerts),
            "first_alert_ts": round(first, 3),
            "seconds_since_first_alert": round(ts - first, 3),
        }

    @property
    def events_seen(self) -> int:
        return self._n_seen


def replay(events: pd.DataFrame, detector: KoronisDetector,
           threshold: float, window_s: float | None = None) -> list[dict]:
    """Replay a whole stream event by event, in order."""
    stream = StreamingKoronis(detector, threshold, window_s)
    return [stream.push(row) for _, row in events.iterrows()]
