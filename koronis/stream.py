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

import pandas as pd
import torch

from .models.koronis import KoronisDetector, node_features
from .validate import _finite, entity_key, problems


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
        # Mirror the detector exactly: the parity claim is meaningless if the
        # stream indexes relations the batch model never looked at.
        self.relations = list(detector.relations)

        # One cache per layer. Layer k of a new event aggregates its
        # neighbours' layer k-1 outputs, so every intermediate layer has to be
        # kept, not just the first. Holding only layer 1 silently limited the
        # stream to a two-layer model - it matched the batch scores exactly
        # right up until the selected depth changed, and then stopped.
        self._x: list[torch.Tensor] = []                       # layer 0
        self._h: list[list[torch.Tensor]] = [
            [] for _ in range(len(detector.net.layers) - 1)     # layers 1..L-1
        ]
        self._ts: list[float] = []
        self._alerted: list[bool] = []
        self._index: dict[str, dict[str, deque]] = {
            rel: defaultdict(deque) for rel in self.relations
        }
        # Insertion order per relation, so eviction can retire index keys as
        # well as cache rows. Without this the deques emptied but the dict kept
        # a key per distinct value seen - and a card-testing campaign mints a
        # fresh entity per attempt, so that dict is exactly what grows.
        self._ins: dict[str, deque] = {rel: deque() for rel in self.relations}

        # Caches are addressed by absolute event number and stored from
        # `_evicted` onwards, so eviction does not renumber anything already
        # handed out. Without this the caches grew with total traffic rather
        # than with the window: 3,120 events through a 60 s window left 3,120
        # entries in every layer while 12 events were actually in scope.
        self._evicted = 0
        self._high_ts = float("-inf")

        self.quarantined = 0
        self.reasons: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ core

    def push(self, event) -> dict:
        """Score one event using only events already seen.

        An event that cannot be scored honestly is quarantined rather than
        scored anyway: `score` comes back None with a reason, instead of the
        NaN that used to compare False against the threshold and report itself
        as "no alert".
        """
        if isinstance(event, pd.Series):
            event = event.to_dict()

        faults = problems(event)
        if faults:
            self.quarantined += 1
            for f in faults:
                self.reasons[f] += 1
            return {
                "ts": float(event["ts"]) if _finite(event.get("ts")) else None,
                "event_id": event.get("event_id"),
                "score": None, "threshold": round(self.threshold, 6),
                "alert": False, "status": "quarantined", "reasons": faults,
                "linked_prior_events": 0, "evidence": {}, "ring": self._ring_summary(self._high_ts),
            }

        ts = float(event["ts"])
        self._high_ts = max(self._high_ts, ts)
        self._evict()
        idx = self._evicted + len(self._x)
        x = torch.from_numpy(node_features(pd.DataFrame([event]))[0])

        neighbours, evidence = self._neighbours(event, ts)

        with torch.no_grad():
            # Layer k reads its neighbours' layer k-1. Backwards-in-time edges
            # guarantee every neighbour is older, so its layer k-1 was already
            # computed from events older still - which is why this reproduces
            # the batch pass exactly at any depth rather than approximately.
            h = x
            hidden = []
            for layer, cache in zip(self.net.layers, [self._x] + self._h):
                h = layer.forward_single(h, [self._stack(n, cache) for n in neighbours])
                hidden.append(h)
            score = float(torch.sigmoid(self.net.head(h)).item())

        alert = bool(score >= self.threshold)
        self._x.append(x)
        for k, cache in enumerate(self._h):        # layers 1..L-1 only
            cache.append(hidden[k])
        self._ts.append(ts)
        self._alerted.append(alert)
        self._remember(event, idx, ts)

        linked = sum(len(n) for n in neighbours)
        return {
            "ts": ts,
            "event_id": event.get("event_id"),
            "score": round(score, 6),
            "threshold": round(self.threshold, 6),
            "alert": alert,
            "status": "scored",
            "linked_prior_events": int(linked),
            "evidence": evidence,
            "ring": self._ring_summary(ts),
        }

    # ------------------------------------------------------------- internals

    def _neighbours(self, event, ts: float):
        """Prior events inside the window sharing an entity, per relation."""
        cutoff = ts - self.window_s
        out, evidence = [], {}
        for rel in self.relations:
            key = entity_key(event.get(rel))
            # An absent value is not a shared value. Interning it as the string
            # "None" linked every event with a missing device fingerprint to
            # every other one, inventing a ring out of missing data.
            bucket = self._index[rel].get(key) if key is not None else None
            picked: list[int] = []
            if bucket:
                while bucket and (bucket[0] < self._evicted
                                  or self._at(bucket[0]) < cutoff):
                    bucket.popleft()               # expire, bounding memory
                picked = list(bucket)[-self.max_degree:]
            out.append(picked)
            evidence[rel] = len(picked)
        return out, evidence

    def _remember(self, event, idx: int, ts: float) -> None:
        for rel in self.relations:
            key = entity_key(event.get(rel))
            if key is not None:
                self._index[rel][key].append(idx)
                self._ins[rel].append((idx, key))

    def _at(self, idx: int) -> float:
        """Timestamp of absolute event `idx`."""
        return self._ts[idx - self._evicted]

    def _evict(self) -> None:
        """Drop events the window can no longer reach.

        `_neighbours` never looks further back than `high_ts - window_s`, so
        anything older is unreachable and dropping it cannot change a score.
        Eviction is driven by the highest timestamp seen rather than by this
        event's, so a mildly out-of-order arrival cannot discard state that a
        later in-order event still needs.
        """
        cutoff = self._high_ts - self.window_s
        drop = 0
        while drop < len(self._ts) and self._ts[drop] < cutoff:
            drop += 1
        if not drop:
            return
        del self._ts[:drop], self._alerted[:drop], self._x[:drop]
        for cache in self._h:
            del cache[:drop]
        self._evicted += drop

        for rel in self.relations:
            log, index = self._ins[rel], self._index[rel]
            while log and log[0][0] < self._evicted:
                idx, key = log.popleft()
                bucket = index.get(key)
                if bucket is None:
                    continue
                if bucket and bucket[0] == idx:
                    bucket.popleft()
                if not bucket:
                    del index[key]

    def _stack(self, indices: list[int], store: list[torch.Tensor]):
        if not indices:
            return None
        return torch.stack([store[i - self._evicted] for i in indices])

    def _ring_summary(self, ts: float) -> dict:
        """Rolling view of what has alerted inside the current window."""
        if ts == float("-inf"):
            return {"alerts_in_window": 0, "first_alert_ts": None,
                    "seconds_since_first_alert": None}
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



def replay(events: pd.DataFrame, detector: KoronisDetector,
           threshold: float, window_s: float | None = None) -> list[dict]:
    """Replay a whole stream event by event, in order."""
    stream = StreamingKoronis(detector, threshold, window_s)
    return [stream.push(row) for _, row in events.iterrows()]
