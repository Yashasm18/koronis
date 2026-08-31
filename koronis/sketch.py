"""Approximate frequency over a sliding window, in fixed memory.

The incident layer needs one thing from history: *how common is this entity
value?* A value covering half the stream is not evidence that two alerts belong
together; a device fingerprint covering 0.03% of it is. Answering that exactly
means a counter per distinct value, and at gateway volumes the distinct-value
count is the thing that grows without bound - a card-testing campaign alone
mints a fresh card id per attempt.

A count-min sketch answers the same question in memory fixed by the sketch
dimensions rather than by cardinality, which is the standard streaming answer to
point-frequency estimation. Windowing is a ring of sketches: each covers a slice
of the window, the estimate sums the ring, and a slice is zeroed and reused when
time moves past it. Nothing is ever deleted individually, so eviction costs
nothing.

**The error is one-sided, and it points the safe way.** A count-min sketch never
underestimates; collisions can only inflate a count. An inflated count makes a
value look *more* common, and a value that looks common is excluded from linking
alerts. So the failure mode is a missed link - two alerts of one campaign kept
apart - rather than a false link that merges two unrelated rings into one
incident and produces a single action for two different attackers. Fragmenting
is recoverable by an analyst; a wrong merge silently hides an attack.

Defense-only: pure arithmetic over in-memory counters.
"""
from __future__ import annotations


# Odd multipliers drawn once, so hashing is deterministic across runs and
# processes - two shards must agree on which counters a value touches.
_MIX = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1)


class SlidingCountMin:
    """Point-frequency estimates over the last `window_s` seconds.

    `width` counters per row, `depth` rows, `slices` ring positions. Memory is
    `depth * width * slices` integers regardless of how many distinct values
    the stream contains.
    """

    def __init__(self, window_s: float, width: int = 2048, depth: int = 4,
                 slices: int = 8):
        if depth > len(_MIX):
            raise ValueError(f"depth must be <= {len(_MIX)}")
        self.window_s = float(window_s)
        self.width, self.depth, self.slices = width, depth, slices
        self.slice_s = self.window_s / slices
        self._ring = [[[0] * width for _ in range(depth)] for _ in range(slices)]
        self._slice_start = [None] * slices        # wall time each slice covers
        self._total = [0] * slices

    # ---------------------------------------------------------------- hashing

    def _cells(self, key: str) -> list[int]:
        h = hash(key) & 0xFFFFFFFF
        return [((h ^ _MIX[d]) * 0x01000193 & 0xFFFFFFFF) % self.width
                for d in range(self.depth)]

    def _slot(self, ts: float) -> int:
        """Ring position for `ts`, recycling any slice that has aged out."""
        epoch = int(ts // self.slice_s)
        pos = epoch % self.slices
        if self._slice_start[pos] != epoch:        # stale slice: reuse it
            for row in self._ring[pos]:
                for i in range(self.width):
                    row[i] = 0
            self._total[pos] = 0
            self._slice_start[pos] = epoch
        return pos

    # ------------------------------------------------------------------- api

    def add(self, key: str, ts: float) -> None:
        pos = self._slot(ts)
        cells = self._cells(key)
        for d, c in enumerate(cells):
            self._ring[pos][d][c] += 1
        self._total[pos] += 1

    def estimate(self, key: str, ts: float) -> int:
        """Upper bound on how often `key` occurred inside the window at `ts`."""
        self._slot(ts)                              # retire anything stale first
        live = self._live(ts)
        cells = self._cells(key)
        return min(sum(self._ring[p][d][cells[d]] for p in live)
                   for d in range(self.depth))

    def total(self, ts: float) -> int:
        """Events observed inside the window - the denominator for a share."""
        self._slot(ts)
        return sum(self._total[p] for p in self._live(ts))

    def share(self, key: str, ts: float) -> float:
        t = self.total(ts)
        return (self.estimate(key, ts) / t) if t else 0.0

    def _live(self, ts: float) -> list[int]:
        epoch = int(ts // self.slice_s)
        return [p for p in range(self.slices)
                if self._slice_start[p] is not None
                and epoch - self._slice_start[p] < self.slices]

    def memory_bytes(self) -> int:
        """Counters only - the point is that this does not depend on the data."""
        return self.depth * self.width * self.slices * 8
