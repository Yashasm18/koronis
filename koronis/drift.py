"""Detect that live traffic no longer resembles what the model was calibrated on.

A detector that always answers is more dangerous than one that sometimes
declines. Thresholds, the incident risk model and the exposure forecast were
all fitted on a single traffic profile; on traffic shaped differently, they may
be confidently wrong. This module measures that gap and lets the policy stand
down.

The measure is the Population Stability Index, which is the standard drift
statistic in credit and payments risk — chosen over a fancier two-sample test
because a risk reviewer can read it, and because per-feature contributions say
*which* aspect of the traffic moved.

The alarm threshold is fitted ONLY on base calibration traffic: repeated
disjoint samples of the base profile give the null distribution of PSI, and the
cut-off is a high quantile of that. Nothing about the shifted profiles informs
it.
"""
import numpy as np
import pandas as pd

# Distributional views of the stream. The first two are inputs the detector
# consumes directly; the reuse counts describe the graph the model will build;
# inter-arrival captures a burst that leaves marginals unchanged.
FEATURES = ("log_amount", "declined", "reuse_device", "reuse_ip",
            "reuse_bin", "log_interarrival")


def signature(events: pd.DataFrame) -> pd.DataFrame:
    """Per-event view of the properties drift would move."""
    amt = events["amount"].to_numpy(dtype=float)
    ts = np.sort(events["ts"].to_numpy(dtype=float))
    gaps = np.diff(ts, prepend=ts[0])
    out = {"log_amount": np.log1p(amt),
           "declined": (~events["approved"].to_numpy().astype(bool)).astype(float),
           "log_interarrival": np.log1p(np.maximum(gaps, 0.0))}
    for col, name in (("device_id", "reuse_device"), ("ip_id", "reuse_ip"),
                      ("bin_id", "reuse_bin")):
        out[name] = events[col].map(events[col].value_counts()).to_numpy(dtype=float)
    return pd.DataFrame(out)


def _psi_one(base: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """PSI for one feature, binned on base quantiles."""
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.histogram(base, bins=edges)[0].astype(float)
    l = np.histogram(live, bins=edges)[0].astype(float)
    b, l = b / max(b.sum(), 1), l / max(l.sum(), 1)
    eps = 1e-4                       # keeps empty bins finite
    b, l = np.maximum(b, eps), np.maximum(l, eps)
    return float(np.sum((l - b) * np.log(l / b)))


def psi(base: pd.DataFrame, live: pd.DataFrame) -> dict:
    """Per-feature PSI plus the mean, which is the reported drift score."""
    per = {f: _psi_one(base[f].to_numpy(), live[f].to_numpy()) for f in FEATURES}
    per["overall"] = float(np.mean([per[f] for f in FEATURES]))
    return per


class DriftMonitor:
    """Flags traffic that does not resemble the calibration distribution.

    `fit` takes base-profile calibration streams only. The reference is their
    pooled signature; the threshold is a high quantile of PSI between disjoint
    base samples, so it answers "how much does base traffic vary against
    itself?" rather than being tuned to catch a particular shift.
    """

    def __init__(self, quantile: float = 0.95, seed: int = 0):
        self.quantile, self.seed = quantile, seed
        self.reference: pd.DataFrame | None = None
        self.threshold: float = float("inf")
        self.null_scores: list[float] = []
        # Mean events-per-entity on base traffic. Reported so a demo can quote
        # an OBSERVED reuse ratio rather than inferring one from a PSI value,
        # which says how much a distribution moved, not by what factor.
        self.base_reuse: dict[str, float] = {}

    def fit(self, base_streams: list[pd.DataFrame]) -> "DriftMonitor":
        sigs = [signature(s) for s in base_streams]
        self.reference = pd.concat(sigs, ignore_index=True)
        rng = np.random.default_rng(self.seed)

        # Null distribution: base against base. Halves of each stream, plus
        # stream-against-pool, so the threshold reflects ordinary variation.
        null = []
        for sg in sigs:
            idx = rng.permutation(len(sg))
            a, b = sg.iloc[idx[: len(idx) // 2]], sg.iloc[idx[len(idx) // 2:]]
            null.append(psi(a, b)["overall"])
            null.append(psi(self.reference, sg)["overall"])
        self.null_scores = sorted(null)
        self.threshold = float(np.quantile(null, self.quantile)) if null else float("inf")
        self.base_reuse = {c: float(self.reference[c].mean())
                           for c in ("reuse_device", "reuse_ip", "reuse_bin")}
        return self

    def score(self, events: pd.DataFrame) -> dict:
        if self.reference is None:
            return {"overall": 0.0}
        return psi(self.reference, signature(events))

    def check(self, events: pd.DataFrame) -> dict:
        s = self.score(events)
        worst = max((f for f in FEATURES), key=lambda f: s[f])
        sig = signature(events)
        reuse = {c: round(float(sig[c].mean()) / max(self.base_reuse.get(c, 1.0), 1e-9), 2)
                 for c in ("reuse_device", "reuse_ip", "reuse_bin")}
        return {
            "psi": round(s["overall"], 4),
            "threshold": round(self.threshold, 4),
            "drifted": bool(s["overall"] > self.threshold),
            "ratio": round(s["overall"] / max(self.threshold, 1e-9), 2),
            "largest_shift": worst,
            "per_feature": {f: round(s[f], 4) for f in FEATURES},
            # Measured, not inferred: observed mean reuse divided by the base
            # mean. Safe to quote in a demo; a PSI value is not.
            "reuse_vs_base": reuse,
        }
