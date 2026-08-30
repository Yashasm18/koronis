import numpy as np
import pandas as pd


def reliability(scores: np.ndarray, labels: np.ndarray,
                bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed positive rate per score bin.

    A model can rank perfectly and still be badly calibrated. If it does, a
    threshold does not mean what its number implies, and any cost calculation
    built on top of it is wrong. Reporting this is the difference between a
    score and a probability.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({
            "bin_mid": (edges[b] + edges[b + 1]) / 2,
            "predicted": float(scores[m].mean()) if m.any() else np.nan,
            "observed": float(labels[m].mean()) if m.any() else np.nan,
            "count": int(m.sum()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray,
                               bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed rates."""
    r = reliability(scores, labels, bins).dropna()
    if r.empty:
        return float("nan")
    w = r["count"] / r["count"].sum()
    return float((w * (r["predicted"] - r["observed"]).abs()).sum())


def cost_optimal_threshold(scores: np.ndarray, labels: np.ndarray,
                           c_fn: float, c_fp: float) -> tuple[float, float]:
    """Threshold minimising total rupee cost, not F1.

    Scanning every candidate cut is O(n log n) here and exact, which is worth
    more than a clever approximation on datasets this size.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    candidates = np.unique(np.round(scores, 4))

    # Degenerate case: a detector that assigns every event the same score
    # carries no information. Thresholding at that value would fire on the
    # entire stream, which reads as a catastrophic false-positive rate rather
    # than what it is - a detector with nothing to say. Return a threshold
    # above the range so it abstains instead.
    if candidates.size <= 1:
        return float(candidates[0] + 1.0 if candidates.size else 1.0), float("inf")

    best_t, best_c = 0.5, float("inf")
    for t in candidates:
        pred = scores >= t
        cost = (c_fn * ((labels == 1) & ~pred).sum()
                + c_fp * ((labels == 0) & pred).sum())
        if cost < best_c:
            best_t, best_c = float(t), float(cost)
    return best_t, best_c
