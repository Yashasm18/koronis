import numpy as np
import pandas as pd

from .cost import COST_PER_ATTEMPT_INR


def detection_times(events: pd.DataFrame, scores: np.ndarray,
                    threshold: float) -> dict[str, float | None]:
    """Seconds from each campaign's first attempt to its first alert.

    Measured from campaign onset, not from stream start: latency only means
    anything relative to when the attack actually began. `None` means the
    campaign was never detected at this threshold.
    """
    out: dict[str, float | None] = {}
    fired = np.asarray(scores) >= threshold
    ts = events["ts"].to_numpy()
    # Group the FULL frame: .indices on a filtered frame returns positions
    # relative to that subset, which would misalign against `fired`.
    # Background rows have a null campaign_id and are dropped by groupby.
    for cid, idx in events.groupby("campaign_id").indices.items():
        idx = np.sort(idx)
        onset = ts[idx[0]]
        hit = idx[fired[idx]]
        out[str(cid)] = float(ts[hit[0]] - onset) if len(hit) else None
    return out


def exposure(events: pd.DataFrame, campaign_id: str) -> float:
    """Total rupees the campaign costs if it is never stopped."""
    n = int((events["campaign_id"] == campaign_id).sum())
    return n * COST_PER_ATTEMPT_INR


def money_prevented(events: pd.DataFrame, detect_s: float | None,
                    campaign_id: str) -> float:
    """Exposure avoided by stopping the campaign `detect_s` after onset.

    Assumes the campaign is halted at the moment of detection, so everything
    that would have followed is prevented. That is the optimistic reading, and
    the README states it: what the number really measures is the *cost of
    latency*, i.e. how much more you lose for every minute you take to notice.
    """
    if detect_s is None:
        return 0.0
    camp = events[events["campaign_id"] == campaign_id]
    if camp.empty:
        return 0.0
    onset = camp["ts"].min()
    remaining = int((camp["ts"] > onset + detect_s).sum())
    return remaining * COST_PER_ATTEMPT_INR


def latency_curve(events: pd.DataFrame, scores: np.ndarray,
                  campaign_id: str, checkpoints_s: list[float],
                  threshold: float) -> pd.DataFrame:
    """Precision, recall and rupees prevented as a function of elapsed time.

    `threshold` must be derived once, from calibration data, and passed in
    frozen. Recomputing an operating point at each checkpoint would not measure
    one deployed detector over time - it would measure a sequence of
    differently tuned detectors, which is a different and much easier question.
    """
    ts = events["ts"].to_numpy()
    y = (events["label"].to_numpy() == 1)
    camp = events[events["campaign_id"] == campaign_id]
    onset = camp["ts"].min()
    scores = np.asarray(scores, dtype=float)

    rows = []
    for t in checkpoints_s:
        seen = ts <= onset + t
        if not seen.any():
            continue
        # A campaign is "caught by t" if any of its attempts seen so far fired.
        fired = seen & (scores >= threshold)
        tp = int((fired & y).sum())
        fp = int((fired & ~y).sum())
        fn = int((seen & y & ~fired).sum())
        detected = tp > 0
        rows.append({
            "t_seconds": t,
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
            "detected": detected,
            "inr_prevented": money_prevented(events, t if detected else None,
                                             campaign_id),
        })
    return pd.DataFrame(rows)
