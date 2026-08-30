"""Forecast how much of an incident is still ahead, using only what it has shown.

The action policy needs to know what inaction would cost. Offline that can be
read off the campaign log, but a live system does not know how many attempts
are still coming — so a policy built on the true remaining count is an *oracle*,
not a product. This module replaces that input with a causal forecast.

The target is deliberately label-free: **how many more alerted events will join
this incident**. That is a structural quantity, observable in hindsight without
anyone deciding whether the incident was genuine. Whether it matters is a
separate question answered by the incident risk model, and the policy multiplies
the two:

    expected remaining exposure  =  P(genuine) × forecast(remaining attempts) × cost

Forecasts come with an interval, because a system that cannot tell the future
should say so rather than guess confidently. The conformal pad is fit on a
held-out subset of CALIBRATION incidents, and coverage is then evaluated on
held-out TEST incidents — so the stated interval is measured on incidents
nothing in the pipeline has seen.
"""
import numpy as np
import pandas as pd

import lightgbm as lgb

from .eval.cost import COST_PER_ATTEMPT_INR
from .incident import Incident

FEATURES = [
    "n_observed", "log_n", "age_s", "log_age", "rate", "rate_trend",
    "mean_score", "score_trend", "decline_ratio",
    "n_devices", "n_ips", "n_bins", "entity_density",
]

# Where along an incident to take snapshots. A forecaster only ever sees a
# prefix, so it must be trained on prefixes of every length it will meet.
SNAPSHOT_AT = (3, 5, 8, 12, 20, 32, 50, 80, 128, 200, 320)


def snapshot_features(events: pd.DataFrame, rows: list[int],
                      scores: np.ndarray, m: int) -> dict:
    """Features from the first `m` events of an incident — nothing later."""
    seen = rows[:m]
    blk = events.iloc[seen]
    ts = blk["ts"].to_numpy()
    sc = np.asarray(scores)[seen]
    age = float(ts[-1] - ts[0])
    half = max(m // 2, 1)
    early_span = max(float(ts[half - 1] - ts[0]), 1e-6)
    late_span = max(float(ts[-1] - ts[half - 1]), 1e-6)
    dev, ip, bn = (int(blk[c].nunique()) for c in ("device_id", "ip_id", "bin_id"))
    return {
        "n_observed": float(m),
        "log_n": float(np.log1p(m)),
        "age_s": age,
        "log_age": float(np.log1p(age)),
        "rate": m / max(age, 1.0),
        "rate_trend": (half / late_span) / max(half / early_span, 1e-9),
        "mean_score": float(sc.mean()),
        "score_trend": float(sc[half:].mean() - sc[:half].mean()) if m > 1 else 0.0,
        "decline_ratio": float((~blk["approved"].to_numpy().astype(bool)).mean()),
        "n_devices": float(dev), "n_ips": float(ip), "n_bins": float(bn),
        "entity_density": m / max(dev + ip + bn, 1),
    }


def build_snapshots(events: pd.DataFrame, incidents: list[Incident],
                    scores: np.ndarray, stream_id: int = 0) -> pd.DataFrame:
    """Prefix snapshots of every incident, with the remaining count as target.

    `stream_id` qualifies the incident id. Ids restart at INC-000 for every
    stream, so grouping on the bare id would silently merge unrelated incidents
    from different streams into one partition key.
    """
    rows = []
    for inc in incidents:
        total = len(inc.rows)
        for m in SNAPSHOT_AT:
            if m >= total:
                break
            f = snapshot_features(events, inc.rows, scores, m)
            f["remaining"] = float(total - m)
            f["incident_id"] = inc.incident_id
            f["stream_id"] = int(stream_id)
            f["group_id"] = f"{stream_id}:{inc.incident_id}"
            f["t_snapshot"] = float(events["ts"].to_numpy()[inc.rows[m - 1]])
            rows.append(f)
    return pd.DataFrame(rows)


class ExposureForecaster:
    """Quantile regression on remaining attempts, with conformalised coverage.

    Two quantile models (median and upper) are fitted on one subset of the
    calibration data; a disjoint subset is used only to measure how far the
    upper model falls short, and to widen it by that amount. Raw quantile
    regression is routinely over-confident, and a stated 90% interval that
    covers 60% of the time is worse than no interval at all.

    **The split is by stream, never by snapshot row.** Snapshots of one
    incident are nested prefixes of the same sequence and are therefore highly
    dependent: putting one prefix in the fit set and another in the conformal
    set measures the residual on data the model has effectively already seen,
    which inflates apparent coverage without being a test-set leak. Whole
    streams go to one side or the other, so the conformal pad is estimated on
    incidents the quantile models never met.
    """

    def __init__(self, upper_q: float = 0.9, seed: int = 0):
        self.upper_q = upper_q
        self.seed = seed
        self.m50 = self.mhi = None
        self.conformal_pad = 0.0
        self.fit_groups_: list = []
        self.conformal_groups_: list = []

    def fit(self, snaps: pd.DataFrame) -> "ExposureForecaster":
        if snaps.empty:
            return self
        x, y = snaps[FEATURES].to_numpy(), snaps["remaining"].to_numpy()

        # Partition by stream where available, else by stream-qualified
        # incident. Never by row.
        key = "stream_id" if "stream_id" in snaps else "group_id"
        groups = snaps[key].to_numpy() if key in snaps else np.arange(len(y))
        uniq = np.unique(groups)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(uniq)
        n_fit = max(int(len(uniq) * 0.6), 1)
        if len(uniq) > 1:
            n_fit = min(n_fit, len(uniq) - 1)     # always leave a conformal set
        fit_groups = set(order[:n_fit].tolist())
        in_fit = np.array([g in fit_groups for g in groups])
        fit_i = np.flatnonzero(in_fit)
        cal_i = np.flatnonzero(~in_fit)
        self.fit_groups_ = sorted(fit_groups)
        self.conformal_groups_ = sorted(set(order[n_fit:].tolist()))
        if fit_i.size == 0:                        # degenerate single group
            fit_i, cal_i = np.arange(len(y)), np.array([], dtype=int)

        def _q(alpha, xi, yi):
            m = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                                  n_estimators=250, learning_rate=0.06,
                                  num_leaves=15, min_child_samples=8,
                                  random_state=self.seed, verbose=-1)
            m.fit(xi, yi)
            return m

        self.m50 = _q(0.5, x[fit_i], y[fit_i])
        self.mhi = _q(self.upper_q, x[fit_i], y[fit_i])

        if cal_i.size:
            # Conformal residual: how far the upper model undershoots on
            # INCIDENTS it never saw. Pad by that quantile so the stated
            # coverage means something on genuinely new incidents.
            short = y[cal_i] - self.mhi.predict(x[cal_i])
            self.conformal_pad = float(max(np.quantile(short, self.upper_q), 0.0))
        return self

    def predict(self, snaps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.m50 is None or snaps.empty:
            n = len(snaps)
            return np.zeros(n), np.zeros(n)
        x = snaps[FEATURES].to_numpy()
        p50 = np.maximum(self.m50.predict(x), 0.0)
        hi = np.maximum(self.mhi.predict(x) + self.conformal_pad, p50)
        return p50, hi

    def predict_one(self, events: pd.DataFrame, rows: list[int],
                    scores: np.ndarray, m: int) -> tuple[float, float]:
        f = snapshot_features(events, rows, scores, m)
        df = pd.DataFrame([f])
        p50, hi = self.predict(df)
        return float(p50[0]), float(hi[0])


def evaluate_forecast(fc: ExposureForecaster, snaps: pd.DataFrame) -> dict:
    """Coverage and error on held-out snapshots.

    Coverage is the number that matters: an interval claiming 90% must contain
    the truth about 90% of the time, or the policy built on it is guessing with
    extra steps.
    """
    if snaps.empty:
        return {"n": 0}
    p50, hi = fc.predict(snaps)
    y = snaps["remaining"].to_numpy()
    return {
        "n_snapshots": int(len(y)),
        "coverage_upper": round(float((y <= hi).mean()), 4),
        "target_coverage": fc.upper_q,
        "mae_p50": round(float(np.abs(y - p50).mean()), 2),
        "median_abs_err_p50": round(float(np.median(np.abs(y - p50))), 2),
        "mean_true_remaining": round(float(y.mean()), 2),
        "mean_p50": round(float(p50.mean()), 2),
        "mean_upper": round(float(hi.mean()), 2),
        "mae_exposure_inr": round(float(np.abs(y - p50).mean() * COST_PER_ATTEMPT_INR), 2),
    }
