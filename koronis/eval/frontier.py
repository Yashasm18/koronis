import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..data.background import load_background
from ..data.campaigns import inject
from ..data.schema import CampaignSpec
from ..models.koronis import KoronisDetector
from ..models.velocity import (
    MultiEntityVelocityDetector, VELOCITY_ENTITIES, tune_velocity,
)


def predicted_boundary_k(n: int, tau: int) -> float:
    """Claim 1: a threshold detector needs n/k > tau, so it goes blind at
    k >= n/tau. Below this k the campaign is loud enough on some entity to
    trip a counter; above it, every counter stays under threshold."""
    return n / tau


def sweep(n_values: list[int], k_values: list[int], fp_budget: float,
          seed: int, window_s: float = 3600.0,
          n_background: int = 4000) -> pd.DataFrame:
    """Run both detectors across the (n, k) grid and compare to the prediction.

    `k` is applied to every counted entity - devices, IPs and BIN ranges -
    because a campaign that spreads only some of them is caught by whichever
    counter it neglected. That finding is why this sweep is meaningful: the
    boundary is about total spread, not device spread.

    Thresholds are tuned once on clean background traffic to the stated
    false-positive budget, so the baseline is the strongest one a real team
    could deploy rather than an arbitrary pick.
    """
    bg = load_background(path=None, n_rows=n_background, seed=seed)
    taus = tune_velocity(bg, window_s=window_s, fp_budget=fp_budget)

    # Fit the graph model ONCE, on its own training stream, then evaluate every
    # cell without refitting. Fitting on each cell's own stream and scoring the
    # same stream would make the graph half of this result non-held-out - the
    # velocity half is a rule and never fits, so it was always clean.
    train_bg = load_background(path=None, n_rows=n_background, seed=seed + 100)
    train_specs = [
        CampaignSpec(n_attempts=400, k_devices=k, k_ips=k, n_bins=k,
                     duration_s=window_s,
                     start_ts=float(train_bg["ts"].min()
                                    + (train_bg["ts"].max() - train_bg["ts"].min())
                                    * (0.1 + 0.25 * i)),
                     camouflage=c)
        for i, (k, c) in enumerate([(4, 0.0), (12, 0.5), (30, 1.0)])
    ]
    train = inject(train_bg, train_specs, seed=seed + 100)
    model = KoronisDetector(seed=seed, window_s=window_s)
    model.fit(train, epochs=60)

    # Operating threshold comes from a THIRD stream - same distribution as
    # training, different draw - not from the training stream itself. A model
    # scores its own training data optimistically, so a quantile taken there
    # sits in the wrong place. Same three-split protocol as the ablation.
    calib_bg = load_background(path=None, n_rows=n_background, seed=seed + 200)
    calib_specs = [
        CampaignSpec(n_attempts=400, k_devices=k, k_ips=k, n_bins=k,
                     duration_s=window_s,
                     start_ts=float(calib_bg["ts"].min()
                                    + (calib_bg["ts"].max() - calib_bg["ts"].min())
                                    * (0.1 + 0.25 * i)),
                     camouflage=c)
        for i, (k, c) in enumerate([(4, 0.0), (12, 0.5), (30, 1.0)])
    ]
    calib = inject(calib_bg, calib_specs, seed=seed + 200)
    cal_scores = model.score_events(calib)
    thr = float(np.quantile(cal_scores, 1.0 - calib["label"].mean()))
    # A multi-entity engine fires if ANY counter trips, so a campaign is blind
    # only once it clears EVERY one. Blindness on entity e needs k >= n/tau_e,
    # so clearing all of them needs k >= max_e(n/tau_e) = n / min_e(tau_e).
    # The binding constraint is therefore the MOST SENSITIVE counter - the
    # smallest tau - not the largest. Using the largest understates the
    # boundary and would credit the baseline with blindness it does not have.
    tau_binding = min(taus[e] for e in VELOCITY_ENTITIES)

    rows = []
    for n in n_values:
        for k in k_values:
            spec = CampaignSpec(n_attempts=n, k_devices=k, k_ips=k, n_bins=k,
                                duration_s=window_s,
                                start_ts=float(bg["ts"].iloc[300]))
            ev = inject(bg, [spec], seed=seed)
            y = ev["label"].to_numpy() == 1

            vel = MultiEntityVelocityDetector(taus, window_s).score_events(ev)
            kor = model.score_events(ev)          # frozen model, unseen stream

            boundary = predicted_boundary_k(n, tau_binding)
            # "Did anything fire" is the right criterion for comparing against
            # a counter, which either trips or does not. It is far too weak to
            # locate the graph model's OWN failure boundary: one event above
            # threshold out of n counts as detection. Recall and PR-AUC are
            # carried alongside so saturation can be read off a graded curve.
            rows.append({
                "n": n,
                "k": k,
                "velocity_detected": bool(vel[y].max() > 0),
                # Same criterion as velocity: did any campaign event fire?
                "koronis_detected": bool((kor[y] >= thr).any()),
                "koronis_recall": round(float((kor[y] >= thr).mean()), 4),
                "koronis_pr_auc": round(float(average_precision_score(y, kor)), 4),
                "predicted_k_boundary": boundary,
                "velocity_blind_predicted": bool(k >= boundary),
                "tau_binding": tau_binding,
                "taus": str(taus),
            })
    return pd.DataFrame(rows)
