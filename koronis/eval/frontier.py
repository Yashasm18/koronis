import numpy as np
import pandas as pd

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
    # The binding constraint is the most permissive counter: the campaign is
    # blind to the engine only once it clears every one of them.
    tau_max = max(taus[e] for e in VELOCITY_ENTITIES)

    rows = []
    for n in n_values:
        for k in k_values:
            spec = CampaignSpec(n_attempts=n, k_devices=k, k_ips=k, n_bins=k,
                                duration_s=window_s,
                                start_ts=float(bg["ts"].iloc[300]))
            ev = inject(bg, [spec], seed=seed)
            y = ev["label"].to_numpy() == 1

            vel = MultiEntityVelocityDetector(taus, window_s).score_events(ev)
            model = KoronisDetector(seed=seed, window_s=window_s)
            model.fit(ev, epochs=30)
            kor = model.score_events(ev)

            boundary = predicted_boundary_k(n, tau_max)
            rows.append({
                "n": n,
                "k": k,
                "velocity_detected": bool(vel[y].max() > 0),
                "koronis_detected": bool(kor[y].mean() > kor[~y].mean()),
                "predicted_k_boundary": boundary,
                "velocity_blind_predicted": bool(k >= boundary),
                "tau_max": tau_max,
            })
    return pd.DataFrame(rows)
