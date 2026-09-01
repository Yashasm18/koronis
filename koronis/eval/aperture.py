"""Does the observer's vantage point change what is detectable?

A merchant sees its own checkout. A payment gateway sees thousands at once,
and a card-testing ring does not politely confine itself to one of them: the
economics push the opposite way, since spreading across merchants is another
axis on which per-merchant counters stay quiet.

This asks whether that wider aperture is worth anything, and predicts the
answer before measuring it.

PREDICTION. A campaign of `n` attempts across `k` entities, split evenly over
`M` merchants, puts `n/M` attempts in any one merchant's view. Co-occurrence
pairs sharing an entity go as attempts squared over spread, so a single
merchant sees about (n/M)^2 / k of them while the pooled stream sees n^2 / k.
The gateway's view therefore carries about `M` times the co-occurrence signal
per campaign, and splitting the same campaign across more merchants should
degrade a merchant-scoped detector roughly as if the campaign had shrunk by M
- while leaving the gateway-scoped one unchanged, since pooling reassembles
exactly the stream it would have seen anyway.

At M = 1 the two views are the same stream by construction, which is the
experiment's own control.
"""
import numpy as np
import pandas as pd

from ..data.background import load_background
from ..data.campaigns import inject
from ..data.schema import CampaignSpec, EVENT_COLUMNS

# Entity ids from the bootstrap sampler are plain ("d17", "i4"), so two
# merchant streams built with different seeds would still collide on them and
# appear to share devices. That is the same defect that once leaked entity
# identity between train and test, wearing a different hat: here it would
# manufacture cross-merchant links that no attacker created, and hand the
# gateway view a signal the experiment is supposed to be measuring.
_ENTITY_COLS = ["device_id", "ip_id", "bin_id", "card_id"]


def _namespaced_merchant(n_rows: int, seed: int, mid: int) -> pd.DataFrame:
    """One merchant's background traffic, with its entities kept its own."""
    bg = load_background(path=None, n_rows=n_rows, seed=seed)
    for col in _ENTITY_COLS:
        bg[col] = f"m{mid}_" + bg[col].astype(str)
    bg["event_id"] = f"m{mid}_" + bg["event_id"].astype(str)
    bg["merchant_id"] = f"m{mid}"
    return bg


def build_split_stream(n_merchants: int, n_attempts: int, k: int,
                       camouflage: float, seed: int, window_s: float,
                       n_background: int) -> pd.DataFrame:
    """One campaign, spread across `n_merchants` independent merchants.

    The campaign is generated once - so its entity pool, spread and timing are
    identical however many merchants it is later split over - and each attempt
    is then dealt to a merchant. That isolates the aperture: the only thing
    changing across the sweep is who gets to see which part of the same attack.
    """
    rng = np.random.default_rng(seed)
    merchants = [_namespaced_merchant(n_background, seed + 31 * m, m)
                 for m in range(n_merchants)]
    pooled_bg = pd.concat(merchants, ignore_index=True)

    # Campaign timing has to sit inside the window every merchant is live in,
    # or a "split" would also be a split in time.
    lo, hi = pooled_bg["ts"].min(), pooled_bg["ts"].max()
    spec = CampaignSpec(n_attempts=n_attempts, k_devices=k, k_ips=k, n_bins=k,
                        duration_s=window_s,
                        start_ts=float(lo + (hi - lo) * 0.25),
                        camouflage=camouflage)
    # inject() needs a background to bootstrap camouflaged marginals from; the
    # pooled frame is the right one, since the attacker blends into whatever
    # traffic it lands in.
    with_camp = inject(pooled_bg[EVENT_COLUMNS], [spec], seed=seed)
    camp = with_camp[with_camp["label"] == 1].copy()
    camp["merchant_id"] = ["m" + str(i) for i in
                           rng.integers(0, n_merchants, len(camp))]

    ev = pd.concat([pooled_bg, camp], ignore_index=True)
    ev = ev.sort_values("ts", kind="mergesort").reset_index(drop=True)
    return ev[EVENT_COLUMNS + ["merchant_id"]]


def _metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import average_precision_score
    fired = s >= thr
    tp = int((fired & y).sum())
    fp = int((fired & ~y).sum())
    fn = int((~fired & y).sum())
    return {
        "pr_auc": float(average_precision_score(y, s)) if y.any() else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "false_positives": fp,
        "detected": bool(tp > 0),
    }


def compare_apertures(model, thr: float, n_merchants_values: list[int],
                      n_attempts: int, k: int, camouflage: float,
                      seed: int, window_s: float,
                      n_background: int) -> pd.DataFrame:
    """Score the same attack twice: once per merchant, once pooled.

    Both views use the SAME frozen model and the SAME frozen threshold. The
    only difference is how much of the stream the detector is allowed to see
    at once, which is the whole question.
    """
    rows = []
    for m in n_merchants_values:
        ev = build_split_stream(m, n_attempts, k, camouflage, seed,
                                window_s, n_background)
        y_all = (ev["label"].to_numpy() == 1)

        # Gateway: one graph over everything.
        s_pooled = model.score_events(ev[EVENT_COLUMNS])

        # Merchant: each builds its own graph, and can only link what it saw.
        s_split = np.zeros(len(ev), dtype=float)
        for mid, part in ev.groupby("merchant_id", sort=False):
            s_split[part.index.to_numpy()] = model.score_events(
                part[EVENT_COLUMNS].reset_index(drop=True))

        # How much of the campaign any single merchant actually sees, and how
        # many campaign links survive the split - the mechanism, not the score.
        per_merchant = ev[ev["label"] == 1].groupby("merchant_id").size()
        for view, s in (("gateway", s_pooled), ("merchant", s_split)):
            rows.append({"n_merchants": m, "view": view,
                         "campaign_share_largest_merchant":
                             float(per_merchant.max() / per_merchant.sum()),
                         **_metrics(y_all, s, thr)})
    return pd.DataFrame(rows)
