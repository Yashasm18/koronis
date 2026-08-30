from pathlib import Path

import numpy as np
import pandas as pd

from .schema import EVENT_COLUMNS


# Traffic density matters more than volume for a graph detector. Spreading the
# background thinly over weeks makes the problem far easier than reality: an
# injected campaign ends up being nearly all of the traffic in its own window,
# so its graph neighbourhood is almost pure campaign and separating it is
# trivial. Measured on an earlier 30-day span, the campaign was 97% of
# concurrent traffic and 96% of a campaign node's neighbours were campaign.
#
# A four-hour slice at this row count gives roughly 1,500 events/hour, so a
# 400-attempt campaign is a minority of the traffic it hides in - which is the
# regime a real merchant is actually in, and a much harder one.
DEFAULT_SPAN_S = 4 * 3600.0


def load_background(path: Path | None, n_rows: int, seed: int,
                    span_s: float = DEFAULT_SPAN_S) -> pd.DataFrame:
    """Return `n_rows` canonical background events, all label=0.

    When `path` points at IEEE-CIS train_transaction.csv the entity columns are
    derived from real fields, preserving real reuse structure. Otherwise a
    bootstrap sampler produces the same contract with plausible reuse.
    """
    rng = np.random.default_rng(seed)
    if path is not None and Path(path).exists():
        df = _from_ieee(Path(path), n_rows, rng)
    else:
        df = _bootstrap(n_rows, rng, span_s)

    df["label"] = 0
    df["campaign_id"] = pd.Series([None] * len(df), dtype="object")
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["event_id"] = [f"bg_{i}" for i in range(len(df))]
    return df[EVENT_COLUMNS]


def _from_ieee(path: Path, n_rows: int, rng) -> pd.DataFrame:
    cols = {"TransactionID", "TransactionDT", "TransactionAmt", "card1",
            "card2", "DeviceInfo", "P_emaildomain", "addr1", "isFraud"}
    raw = pd.read_csv(path, usecols=lambda c: c in cols, nrows=n_rows * 3)
    raw = raw.head(n_rows).copy()
    return pd.DataFrame({
        "ts": raw["TransactionDT"].astype(float).to_numpy(),
        "amount": raw["TransactionAmt"].astype(float).to_numpy(),
        "card_id": raw["card1"].astype("string").fillna("unk").to_numpy(),
        "bin_id": raw["card2"].astype("string").fillna("unk").to_numpy(),
        "device_id": raw["DeviceInfo"].astype("string").fillna("unk").to_numpy(),
        "ip_id": raw["addr1"].astype("string").fillna("unk").to_numpy(),
        "email_domain": raw["P_emaildomain"].astype("string").fillna("unk").to_numpy(),
        # IEEE-CIS has no auth outcome; approval is modelled from its fraud flag
        "approved": (raw["isFraud"].to_numpy() == 0),
    })


# How concentrated each entity type is, as (pool fraction, power-law exponent).
# Entity types do NOT share a distribution in reality, and treating them alike
# distorts the whole problem:
#   device  - near-unique per customer, a modest tail of repeat visitors
#   ip      - shared by offices and CGNAT, so a heavier tail than devices
#   bin     - genuinely shared: one issuer's BIN covers millions of cards
# A single earlier setting (exponent 1.6 for everything) put 42% of all traffic
# on one device. That is not a merchant's traffic, and it forced the velocity
# threshold so high that no campaign of any shape could trip it - which looked
# like a strong result for this project and was actually a broken simulation.
# (pool as a fraction of rows, power-law exponent alpha). Ids are drawn from an
# explicit p_i ~ 1/i^alpha over a fixed pool rather than from rng.zipf with a
# clamp: the clamp piles the entire tail onto a single id, which is how one
# device ended up carrying 42% of all traffic. Smaller alpha = flatter.
_ENTITY_SHAPE = {
    "device": (0.80, 0.45),
    "ip": (0.25, 0.70),
    "bin": (0.02, 0.90),
}


def _bootstrap(n_rows: int, rng, span_s: float = DEFAULT_SPAN_S) -> pd.DataFrame:
    """Realistic entity reuse: a long tail, but no single dominant entity.

    The reuse structure is the part that matters — it is what produces
    legitimate dense subgraphs, and therefore honest false positives.
    """
    def zipf_ids(prefix: str, kind: str) -> np.ndarray:
        frac, alpha = _ENTITY_SHAPE[kind]
        pool = max(int(n_rows * frac), 2)
        ranks = np.arange(1, pool + 1, dtype=float)
        probs = ranks ** (-alpha)
        probs /= probs.sum()
        idx = rng.choice(pool, size=n_rows, p=probs)
        return np.array([f"{prefix}{i}" for i in idx])

    ts = np.sort(rng.uniform(0, span_s, n_rows))
    return pd.DataFrame({
        "ts": ts,
        "amount": np.round(rng.lognormal(6.2, 1.1, n_rows), 2),
        "card_id": np.array([f"c{i}" for i in range(n_rows)]),
        "bin_id": zipf_ids("b", "bin"),
        "device_id": zipf_ids("d", "device"),
        "ip_id": zipf_ids("i", "ip"),
        "email_domain": rng.choice(
            ["gmail.com", "yahoo.com", "outlook.com", "rediff.com", "proton.me"],
            n_rows, p=[0.55, 0.15, 0.15, 0.10, 0.05]),
        "approved": rng.random(n_rows) > 0.08,
    })
