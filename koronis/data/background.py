from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .schema import EVENT_COLUMNS

if TYPE_CHECKING:                      # avoids a circular import at runtime
    from ..profiles import Profile


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
                    span_s: float | None = None,
                    profile: "Profile | None" = None) -> pd.DataFrame:
    """Return `n_rows` canonical background events, all label=0.

    `path` selects an unfinished IEEE-CIS loader; see `_from_ieee`, which raises
    rather than pretending. Every published number comes from the bootstrap
    sampler, which produces the canonical contract with plausible reuse.
    """
    rng = np.random.default_rng(seed)
    if path is not None and Path(path).exists():
        df = _from_ieee(Path(path), n_rows, rng)
    else:
        from ..profiles import BASE
        prof = profile or BASE
        df = _bootstrap(n_rows, rng,
                        span_s if span_s is not None else prof.span_s,
                        prof.entity_shape, prof.decline_rate)

    df["label"] = 0
    df["campaign_id"] = pd.Series([None] * len(df), dtype="object")
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["event_id"] = [f"bg_{i}" for i in range(len(df))]
    return df[EVENT_COLUMNS]


#: Columns `_from_ieee` needs, and the file each actually lives in.
_IEEE_COLUMNS = {
    "TransactionID": "train_transaction.csv",
    "TransactionDT": "train_transaction.csv",
    "TransactionAmt": "train_transaction.csv",
    "card1": "train_transaction.csv",
    "card2": "train_transaction.csv",
    "P_emaildomain": "train_transaction.csv",
    "addr1": "train_transaction.csv",
    "isFraud": "train_transaction.csv",
    "DeviceInfo": "train_identity.csv",        # NOT in train_transaction
}


def _from_ieee(path: Path, n_rows: int, rng) -> pd.DataFrame:
    """Unfinished. Reads IEEE-CIS if the columns are there, and says so if not.

    This used to be described as a working alternative background. It is not:
    `DeviceInfo` lives in `train_identity.csv`, not `train_transaction.csv`, so
    on the file the old docstring named `usecols` silently dropped it and the
    frame lookup below raised a bare KeyError. Joining identity supplies it on
    only ~24% of rows, and `device_id` is one of three model relations.

    Two further reasons the path was never finished, both measured and recorded
    in docs/limitations.md: a contiguous slice runs at ~210 events/hour against
    the ~1,500 the simulator is tuned to, which is the thin-traffic regime that
    made an injected campaign trivially separable (defect 6); and IEEE-CIS has
    no authorisation outcome at all, so `approved` would be modelled from
    `isFraud` - synthesising the very mechanism the detector leans on.

    It raises instead of half-working, because a loader that silently produces a
    degraded background is how a published number goes quietly wrong.
    """
    raw = pd.read_csv(path, usecols=lambda c: c in _IEEE_COLUMNS, nrows=n_rows * 3)
    missing = [c for c in _IEEE_COLUMNS if c not in raw.columns]
    if missing:
        where = {c: _IEEE_COLUMNS[c] for c in missing}
        raise NotImplementedError(
            f"the IEEE-CIS loader is unfinished: {path.name} has no {missing}. "
            f"Those columns live in {sorted(set(where.values()))} and joining them "
            "is not implemented - DeviceInfo covers only ~24% of rows even after "
            "the join, and device_id is one of three model relations. See "
            "docs/limitations.md for the density measurement behind not using "
            "this dataset. Pass path=None to use the bootstrap sampler, which is "
            "what every published number uses.")
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


def _bootstrap(n_rows: int, rng, span_s: float = DEFAULT_SPAN_S,
               entity_shape: dict | None = None,
               decline_rate: float = 0.08) -> pd.DataFrame:
    """Realistic entity reuse: a long tail, but no single dominant entity.

    The reuse structure is the part that matters — it is what produces
    legitimate dense subgraphs, and therefore honest false positives.
    """
    shape = entity_shape or _ENTITY_SHAPE

    def zipf_ids(prefix: str, kind: str) -> np.ndarray:
        frac, alpha = shape[kind]
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
        "approved": rng.random(n_rows) > decline_rate,
    })
