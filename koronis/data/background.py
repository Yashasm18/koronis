from pathlib import Path

import numpy as np
import pandas as pd

from .schema import EVENT_COLUMNS


def load_background(path: Path | None, n_rows: int, seed: int) -> pd.DataFrame:
    """Return `n_rows` canonical background events, all label=0.

    When `path` points at IEEE-CIS train_transaction.csv the entity columns are
    derived from real fields, preserving real reuse structure. Otherwise a
    bootstrap sampler produces the same contract with plausible reuse.
    """
    rng = np.random.default_rng(seed)
    if path is not None and Path(path).exists():
        df = _from_ieee(Path(path), n_rows, rng)
    else:
        df = _bootstrap(n_rows, rng)

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


def _bootstrap(n_rows: int, rng) -> pd.DataFrame:
    """Zipfian entity reuse: a few heavy sharers (offices, CGNAT), a long tail.

    The reuse structure is the part that matters — it is what produces
    legitimate dense subgraphs, and therefore honest false positives.
    """
    def zipf_ids(prefix: str, pool: int) -> np.ndarray:
        idx = np.minimum(rng.zipf(1.6, n_rows), max(pool, 1))
        return np.array([f"{prefix}{i}" for i in idx])

    ts = np.sort(rng.uniform(0, 30 * 86400, n_rows))
    return pd.DataFrame({
        "ts": ts,
        "amount": np.round(rng.lognormal(6.2, 1.1, n_rows), 2),
        "card_id": zipf_ids("c", n_rows),
        "bin_id": zipf_ids("b", 400),
        "device_id": zipf_ids("d", int(n_rows * 0.55)),
        "ip_id": zipf_ids("i", int(n_rows * 0.35)),
        "email_domain": rng.choice(
            ["gmail.com", "yahoo.com", "outlook.com", "rediff.com", "proton.me"],
            n_rows, p=[0.55, 0.15, 0.15, 0.10, 0.05]),
        "approved": rng.random(n_rows) > 0.08,
    })
