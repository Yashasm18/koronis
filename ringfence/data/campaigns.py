import numpy as np
import pandas as pd

from .schema import CampaignSpec, EVENT_COLUMNS


def inject(background: pd.DataFrame, specs: list[CampaignSpec],
           seed: int) -> pd.DataFrame:
    """Add labeled card-testing campaigns to a background stream.

    Defense-only: this operates on in-memory dataframes, makes no network
    calls, and uses no real BIN ranges. It exists to produce labeled test data,
    which is what makes measured precision and recall possible at all.
    """
    rng = np.random.default_rng(seed)
    frames = [background]
    for c, spec in enumerate(specs):
        frames.append(_one_campaign(spec, f"camp_{c}", rng))
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.sort_values("ts", kind="mergesort").reset_index(drop=True)
    ev["event_id"] = [f"e_{i}" for i in range(len(ev))]
    return ev[EVENT_COLUMNS]


def _one_campaign(spec: CampaignSpec, cid: str, rng) -> pd.DataFrame:
    n = spec.n_attempts
    devices = np.array([f"{cid}_d{i}" for i in range(spec.k_devices)])
    ips = np.array([f"{cid}_i{i}" for i in range(spec.k_ips)])
    bins = np.array([f"{cid}_b{i}" for i in range(spec.n_bins)])

    # Round-robin assignment guarantees exactly k distinct entities appear,
    # which is what makes the (n, k) frontier sweep well defined.
    dev = devices[np.arange(n) % spec.k_devices]
    ip = ips[np.arange(n) % spec.k_ips]

    return pd.DataFrame({
        "event_id": [f"{cid}_{i}" for i in range(n)],
        "ts": np.sort(rng.uniform(spec.start_ts, spec.start_ts + spec.duration_s, n)),
        "amount": np.round(rng.uniform(1.0, 20.0, n), 2),
        "card_id": [f"{cid}_c{i}" for i in range(n)],   # a fresh card each attempt
        "bin_id": rng.choice(bins, n),
        "device_id": dev,
        "ip_id": ip,
        "email_domain": rng.choice(["gmail.com", "outlook.com"], n),
        "approved": rng.random(n) < 0.04,               # ~96% decline
        "label": 1,
        "campaign_id": cid,
    })
