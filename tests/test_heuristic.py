import numpy as np

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec
from koronis.models.heuristic import DeclineBurstDetector, SharedEntityDetector


def _data(k=60, camo=1.0, seed=0):
    bg = load_background(path=None, n_rows=4000, seed=seed)
    spec = CampaignSpec(n_attempts=300, k_devices=k, k_ips=k, n_bins=k,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[400]),
                        camouflage=camo)
    return inject(bg, [spec], seed=seed)


def test_shared_entity_returns_one_score_per_row():
    ev = _data()
    assert SharedEntityDetector().score_events(ev).shape == (len(ev),)


def test_shared_entity_ranks_campaign_above_background():
    """The graph signal must exist without learning, or the GNN has nothing
    to sharpen."""
    ev = _data()
    s = SharedEntityDetector().score_events(ev)
    y = ev["label"].to_numpy() == 1
    assert s[y].mean() > s[~y].mean()


def test_decline_burst_ranks_campaign_above_background():
    ev = _data()
    s = DeclineBurstDetector().score_events(ev)
    y = ev["label"].to_numpy() == 1
    assert s[y].mean() > s[~y].mean()


def test_decline_burst_uses_only_past_events():
    """No lookahead: an event's score may not depend on anything after it."""
    ev = _data()
    full = DeclineBurstDetector().score_events(ev)
    half = len(ev) // 2
    prefix = DeclineBurstDetector().score_events(ev.iloc[:half].reset_index(drop=True))
    assert np.allclose(full[:half], prefix, atol=1e-9)
