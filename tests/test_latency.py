import numpy as np

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec
from koronis.eval.latency import detection_times, money_prevented, exposure


def _data():
    bg = load_background(path=None, n_rows=2000, seed=0)
    spec = CampaignSpec(n_attempts=200, k_devices=10, k_ips=5,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[200]))
    return inject(bg, [spec], seed=0)


def test_perfect_scores_detect_at_first_attempt():
    ev = _data()
    scores = ev["label"].to_numpy().astype(float)
    assert detection_times(ev, scores, threshold=0.5)["camp_0"] == 0.0


def test_never_detected_returns_none():
    ev = _data()
    assert detection_times(ev, np.zeros(len(ev)), threshold=0.5)["camp_0"] is None


def test_earlier_detection_prevents_more_money():
    ev = _data()
    early = money_prevented(ev, 60.0, "camp_0")
    late = money_prevented(ev, 3000.0, "camp_0")
    assert early > late >= 0.0


def test_never_detecting_prevents_nothing():
    ev = _data()
    assert money_prevented(ev, None, "camp_0") == 0.0


def test_instant_detection_prevents_almost_all_exposure():
    ev = _data()
    total = exposure(ev, "camp_0")
    assert money_prevented(ev, 0.0, "camp_0") >= total * 0.95


def test_detection_time_is_measured_from_campaign_onset():
    """Not from stream start — latency is only meaningful relative to onset."""
    ev = _data()
    camp = ev[ev["label"] == 1]
    onset_idx = camp.index[0]
    scores = np.zeros(len(ev))
    # fire on the 50th campaign attempt
    fire_idx = camp.index[50]
    scores[fire_idx] = 1.0
    got = detection_times(ev, scores, threshold=0.5)["camp_0"]
    expected = ev["ts"].iloc[fire_idx] - ev["ts"].iloc[onset_idx]
    assert abs(got - expected) < 1e-6
