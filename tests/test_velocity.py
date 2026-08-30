from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec
from koronis.models.velocity import VelocityDetector


def test_catches_concentrated_campaign():
    bg = load_background(path=None, n_rows=3000, seed=0)
    spec = CampaignSpec(n_attempts=300, k_devices=2, k_ips=2,
                        duration_s=1800.0, start_ts=float(bg["ts"].iloc[100]))
    ev = inject(bg, [spec], seed=0)
    s = VelocityDetector(tau=40, window_s=3600.0).score_events(ev)
    assert s[ev["label"].to_numpy() == 1].max() > 0


def test_blind_to_spread_campaign():
    """Claim 1: with k >= n/tau every entity stays under threshold."""
    bg = load_background(path=None, n_rows=3000, seed=0)
    spec = CampaignSpec(n_attempts=300, k_devices=200, k_ips=200,
                        duration_s=1800.0, start_ts=float(bg["ts"].iloc[100]))
    ev = inject(bg, [spec], seed=0)
    s = VelocityDetector(tau=40, window_s=3600.0).score_events(ev)
    assert s[ev["label"].to_numpy() == 1].max() == 0


def test_returns_one_score_per_row():
    ev = load_background(path=None, n_rows=500, seed=1)
    assert VelocityDetector(tau=10, window_s=600.0).score_events(ev).shape == (500,)
