import numpy as np

from ringfence.data.background import load_background
from ringfence.data.campaigns import inject
from ringfence.data.schema import CampaignSpec
from ringfence.models.velocity import (
    MultiEntityVelocityDetector, tune_velocity, false_positive_rate,
)


def _bg(seed=0, n=4000):
    return load_background(path=None, n_rows=n, seed=seed)


def test_tuning_respects_the_false_positive_budget():
    bg = _bg()
    taus = tune_velocity(bg, window_s=3600.0, fp_budget=0.01)
    det = MultiEntityVelocityDetector(taus, window_s=3600.0)
    assert false_positive_rate(det.score_events(bg)) <= 0.01


def test_tighter_budget_forces_a_higher_threshold():
    """This is the tau-floor of Claim 1, measured rather than assumed."""
    bg = _bg()
    loose = tune_velocity(bg, window_s=3600.0, fp_budget=0.05)
    tight = tune_velocity(bg, window_s=3600.0, fp_budget=0.001)
    assert all(tight[e] >= loose[e] for e in loose)


def test_tuned_baseline_still_catches_a_concentrated_campaign():
    """A fair baseline must be genuinely strong where it should be strong."""
    bg = _bg()
    taus = tune_velocity(bg, window_s=3600.0, fp_budget=0.01)
    spec = CampaignSpec(n_attempts=400, k_devices=2, k_ips=2,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[300]))
    ev = inject(bg, [spec], seed=0)
    s = MultiEntityVelocityDetector(taus, window_s=3600.0).score_events(ev)
    assert s[ev["label"].to_numpy() == 1].max() > 0


def test_partial_spread_is_still_caught():
    """The sharpened Claim 1: spreading SOME entities is not enough.

    An attacker who rotates 400 devices and 400 IPs but enumerates only 2 BIN
    ranges is caught by the BIN counter. This is a result, not a limitation —
    it says which counters actually carry weight, and it is why the baseline
    has to be multi-entity to be fair.
    """
    bg = _bg()
    taus = tune_velocity(bg, window_s=3600.0, fp_budget=0.01)
    spec = CampaignSpec(n_attempts=400, k_devices=400, k_ips=400, n_bins=2,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[300]))
    ev = inject(bg, [spec], seed=0)
    s = MultiEntityVelocityDetector(taus, window_s=3600.0).score_events(ev)
    assert s[ev["label"].to_numpy() == 1].max() > 0


def test_tuned_baseline_blind_to_fully_spread_campaign():
    """Claim 1: the engine is blind only when k >= n/tau on EVERY counted
    entity. That is the morphology the graph model has to earn its keep on."""
    bg = _bg()
    taus = tune_velocity(bg, window_s=3600.0, fp_budget=0.01)
    spec = CampaignSpec(n_attempts=400, k_devices=400, k_ips=400, n_bins=400,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[300]))
    ev = inject(bg, [spec], seed=0)
    s = MultiEntityVelocityDetector(taus, window_s=3600.0).score_events(ev)
    assert s[ev["label"].to_numpy() == 1].max() == 0
