from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec, EVENT_COLUMNS


def _bg():
    return load_background(path=None, n_rows=4000, seed=0)


def test_injects_exact_attempt_count_and_spread():
    bg = _bg()
    spec = CampaignSpec(n_attempts=200, k_devices=25, k_ips=10,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[100]))
    ev = inject(bg, [spec], seed=3)
    camp = ev[ev["label"] == 1]
    assert len(camp) == 200
    assert camp["device_id"].nunique() == 25
    assert camp["ip_id"].nunique() == 10
    assert camp["campaign_id"].nunique() == 1


def test_campaign_rows_look_like_card_testing():
    bg = _bg()
    spec = CampaignSpec(n_attempts=300, k_devices=20, k_ips=8,
                        duration_s=1800.0, start_ts=float(bg["ts"].iloc[50]))
    camp = inject(bg, [spec], seed=1).query("label == 1")
    assert camp["amount"].max() <= 25.0            # micro-amounts
    assert camp["card_id"].nunique() > 250          # many cards, few devices
    assert camp["approved"].mean() < 0.15           # mostly declines


def test_output_is_sorted_and_contract_preserved():
    bg = _bg()
    spec = CampaignSpec(n_attempts=50, k_devices=5, k_ips=3,
                        duration_s=600.0, start_ts=float(bg["ts"].iloc[10]))
    ev = inject(bg, [spec], seed=0)
    assert list(ev.columns) == EVENT_COLUMNS
    assert ev["ts"].is_monotonic_increasing
    assert len(ev) == len(bg) + 50
    assert ev["event_id"].is_unique
