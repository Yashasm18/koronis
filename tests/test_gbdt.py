from ringfence.data.background import load_background
from ringfence.data.campaigns import inject
from ringfence.data.schema import CampaignSpec
from ringfence.models.gbdt import GBDTDetector, transaction_features


def _data(seed=0):
    bg = load_background(path=None, n_rows=6000, seed=seed)
    specs = [CampaignSpec(n_attempts=250, k_devices=8, k_ips=4,
                          duration_s=3600.0, start_ts=float(bg["ts"].iloc[500]))]
    return inject(bg, specs, seed=seed)


def test_features_have_no_leakage_columns():
    f = transaction_features(_data())
    for banned in ("label", "campaign_id", "event_id"):
        assert banned not in f.columns


def test_learns_something_on_concentrated_campaign():
    tr, te = _data(0), _data(1)
    m = GBDTDetector(seed=0)
    m.fit(tr)
    s = m.score_events(te)
    y = te["label"].to_numpy()
    assert s[y == 1].mean() > s[y == 0].mean()


def test_returns_one_score_per_row():
    ev = _data()
    m = GBDTDetector(seed=0)
    m.fit(ev)
    assert m.score_events(ev).shape == (len(ev),)
