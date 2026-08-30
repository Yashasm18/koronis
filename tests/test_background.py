import pandas as pd
from ringfence.data.background import load_background
from ringfence.data.schema import EVENT_COLUMNS


def test_background_has_contract_columns_and_no_labels():
    ev = load_background(path=None, n_rows=2000, seed=0)
    assert list(ev.columns) == EVENT_COLUMNS
    assert len(ev) == 2000
    assert (ev["label"] == 0).all()
    assert ev["campaign_id"].isna().all()
    assert ev["ts"].is_monotonic_increasing


def test_background_is_deterministic_under_seed():
    a = load_background(path=None, n_rows=500, seed=7)
    b = load_background(path=None, n_rows=500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_background_entities_are_reused_not_unique():
    ev = load_background(path=None, n_rows=5000, seed=1)
    # real traffic shares devices; unique-per-row would make the graph trivial
    assert ev["device_id"].nunique() < len(ev) * 0.8
