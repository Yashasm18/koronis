import numpy as np
import torch

from ringfence.data.background import load_background
from ringfence.data.campaigns import inject
from ringfence.data.schema import CampaignSpec
from ringfence.models.loss import expected_cost_loss
from ringfence.models.ringfence import RingfenceDetector, node_features


def _data(seed):
    bg = load_background(path=None, n_rows=4000, seed=seed)
    spec = CampaignSpec(n_attempts=300, k_devices=30, k_ips=15,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[400]))
    return inject(bg, [spec], seed=seed)


def test_loss_penalises_confident_mistakes_most():
    y = torch.tensor([1.0, 1.0])
    confident_wrong = expected_cost_loss(torch.tensor([-6.0, -6.0]), y, 100.0, 10.0)
    confident_right = expected_cost_loss(torch.tensor([6.0, 6.0]), y, 100.0, 10.0)
    assert confident_wrong > confident_right


def test_loss_respects_cost_asymmetry():
    y = torch.tensor([0.0])
    logits = torch.tensor([3.0])                      # a false positive
    cheap = expected_cost_loss(logits, y, c_fn=100.0, c_fp=1.0)
    dear = expected_cost_loss(logits, y, c_fn=100.0, c_fp=50.0)
    assert dear > cheap


def test_node_features_shape_and_finiteness():
    f = node_features(_data(0))
    assert f.shape[1] == 6 and np.isfinite(f).all()


def test_ranks_campaign_above_background():
    tr, te = _data(0), _data(1)
    m = RingfenceDetector(seed=0)
    m.fit(tr, epochs=40)
    s = m.score_events(te)
    y = te["label"].to_numpy()
    assert s[y == 1].mean() > s[y == 0].mean()


def test_scores_unseen_entities_inductively():
    """Test entities share no ids with train — a transductive model would fail."""
    tr, te = _data(0), _data(1)
    assert not set(tr["device_id"]) & set(te.query("label == 1")["device_id"])
    m = RingfenceDetector(seed=0)
    m.fit(tr, epochs=20)
    assert np.isfinite(m.score_events(te)).all()
