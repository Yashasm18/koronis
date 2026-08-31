"""The architecture switches must actually change the architecture.

An ablation that silently does nothing produces a null result and an
unjustified sense of safety, so each switch is checked at the tensor level
rather than by trusting the flag.
"""
import numpy as np
import torch

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec, RELATIONS
from koronis.models.koronis import KoronisDetector
from koronis.models.layers import RelationalLayer


def _stream(seed=0, n=800):
    bg = load_background(path=None, n_rows=n, seed=seed)
    spec = CampaignSpec(n_attempts=80, k_devices=8, k_ips=8, n_bins=8,
                        duration_s=600.0, start_ts=float(bg["ts"].iloc[60]),
                        camouflage=1.0)
    return inject(bg, [spec], seed=seed)


def test_uniform_attention_is_actually_uniform():
    layer = RelationalLayer(4, 4, len(RELATIONS), use_rel_attention=False)
    with torch.no_grad():
        layer.rel_att.copy_(torch.tensor([3.0, -2.0, 0.5, 1.0]))  # far from equal
    att = layer._attention()
    assert torch.allclose(att, torch.full_like(att, 1.0 / len(RELATIONS)))


def test_learned_attention_responds_to_its_parameter():
    layer = RelationalLayer(4, 4, len(RELATIONS), use_rel_attention=True)
    with torch.no_grad():
        layer.rel_att.copy_(torch.tensor([3.0, -2.0, 0.5, 1.0]))
    att = layer._attention().detach()
    assert att.argmax().item() == 0 and abs(float(att.sum()) - 1.0) < 1e-6


def test_gate_off_weights_every_edge_equally():
    """With the gate off the layer must ignore its gate parameters entirely."""
    x = torch.randn(6, 4)
    edges = [torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
             for _ in RELATIONS]
    layer = RelationalLayer(4, 4, len(RELATIONS), use_gate=False)
    before = layer(x, edges)
    with torch.no_grad():                       # perturb the gate hard
        for prm in layer.gate.parameters():
            prm.add_(5.0)
    assert torch.allclose(before, layer(x, edges)), "gate still influencing output"


def test_gate_on_does_influence_the_output():
    x = torch.randn(6, 4)
    edges = [torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
             for _ in RELATIONS]
    layer = RelationalLayer(4, 4, len(RELATIONS), use_gate=True)
    before = layer(x, edges)
    with torch.no_grad():
        for prm in layer.gate.parameters():
            prm.add_(5.0)
    assert not torch.allclose(before, layer(x, edges))


def test_switches_reach_the_detector_and_change_its_scores():
    ev = _stream()
    base = KoronisDetector(seed=0, window_s=600.0)
    base.fit(ev, epochs=8)
    for kw in (dict(use_gate=False), dict(use_rel_attention=False), dict(layers=1)):
        alt = KoronisDetector(seed=0, window_s=600.0, **kw)
        alt.fit(ev, epochs=8)
        assert not np.allclose(base.score_events(ev), alt.score_events(ev)), \
            f"{kw} produced identical scores — the switch is not wired through"


def test_one_layer_really_builds_one_layer():
    ev = _stream()
    m = KoronisDetector(seed=0, window_s=600.0, layers=1)
    m.fit(ev, epochs=2)
    assert len(m.net.layers) == 1
