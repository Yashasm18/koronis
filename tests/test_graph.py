import numpy as np

from ringfence.data.background import load_background
from ringfence.data.schema import RELATIONS
from ringfence.graph.build import build_edges


def test_edges_only_join_same_entity_within_window():
    ev = load_background(path=None, n_rows=800, seed=0)
    edges = build_edges(ev, window_s=3600.0)
    assert set(edges.keys()) == set(RELATIONS)
    for rel, ei in edges.items():
        assert ei.shape[0] == 2
        src, dst = ei
        assert (ev[rel].to_numpy()[src] == ev[rel].to_numpy()[dst]).all()
        dt = np.abs(ev["ts"].to_numpy()[src] - ev["ts"].to_numpy()[dst])
        assert (dt <= 3600.0).all()


def test_no_self_loops():
    ev = load_background(path=None, n_rows=400, seed=1)
    for ei in build_edges(ev, window_s=3600.0).values():
        assert (ei[0] != ei[1]).all()


def test_edges_point_backwards_in_time():
    """No lookahead: a node may only aggregate from its own past."""
    ev = load_background(path=None, n_rows=600, seed=3)
    ts = ev["ts"].to_numpy()
    for ei in build_edges(ev, window_s=3600.0).values():
        if ei.shape[1]:
            assert (ts[ei[0]] <= ts[ei[1]]).all()


def test_degree_is_capped():
    ev = load_background(path=None, n_rows=1500, seed=2)
    edges = build_edges(ev, window_s=10**9, max_degree=4)
    for ei in edges.values():
        if ei.shape[1]:
            _, counts = np.unique(ei[1], return_counts=True)
            assert counts.max() <= 4
