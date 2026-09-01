"""Partitioning the stream is a modelling decision, so it is tested like one."""
import numpy as np
import pytest

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec, RELATIONS
from koronis.eval.sharding import STRATEGIES, assign_shards, edges_preserved


@pytest.fixture(scope="module")
def stream():
    bg = load_background(path=None, n_rows=2500, seed=0)
    spec = CampaignSpec(n_attempts=200, k_devices=20, k_ips=20, n_bins=20,
                        duration_s=1200.0, start_ts=float(bg["ts"].iloc[200]),
                        camouflage=1.0)
    return inject(bg, [spec], seed=0)


def test_one_shard_is_the_undivided_stream(stream):
    for strat in STRATEGIES:
        assert (assign_shards(stream, 1, strat) == 0).all()
        assert edges_preserved(stream, assign_shards(stream, 1, strat),
                               window_s=1200.0)["all"] == 1.0


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_the_hash_actually_distributes(stream, n):
    """Regression: interpreting the id bytes as a little-endian integer made
    `value % 2**k` depend only on the first character, so every background BIN
    landed on one shard and every campaign entity on another. 94% of events sat
    on a single shard and the sweep reported entity routing as free."""
    for strat in STRATEGIES:
        shard = assign_shards(stream, n, strat)
        largest = np.bincount(shard, minlength=n).max() / len(stream)
        assert largest < (1.0 / n) * 3 + 0.05, (
            f"{strat} at {n} shards puts {largest:.0%} on one shard; "
            f"the routing hash is not distributing")
        assert len(np.unique(shard)) > 1


@pytest.mark.parametrize("rel", ["bin_id", "device_id"])
def test_routing_on_a_relation_preserves_exactly_that_relation(stream, rel):
    """The point of entity routing: co-located values keep all their edges."""
    shard = assign_shards(stream, 8, rel)
    kept = edges_preserved(stream, shard, window_s=1200.0)
    assert kept[rel] == pytest.approx(1.0), f"{rel} edges were cut"
    others = [kept[r] for r in RELATIONS if r != rel]
    assert max(others) < 0.9, "another relation survived intact; routing is not partitioning"


def test_random_routing_keeps_about_one_shard_worth_of_edges(stream):
    """With no affinity, an edge survives only if both ends land together."""
    for n in (4, 8):
        kept = edges_preserved(stream, assign_shards(stream, n, "random"),
                               window_s=1200.0)["all"]
        assert abs(kept - 1.0 / n) < 0.05, f"{kept:.3f} is not ~1/{n}"


def test_routing_is_stable_across_calls(stream):
    """Two workers must agree on where a value belongs, in any process."""
    a = assign_shards(stream, 8, "bin_id")
    b = assign_shards(stream, 8, "bin_id")
    assert (a == b).all()
