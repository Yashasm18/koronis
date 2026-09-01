"""The online consolidator must be causal, bounded, and still separate rings.

`build_incidents` counts entity values across the whole frame, so its link
decisions depend on traffic that had not happened yet. These hold the streaming
replacement to the property the batch version cannot have.
"""
import numpy as np
import pytest

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec, RELATIONS
from koronis.incident import StreamingIncidents, build_incidents
from koronis.sketch import SlidingCountMin


# ------------------------------------------------------------------- sketch
def test_sketch_never_underestimates():
    """The one-sided error is the whole safety argument: an inflated count can
    only make a value look too common to link on, which fragments rather than
    wrongly merges."""
    cm = SlidingCountMin(window_s=1000.0, width=256, depth=4, slices=4)
    rng = np.random.default_rng(0)
    truth = {}
    for i in range(3000):
        k = f"v{rng.integers(0, 400)}"
        cm.add(k, i * 0.1)
        truth[k] = truth.get(k, 0) + 1
    for k, v in truth.items():
        assert cm.estimate(k, 299.0) >= v, f"{k} underestimated"


def test_sketch_memory_does_not_grow_with_cardinality():
    a = SlidingCountMin(window_s=100.0, width=256, depth=4, slices=4)
    b = SlidingCountMin(window_s=100.0, width=256, depth=4, slices=4)
    for i in range(50):
        a.add("same-key", i * 0.5)
    for i in range(50):
        b.add(f"unique-{i}", i * 0.5)
    assert a.memory_bytes() == b.memory_bytes()


def test_sketch_forgets_outside_the_window():
    cm = SlidingCountMin(window_s=40.0, width=128, depth=3, slices=4)
    for i in range(20):
        cm.add("x", float(i))                      # t = 0..19
    assert cm.estimate("x", 20.0) == 20
    assert cm.estimate("x", 500.0) == 0            # far past the window


# ------------------------------------------------------- streaming incidents
@pytest.fixture(scope="module")
def two_rings():
    bg = load_background(path=None, n_rows=3000, seed=0)
    start = float(bg["ts"].iloc[300])
    specs = [CampaignSpec(n_attempts=200, k_devices=20, k_ips=20, n_bins=20,
                          duration_s=1200.0, start_ts=start, camouflage=1.0),
             CampaignSpec(n_attempts=200, k_devices=20, k_ips=20, n_bins=20,
                          duration_s=1200.0, start_ts=start + 60.0, camouflage=1.0)]
    ev = inject(bg, specs, seed=0)
    return ev, ev["label"].to_numpy().astype(float)   # perfect detector


def _run(ev, sc, **kw):
    st = StreamingIncidents(threshold=0.5, **kw)
    for i, (_, e) in enumerate(ev.iterrows()):
        st.push(e, float(sc[i]), row=i)
    return st


def test_link_decisions_do_not_depend_on_the_future(two_rings):
    """Push a prefix, then push the same prefix followed by more traffic. The
    grouping of the prefix's own alerts must be identical - if a later event
    could change an earlier decision, the consolidator is not causal."""
    ev, sc = two_rings
    k = 1200
    short = _run(ev.iloc[:k], sc[:k])
    long = _run(ev, sc)

    def partition(st, limit):
        """{row: frozenset of co-grouped rows}, restricted to rows < limit."""
        out = {}
        for members in st.groups().values():
            m = frozenset(r for r in members if r < limit)
            for r in m:
                out[r] = m
        return out

    a, b = partition(short, k), partition(long, k)
    assert a, "prefix produced no alerts; test is vacuous"
    # every prefix alert must sit with exactly the same prefix companions
    assert a == b


def test_two_concurrent_rings_stay_apart(two_rings):
    ev, sc = two_rings
    st = _run(ev, sc)
    for members in st.groups().values():
        if len(members) < 50:
            continue                                # ignore small fragments
        camps = set(ev.iloc[members]["campaign_id"])
        assert len(camps) == 1, f"incident mixed campaigns {camps}"


def test_a_ubiquitous_value_cannot_link_alerts(two_rings):
    """Email domain covers most of the stream; it must never bridge alerts."""
    ev, sc = two_rings
    st = _run(ev, sc)
    ts = float(ev["ts"].iloc[-1])
    for dom in ev["email_domain"].value_counts().head(2).index:
        assert st.sketch["email_domain"].share(str(dom), ts) > st.max_link_share


def test_memory_is_bounded_by_the_window_not_the_stream(two_rings):
    ev, sc = two_rings
    half = _run(ev.iloc[:len(ev)//2], sc[:len(ev)//2]).stats()
    full = _run(ev, sc).stats()
    assert full["sketch_kb"] == half["sketch_kb"]     # fixed by construction
    # buffered alert references must not scale with the whole stream
    assert full["buffered_alert_refs"] < full["alerts"] * len(RELATIONS)


def test_online_grouping_tracks_the_batch_grouping(two_rings):
    """Not identical - batch uses the future - but it must not fall apart."""
    ev, sc = two_rings
    batch = build_incidents(ev, sc, threshold=0.5)
    online = _run(ev, sc).groups()
    b_big = max((i.rows for i in batch), key=len)
    o_big = max(online.values(), key=len)
    overlap = len(set(b_big) & set(o_big)) / len(set(b_big) | set(o_big))
    assert overlap > 0.8, f"largest incidents only agree on {overlap:.0%}"
