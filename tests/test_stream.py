import numpy as np
import pytest

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec
from koronis.models.koronis import KoronisDetector
from koronis.stream import StreamingKoronis, replay


@pytest.fixture(scope="module")
def fitted():
    bg = load_background(path=None, n_rows=1500, seed=0)
    spec = CampaignSpec(n_attempts=200, k_devices=8, k_ips=8, n_bins=8,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[200]))
    ev = inject(bg, [spec], seed=0)
    det = KoronisDetector(seed=0, window_s=3600.0)
    det.fit(ev, epochs=15)
    return det, ev


def test_streaming_matches_batch_scores(fitted):
    """The parity guarantee: replaying one event at a time must reproduce the
    batch scores. Backwards-pointing edges are what make this possible; if that
    invariant ever breaks, this test is what catches it."""
    det, ev = fitted
    batch = det.score_events(ev)
    streamed = np.array([r["score"] for r in replay(ev, det, threshold=0.5)])
    assert streamed.shape == batch.shape
    assert np.allclose(streamed, batch, atol=1e-5), \
        f"max |diff| = {np.abs(streamed - batch).max():.3e}"


def test_push_never_sees_future_events(fitted):
    """Scoring a prefix must give identical results to scoring the whole
    stream, for the events in that prefix."""
    det, ev = fitted
    half = len(ev) // 2
    full = [r["score"] for r in replay(ev, det, threshold=0.5)][:half]
    prefix = [r["score"] for r in
              replay(ev.iloc[:half].reset_index(drop=True), det, threshold=0.5)]
    assert np.allclose(full, prefix, atol=1e-6)


def test_emits_the_required_fields(fitted):
    det, ev = fitted
    out = StreamingKoronis(det, threshold=0.5).push(ev.iloc[0])
    for key in ("ts", "event_id", "score", "threshold", "alert",
                "linked_prior_events", "evidence", "ring"):
        assert key in out
    # the stream mirrors the DETECTOR's relations, which are narrower than
    # the data's - that mirroring is what makes the parity claim meaningful
    assert set(out["evidence"]) == set(det.relations)


def test_first_event_has_no_links(fitted):
    det, ev = fitted
    out = StreamingKoronis(det, threshold=0.5).push(ev.iloc[0])
    assert out["linked_prior_events"] == 0
    assert out["ring"]["alerts_in_window"] in (0, 1)


def test_alert_follows_the_frozen_threshold(fitted):
    det, ev = fitted
    rows = replay(ev, det, threshold=0.5)
    for r in rows:
        assert r["alert"] == (r["score"] >= 0.5)
        assert r["threshold"] == 0.5


def test_window_bounds_memory(fitted):
    """Entity buckets must expire, or a long stream grows without limit."""
    det, ev = fitted
    s = StreamingKoronis(det, threshold=0.5, window_s=60.0)
    for _, row in ev.iterrows():
        s.push(row)
    live = sum(len(d) for rel in s._index.values() for d in rel.values())
    assert live < len(ev) * len(s._index)


@pytest.mark.parametrize("layers", [1, 2, 3, 4])
def test_batch_parity_holds_at_any_depth(layers):
    """The stream kept only layer 1, so it happened to be exact for a
    two-layer model and silently stopped being exact the moment the selected
    depth changed. Parity is a property of the backwards-in-time edge rule,
    not of a particular depth, so it is asserted across depths."""
    bg = load_background(path=None, n_rows=900, seed=0)
    spec = CampaignSpec(n_attempts=90, k_devices=9, k_ips=9, n_bins=9,
                        duration_s=900.0, start_ts=float(bg["ts"].iloc[80]),
                        camouflage=1.0)
    ev = inject(bg, [spec], seed=0)
    det = KoronisDetector(seed=0, window_s=900.0, layers=layers)
    det.fit(ev, epochs=5)
    batch = det.score_events(ev)
    stream = StreamingKoronis(det, threshold=0.5, window_s=900.0)
    online = np.array([stream.push(r)["score"] for _, r in ev.iterrows()])
    assert np.allclose(batch, online, atol=1e-5), (
        f"streaming diverged from batch at {layers} layers "
        f"(max delta {np.abs(batch - online).max():.2e})")


def test_evidence_names_the_linked_attempts(fitted):
    """A count is a schematic; ids are evidence. The panel must be able to say
    which prior attempts a decision rested on, and through which entity."""
    det, ev = fitted
    rows = replay(ev, det, threshold=0.5)
    linked = [r for r in rows if r["linked_prior_events"] > 0]
    assert linked, "fixture should produce at least one linked event"

    seen_ids = set()
    for r in rows:
        for rel, d in r["evidence_ids"].items():
            assert set(d) == {"value", "ids", "total"}
            assert d["total"] == r["evidence"][rel]
            assert 0 < len(d["ids"]) <= min(d["total"], 6)
            # every named id must be an event that actually preceded this one
            assert all(i in seen_ids for i in d["ids"]), (r["event_id"], rel)
        seen_ids.add(r["event_id"])


def test_evidence_ids_do_not_disturb_the_counts(fitted):
    """`evidence` is consumed by the replay artifact and the site; adding ids
    must not change what it reports."""
    det, ev = fitted
    for r in replay(ev, det, threshold=0.5):
        assert sum(r["evidence"].values()) == r["linked_prior_events"]
