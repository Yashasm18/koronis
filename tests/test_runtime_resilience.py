"""What the streaming path does when the stream is not clean.

Each case here was a live defect, found by injecting faults into `push` rather
than by reading the code, and each one was silent - the stream kept running and
kept returning answers:

* a non-finite feature produced a NaN score, and `NaN >= threshold` is False, so
  the event reported itself as "no alert";
* a missing entity was interned as the string "None", so every event without a
  device fingerprint linked to every other one;
* nothing was evicted, so the caches tracked total traffic and not the window -
  3,120 rows held for a 60-second window containing 12 events.

The rule these assert is that an event which cannot be scored honestly is
escalated, not scored anyway, and that missing data can never become evidence.
"""
import numpy as np
import pytest

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec
from koronis.models.koronis import KoronisDetector
from koronis.stream import StreamingKoronis


@pytest.fixture(scope="module")
def fitted():
    bg = load_background(path=None, n_rows=2500, seed=0)
    ev = inject(bg, [CampaignSpec(n_attempts=100, k_devices=6, k_ips=3,
                                  duration_s=1800.0,
                                  start_ts=float(bg["ts"].iloc[250]))], seed=0)
    det = KoronisDetector(seed=0)
    det.fit(ev, epochs=3)
    return det, ev


def _warm(det, ev, n=150, **kw):
    s = StreamingKoronis(det, threshold=0.5, **kw)
    for _, r in ev.iloc[:n].iterrows():
        s.push(r)
    return s


UNSCOREABLE = [
    ("amount is NaN", lambda r: {**r, "amount": float("nan")}),
    ("amount is None", lambda r: {**r, "amount": None}),
    ("approved is None", lambda r: {**r, "approved": None}),
    ("ts is None", lambda r: {**r, "ts": None}),
    ("ts is unparseable", lambda r: {**r, "ts": "not-a-time"}),
    ("ts field absent", lambda r: {k: v for k, v in r.items() if k != "ts"}),
    ("device field absent", lambda r: {k: v for k, v in r.items() if k != "device_id"}),
]


@pytest.mark.parametrize("name,mutate", UNSCOREABLE, ids=[n for n, _ in UNSCOREABLE])
def test_an_unscoreable_event_is_quarantined_not_scored(fitted, name, mutate):
    det, ev = fitted
    s = _warm(det, ev)
    before = s.quarantined
    out = s.push(mutate(ev.iloc[150].to_dict()))

    assert out["status"] == "quarantined", f"{name} was scored anyway: {out}"
    assert out["score"] is None, (
        f"{name} produced a score of {out['score']}. A NaN here compares False "
        "against the threshold and reports itself as 'no alert'.")
    assert out["alert"] is False
    assert out["reasons"], "quarantined without saying why"
    assert s.quarantined == before + 1, "a quarantined event was not counted"


def test_a_scored_event_never_carries_a_nan_score(fitted):
    det, ev = fitted
    s = StreamingKoronis(det, threshold=0.5)
    for _, r in ev.iterrows():
        out = s.push(r)
        if out["status"] == "scored":
            assert np.isfinite(out["score"]), f"NaN score on {out['event_id']}"


def test_a_missing_entity_links_to_nothing(fitted):
    """The null case, and the placeholder control that shows why it matters."""
    det, ev = fitted

    def links(value):
        s = _warm(det, ev)
        last = None
        for i in range(6):
            row = ev.iloc[150 + i].to_dict()
            row["device_id"] = value
            row["ip_id"] = f"only_mine_{i}"
            row["bin_id"] = f"only_mine_{i}"
            last = s.push(row)
        return last["evidence"]["device_id"]

    assert links(None) == 0, (
        "events with no device fingerprint were linked to each other. Interning "
        "an absent value as a key invents a ring out of missing data.")
    assert links(float("nan")) == 0, "a NaN device was interned as a value"
    assert links("unk") == 0, "the loader's own null marker was interned as a value"

    # If a placeholder is substituted upstream the detector cannot tell, and
    # does link. That is the reason the null has to survive intact to entity_key.
    assert links("MISSING_DEVICE") == 5, (
        "the placeholder control no longer links, so this test would pass even "
        "if the null rule were removed")


def test_memory_tracks_the_window_and_not_the_stream(fitted):
    det, ev = fitted
    window = 60.0
    s = StreamingKoronis(det, threshold=0.5, window_s=window)
    for _, r in ev.iterrows():
        s.push(r)

    in_window = int((ev["ts"] >= ev["ts"].max() - window).sum())
    held = len(s._x)
    assert held <= max(in_window * 3, 50), (
        f"{held} rows cached for a {window:.0f}s window holding {in_window} "
        f"events, out of {len(ev)} seen. The caches are tracking total traffic.")
    assert len(s._ts) == held and all(len(c) == held for c in s._h), \
        "the per-layer caches fell out of step with each other"

    live = {rel: len(idx) for rel, idx in s._index.items()}
    assert all(n <= held for n in live.values()), (
        f"the entity index kept keys for events already evicted: {live}, "
        f"{held} rows live. A campaign mints a fresh entity per attempt, so "
        "this dict is what grows first.")


def test_quarantine_does_not_disturb_the_events_around_it(fitted):
    """A bad event must not corrupt the state the good ones depend on."""
    det, ev = fitted

    clean = StreamingKoronis(det, threshold=0.5)
    scores_clean = [clean.push(r)["score"] for _, r in ev.iloc[:200].iterrows()]

    mixed = StreamingKoronis(det, threshold=0.5)
    scores_mixed = []
    for i, (_, r) in enumerate(ev.iloc[:200].iterrows()):
        if i in (37, 88, 150):
            mixed.push({**r.to_dict(), "amount": float("nan")})    # quarantined
        scores_mixed.append(mixed.push(r)["score"])

    assert scores_clean == scores_mixed, (
        "interleaving unscoreable events changed the scores of the good ones")
