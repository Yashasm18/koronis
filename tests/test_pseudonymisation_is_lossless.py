"""Which identifiers can be tokenised before they reach the detector, and which cannot.

The detector links two events when they *share* an entity value: `build_edges`
groups by the column and never reads the value itself. So a bijective relabelling
of an identifier - a salted hash, a token, an opaque surrogate key - must produce
the same graph and the same scores.

That holds for the three relations the detector consumes as structure
(`device_id`, `ip_id`, `bin_id`), bit for bit. It does **not** hold for
`email_domain`, and the exception is not a defect: one node feature asks whether
the domain is a free-mail provider, so that column carries meaning beyond
equality. An integrator can tokenise the first three freely and must not tokenise
the fourth without recomputing that feature upstream.

Both halves are asserted. The invariance is the security claim; the exception
pins the single place where an identifier's *value* is read, so that adding a
second such place - an ordering, a hash bucket, a learned embedding on
`device_id` - fails here instead of silently coupling the model to values that
are opaque in production and unseen at fit time.
"""
import hashlib

import numpy as np
import pytest

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import MODEL_RELATIONS, RELATIONS, CampaignSpec
from koronis.graph.build import build_edges
from koronis.models.koronis import KoronisDetector

SALT = "a-salt-the-detector-never-sees"


def _data(seed=0):
    bg = load_background(path=None, n_rows=4000, seed=seed)
    specs = [CampaignSpec(n_attempts=200, k_devices=8, k_ips=4, duration_s=3600.0,
                          start_ts=float(bg["ts"].iloc[400]), camouflage=1.0)]
    return inject(bg, specs, seed=seed)


def _pseudonymise(events, cols):
    """Relabel the named columns with a salted digest. Bijective by construction."""
    out = events.copy()
    for col in cols:
        out[col] = [hashlib.sha256(f"{SALT}|{col}|{v}".encode()).hexdigest()
                    for v in events[col]]
    return out


@pytest.fixture(scope="module")
def fitted():
    det = KoronisDetector(seed=0)
    det.fit(_data(0), epochs=3)
    return det, _data(1)


def test_the_relabelling_really_is_bijective():
    """Guard the guard: a collision would make every assertion below vacuous."""
    ev = _data()
    ps = _pseudonymise(ev, RELATIONS)
    for rel in RELATIONS:
        assert ev[rel].nunique() == ps[rel].nunique(), f"{rel} collided under hashing"
        assert not set(ev[rel]) & set(ps[rel]), f"{rel} was not actually relabelled"


def test_the_graph_is_identical_after_relabelling_every_identifier():
    ev = _data()
    a = build_edges(ev, window_s=3600.0)
    b = build_edges(_pseudonymise(ev, RELATIONS), window_s=3600.0)
    assert a.keys() == b.keys()
    for rel in a:
        assert np.array_equal(a[rel], b[rel]), (
            f"the {rel} graph changed when the identifiers were renamed, so "
            "something reads the identifier's value rather than its equality")


def test_tokenising_the_graph_relations_changes_no_score(fitted):
    det, test = fitted
    plain = det.score_events(test)
    hashed = det.score_events(_pseudonymise(test, MODEL_RELATIONS))
    assert np.array_equal(plain, hashed), (
        "scores moved when device/IP/BIN were tokenised, so the model has "
        "acquired a dependence on identifier values (max delta "
        f"{np.abs(plain - hashed).max():.3e})")


def test_email_domain_is_the_one_identifier_that_carries_meaning(fitted):
    """The documented exception, asserted so it stays a known one."""
    det, test = fitted
    plain = det.score_events(test)
    hashed = det.score_events(_pseudonymise(test, ["email_domain"]))
    assert not np.array_equal(plain, hashed), (
        "tokenising email_domain no longer changes any score. The free-mail "
        "feature in koronis/models/koronis.py must have stopped reading the "
        "domain - either it was removed, or it silently stopped matching.")
