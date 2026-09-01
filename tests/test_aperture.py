import numpy as np
import pytest

from koronis.data.schema import RELATIONS
from koronis.eval.aperture import build_split_stream, compare_apertures


def test_merchant_entities_do_not_collide_across_merchants():
    """Two merchants must not appear to share a device they never shared.

    The bootstrap sampler names entities plainly, so without namespacing every
    merchant would reuse "d17" and the pooled graph would link strangers - the
    gateway view would then win on an artefact rather than on real overlap.
    """
    ev = build_split_stream(n_merchants=3, n_attempts=60, k=10, camouflage=1.0,
                            seed=0, window_s=600.0, n_background=400)
    bg = ev[ev["label"] == 0]
    for col in ("device_id", "ip_id", "bin_id"):
        owners = bg.groupby(col)["merchant_id"].nunique()
        assert owners.max() == 1, f"{col} shared across merchants in background"


def test_campaign_is_split_across_merchants():
    ev = build_split_stream(n_merchants=4, n_attempts=200, k=20, camouflage=1.0,
                            seed=1, window_s=600.0, n_background=400)
    camp = ev[ev["label"] == 1]
    assert camp["merchant_id"].nunique() == 4
    # no single merchant should hold the whole campaign
    assert camp["merchant_id"].value_counts().max() < len(camp)


class _Overlap:
    """A stand-in detector that scores an event by how many other events in
    the frame it shares an entity with. It has no notion of merchants, so any
    difference between the two views comes purely from what each one can see."""

    def score_events(self, ev):
        s = np.zeros(len(ev))
        for rel in RELATIONS:
            counts = ev[rel].map(ev[rel].value_counts())
            s += counts.to_numpy()
        return s / s.max() if s.max() else s


def test_one_merchant_makes_the_two_views_identical():
    """The experiment's own control: with a single merchant, 'pooled' and
    'per-merchant' are the same stream, so any gap at M=1 is a bug in the
    harness rather than a finding about apertures."""
    df = compare_apertures(_Overlap(), thr=0.5, n_merchants_values=[1],
                           n_attempts=80, k=10, camouflage=1.0, seed=2,
                           window_s=600.0, n_background=400)
    g = df[df["view"] == "gateway"].iloc[0]
    m = df[df["view"] == "merchant"].iloc[0]
    for col in ("pr_auc", "precision", "recall", "false_positives"):
        assert g[col] == pytest.approx(m[col]), f"{col} differs at M=1"


def test_splitting_costs_the_merchant_view_co_occurrence():
    """Splitting one campaign over more merchants must not make any single
    merchant's view of it larger."""
    df = compare_apertures(_Overlap(), thr=0.5, n_merchants_values=[1, 4],
                           n_attempts=160, k=20, camouflage=1.0, seed=3,
                           window_s=600.0, n_background=400)
    share = df.groupby("n_merchants")["campaign_share_largest_merchant"].first()
    assert share[4] < share[1]
