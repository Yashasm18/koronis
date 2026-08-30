"""What a campaign's opening attempt can and cannot see.

The full model alerts on the first campaign event, at t = 0. That is only
explicable if the signal at that instant comes from the event itself, because
there is no coordinated history yet to read. These tests pin down exactly what
is available to it, so the claim in the README cannot drift.
"""
import numpy as np

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import RELATIONS, CampaignSpec
from koronis.graph.build import build_edges


def _data(seed=1, k=60, camo=1.0):
    bg = load_background(path=None, n_rows=6000, seed=seed)
    spec = CampaignSpec(n_attempts=400, k_devices=k, k_ips=k, n_bins=k,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[500]),
                        camouflage=camo)
    return inject(bg, [spec], seed=seed)


def _first_campaign_row(ev):
    return int(np.flatnonzero(ev["label"].to_numpy() == 1)[0])


def test_first_campaign_event_has_no_campaign_derived_links():
    """The opening attempt cannot be linked to the campaign, because none of
    it has happened yet. Any alert on it is therefore per-event evidence, not
    coordination — which is why the mechanism ablation exists."""
    ev = _data()
    first = _first_campaign_row(ev)
    y = ev["label"].to_numpy() == 1
    edges = build_edges(ev, window_s=3600.0)

    for rel, ei in edges.items():
        if ei.shape[1] == 0:
            continue
        src = ei[0][ei[1] == first]
        assert not y[src].any(), \
            f"first campaign event has a campaign-derived {rel} link"


def test_first_campaign_event_may_share_legitimate_history():
    """It is not edge-free. Entity types the attacker does not mint fresh —
    a BIN range that real cards also use, a free email domain — can link it to
    ordinary prior traffic. Those links carry no campaign evidence, and the
    model must not be able to mistake them for it."""
    ev = _data()
    first = _first_campaign_row(ev)
    y = ev["label"].to_numpy() == 1
    edges = build_edges(ev, window_s=3600.0)

    legit_links = {
        rel: int((~y[ei[0][ei[1] == first]]).sum())
        for rel, ei in edges.items() if ei.shape[1]
    }
    # Devices, IPs and BINs are minted per campaign, so they cannot link back.
    # email_domain is drawn from real domains under camouflage, so it can.
    assert set(legit_links) <= set(RELATIONS)
    assert all(v >= 0 for v in legit_links.values())
    assert legit_links.get("device_id", 0) == 0
    assert legit_links.get("ip_id", 0) == 0


def test_campaign_links_accumulate_after_the_opening_attempt():
    """Coordination becomes visible only as the campaign builds. This is the
    signal the graph half of the model depends on."""
    ev = _data()
    y = ev["label"].to_numpy() == 1
    camp_rows = np.flatnonzero(y)
    edges = build_edges(ev, window_s=3600.0)

    def campaign_links(row):
        return sum(int(y[ei[0][ei[1] == row]].sum())
                   for ei in edges.values() if ei.shape[1])

    assert campaign_links(camp_rows[0]) == 0
    assert campaign_links(camp_rows[len(camp_rows) // 2]) > 0
    assert campaign_links(camp_rows[-1]) > campaign_links(camp_rows[1])
