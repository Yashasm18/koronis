import numpy as np
import pytest

from koronis.data.background import load_background
from koronis.drift import DriftMonitor, psi, signature
from koronis.incident import ACTION_BY_NAME, AUTONOMOUS_ACTIONS, choose_action
from koronis.profiles import BASE, BY_NAME, SHIFTED


def _bg(profile, seed=0, n=5000):
    return load_background(path=None, n_rows=n, seed=seed, profile=profile)


@pytest.fixture(scope="module")
def monitor():
    return DriftMonitor(quantile=0.95, seed=0).fit(
        [_bg(BASE, seed=s) for s in (0, 1, 2, 3)])


def test_base_traffic_is_not_flagged(monitor):
    """The cut-off is fitted on base-versus-base variation, so ordinary base
    traffic must sit under it."""
    assert not monitor.check(_bg(BASE, seed=11))["drifted"]


@pytest.mark.parametrize("name", [p.name for p in SHIFTED])
def test_every_shifted_profile_is_flagged(monitor, name):
    assert monitor.check(_bg(BY_NAME[name], seed=7))["drifted"], name


def test_drift_score_orders_by_how_different_the_profile_is(monitor):
    base = monitor.check(_bg(BASE, seed=5))["psi"]
    for p in SHIFTED:
        assert monitor.check(_bg(p, seed=5))["psi"] > base, p.name


def test_it_names_which_aspect_moved(monitor):
    """A drift alarm that cannot say what changed is hard to act on.

    The assertion is that the largest shift lies among the features the
    profile genuinely alters, not that it is one specific feature: the
    marketplace profile moves device, IP and BIN concentration together, so
    which of them tops the list is a matter of sampling.
    """
    plausible = {
        "subscription": {"reuse_device", "reuse_ip"},
        "marketplace": {"reuse_device", "reuse_ip", "reuse_bin"},
        "flash_sale": {"log_interarrival", "declined"},
    }
    for name, feats in plausible.items():
        for seed in (3, 4, 5):
            got = monitor.check(_bg(BY_NAME[name], seed=seed))["largest_shift"]
            assert got in feats, f"{name} seed {seed}: {got} not in {feats}"


def test_psi_of_a_distribution_against_itself_is_near_zero():
    sig = signature(_bg(BASE, seed=0))
    assert psi(sig, sig)["overall"] < 1e-6


def test_threshold_uses_only_base_traffic(monitor):
    """Fitting the cut-off on shifted traffic would tune the guardrail to the
    thing it is meant to discover."""
    assert monitor.null_scores
    assert monitor.threshold == pytest.approx(
        float(np.quantile(monitor.null_scores, 0.95)))


def test_review_only_is_not_selectable_by_the_cost_policy():
    """The guardrail action is a decision to stop automating, not an option to
    be weighed. In the argmin it would be picked whenever doing nothing while
    billing an analyst looked cheap."""
    assert "review_only" not in [a.name for a in AUTONOMOUS_ACTIONS]
    for risk in (0.01, 0.5, 0.99):
        for remaining in (0, 50, 900):
            action, costs = choose_action(risk, remaining)
            assert action.name != "review_only"
            assert "review_only" not in dict(costs)


def test_review_only_lowers_automation_rather_than_raising_it():
    """Standing down must stop nothing automatically; that is its honest cost."""
    a = ACTION_BY_NAME["review_only"]
    assert a.stops == 0.0
    assert a.analyst_minutes > 0
    assert a.friction_inr == 0.0 and a.false_harm_inr == 0.0
