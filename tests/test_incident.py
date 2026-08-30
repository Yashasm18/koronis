import numpy as np
import pytest

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec
from koronis.eval.policy import evaluate_policies, incident_reliability
from koronis.incident import (
    ACTION_BY_NAME, IncidentRisk, build_incidents, choose_action, expected_cost,
)


@pytest.fixture(scope="module")
def stream():
    bg = load_background(path=None, n_rows=4000, seed=0)
    spec = CampaignSpec(n_attempts=300, k_devices=30, k_ips=30, n_bins=30,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[400]),
                        camouflage=1.0)
    ev = inject(bg, [spec], seed=0)
    # a perfect detector, so the test exercises consolidation rather than the model
    return ev, ev["label"].to_numpy().astype(float)


def test_campaign_consolidates_into_one_incident(stream):
    """The whole point: hundreds of alerts must become one thing to act on."""
    ev, sc = stream
    incs = build_incidents(ev, sc, threshold=0.5)
    assert len(incs) == 1
    assert incs[0].n_attempts == 300
    assert incs[0].is_true()


def test_incident_reports_its_entity_spread(stream):
    ev, sc = stream
    inc = build_incidents(ev, sc, threshold=0.5)[0]
    assert inc.n_devices == 30 and inc.n_ips == 30 and inc.n_bins == 30
    assert inc.t_end > inc.t_start


def test_no_alerts_means_no_incidents(stream):
    ev, _ = stream
    assert build_incidents(ev, np.zeros(len(ev)), threshold=0.5) == []


def test_unrelated_alerts_stay_separate(stream):
    """Consolidation must not merge everything into one blob."""
    ev, _ = stream
    sc = np.zeros(len(ev))
    bg_rows = np.flatnonzero(ev["label"].to_numpy() == 0)
    far = [bg_rows[0], bg_rows[len(bg_rows) // 2], bg_rows[-1]]
    sc[far] = 1.0
    assert len(build_incidents(ev, sc, threshold=0.5)) >= 2


def test_action_is_chosen_by_cost_not_by_risk():
    """A high risk score with nothing left to prevent should not trigger the
    most expensive intervention - that is the whole argument for a cost policy."""
    nothing_left, _ = choose_action(risk=0.99, remaining_attempts=0)
    lots_left, _ = choose_action(risk=0.99, remaining_attempts=500)
    assert nothing_left.name == "monitor"
    assert lots_left.name == "hold_review"


def test_low_risk_never_escalates():
    action, _ = choose_action(risk=0.02, remaining_attempts=500)
    assert action.name == "monitor"


def test_expected_cost_rises_with_false_positive_harm():
    cheap = expected_cost(ACTION_BY_NAME["hold_review"], risk=0.1, remaining_attempts=10)
    dear = expected_cost(ACTION_BY_NAME["monitor"], risk=0.1, remaining_attempts=10)
    assert cheap > dear      # holding a probably-legitimate incident is expensive


class _OracleRisk:
    """A stand-in for a well-fitted risk model.

    The policy and the risk model are separate claims and are tested
    separately. Fitting IncidentRisk on a single incident shrinks it toward
    0.5 under L2, and the policy then correctly picks a cheaper-but-weaker
    action - which is the risk model being under-determined, not the policy
    being wrong. `koronis.cli incidents` pools incidents across streams for
    exactly this reason.
    """

    @staticmethod
    def predict(incs):
        return np.array([i.campaign_share for i in incs])


def test_risk_model_is_underdetermined_by_a_single_incident():
    """Documents why the CLI pools calibration incidents across streams."""
    bg = load_background(path=None, n_rows=1500, seed=3)
    spec = CampaignSpec(n_attempts=200, k_devices=20, k_ips=20, n_bins=20,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[200]))
    ev = inject(bg, [spec], seed=3)
    incs = build_incidents(ev, ev["label"].to_numpy().astype(float), threshold=0.5)
    r = IncidentRisk().fit(incs).predict(incs)
    assert len(incs) == 1
    assert r[0] < 0.95        # cannot be confident from one example


class _PerfectForecast:
    """A forecaster that knows the remaining count exactly.

    Used to test the POLICY in isolation. The forecaster's own accuracy is
    tested in tests/test_forecast.py, and the cost of its real error is
    reported as action regret by `koronis.cli incidents`.
    """

    upper_q = 0.9

    @staticmethod
    def predict_one(events, rows, scores, m):
        remaining = float(len(rows) - m)
        return remaining, remaining


def test_oracle_policy_beats_the_alternatives(stream):
    """The upper bound: given the true future, the policy is the cheapest."""
    ev, sc = stream
    summary, _ = evaluate_policies(ev, sc, 0.5, _OracleRisk())
    cost = dict(zip(summary["policy"], summary["merchant_cost_inr"]))
    assert cost["oracle_policy"] <= cost["always_allow"]
    assert cost["oracle_policy"] <= cost["always_hold"]
    assert cost["oracle_policy"] <= cost["event_thresholding"]


def test_causal_policy_matches_oracle_given_a_perfect_forecast(stream):
    """Isolates policy from forecast: with the future known, the causal path
    must reproduce the oracle exactly. Any gap is a bug in the policy, not
    forecast error."""
    ev, sc = stream
    summary, detail = evaluate_policies(ev, sc, 0.5, _OracleRisk(), _PerfectForecast())
    cost = dict(zip(summary["policy"], summary["merchant_cost_inr"]))
    assert cost["causal_policy"] == pytest.approx(cost["oracle_policy"])
    assert all(d["action"] == d["oracle_action"] for d in detail)


def test_causal_policy_cuts_analyst_workload(stream):
    ev, sc = stream
    summary, _ = evaluate_policies(ev, sc, 0.5, _OracleRisk(), _PerfectForecast())
    mins = dict(zip(summary["policy"], summary["analyst_minutes"]))
    assert mins["causal_policy"] < mins["event_thresholding"]


def test_reliability_is_measured_at_incident_level(stream):
    """Event calibration does not transfer to incidents, so it is measured."""
    ev, sc = stream
    incs = build_incidents(ev, sc, threshold=0.5)
    risk = IncidentRisk().fit(incs)
    rel = incident_reliability(incs, risk.predict(incs))
    assert set(rel.columns) == {"bin_mid", "predicted", "observed", "count"}
    assert rel["count"].sum() == len(incs)


def _two_ring_stream(seed=1, n=300, gap_s=600.0):
    """One merchant, two independent campaigns overlapping in time."""
    bg = load_background(path=None, n_rows=4000, seed=seed)
    t0 = float(bg["ts"].iloc[400])
    specs = [CampaignSpec(n_attempts=n, k_devices=30, k_ips=30, n_bins=30,
                          duration_s=3600.0, start_ts=t0, camouflage=1.0),
             CampaignSpec(n_attempts=n, k_devices=30, k_ips=30, n_bins=30,
                          duration_s=3600.0, start_ts=t0 + gap_s, camouflage=1.0)]
    ev = inject(bg, specs, seed=seed)
    return ev, ev["label"].to_numpy().astype(float)


def test_two_concurrent_rings_become_two_incidents():
    """A merchant attacked by two rings needs two actionable incidents, not one
    merged blob. Before the linking guard they merged into a single 597-attempt
    incident bridged entirely by shared email domain."""
    ev, sc = _two_ring_stream()
    big = [i for i in build_incidents(ev, sc, 0.5) if i.n_attempts >= 50]
    assert len(big) == 2, [i.n_attempts for i in big]


def test_each_incident_belongs_to_exactly_one_campaign():
    """Isolation has to be clean: an incident mixing two rings would produce
    one action and an evidence card describing neither."""
    ev, sc = _two_ring_stream()
    for inc in build_incidents(ev, sc, 0.5):
        if inc.n_attempts < 50:
            continue
        camps = ev.iloc[inc.rows]["campaign_id"].dropna().unique()
        assert len(camps) == 1, f"{inc.incident_id} mixes {list(camps)}"


def test_a_ubiquitous_entity_cannot_link_an_incident():
    """Sharing gmail.com is not evidence, and the guard measures a value's
    share of the WHOLE stream because that is what makes it common.

    Both rings draw from the same handful of email domains, and one covers over
    half the stream. Linking through it would collapse them into a single
    incident, so two surviving separately is the property under test.
    """
    ev, sc = _two_ring_stream()
    top = ev["email_domain"].value_counts().iloc[0] / len(ev)
    assert top > 0.2, "fixture should contain a dominant email domain"
    assert len(build_incidents(ev, sc, 0.5)) == 2


def test_lifting_the_link_guard_reproduces_the_merge():
    """The guard is load-bearing: without it the two rings merge, which is the
    defect this whole mechanism exists to prevent."""
    ev, sc = _two_ring_stream()
    merged = build_incidents(ev, sc, 0.5, max_link_share=1.0)
    assert len(merged) == 1
    assert merged[0].n_attempts == 600
