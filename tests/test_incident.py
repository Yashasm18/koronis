import pathlib

import numpy as np
import pytest

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec
from koronis.eval.policy import evaluate_policies, incident_reliability
from koronis.incident import (
    ACTION_BY_NAME,
    AUTONOMOUS_ACTIONS,
    IncidentRisk,
    build_incidents,
    choose_action,
    dossier,
    expected_cost,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _detail_record(**over):
    d = dict(incident_id="INC-007", risk=1.0, n_attempts=395, n_devices=60,
             n_ips=60, n_bins=60, observed_at_decision=12,
             forecast_remaining_p50=309.4, forecast_remaining_p90=557.0,
             forecast_exposure_p50_inr=22583.59, forecast_exposure_p90_inr=40657.51,
             forecast_uncertain=False, true_remaining_attempts=399,
             true_remaining_exposure_inr=29127.0, genuine=True,
             action="hold_review", oracle_action="hold_review",
             option_costs={"monitor": 22557.0, "rate_limit": 10270.65,
                           "step_up": 3733.55, "hold_review": 1684.71})
    d.update(over)
    return d


def test_dossier_reports_only_computed_fields():
    text = dossier(_detail_record())
    # spread and the reuse ratio it implies
    assert "395 alerted attempts - 60 devices - 60 IPs - 60 BINs" in text
    assert "6.6/device" in text
    # the chosen action and the monitor comparison, straight from option_costs
    assert "hold_review" in text and "1,685" in text and "22,557" in text
    assert "(matches)" in text          # oracle agrees on this incident


def test_dossier_flags_a_disagreeing_oracle_and_wide_interval():
    text = dossier(_detail_record(oracle_action="monitor", forecast_uncertain=True))
    assert "differs - oracle: monitor" in text
    assert "escalate to review" in text


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


def test_the_policy_applies_p_genuine_exactly_once():
    """The causal and oracle arms must use the same cost formula.

    forecast.py and docs/evaluation.md both state it:

        expected remaining exposure = P(genuine) x forecast(remaining) x cost

    `expected_cost` implements that with `risk * exposure`. The policy used to
    ALSO pass `risk * p50` as the remaining count, so the causal arm computed
    risk^2 * p50 while the oracle arm passed an unconditional count and got
    risk * true_remaining. Two arms of the published regret, computed
    differently - and the error changed the chosen action on 6 of 7 incidents.

    No existing test could see it: the oracle-comparison fixture drives the
    policy with `_OracleRisk`, which returns `campaign_share` = 1.0, and
    risk^2 == risk at 1.0. This asserts against the published artifact, where
    risk is genuinely fractional.
    """
    import json

    detail = json.loads((ROOT / "results" / "policy.json").read_text())["detail"]
    fractional = [d for d in detail if 0.0 < d["risk"] < 1.0]
    assert fractional, (
        "every incident in policy.json has risk 0 or 1, so this test cannot "
        "distinguish risk from risk^2 - the exact blindness it exists to fix")

    # Compare against BOTH candidate formulas rather than to a tolerance: the
    # artifact rounds risk to four decimals, so an exact match is not available,
    # but "which formula is this closer to" is unambiguous and is the actual
    # question.
    wrong = []
    for d in fractional:
        once = int(round(d["forecast_remaining_p50"]))
        twice = int(round(d["risk"] * d["forecast_remaining_p50"]))
        for action in AUTONOMOUS_ACTIONS:
            got = d["option_costs"][action.name]
            d_once = abs(expected_cost(action, d["risk"], once) - got)
            d_twice = abs(expected_cost(action, d["risk"], twice) - got)
            if d_twice < d_once:
                wrong.append(
                    f"{d['incident_id']}/{action.name}: published {got:.2f} is "
                    f"closer to risk-applied-twice "
                    f"({expected_cost(action, d['risk'], twice):.2f}) than to the "
                    f"documented formula "
                    f"({expected_cost(action, d['risk'], once):.2f})")
    assert not wrong, (
        "published option costs apply P(genuine) twice - the forecast is being "
        "scaled by risk before expected_cost scales by it again:\n  "
        + "\n  ".join(wrong))


def test_the_oracle_has_no_regret_against_itself():
    """`regret_vs_oracle_inr` and `actions_matching_oracle` are per policy.

    They used to be computed once from the causal policy and broadcast to every
    row with `df[col] = scalar`, so results/policy.csv told anyone who opened it
    that the oracle carried Rs4,750 of regret against itself and matched its own
    actions on 1 of 7 incidents. Nothing published was wrong - every consumer
    selected the causal figure - but the artifact was not readable on its own
    terms, and an artifact nobody can read is not evidence.

    The oracle is the fixed point: zero regret, and it agrees with itself
    everywhere. Any broadcast breaks both halves at once.
    """
    import csv

    rows = {r["policy"]: r
            for r in csv.DictReader((ROOT / "results" / "policy.csv").open())}
    oracle, causal = rows["oracle_policy"], rows["causal_policy"]
    n_incidents = int(causal["incidents_formed"])

    assert float(oracle["regret_vs_oracle_inr"]) == 0.0, (
        "the oracle is carrying regret against itself, so this column is a "
        "broadcast of some other policy's figure")
    assert int(oracle["actions_matching_oracle"]) == n_incidents, (
        "the oracle does not match its own actions on every incident")

    # And the column must actually vary, or a broadcast of zeros would pass.
    regrets = {p: float(r["regret_vs_oracle_inr"]) for p, r in rows.items()}
    assert len(set(regrets.values())) > 1, f"every policy has the same regret: {regrets}"
    assert regrets["always_hold"] > regrets["causal_policy"] > 0, (
        f"regret is not ordered as the costs are: {regrets}")
