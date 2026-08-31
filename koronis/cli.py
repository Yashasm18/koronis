"""Experiment entry points.

    python -m koronis.cli ablation    # the headline comparison
    python -m koronis.cli frontier    # predicted vs measured boundary
    python -m koronis.cli latency     # precision/recall/INR over time
    python -m koronis.cli seeds       # repeat across seeds, report intervals
    python -m koronis.cli replay      # causal event-by-event replay -> JSON
    python -m koronis.cli benchmark   # p50/p95 per-event inference latency
    python -m koronis.cli mechanism   # which mechanism actually carries the signal
    python -m koronis.cli incidents   # consolidate alerts -> incidents -> actions
    python -m koronis.cli drift       # traffic-profile transfer stress test
    python -m koronis.cli relations   # which entity type carries the signal
    python -m koronis.cli aperture    # merchant view vs gateway view
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from .data.background import load_background
from .data.campaigns import inject
from .data.schema import CampaignSpec
from .eval.aperture import compare_apertures
from .eval.calibration import cost_optimal_threshold, expected_calibration_error
from .eval.cost import COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR
from .eval.latency import detection_times, exposure, latency_curve, money_prevented
from .models.gbdt import GBDTDetector
from .models.heuristic import DeclineBurstDetector, SharedEntityDetector
from .models.koronis import KoronisDetector
from .models.velocity import MultiEntityVelocityDetector, tune_velocity
from .drift import DriftMonitor
from .forecast import ExposureForecaster, build_snapshots, evaluate_forecast
from .profiles import BASE, SHIFTED
from .incident import ACTION_BY_NAME, IncidentRisk, build_incidents, dossier
from .eval.policy import evaluate_policies, incident_reliability
from .stream import StreamingKoronis

RESULTS = Path("results")
WINDOW_S = 3600.0
FP_BUDGET = 0.01

# The held-out axis is SPREAD, and the hold-out is extrapolation, not
# interpolation: training contains only campaigns concentrated enough that a
# tuned velocity engine still catches them (k <= 30, below the k = n/tau
# boundary of ~44). The test campaign is spread past that boundary, into the
# region Claim 1 proves no threshold rule can reach.
#
# Camouflage IS varied in training. Without camouflaged examples the model has
# no way to discover that coordination structure - rather than micro-amounts -
# is the invariant, and it simply memorises the naive campaign's per-row
# giveaways. That was a real failure observed before this design: trained on
# one loud campaign it scored 0.998 on train and 0.000 on a camouflaged test.
TRAIN_KS = (4, 12, 30)
TRAIN_CAMOS = (0.0, 0.5, 1.0)
TEST_K, TEST_CAMO = 60, 1.0
N_ATTEMPTS = 400
N_BACKGROUND = 6000


def _dataset(seed: int, k: int, camouflage: float = 0.0,
             n_attempts: int | None = None) -> pd.DataFrame:
    bg = load_background(path=None, n_rows=N_BACKGROUND, seed=seed)
    spec = CampaignSpec(n_attempts=n_attempts or N_ATTEMPTS,
                        k_devices=k, k_ips=k, n_bins=k,
                        duration_s=WINDOW_S, start_ts=float(bg["ts"].iloc[500]),
                        camouflage=camouflage)
    return inject(bg, [spec], seed=seed)


# Campaign length is FIXED in the detection experiments, where the held-out
# axis is spread and camouflage and size is deliberately controlled. It must
# NOT be fixed for the forecaster: with every campaign exactly N_ATTEMPTS long,
# "how many attempts remain" collapses to N_ATTEMPTS minus what you have seen,
# and a forecaster scores beautifully by memorising a constant of the
# simulation. Sizes are drawn per stream so the forecast has to be inferred
# from the observed prefix.
CAMPAIGN_SIZES = (150, 240, 380, 520, 700, 900, 300, 460)


def _sized_stream(seed: int, k: int, camouflage: float, idx: int) -> pd.DataFrame:
    return _dataset(seed, k, camouflage, n_attempts=CAMPAIGN_SIZES[idx % len(CAMPAIGN_SIZES)])


def _calibration_set(seed: int = 2) -> pd.DataFrame:
    """A third split, used only to pick the operating threshold.

    Choosing a threshold with test labels optimises detection time, false
    positives and rupees prevented against the answers - the operating-point
    results stop being held out. The campaign morphology comes from the
    TRAINING range, because that is what a deployed system would actually have
    seen: you tune on attacks you already know, then meet a new one.

    Crucially it carries ONE campaign, not the nine that training needs for
    variety. Prevalence matters here in a way it does not for fitting: with
    nine campaigns the positive rate reaches 37%, and under the rupee cost
    model "alert on every event" then becomes genuinely optimal - so the
    threshold search returns a detector that fires on the whole stream. A
    single campaign gives ~6% prevalence, matching the test split and a
    plausible deployment, so the chosen threshold means something.
    """
    return _dataset(seed, k=max(TRAIN_KS), camouflage=1.0)


def _train_set(seed: int = 0) -> pd.DataFrame:
    """Background plus one campaign per (spread, camouflage) combination."""
    bg = load_background(path=None, n_rows=N_BACKGROUND, seed=seed)
    span = float(bg["ts"].max() - bg["ts"].min())
    specs, i = [], 0
    for k in TRAIN_KS:
        for camo in TRAIN_CAMOS:
            specs.append(CampaignSpec(
                n_attempts=N_ATTEMPTS, k_devices=k, k_ips=k, n_bins=k,
                duration_s=WINDOW_S,
                start_ts=float(bg["ts"].min() + span * (0.05 + 0.1 * i)),
                camouflage=camo))
            i += 1
    return inject(bg, specs, seed=seed)


# Detectors whose raw output is a calibrated probability in [0, 1]. Expected
# calibration error is only meaningful for these; velocity counts and
# co-occurrence sums are ordinal scores, not probabilities, so reporting an
# ECE for them would be a category error.
PROBABILISTIC = {"gbdt_per_txn", "koronis_graph"}


def _raw(s: np.ndarray) -> np.ndarray:
    """Detector scores, used exactly as produced.

    Scores are deliberately NOT rescaled per split. Dividing calibration and
    test each by their own maximum lets every split redefine what a score of
    1.0 means, so a "frozen" threshold silently refers to a different absolute
    quantity on each one. It is not label leakage, but it does hollow out the
    claim that the operating point was fixed in advance.
    """
    return np.asarray(s, dtype=float)


def _fit_all(train: pd.DataFrame, seed: int = 0):
    clean = train[train["label"] == 0]
    taus = tune_velocity(clean, window_s=WINDOW_S, fp_budget=FP_BUDGET)
    gbdt = GBDTDetector(seed=seed)
    gbdt.fit(train)
    kor = KoronisDetector(seed=seed, window_s=WINDOW_S)
    kor.fit(train, epochs=60)
    return taus, gbdt, kor


def _run_once(seed: int = 0) -> pd.DataFrame:
    """One complete train / calibrate / test cycle at a given seed.

    Every split gets its own seed derived from this one, so repeating across
    seeds resamples the background traffic, the campaign entities and the
    model initialisation together. A single run cannot distinguish a real
    5x gap from a lucky draw; repeating this is what turns the headline into
    an interval.
    """
    train = _train_set(seed * 10)
    calib = _calibration_set(seed * 10 + 2)
    test = _dataset(seed * 10 + 1, TEST_K, TEST_CAMO)
    taus, gbdt, kor = _fit_all(train, seed=seed)
    y = test["label"].to_numpy()
    y_cal = calib["label"].to_numpy()

    scorers = {
        "velocity_tuned": MultiEntityVelocityDetector(taus, WINDOW_S).score_events,
        "decline_burst": DeclineBurstDetector().score_events,
        "shared_entity": SharedEntityDetector(window_s=WINDOW_S).score_events,
        "gbdt_per_txn": gbdt.score_events,
        "koronis_graph": kor.score_events,
    }

    def score(detector_name, frame):
        return scorers[detector_name](frame)

    total = exposure(test, "camp_0")
    rows = []
    for name in scorers:
        # Threshold is chosen on RAW calibration scores and FROZEN before test
        # is touched, then applied unchanged to raw test scores.
        thr, _ = cost_optimal_threshold(
            _raw(score(name, calib)), y_cal,
            COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)
        s = _raw(score(name, test))
        fired = s >= thr
        dt = detection_times(test, s, thr)["camp_0"]
        rows.append({
            "detector": name,
            # The brief asks for precision and recall on a held-out test set;
            # both are reported at the cost-optimal operating point, with
            # PR-AUC alongside as the threshold-free summary.
            "precision": round(float(precision_score(y, fired, zero_division=0)), 4),
            "recall": round(float(recall_score(y, fired, zero_division=0)), 4),
            "pr_auc": round(float(average_precision_score(y, s)), 4),
            "threshold": round(thr, 4),
            "detect_s": None if dt is None else round(dt, 1),
            "false_positives": int((fired & (y == 0)).sum()),
            "fp_cost_inr": round(float((fired & (y == 0)).sum() * COST_PER_FALSE_BLOCK_INR), 2),
            "inr_prevented": round(money_prevented(test, dt, "camp_0"), 2),
            "inr_exposure": round(total, 2),
            "ece": (round(expected_calibration_error(s, y), 4)
                    if name in PROBABILISTIC else None),
        })

    df = pd.DataFrame(rows)
    df.insert(0, "seed", seed)
    df.attrs["taus"] = taus
    df.attrs["attention"] = kor.relation_attention()
    return df


def ablation() -> pd.DataFrame:
    df = _run_once(0)
    RESULTS.mkdir(exist_ok=True)
    df.drop(columns=["seed"]).to_csv(RESULTS / "ablation.csv", index=False)
    print(f"\ntrain: k in {TRAIN_KS} x camo in {TRAIN_CAMOS} | "
          f"calibration: same distribution, thresholds frozen there | "
          f"\ntest: k={TEST_K} camo={TEST_CAMO} (extrapolated) | taus={df.attrs['taus']}")
    print(df.drop(columns=["seed"]).to_string(index=False))
    print(f"\nrelation attention: {df.attrs['attention']}")
    return df


def seeds(n_seeds: int = 10) -> pd.DataFrame:
    """Repeat the whole protocol across independent trials and report intervals.

    Each trial redraws the background traffic, the campaign entities, the
    calibration stream and the model initialisation. The held-out morphology is
    preserved throughout: train and calibration stay at k <= 30, test at k = 60
    with full camouflage, so every trial asks the same extrapolation question
    of a differently-sampled world.

    Intervals are the median with the 2.5th/97.5th percentiles OBSERVED ACROSS
    RUNS. They are an empirical spread, not a population confidence interval:
    ten draws is good evidence of stability, not statistical certainty. A
    normal approximation would be worse still here, since a detector that
    collapses on some draws has a long left tail.
    """
    runs = [_run_once(s) for s in range(n_seeds)]
    allr = pd.concat(runs, ignore_index=True)

    # False-positive reduction against the GBDT baseline, per trial.
    ref = (allr[allr["detector"] == "gbdt_per_txn"]
           .set_index("seed")["false_positives"])
    allr["fp_reduction_vs_gbdt"] = allr.apply(
        lambda r: (ref[r["seed"]] / r["false_positives"]
                   if r["false_positives"] > 0 else np.nan), axis=1).round(3)

    RESULTS.mkdir(exist_ok=True)
    allr.to_csv(RESULTS / "seeds_raw.csv", index=False)

    metrics = ["pr_auc", "precision", "recall", "false_positives",
               "detect_s", "fp_reduction_vs_gbdt"]
    rows = []
    for name, g in allr.groupby("detector", sort=False):
        row = {"detector": name, "n_trials": n_seeds}
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce").dropna()
            if v.empty:
                row[f"{m}_median"] = row[f"{m}_lo95"] = row[f"{m}_hi95"] = np.nan
                continue
            row[f"{m}_median"] = round(float(v.median()), 4)
            row[f"{m}_lo95"] = round(float(v.quantile(0.025)), 4)
            row[f"{m}_hi95"] = round(float(v.quantile(0.975)), 4)
        # how often the detector found the campaign at all
        row["detected_rate"] = round(float(g["detect_s"].notna().mean()), 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "seeds_summary.csv", index=False)

    print(f"\n{n_seeds} independent trials; median [2.5th, 97.5th percentile across runs]\n")
    for _, r in out.iterrows():
        print(f"  {r['detector']:>15}"
              f"  PR-AUC {r['pr_auc_median']:.3f} [{r['pr_auc_lo95']:.3f}, {r['pr_auc_hi95']:.3f}]"
              f"  P {r['precision_median']:.3f} [{r['precision_lo95']:.3f}, {r['precision_hi95']:.3f}]"
              f"  R {r['recall_median']:.3f}"
              f"  FP {r['false_positives_median']:.0f} [{r['false_positives_lo95']:.0f}, {r['false_positives_hi95']:.0f}]"
              f"  detect {r['detected_rate']:.0%}")
    print("\nper-trial values: results/seeds_raw.csv")
    return out


MECHANISM_VARIANTS = {
    "koronis_full": dict(use_edges=True, use_approved=True),
    "no_edges": dict(use_edges=False, use_approved=True),
    "no_approved": dict(use_edges=True, use_approved=False),
    "no_edges_no_approved": dict(use_edges=False, use_approved=False),
}


def mechanism(n_seeds: int = 5) -> pd.DataFrame:
    """Which mechanism actually carries the signal: graph or event features?

    The full model raises its first campaign alert on the campaign's opening
    attempt, which has no prior campaign neighbours. That is worth explaining
    rather than celebrating: with no coordinated history to read, any signal at
    that instant must come from the event itself - and the only per-event
    feature that separates a camouflaged campaign is the authorisation outcome.

    Note also what an early alert can and cannot buy. The outcome of an attempt
    is observed only after it is submitted, so no detector can prevent the
    attempt it learns from. The value is in stopping the ones that follow.

    Each variant is trained and evaluated under the same three-split protocol.
    """
    rows = []
    for seed in range(n_seeds):
        train = _train_set(seed * 10)
        calib = _calibration_set(seed * 10 + 2)
        test = _dataset(seed * 10 + 1, TEST_K, TEST_CAMO)
        y, y_cal = test["label"].to_numpy(), calib["label"].to_numpy()

        for name, kw in MECHANISM_VARIANTS.items():
            m = KoronisDetector(seed=seed, window_s=WINDOW_S, **kw)
            m.fit(train, epochs=60)
            thr, _ = cost_optimal_threshold(_raw(m.score_events(calib)), y_cal,
                                            COST_PER_ATTEMPT_INR,
                                            COST_PER_FALSE_BLOCK_INR)
            sc = _raw(m.score_events(test))
            fired = sc >= thr
            dt = detection_times(test, sc, thr)["camp_0"]
            rows.append({
                "seed": seed, "variant": name,
                "pr_auc": round(float(average_precision_score(y, sc)), 4),
                "precision": round(float(precision_score(y, fired, zero_division=0)), 4),
                "recall": round(float(recall_score(y, fired, zero_division=0)), 4),
                "false_positives": int((fired & (y == 0)).sum()),
                "detect_s": None if dt is None else round(dt, 1),
            })

    allr = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    allr.to_csv(RESULTS / "mechanism_raw.csv", index=False)

    out = []
    for name, g in allr.groupby("variant", sort=False):
        r = {"variant": name, "n_trials": n_seeds}
        for m_ in ("pr_auc", "precision", "recall", "false_positives", "detect_s"):
            v = pd.to_numeric(g[m_], errors="coerce").dropna()
            r[f"{m_}_median"] = round(float(v.median()), 4) if not v.empty else np.nan
        r["detected_rate"] = round(float(g["detect_s"].notna().mean()), 2)
        out.append(r)
    summary = pd.DataFrame(out)
    summary.to_csv(RESULTS / "mechanism.csv", index=False)

    print(f"\nmechanism ablation, {n_seeds} trials, medians\n")
    print(summary.to_string(index=False))
    return summary


APERTURE_MERCHANTS = (1, 2, 4, 8, 16)


def aperture() -> pd.DataFrame:
    """Does seeing more merchants at once make the same attack more visible?

    A gateway observes many merchants; a merchant observes one. The same ring
    hits several, so the two vantage points see different fractions of it.
    The prediction is stated in eval/aperture.py before any of this runs: the
    pooled view should carry about M times the co-occurrence signal, so the
    per-merchant view should degrade as M grows while the pooled view does not.

    Model and threshold are fitted once on the ordinary training and
    calibration splits and frozen - the aperture streams are held out from
    both, and the two views differ only in what the detector may see at once.
    """
    train = _train_set(0)
    _, _, kor = _fit_all(train)
    calib = _calibration_set(2)
    thr, _ = cost_optimal_threshold(_raw(kor.score_events(calib)),
                                    calib["label"].to_numpy(),
                                    COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)

    df = compare_apertures(kor, thr, list(APERTURE_MERCHANTS),
                           n_attempts=N_ATTEMPTS, k=TEST_K, camouflage=TEST_CAMO,
                           seed=11, window_s=WINDOW_S,
                           n_background=N_BACKGROUND // 2)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "aperture.csv", index=False)

    wide = df.pivot(index="n_merchants", columns="view", values="pr_auc")
    print("\nthe same campaign, seen from two vantage points")
    print(f"threshold {thr:.4f} frozen on calibration; model never refitted\n")
    print(df.to_string(index=False))
    print("\nPR-AUC by aperture:")
    print(wide.to_string())
    return df


N_POLICY_STREAMS = 8


def incidents() -> pd.DataFrame:
    """Consolidate event alerts into incidents and pick a response for each.

    The incident risk model is fitted on CALIBRATION incidents only, never on
    test - the same three-way protocol the detector uses.

    A single calibration stream yields only a couple of incidents, which cannot
    determine a risk model or support a reliability claim. Calibration and
    evaluation therefore pool incidents across several independent streams.
    The detector is trained once and reused; only scoring is repeated, so this
    costs little. Incident-level reliability is MEASURED here rather than
    inherited from the event model: the events inside an incident are strongly
    dependent, so event calibration says nothing about the aggregate.
    """
    train = _train_set(0)
    taus, gbdt, kor = _fit_all(train)

    def _incidents_for(frame):
        sc = _raw(kor.score_events(frame))
        return sc, build_incidents(frame, sc, thr)

    calib0 = _calibration_set(2)
    thr, _ = cost_optimal_threshold(_raw(kor.score_events(calib0)),
                                    calib0["label"].to_numpy(),
                                    COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)

    cal_inc, cal_snaps = [], []
    for j in range(N_POLICY_STREAMS):
        f = _sized_stream(500 + j * 7, max(TRAIN_KS), 1.0, j)
        sc, inc = _incidents_for(f)
        cal_inc.extend(inc)
        cal_snaps.append(build_snapshots(f, inc, sc, stream_id=j))
    risk = IncidentRisk().fit(cal_inc)

    # The forecaster is fitted on CALIBRATION snapshots only. Its target -
    # how many more alerted events join an incident - needs no labels at all,
    # so this stays clean even before the risk model has an opinion.
    cal_snaps = pd.concat(cal_snaps, ignore_index=True) if cal_snaps else pd.DataFrame()
    fc = ExposureForecaster(seed=0).fit(cal_snaps)

    # Headline stream, the one the demo replays.
    test = _dataset(1, TEST_K, TEST_CAMO)
    scores = _raw(kor.score_events(test))
    summary, detail = evaluate_policies(test, scores, thr, risk, fc)

    # Policy comparison and reliability pooled over independent test streams.
    pooled, pool_inc, pool_risk, test_snaps = [], [], [], []
    for j in range(N_POLICY_STREAMS):
        f = _sized_stream(900 + j * 11, TEST_K, TEST_CAMO, j + 3)
        sc, inc = _incidents_for(f)
        r = risk.predict(inc)
        for i2, rv in zip(inc, r):
            i2.risk = float(rv)
        pool_inc.extend(inc); pool_risk.extend(list(r))
        test_snaps.append(build_snapshots(f, inc, sc, stream_id=100 + j))
        sm, _ = evaluate_policies(f, sc, thr, risk, fc)
        sm["stream"] = j
        pooled.append(sm)

    pooled = pd.concat(pooled, ignore_index=True)
    across = (pooled.groupby("policy", sort=False)
              [["incidents_actioned", "false_incidents", "analyst_minutes",
                "merchant_cost_inr"]].median().round(1).reset_index())
    rel = incident_reliability(pool_inc, np.array(pool_risk))
    test_snaps = pd.concat(test_snaps, ignore_index=True) if test_snaps else pd.DataFrame()
    fcast = evaluate_forecast(fc, test_snaps)

    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "policy.csv", index=False)
    across.to_csv(RESULTS / "policy_across_streams.csv", index=False)
    pd.DataFrame(detail).to_csv(RESULTS / "incidents.csv", index=False)
    rel.to_csv(RESULTS / "incident_reliability.csv", index=False)
    json.dump({"threshold": thr, "detail": detail,
               "summary": summary.to_dict("records"),
               "across_streams": across.to_dict("records"),
               "reliability": rel.replace({np.nan: None}).to_dict("records"),
               "n_calibration_incidents": len(cal_inc),
               "n_pooled_test_incidents": len(pool_inc),
               "n_streams": N_POLICY_STREAMS,
               "forecast": fcast,
               "n_calibration_snapshots": int(len(cal_snaps)),
               "campaign_sizes": list(CAMPAIGN_SIZES),
               "forecast_fit_streams": [int(g) for g in fc.fit_groups_],
               "forecast_conformal_streams": [int(g) for g in fc.conformal_groups_]},
              open(RESULTS / "policy.json", "w"), separators=(",", ":"))

    print(f"\n{summary['events_alerted'].iloc[0]} event alerts -> "
          f"{summary['incidents_formed'].iloc[0]} incidents on the demo stream")
    print(f"risk model fitted on {len(cal_inc)} calibration incidents "
          f"from {N_POLICY_STREAMS} streams; reliability measured on "
          f"{len(pool_inc)} held-out incidents\n")
    print(summary.drop(columns=["exposure_if_unstopped_inr"]).to_string(index=False))
    print(f"\nmedian across {N_POLICY_STREAMS} independent test streams:")
    print(across.to_string(index=False))
    print("\nincidents on the demo stream:")
    for d in detail:
        print(f"  {d['incident_id']}  risk {d['risk']:.3f}  {d['n_attempts']:>4} attempts  "
              f"{d['n_devices']:>3}dev {d['n_ips']:>3}ip {d['n_bins']:>3}bin  "
              f"genuine={str(d['genuine']):>5}  -> {d['action']}")

    if detail:
        lead = max(detail, key=lambda x: (x["risk"], x["n_attempts"]))
        print()
        print(dossier(lead))
    print("\nincident-level reliability (measured, not inherited):")
    print(rel.dropna().to_string(index=False))
    print(f"\nremaining-exposure forecast, fitted on {len(cal_snaps)} calibration "
          f"snapshots, evaluated on {fcast.get('n_snapshots', 0)} held-out:")
    print(f"  fit streams {fc.fit_groups_} | conformal streams {fc.conformal_groups_}")
    print(f"  P{int(fc.upper_q*100)} coverage {fcast['coverage_upper']:.1%} "
          f"(target {fc.upper_q:.0%})   median abs error "
          f"{fcast['median_abs_err_p50']:.1f} attempts   "
          f"mean true remaining {fcast['mean_true_remaining']:.1f}")
    reg = summary["regret_vs_oracle_inr"].iloc[0]
    match = summary["actions_matching_oracle"].iloc[0]
    print(f"\naction regret vs oracle: INR {reg:,.0f}  "
          f"({match}/{len(detail)} incidents get the same action)")
    return summary


def _profile_stream(seed: int, profile, k: int, camo: float,
                    n_attempts: int) -> pd.DataFrame:
    bg = load_background(path=None, n_rows=N_BACKGROUND, seed=seed, profile=profile)
    spec = CampaignSpec(n_attempts=n_attempts, k_devices=k, k_ips=k, n_bins=k,
                        duration_s=WINDOW_S, start_ts=float(bg["ts"].iloc[500]),
                        camouflage=camo)
    return inject(bg, [spec], seed=seed)


def relations(n_seeds: int = 5) -> pd.DataFrame:
    """Drop each relation in turn and measure what it was worth.

    The model reports a learned attention weight per relation, which is
    suggestive but is not evidence: attention says where the model looked, not
    what it gained. Removing a relation and re-fitting says what it gained.

    Each variant is trained and evaluated under the same three-split protocol.
    """
    from .data import schema

    original = list(schema.RELATIONS)
    variants = {"all": original}
    for rel in original:
        variants[f"no_{rel}"] = [r for r in original if r != rel]

    rows = []
    for seed in range(n_seeds):
        train = _train_set(seed * 10)
        calib = _calibration_set(seed * 10 + 2)
        test = _dataset(seed * 10 + 1, TEST_K, TEST_CAMO)
        y, y_cal = test["label"].to_numpy(), calib["label"].to_numpy()

        for name, rels in variants.items():
            # RELATIONS is read at call time by build_edges and the model, so
            # patching it here changes which entity types exist for this fit.
            schema.RELATIONS[:] = rels
            try:
                m = KoronisDetector(seed=seed, window_s=WINDOW_S)
                m.fit(train, epochs=60)
                thr, _ = cost_optimal_threshold(_raw(m.score_events(calib)), y_cal,
                                                COST_PER_ATTEMPT_INR,
                                                COST_PER_FALSE_BLOCK_INR)
                sc = _raw(m.score_events(test))
                fired = sc >= thr
                rows.append({
                    "seed": seed, "variant": name, "n_relations": len(rels),
                    "pr_auc": round(float(average_precision_score(y, sc)), 4),
                    "precision": round(float(precision_score(y, fired, zero_division=0)), 4),
                    "recall": round(float(recall_score(y, fired, zero_division=0)), 4),
                    "false_positives": int((fired & (y == 0)).sum()),
                })
            finally:
                schema.RELATIONS[:] = original

    allr = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    allr.to_csv(RESULTS / "relations_raw.csv", index=False)

    med = (allr.groupby("variant", sort=False)
           [["pr_auc", "precision", "recall", "false_positives"]]
           .median().round(4).reset_index())
    base = med.loc[med["variant"] == "all", "pr_auc"].iloc[0]
    med["pr_auc_drop"] = (base - med["pr_auc"]).round(4)
    med.to_csv(RESULTS / "relations.csv", index=False)

    print(f"\nper-relation ablation, {n_seeds} trials, medians\n")
    print(med.to_string(index=False))
    return med


def drift() -> pd.DataFrame:
    """Traffic-profile transfer stress test with an automation guardrail.

    These are synthetic merchant SHAPES, not real merchants. Surviving this is
    evidence the detector is not tuned to one profile; it is not evidence of
    production cross-merchant transfer.

    Everything is fitted on the base profile and frozen before any shifted
    traffic is scored: detector weights, the alert threshold, the incident risk
    model, the exposure forecaster, and the drift cut-off. The three shifted
    profiles were defined in koronis/profiles.py before being run.
    """
    train = _train_set(0)
    taus, gbdt, kor = _fit_all(train)

    # Cut-off fitted on many independent base streams; the base false-flag
    # rate is then measured on a DISJOINT set of base streams. Three streams
    # cannot estimate a 95th percentile, and the earlier 1-in-3 base false flag
    # was as much small-sample noise as it was a real alarm rate.
    N_DRIFT_CALIB, N_DRIFT_EVAL = 16, 12
    base_calibs = [_calibration_set(2)] + [
        _sized_stream(500 + j * 7, max(TRAIN_KS), 1.0, j)
        for j in range(N_POLICY_STREAMS)]
    thr, _ = cost_optimal_threshold(_raw(kor.score_events(base_calibs[0])),
                                    base_calibs[0]["label"].to_numpy(),
                                    COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)

    cal_inc, cal_snaps = [], []
    for j, f in enumerate(base_calibs[1:]):
        sc = _raw(kor.score_events(f))
        inc = build_incidents(f, sc, thr)
        cal_inc.extend(inc)
        cal_snaps.append(build_snapshots(f, inc, sc, stream_id=j))
    risk = IncidentRisk().fit(cal_inc)
    fc = ExposureForecaster(seed=0).fit(pd.concat(cal_snaps, ignore_index=True))

    # Drift cut-off from base calibration traffic ONLY, over many streams.
    drift_calib = [_sized_stream(3000 + j * 17, max(TRAIN_KS), 1.0, j)
                   for j in range(N_DRIFT_CALIB)]
    monitor = DriftMonitor(quantile=0.95, seed=0).fit(drift_calib)

    # Base false-flag rate on streams disjoint from the ones that set the
    # cut-off. This is the number that decides whether the guardrail is a
    # safety control or an experiment.
    base_eval = [_sized_stream(7000 + j * 23, TEST_K, TEST_CAMO, j)
                 for j in range(N_DRIFT_EVAL)]
    base_flags = [monitor.check(f) for f in base_eval]
    base_false_flag_rate = float(np.mean([c["drifted"] for c in base_flags]))

    # Confound diagnostic. A campaign shifts the very statistics the monitor
    # watches, so a "false flag" on base traffic may be the attack rather than
    # the merchant. Holding the merchant fixed and varying only the campaign
    # separates the two.
    def _flag_rate(streams):
        f = [monitor.check(x)["drifted"] for x in streams]
        return round(float(np.mean(f)), 4)

    confound = {
        "campaign_matches_calibration_k30": _flag_rate(
            [_sized_stream(7000 + j * 23, max(TRAIN_KS), 1.0, j)
             for j in range(N_DRIFT_EVAL)]),
        "background_only_no_campaign": _flag_rate(
            [load_background(path=None, n_rows=N_BACKGROUND, seed=7000 + j * 23,
                             profile=BASE) for j in range(N_DRIFT_EVAL)]),
        "campaign_unseen_morphology_k60": base_false_flag_rate,
    }

    rows = []
    for prof in [BASE, *SHIFTED]:
        for j in range(3):
            ev = _profile_stream(1200 + j * 13, prof, TEST_K, TEST_CAMO,
                                 CAMPAIGN_SIZES[j])
            sc = _raw(kor.score_events(ev))
            chk = monitor.check(ev)
            summary, detail = evaluate_policies(ev, sc, thr, risk, fc)

            # Raw behaviour, then guarded: under drift the policy stands down
            # from automated intervention to analyst review.
            raw_actions = [d["action"] for d in detail]
            guarded = ["review_only" if (chk["drifted"] and a != "monitor") else a
                       for a in raw_actions]
            auto_raw = sum(1 for a in raw_actions if a != "monitor")
            false_auto = sum(1 for a, d in zip(raw_actions, detail)
                             if a != "monitor" and not d["genuine"])
            true_downgraded = sum(1 for a, g, d in zip(raw_actions, guarded, detail)
                                  if a != g and d["genuine"])
            added_minutes = sum(
                ACTION_BY_NAME[g].analyst_minutes - ACTION_BY_NAME[a].analyst_minutes
                for a, g in zip(raw_actions, guarded))
            rows.append({
                "profile": prof.name, "stream": j,
                "psi": chk["psi"], "psi_threshold": chk["threshold"],
                "drifted": chk["drifted"], "largest_shift": chk["largest_shift"],
                "incidents": len(detail),
                "auto_actions_raw": auto_raw,
                "false_auto_actions_raw": false_auto,
                "false_escalations_avoided": false_auto if chk["drifted"] else 0,
                "true_responses_downgraded": true_downgraded,
                "analyst_minutes_added": round(added_minutes, 1),
            })

    df = pd.DataFrame(rows)
    per = (df.groupby("profile", sort=False)
           .agg(psi=("psi", "median"), drifted=("drifted", "mean"),
                incidents=("incidents", "median"),
                auto_raw=("auto_actions_raw", "median"),
                false_auto_raw=("false_auto_actions_raw", "sum"),
                false_avoided=("false_escalations_avoided", "sum"),
                true_downgraded=("true_responses_downgraded", "sum"),
                minutes_added=("analyst_minutes_added", "sum"))
           .round(3).reset_index())

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "drift_raw.csv", index=False)
    per.to_csv(RESULTS / "drift.csv", index=False)
    json.dump({"threshold_psi": monitor.threshold,
               "n_calibration_streams": N_DRIFT_CALIB,
               "n_base_eval_streams": N_DRIFT_EVAL,
               "base_false_flag_rate": round(base_false_flag_rate, 4),
               "base_eval_psi": [c["psi"] for c in base_flags],
               "confound_diagnostic": confound,
               "status": ("experimental decision support - base false-flag rate is "
                          "too high for a default safety control, and the signal "
                          "partly tracks campaign shape rather than merchant shift"),
               "base_reuse_reference": monitor.base_reuse,
               "null_scores": [round(x, 4) for x in monitor.null_scores],
               "profiles": {p.name: p.blurb for p in [BASE, *SHIFTED]},
               "per_profile": per.to_dict("records"),
               "streams": df.to_dict("records")},
              open(RESULTS / "drift.json", "w"), separators=(",", ":"))

    print(f"\ndrift cut-off (95th pct of base-vs-base PSI over "
          f"{N_DRIFT_CALIB} streams): {monitor.threshold:.4f}")
    print(f"base false-flag rate on {N_DRIFT_EVAL} disjoint base streams: "
          f"{base_false_flag_rate:.1%}  "
          f"(median base PSI {np.median([c['psi'] for c in base_flags]):.4f})")
    print("\nconfound diagnostic - merchant fixed, campaign varied:")
    for k, v in confound.items():
        print(f"  {k:>36}: {v:.1%}")
    print("  -> the flag rate tracks CAMPAIGN shape, not merchant variation\n")
    print(per.to_string(index=False))
    print("\nper stream:")
    print(df[["profile", "stream", "psi", "drifted", "largest_shift",
              "incidents", "auto_actions_raw", "false_auto_actions_raw"]]
          .to_string(index=False))
    return per


def frontier() -> pd.DataFrame:
    from .eval.frontier import sweep
    df = sweep(n_values=[200, 400, 800, 1600], k_values=[2, 10, 50, 200],
               fp_budget=FP_BUDGET, seed=0, window_s=WINDOW_S,
               n_background=N_BACKGROUND)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "frontier.csv", index=False)
    print(df.to_string(index=False))
    agree = (df["velocity_blind_predicted"] == ~df["velocity_detected"]).mean()
    print(f"\nprediction agrees with observation on {agree:.0%} of the grid")
    return df


def latency() -> pd.DataFrame:
    train = _train_set(0)
    calib = _calibration_set(2)
    test = _dataset(1, TEST_K, TEST_CAMO)
    taus, gbdt, kor = _fit_all(train)
    y_cal = calib["label"].to_numpy()
    checkpoints = [60, 300, 600, 1200, 1800, 2400, 3000, 3600]

    scorers = {
        "velocity_tuned": MultiEntityVelocityDetector(taus, WINDOW_S).score_events,
        "decline_burst": DeclineBurstDetector().score_events,
        "shared_entity": SharedEntityDetector(window_s=WINDOW_S).score_events,
        "gbdt_per_txn": gbdt.score_events,
        "koronis_graph": kor.score_events,
    }

    frames = []
    for name, fn in scorers.items():
        # One threshold per detector, derived from calibration, held fixed at
        # every checkpoint. This measures one deployed detector over time,
        # rather than a sequence of differently tuned ones.
        thr, _ = cost_optimal_threshold(_raw(fn(calib)), y_cal,
                                        COST_PER_ATTEMPT_INR,
                                        COST_PER_FALSE_BLOCK_INR)
        c = latency_curve(test, _raw(fn(test)), "camp_0", checkpoints,
                          threshold=thr)
        c.insert(0, "detector", name)
        c.insert(1, "threshold", round(thr, 6))
        frames.append(c)

    df = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "latency.csv", index=False)
    print(df.to_string(index=False))
    return df


def _replay_setup():
    """Fit, freeze the threshold on calibration, and return the test stream."""
    train, calib = _train_set(0), _calibration_set(2)
    test = _dataset(1, TEST_K, TEST_CAMO)
    taus, gbdt, kor = _fit_all(train)
    thr, _ = cost_optimal_threshold(_raw(kor.score_events(calib)),
                                    calib["label"].to_numpy(),
                                    COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)
    return test, kor, taus, float(thr)


def replay() -> dict:
    """Replay the held-out stream one event at a time and write the artifact.

    Strictly causal: the scorer sees no event before it arrives. Velocity
    scores are included alongside so the demo can show the rules engine
    staying silent on the same stream.
    """
    test, kor, taus, thr = _replay_setup()
    vel = MultiEntityVelocityDetector(taus, WINDOW_S).score_events(test)
    vel_thr, _ = cost_optimal_threshold(
        _raw(MultiEntityVelocityDetector(taus, WINDOW_S).score_events(_calibration_set(2))),
        _calibration_set(2)["label"].to_numpy(),
        COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)

    stream = StreamingKoronis(kor, threshold=thr, window_s=WINDOW_S)
    onset = float(test[test["label"] == 1]["ts"].min())

    events = []
    for i, (_, row) in enumerate(test.iterrows()):
        out = stream.push(row)
        events.append({
            "t": round(out["ts"] - onset, 2),
            "score": out["score"],
            "alert": out["alert"],
            "linked": out["linked_prior_events"],
            "ev": out["evidence"],
            "vel": round(float(vel[i]), 3),
            "vel_alert": bool(vel[i] >= vel_thr),
            "amount": round(float(row["amount"]), 2),
            "approved": bool(row["approved"]),
            "campaign": bool(row["label"] == 1),
            "ring": out["ring"]["alerts_in_window"],
        })

    first = next((e for e in events if e["alert"] and e["campaign"]), None)
    artifact = {
        "meta": {
            "generated_by": "koronis.cli replay",
            "defense_only": "synthetic in-memory stream; no gateway, no real card data",
            "threshold": round(thr, 6),
            "velocity_threshold": round(float(vel_thr), 6),
            "window_s": WINDOW_S,
            "test_morphology": {"k": TEST_K, "camouflage": TEST_CAMO,
                                "n_attempts": N_ATTEMPTS},
            "n_events": len(events),
            "n_campaign": int((test["label"] == 1).sum()),
            "campaign_onset_t": 0.0,
            "koronis_first_alert_t": None if first is None else first["t"],
            "velocity_ever_alerts": any(e["vel_alert"] and e["campaign"] for e in events),
            "cost_per_attempt_inr": COST_PER_ATTEMPT_INR,
            "cost_per_false_block_inr": COST_PER_FALSE_BLOCK_INR,
        },
        "events": events,
    }

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "replay.json"
    path.write_text(json.dumps(artifact, separators=(",", ":")))
    _write_demo_subset(artifact)
    m = artifact["meta"]
    print(f"wrote {path}  ({path.stat().st_size/1024:.0f} KB, {m['n_events']} events)")
    print(f"  koronis threshold {m['threshold']:.4f}, first campaign alert at "
          f"t = {m['koronis_first_alert_t']}s")
    print(f"  velocity ever alerts on the campaign: {m['velocity_ever_alerts']}")
    return artifact


DEMO_PRE_S, DEMO_POST_S, DEMO_BG_STRIDE = 60.0, 120.0, 4
# The visualisation timeline covers the campaign, not the 30 days of
# background traffic surrounding it. Events outside this band are ambient
# noise for the demo and would flatten the time axis to uselessness.
DEMO_BAND_S = (-300.0, 3900.0)


def _write_demo_subset(artifact: dict) -> None:
    """A smaller, deterministic subset for the demo site.

    The full replay stays untouched and auditable. This exists only so a web
    page can animate a readable stream instead of six thousand rows. Nothing
    is recomputed: every field is copied verbatim, and the metadata says
    plainly that reported metrics come from the full replay, so a viewer can
    never mistake the visualisation for the experiment.

    Kept: every campaign event, and every event - campaign or not - inside a
    window around the first alert, so the moment of detection is shown with
    its real surrounding traffic. Outside that window, background is sampled
    on a fixed stride, which is deterministic and needs no RNG.
    """
    first = artifact["meta"]["koronis_first_alert_t"]
    lo = (first if first is not None else 0.0) - DEMO_PRE_S
    hi = (first if first is not None else 0.0) + DEMO_POST_S

    band_lo, band_hi = DEMO_BAND_S
    kept, bg_i = [], 0
    for e in artifact["events"]:
        if e["campaign"]:
            kept.append(e)
            continue
        if not (band_lo <= e["t"] <= band_hi):
            continue                       # outside the visualised band
        if lo <= e["t"] <= hi:
            kept.append(e)                 # full traffic around the alert
        else:
            if bg_i % DEMO_BG_STRIDE == 0:
                kept.append(e)
            bg_i += 1
    kept.sort(key=lambda e: e["t"])

    meta = dict(artifact["meta"])
    meta["subset"] = {
        "note": ("Visualisation subset only; metrics are computed from the "
                 "full 6,400-event replay in results/replay.json."),
        "kept_events": len(kept),
        "full_events": len(artifact["events"]),
        "rule": (f"all campaign events; within t in [{DEMO_BAND_S[0]:.0f}s, "
                 f"{DEMO_BAND_S[1]:.0f}s]: all traffic within -{DEMO_PRE_S:.0f}s/"
                 f"+{DEMO_POST_S:.0f}s of the first alert, background elsewhere "
                 f"sampled every {DEMO_BG_STRIDE}th event"),
        "band_s": list(DEMO_BAND_S),
        "deterministic": True,
    }
    out = RESULTS / "replay_demo.json"
    out.write_text(json.dumps({"meta": meta, "events": kept},
                              separators=(",", ":")))
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, "
          f"{len(kept)} of {len(artifact['events'])} events)")


def benchmark(n_warmup: int = 200) -> dict:
    """Per-event inference latency, measured after warm-up.

    Timing covers only StreamingKoronis.push - neighbour lookup plus two
    message-passing steps. Dataset construction and model fitting are excluded,
    since neither happens per event in a deployment.
    """
    test, kor, _, thr = _replay_setup()
    rows = [row for _, row in test.iterrows()]

    stream = StreamingKoronis(kor, threshold=thr, window_s=WINDOW_S)
    for row in rows[:n_warmup]:
        stream.push(row)

    times = []
    for row in rows[n_warmup:]:
        t0 = time.perf_counter()
        stream.push(row)
        times.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times)
    out = {
        "n_measured": int(t.size),
        "n_warmup": n_warmup,
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p95_ms": round(float(np.percentile(t, 95)), 3),
        "p99_ms": round(float(np.percentile(t, 99)), 3),
        "mean_ms": round(float(t.mean()), 3),
        "throughput_eps": round(1000.0 / float(t.mean()), 1),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(json.dumps(out, indent=2))
    print(f"\nper-event inference, {out['n_measured']} events after "
          f"{n_warmup} warm-up")
    print(f"  p50 {out['p50_ms']:.2f} ms   p95 {out['p95_ms']:.2f} ms   "
          f"p99 {out['p99_ms']:.2f} ms")
    print(f"  mean {out['mean_ms']:.2f} ms  ->  {out['throughput_eps']:.0f} events/sec")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    {"ablation": ablation, "frontier": frontier, "latency": latency,
     "seeds": seeds, "replay": replay, "benchmark": benchmark,
     "mechanism": mechanism, "incidents": incidents, "drift": drift,
     "relations": relations, "aperture": aperture}[cmd]()
