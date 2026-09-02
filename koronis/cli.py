"""Experiment entry points.

    python -m koronis.cli ablation    # the headline comparison
    python -m koronis.cli frontier    # the per-entity blind region, k >= n/tau
    python -m koronis.cli latency     # precision/recall/INR over time
    python -m koronis.cli seeds       # repeat across seeds, report intervals
    python -m koronis.cli replay      # causal event-by-event replay -> JSON
    python -m koronis.cli benchmark   # p50/p95 per-event inference latency
    python -m koronis.cli mechanism   # which mechanism actually carries the signal
    python -m koronis.cli incidents   # consolidate alerts -> incidents -> actions
    python -m koronis.cli drift       # traffic-profile transfer stress test
    python -m koronis.cli relations   # which entity type carries the signal
    python -m koronis.cli aperture    # merchant view vs gateway view
    python -m koronis.cli architecture # do the gate and the attention earn their place
    python -m koronis.cli online      # online consolidation vs the batch grouping
    python -m koronis.cli sharding    # does the graph survive being split across machines
    python -m koronis.cli select      # model selection on calibration, evaluated once on test
    python -m koronis.cli replicate   # can replication recover what sharding deletes
    python -m koronis.cli capacity    # was the model sized, or just chosen
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
from .eval.sharding import sweep as shard_sweep, sweep_replication
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
from .incident import (
    ACTION_BY_NAME, MAX_LINK_SHARE, IncidentRisk, StreamingIncidents,
    build_incidents, dossier,
)
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


# Candidate architectures. Three components have been measured as net-negative
# or neutral - the device relation, the email relation, and the heterophily
# gate - each in an experiment that used the TEST split. Acting on those
# findings directly would be selecting an architecture on test results, which
# is the leakage this project refuses everywhere else. So the candidates are
# re-scored on calibration, the winner is chosen there, and test is touched
# once at the end to report what that choice was worth.
SELECT_CANDIDATES = {
    "full":                  (["device_id", "ip_id", "bin_id", "email_domain"], True),
    "no_device":             (["ip_id", "bin_id", "email_domain"], True),
    "no_email":              (["device_id", "ip_id", "bin_id"], True),
    "no_device_no_email":    (["ip_id", "bin_id"], True),
    "no_gate":               (["device_id", "ip_id", "bin_id", "email_domain"], False),
    "no_device_no_gate":     (["ip_id", "bin_id", "email_domain"], False),
    "no_email_no_gate":      (["device_id", "ip_id", "bin_id"], False),
    "lean":                  (["ip_id", "bin_id"], False),
}


def _decision_cost(y: np.ndarray, scores: np.ndarray, thr: float) -> float:
    """What the operating point costs, in the currency the model is trained on."""
    fired = scores >= thr
    fn = int((~fired & (y == 1)).sum())
    fp = int((fired & (y == 0)).sum())
    return fn * COST_PER_ATTEMPT_INR + fp * COST_PER_FALSE_BLOCK_INR


def select(n_seeds: int = 5) -> pd.DataFrame:
    """Choose an architecture on held-out calibration data, then report on test.

    Two independent calibration draws are used, not one: the threshold is fitted
    on the first and the selection score is measured on the second. Scoring a
    variant at a threshold fitted on the same events flatters whichever variant
    happens to suit that draw, which is the same mistake as tuning on test, one
    level down.
    """
    rows = []
    for seed in range(n_seeds):
        train = _train_set(seed * 10)
        calib_thr = _calibration_set(seed * 10 + 2)      # fits the threshold
        calib_sel = _calibration_set(seed * 10 + 5)      # scores the candidate
        test = _dataset(seed * 10 + 1, TEST_K, TEST_CAMO)
        y_t, y_a, y_b = (test["label"].to_numpy(),
                         calib_thr["label"].to_numpy(),
                         calib_sel["label"].to_numpy())

        for name, (rels, gate) in SELECT_CANDIDATES.items():
            m = KoronisDetector(seed=seed, window_s=WINDOW_S,
                                use_gate=gate, relations=rels)
            m.fit(train, epochs=60)
            thr, _ = cost_optimal_threshold(_raw(m.score_events(calib_thr)), y_a,
                                            COST_PER_ATTEMPT_INR,
                                            COST_PER_FALSE_BLOCK_INR)
            sel = _raw(m.score_events(calib_sel))
            sc = _raw(m.score_events(test))
            fired = sc >= thr
            rows.append({
                "seed": seed, "variant": name,
                "select_cost_inr": round(_decision_cost(y_b, sel, thr), 1),
                "select_pr_auc": round(float(average_precision_score(y_b, sel)), 4),
                "test_cost_inr": round(_decision_cost(y_t, sc, thr), 1),
                "test_pr_auc": round(float(average_precision_score(y_t, sc)), 4),
                "test_precision": round(float(precision_score(y_t, fired, zero_division=0)), 4),
                "test_recall": round(float(recall_score(y_t, fired, zero_division=0)), 4),
                "test_false_positives": int((fired & (y_t == 0)).sum()),
            })

    allr = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    allr.to_csv(RESULTS / "select_raw.csv", index=False)
    med = (allr.groupby("variant", sort=False).median(numeric_only=True)
           .drop(columns=["seed"]).round(4).reset_index())
    med.to_csv(RESULTS / "select.csv", index=False)

    winner = med.loc[med["select_cost_inr"].idxmin(), "variant"]
    full = med[med["variant"] == "full"].iloc[0]
    won = med[med["variant"] == winner].iloc[0]
    verdict = {
        "selected_on_calibration": winner,
        "selection_beat_full_on_calibration":
            bool(won["select_cost_inr"] < full["select_cost_inr"]),
        "held_up_on_test": bool(won["test_cost_inr"] < full["test_cost_inr"]),
        "full_test_cost_inr": float(full["test_cost_inr"]),
        "selected_test_cost_inr": float(won["test_cost_inr"]),
        "full_test_pr_auc": float(full["test_pr_auc"]),
        "selected_test_pr_auc": float(won["test_pr_auc"]),
        "n_seeds": n_seeds,
    }
    json.dump(verdict, open(RESULTS / "select.json", "w"), indent=2)

    print(f"\nmodel selection, {n_seeds} trials, medians")
    print("selection uses CALIBRATION only; test is reported once, after.\n")
    print(med.to_string(index=False))
    print(f"\nchosen on calibration: {winner}")
    print(f"  calibration cost  full INR {full['select_cost_inr']:,.0f}"
          f"  ->  {winner} INR {won['select_cost_inr']:,.0f}")
    print(f"  test cost         full INR {full['test_cost_inr']:,.0f}"
          f"  ->  {winner} INR {won['test_cost_inr']:,.0f}"
          f"   ({'held up' if verdict['held_up_on_test'] else 'DID NOT hold up'})")
    return med


SHARD_COUNTS = (1, 2, 4, 8, 16)


def sharding() -> pd.DataFrame:
    """Detection quality against shard count, by routing key.

    The prediction is stated in eval/sharding.py before this runs, and it is
    derived from the per-relation ablation rather than from intuition: BIN
    carries the signal, so routing by BIN should hold up while routing at
    random should not.
    """
    train = _train_set(0)
    _, _, kor = _fit_all(train)
    calib = _calibration_set(2)
    thr, _ = cost_optimal_threshold(_raw(kor.score_events(calib)),
                                    calib["label"].to_numpy(),
                                    COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)
    test = _dataset(1, TEST_K, TEST_CAMO)
    df = shard_sweep(kor, test, thr, list(SHARD_COUNTS), window_s=WINDOW_S)

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "sharding.csv", index=False)
    wide = df.pivot(index="n_shards", columns="strategy", values="pr_auc")
    wide.to_csv(RESULTS / "sharding_pr_auc.csv")

    print("\nsharding sweep, frozen model and threshold\n")
    print(df.to_string(index=False))
    print("\nPR-AUC by routing key:")
    print(wide.to_string())
    return df


# Capacity grid. Width and depth were defaults for most of this project's life,
# and a default is not a decision. Swept on the SELECTED relation set, so this
# asks how big the chosen architecture should be rather than re-opening which
# architecture it is.
#
# PREDICTION, before the run. The input is six features and the signal is
# structural - coordination between events, not a rich per-event
# representation - so width should saturate almost immediately. Depth should
# matter more, since a second hop is what reaches coordination that survives
# camouflage, and the architecture ablation already measured exactly that. A
# third layer should not help and may hurt: repeated neighbourhood averaging
# drives node representations together, which is over-smoothing, and a graph
# whose legitimate traffic is dense is a good place for it to bite.
CAPACITY_HIDDEN = (16, 32, 64)
CAPACITY_LAYERS = (1, 2, 3)


def capacity(n_seeds: int = 5) -> pd.DataFrame:
    """How large should the selected architecture be?

    Same protocol as `select`: candidates scored on calibration, threshold
    fitted on a separate calibration draw, test read once at the end.
    """
    rows = []
    for seed in range(n_seeds):
        train = _train_set(seed * 10)
        calib_thr = _calibration_set(seed * 10 + 2)
        calib_sel = _calibration_set(seed * 10 + 5)
        test = _dataset(seed * 10 + 1, TEST_K, TEST_CAMO)
        y_t, y_a, y_b = (test["label"].to_numpy(), calib_thr["label"].to_numpy(),
                         calib_sel["label"].to_numpy())
        for h in CAPACITY_HIDDEN:
            for L in CAPACITY_LAYERS:
                m = KoronisDetector(seed=seed, window_s=WINDOW_S, hidden=h, layers=L)
                m.fit(train, epochs=60)
                thr, _ = cost_optimal_threshold(_raw(m.score_events(calib_thr)), y_a,
                                                COST_PER_ATTEMPT_INR,
                                                COST_PER_FALSE_BLOCK_INR)
                sc = _raw(m.score_events(test))
                fired = sc >= thr
                params = sum(p.numel() for p in m.net.parameters())
                rows.append({
                    "seed": seed, "hidden": h, "layers": L, "params": params,
                    "select_cost_inr": round(_decision_cost(
                        y_b, _raw(m.score_events(calib_sel)), thr), 1),
                    "test_cost_inr": round(_decision_cost(y_t, sc, thr), 1),
                    "test_pr_auc": round(float(average_precision_score(y_t, sc)), 4),
                    "test_precision": round(float(precision_score(y_t, fired, zero_division=0)), 4),
                    "test_recall": round(float(recall_score(y_t, fired, zero_division=0)), 4),
                    "test_false_positives": int((fired & (y_t == 0)).sum()),
                })
    allr = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    allr.to_csv(RESULTS / "capacity_raw.csv", index=False)
    med = (allr.groupby(["hidden", "layers"], sort=False).median(numeric_only=True)
           .drop(columns=["seed"]).round(4).reset_index())
    med.to_csv(RESULTS / "capacity.csv", index=False)

    win = med.loc[med["select_cost_inr"].idxmin()]
    cur = med[(med["hidden"] == 32) & (med["layers"] == 2)].iloc[0]
    json.dump({"selected_hidden": int(win["hidden"]), "selected_layers": int(win["layers"]),
               "selected_params": int(win["params"]),
               "selected_select_cost_inr": float(win["select_cost_inr"]),
               "selected_test_cost_inr": float(win["test_cost_inr"]),
               "default_test_cost_inr": float(cur["test_cost_inr"]),
               "default_is_the_winner": bool(win["hidden"] == 32 and win["layers"] == 2),
               "n_seeds": n_seeds},
              open(RESULTS / "capacity.json", "w"), indent=2)

    print(f"\ncapacity sweep, {n_seeds} trials, medians")
    print("chosen on CALIBRATION; test shown after, never used to choose\n")
    print(med.to_string(index=False))
    print("\ncalibration cost by size (the column that selects):")
    print(med.pivot(index="hidden", columns="layers", values="select_cost_inr").to_string())
    print(f"\nchosen: hidden={int(win['hidden'])} layers={int(win['layers'])} "
          f"({int(win['params'])} params)")
    return med


def replicate() -> pd.DataFrame:
    """Can copying a minority of events restore the edges a partition deletes?

    Routing by BIN keeps precision and loses recall as shards multiply, because
    device and IP edges are cut. The prediction is stated in eval/sharding.py
    before this runs: since entity frequencies are heavy-tailed, replicating
    only events whose device or IP actually recurs should restore most of those
    edges while copying a minority of traffic - and campaign events, whose
    entities are shared by construction, should be copied preferentially.
    """
    train = _train_set(0)
    _, _, kor = _fit_all(train)
    calib = _calibration_set(2)
    thr, _ = cost_optimal_threshold(_raw(kor.score_events(calib)),
                                    calib["label"].to_numpy(),
                                    COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)
    test = _dataset(1, TEST_K, TEST_CAMO)
    df = sweep_replication(kor, test, thr, list(SHARD_COUNTS), window_s=WINDOW_S)

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "replication.csv", index=False)
    print("\nrecovering sharded edges by replication, frozen model and threshold\n")
    print(df.to_string(index=False))
    print("\nrecall by routing:")
    print(df.pivot(index="n_shards", columns="routing", values="recall").to_string())
    print("\ncompute cost (duplication factor):")
    print(df.pivot(index="n_shards", columns="routing", values="duplication").to_string())
    return df


def online(n_streams: int = 6) -> pd.DataFrame:
    """How much does making consolidation causal actually cost?

    `build_incidents` decides whether an entity value is too common to link on
    by counting it across the whole frame - including events that had not
    happened when the alert fired. `StreamingIncidents` replaces that with a
    sliding count-min sketch fed event by event, so the decision at time t uses
    only what was known at time t, in memory fixed by the sketch rather than by
    how many distinct entity values the stream contains.

    The two will not agree perfectly, and the batch one is not the ground
    truth: it is the one using the future. What matters is whether the online
    grouping still separates campaigns from background, which is measured here
    by comparing both against the labels.
    """
    train = _train_set(0)
    _, _, kor = _fit_all(train)
    calib = _calibration_set(2)
    thr, _ = cost_optimal_threshold(_raw(kor.score_events(calib)),
                                    calib["label"].to_numpy(),
                                    COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)

    rows = []
    for j in range(n_streams):
        ev = _sized_stream(900 + j * 11, TEST_K, TEST_CAMO, j)
        sc = _raw(kor.score_events(ev))
        labels = ev["label"].to_numpy()

        batch = build_incidents(ev, sc, thr)
        st = StreamingIncidents(threshold=thr, freq_window_s=WINDOW_S)
        for i, (_, e) in enumerate(ev.iterrows()):
            st.push(e, float(sc[i]), row=i)
        groups = st.groups()

        def purity(members: list[list[int]]) -> tuple[int, float, int]:
            """Incidents formed, purity of the largest, and campaign recall."""
            if not members:
                return 0, float("nan"), 0
            big = max(members, key=len)
            pure = float((labels[big] == 1).mean())
            found = int((labels[big] == 1).sum())
            return len(members), pure, found

        b_n, b_pure, b_found = purity([i.rows for i in batch])
        o_n, o_pure, o_found = purity(list(groups.values()))
        n_camp = int((labels == 1).sum())
        stats = st.stats()
        rows.append({
            "stream": j, "campaign_attempts": n_camp,
            "batch_incidents": b_n, "online_incidents": o_n,
            "batch_purity": round(b_pure, 4), "online_purity": round(o_pure, 4),
            "batch_recall": round(b_found / n_camp, 4),
            "online_recall": round(o_found / n_camp, 4),
            "sketch_kb": stats["sketch_kb"],
            "buffered_alert_refs": stats["buffered_alert_refs"],
        })

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "online.csv", index=False)
    med = df.drop(columns=["stream"]).median().round(4)
    json.dump({"per_stream": df.to_dict("records"),
               "median": med.to_dict()},
              open(RESULTS / "online.json", "w"), separators=(",", ":"))

    print("\nonline consolidation vs batch, per stream\n")
    print(df.to_string(index=False))
    print("\nmedians:")
    print(med.to_string())
    return df





# ----------------------------------------------------------- feature parity
def feature_parity(trials: int = 3) -> pd.DataFrame:
    """One disclosed asymmetry between the detector and its baseline, measured.

    The two feature sets are meant to match, so that all coordination signal has
    to reach the detector through the graph. One difference survives: the
    baseline's free-mail flag covers gmail/yahoo/outlook where the detector's
    covers gmail/outlook, and yahoo is about 15% of generated traffic.

    Rather than assert the difference is immaterial, this measures it. Aligning
    the two lists would mean re-running every published figure, so the honest
    move is to publish the size of the gap and let a reader judge it.
    """
    import koronis.models.gbdt as gbdt_mod

    train = _train_set(0)
    original = gbdt_mod.transaction_features

    def matched(events):
        f = original(events).copy()
        f["email_is_free"] = (events["email_domain"]
                              .isin(["gmail.com", "outlook.com"])
                              .to_numpy().astype(float))
        return f

    rows = []
    try:
        for label, fn in (("as_shipped_gmail_yahoo_outlook", original),
                          ("matched_to_detector_gmail_outlook", matched)):
            gbdt_mod.transaction_features = fn
            for seed in range(trials):
                ev = _sized_stream(900 + seed * 11, TEST_K, TEST_CAMO, seed)
                y = ev["label"].to_numpy()
                m = gbdt_mod.GBDTDetector(seed=seed)
                m.fit(train)
                p = m.score_events(ev)
                rows.append({"baseline_free_mail_list": label, "seed": seed,
                             "pr_auc": average_precision_score(y, p),
                             "false_positives": int(((p >= 0.5) & (y == 0)).sum())})
    finally:
        gbdt_mod.transaction_features = original

    df = pd.DataFrame(rows)
    med = (df.groupby("baseline_free_mail_list")[["pr_auc", "false_positives"]]
             .median().round(4).reset_index())
    RESULTS.mkdir(exist_ok=True)
    med.to_csv(RESULTS / "feature_parity.csv", index=False)
    df.to_csv(RESULTS / "feature_parity_raw.csv", index=False)

    print(f"\nbaseline free-mail list, {trials} trials, medians\n")
    print(med.to_string(index=False))
    print("\nThe detector's own list is gmail/outlook. A positive gap here means the "
          "\nasymmetry favours the baseline; a negative one means it costs the baseline.")
    return med


# ------------------------------------------------------------------- ceiling
def ceiling(trials: int = 3) -> pd.DataFrame:
    """Is the gap a modelling gap, or an information gap?

    Every headline comparison in this repo is against per-transaction models, so
    the obvious objection is that the baseline was simply too small - that a
    bigger learner, or a more fashionable one, would close it. That is testable
    without guessing: hold the per-event feature set fixed, scale capacity over
    two unrelated model families, and see where each one stops.

    The families are chosen to fail differently. LightGBM at increasing trees
    and leaves is a strong tabular learner with a different inductive bias from
    a neural net; the Koronis architecture with `use_edges=False` is the *same*
    network as the graph model with only the edges removed, which isolates the
    edges rather than the architecture.

    This is also the honest answer to "why is there no language model in here".
    A transformer reading one transaction is another per-event model, and the
    ceiling below is a property of what a single authorisation contains, not of
    who is reading it. A model given the neighbourhood as text is no longer a
    per-event model - it is doing the graph's job, at a per-event budget
    measured here at 0.91 ms (`koronis.cli benchmark`).
    """
    train = _train_set(0)
    rows = []

    grid_gbdt = [(50, 31), (300, 31), (1500, 63), (4000, 255)]
    grid_net = [(8, 1), (32, 2), (32, 3), (128, 3), (256, 3)]

    for seed in range(trials):
        ev = _sized_stream(900 + seed * 11, TEST_K, TEST_CAMO, seed)
        y = ev["label"].to_numpy()

        for n_est, leaves in grid_gbdt:
            m = GBDTDetector(seed=seed, n_estimators=n_est, num_leaves=leaves)
            m.fit(train)
            rows.append({"family": "per-event GBDT",
                         "capacity": f"{n_est} trees x {leaves} leaves",
                         "params": n_est * leaves, "seed": seed,
                         "pr_auc": average_precision_score(y, m.score_events(ev))})

        for hidden, layers in grid_net:
            for use_edges, family in ((False, "per-event net"), (True, "graph net")):
                m = KoronisDetector(hidden=hidden, layers=layers, seed=seed,
                                    use_edges=use_edges, window_s=WINDOW_S)
                # Same training budget as every published number (`_fit_all`).
                # An unfair budget would make this comparison meaningless in
                # the direction that flatters the conclusion.
                m.fit(train, epochs=60)
                rows.append({"family": family,
                             "capacity": f"{hidden} wide x {layers} deep",
                             "params": sum(p.numel() for p in m.net.parameters()),
                             "seed": seed,
                             "pr_auc": average_precision_score(y, _raw(m.score_events(ev)))})

    df = pd.DataFrame(rows)
    med = (df.groupby(["family", "capacity", "params"])["pr_auc"]
             .median().round(4).reset_index()
             .sort_values(["family", "params"]))
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "ceiling_raw.csv", index=False)
    med.to_csv(RESULTS / "ceiling.csv", index=False)

    best = med.groupby("family")["pr_auc"].max()
    json.dump({"best_by_family": {k: float(v) for k, v in best.items()},
               "trials": trials},
              open(RESULTS / "ceiling.json", "w"), separators=(",", ":"))

    print(f"\nper-event ceiling vs the graph, {trials} trials, medians\n")
    print(med.to_string(index=False))
    print("\nbest PR-AUC by family:")
    print(best.round(4).to_string())
    return med


# ---------------------------------------------------------------- resilience
# Fault injection. Every fault here was found by probing the live streaming
# path, not imagined: each one used to be silent, which is the property that
# made them dangerous. A detector that answers "no alert" when it means "I
# could not read this event" is worse than one that stops.
def _corrupt(events: pd.DataFrame, kind: str, rate: float, seed: int):
    ev = events.copy()
    rng = np.random.default_rng(seed)
    hit = rng.random(len(ev)) < rate

    if kind == "clean":
        return ev, np.zeros(len(ev), dtype=bool)
    if kind == "null_device":
        ev["device_id"] = ev["device_id"].astype(object)
        ev.loc[hit, "device_id"] = None
    elif kind == "placeholder_device":
        ev["device_id"] = ev["device_id"].astype(object)
        # The upstream mistake this guards against: a null replaced by a
        # constant before it ever reaches the detector. It is then an ordinary
        # value, indistinguishable from a device that really is shared.
        ev.loc[hit, "device_id"] = "MISSING_DEVICE"
    elif kind == "nan_amount":
        ev.loc[hit, "amount"] = float("nan")
    elif kind == "dropped_approved":
        # `approved` is a bool column; a null cannot live in it without a cast,
        # which is itself how this fault reaches production - the null is lost
        # at the boundary rather than inside the model.
        ev["approved"] = ev["approved"].astype(object)
        ev.loc[hit, "approved"] = None
    elif kind == "entity_explosion":
        # A fresh device per attempt - the shape a card-testing ring already
        # has, pushed to every event, so the index cannot amortise anything.
        ev["device_id"] = [f"x{i}" for i in range(len(ev))]
    else:
        raise ValueError(kind)
    return ev, hit


# The placeholder is swept either side of the link-share cap on purpose. Above
# the cap the frequency guard already refuses to link on it; below the cap that
# guard is silent, and `entity_key` is the only thing standing between a missing
# device fingerprint and an invented ring.
SCENARIOS = [
    ("clean", 0.00),
    ("null_device", 0.01),
    ("placeholder_device", 0.01),
    ("null_device", 0.10),
    ("placeholder_device", 0.10),
    ("nan_amount", 0.05),
    ("dropped_approved", 0.05),
    ("entity_explosion", 1.00),
]


def resilience(n_streams: int = 4) -> pd.DataFrame:
    """What the detector does when the stream is not clean.

    Injects one fault class at a time into a held-out stream and measures the
    consequence, with the model and threshold frozen throughout. The claim being
    tested is not "nothing changes" - a corrupted event genuinely cannot be
    scored - but that the failure is **loud and bounded**: quarantined and
    counted rather than scored NaN, unable to invent links out of missing data,
    and unable to grow memory past the window.

    `placeholder_device` is the control that makes the point. It is the same
    missing data as `null_device`, except a well-meaning upstream step replaced
    the null with a constant first. The detector cannot tell that constant from
    a device that really is shared, which is why the null has to survive intact
    all the way to `entity_key`.
    """
    train = _train_set(0)
    _, _, kor = _fit_all(train)
    calib = _calibration_set(2)
    thr, _ = cost_optimal_threshold(_raw(kor.score_events(calib)),
                                    calib["label"].to_numpy(),
                                    COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)

    rows = []
    for kind, rate in SCENARIOS:
        per = []
        for j in range(n_streams):
            ev = _sized_stream(900 + j * 11, TEST_K, TEST_CAMO, j)
            labels = ev["label"].to_numpy()
            bad, hit = _corrupt(ev, kind, rate, seed=j)

            stream = StreamingKoronis(kor, threshold=thr, window_s=WINDOW_S)
            inc = StreamingIncidents(threshold=thr, freq_window_s=WINDOW_S)
            peak, alerts, nan_scores, hit_links = 0, 0, 0, 0
            for i, (_, row) in enumerate(bad.iterrows()):
                out = stream.push(row)
                peak = max(peak, len(stream._x))
                if out["score"] is None:
                    continue
                # Device links reported for an event whose device was destroyed.
                # This is coordination read out of missing data, counted before
                # the alert threshold can hide it - the incident-level view
                # cannot see it, because linking only happens above threshold.
                if hit[i]:
                    hit_links += out["evidence"].get("device_id", 0)
                if not np.isfinite(out["score"]):
                    nan_scores += 1                # must never happen
                if out["alert"]:
                    alerts += 1
                inc.push(row, float(out["score"]), row=i)

            groups = list(inc.groups().values())
            biggest = max(groups, key=len) if groups else []
            # An incident of two or more events, none of which is a campaign
            # event, is coordination the detector invented. This is the number
            # the missing-entity rule exists to hold at zero.

            scored = len(bad) - stream.quarantined
            # Recall is over campaign events the detector was actually given a
            # readable copy of; the quarantined ones are reported separately
            # rather than folded in, because losing them is the documented
            # behaviour, not a detection failure.
            found = int(labels[biggest].sum()) if len(biggest) else 0
            in_window = int(bad["ts"].gt(bad["ts"].max() - WINDOW_S).sum())

            per.append({
                "quarantined": stream.quarantined,
                "scored": scored,
                "nan_scores": nan_scores,
                "alerts": alerts,
                "incidents": len(groups),
                "device_links_on_corrupted": hit_links,
                "largest_incident": len(biggest),
                "largest_purity": round(float(labels[biggest].mean()), 4) if len(biggest) else float("nan"),
                "campaign_recall": round(found / max(int(labels.sum()), 1), 4),
                "peak_cache_rows": peak,
                "events_in_window": in_window,
                "events_total": len(bad),
            })

        med = pd.DataFrame(per).median().round(4)
        rows.append({"fault": kind, "rate": rate, **med.to_dict()})

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "resilience.csv", index=False)

    print("\nfault injection, frozen model and threshold, medians over "
          f"{n_streams} held-out streams\n")
    print(df.to_string(index=False))
    print("\ninvariants, both of which used to be violated silently:")
    print("  nan_scores        must be 0 - a NaN score compares False against "
          "the threshold and reports itself as 'no alert'")
    print("  device_links_on_corrupted must be 0 wherever the device was "
          "destroyed - any link there is coordination read out of missing data")
    return df


# Architecture ablations. These test the MODEL, not the data sources the
# mechanism ablation covers: the heterophily gate and the learned relation
# attention are design claims made in layers.py, and until they are removed
# and re-measured they are only assertions.
# Expressed as departures from the SELECTED architecture, which is the default.
# When the gate was still on by default this read "no_gate"; after selection
# removed it, the honest question is what putting it back costs - and a variant
# dict of `{}` would silently be the baseline compared against itself. Depth
# moved here too: `koronis.cli capacity` now owns the full width x depth grid,
# so this only needs the adjacent step down from the selected three layers.
ARCH_VARIANTS = {
    "selected": dict(),
    "add_gate": dict(use_gate=True),
    "uniform_relation_attention": dict(use_rel_attention=False),
    "two_layers": dict(layers=2),
}

# The gate's justification is specific, so the test can be too. It exists to
# damp edges joining DISSIMILAR nodes, and camouflage is exactly what creates
# those: a camouflaged attempt draws its amount and email domain from the
# background, so it links to legitimate traffic that looks nothing like the
# rest of the ring. The prediction is therefore conditional - the gate should
# buy more as camouflage rises, and little at camouflage 0 where the campaign
# is separable per-event anyway. A flat difference would mean the gate helps
# for some other reason and the stated justification is wrong.
ARCH_CAMOS = (0.0, 0.5, 1.0)


def architecture(n_seeds: int = 5) -> pd.DataFrame:
    """Do the two architectural claims survive being removed?

    Same three-split protocol as every other ablation: fit on train, freeze a
    cost-optimal threshold on calibration, report on the held-out test stream.
    Swept across camouflage, because the gate's justification predicts its
    benefit should depend on it.
    """
    rows = []
    for seed in range(n_seeds):
        train = _train_set(seed * 10)
        calib = _calibration_set(seed * 10 + 2)
        y_cal = calib["label"].to_numpy()
        # Training does not depend on the TEST camouflage, so each variant is
        # fitted once per seed and then met with all three test streams. That
        # is also the cleaner comparison: one model, varying only the attack.
        tests = {c: _dataset(seed * 10 + 1, TEST_K, c) for c in ARCH_CAMOS}
        for name, kw in ARCH_VARIANTS.items():
            m = KoronisDetector(seed=seed, window_s=WINDOW_S, **kw)
            m.fit(train, epochs=60)
            thr, _ = cost_optimal_threshold(_raw(m.score_events(calib)), y_cal,
                                            COST_PER_ATTEMPT_INR,
                                            COST_PER_FALSE_BLOCK_INR)
            for camo, test in tests.items():
                y = test["label"].to_numpy()
                sc = _raw(m.score_events(test))
                fired = sc >= thr
                rows.append({
                    "seed": seed, "camouflage": camo, "variant": name,
                    "pr_auc": round(float(average_precision_score(y, sc)), 4),
                    "precision": round(float(precision_score(y, fired, zero_division=0)), 4),
                    "recall": round(float(recall_score(y, fired, zero_division=0)), 4),
                    "false_positives": int((fired & (y == 0)).sum()),
                })

    allr = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    allr.to_csv(RESULTS / "architecture_raw.csv", index=False)

    med = (allr.groupby(["camouflage", "variant"], sort=False)
           [["pr_auc", "precision", "recall", "false_positives"]]
           .median().round(4).reset_index())
    med.to_csv(RESULTS / "architecture.csv", index=False)

    wide = med.pivot(index="camouflage", columns="variant", values="pr_auc")
    for v in ARCH_VARIANTS:
        if v != "selected":
            wide[f"delta_{v}"] = (wide["selected"] - wide[v]).round(4)
    wide.to_csv(RESULTS / "architecture_delta.csv")

    print(f"\narchitecture ablation, {n_seeds} trials, medians\n")
    print(med.to_string(index=False))
    print("\nPR-AUC, and what removing each piece costs:")
    print(wide.to_string())
    return med


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
    from .data.schema import RELATIONS

    original = list(RELATIONS)
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
            # The relation set is a constructor argument, so a variant is a
            # different model rather than a mutated global - which used to be
            # patched here and could leak into any later experiment.
            m = KoronisDetector(seed=seed, window_s=WINDOW_S, relations=rels,
                                use_gate=True)
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



def bin_concentration(n_streams: int = 6) -> pd.DataFrame:
    """Does legitimate BIN concentration alone produce false alarms?

    The per-relation ablation says BIN carries most of the signal, and until
    `bin_dense` was added no shifted profile concentrated it: two held it at base
    and one made it more diffuse. So the relation the detector leans on hardest
    had no adversarial legitimate profile. A domestic sale event on a handful of
    issuers is exactly that case, and it is the strongest criticism available
    against this design.

    The drift sweep cannot answer it, because every stream it builds contains an
    injected campaign. This one contains **none**: pure legitimate traffic,
    scored with the frozen threshold, so every alert is a false one. Both layers
    are measured, because they defend separately - the detector's own score, and
    the link-share cap that refuses to consolidate on a value covering more than
    2% of the stream.

    The prediction under test, stated before running: a dense legitimate BIN
    component and a card-testing ring differ on the authorisation outcome, which
    is a model feature, so concentration alone should not be enough. If that is
    wrong the number says so.
    """
    from .profiles import BY_NAME

    train = _train_set(0)
    _, _, kor = _fit_all(train)
    calib = _calibration_set(2)
    thr, _ = cost_optimal_threshold(_raw(kor.score_events(calib)),
                                    calib["label"].to_numpy(),
                                    COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR)

    rows = []
    for name in ("base", "bin_dense"):
        prof = BY_NAME[name]
        for j in range(n_streams):
            # No campaign injected: every alert below is a false positive.
            ev = load_background(path=None, n_rows=N_BACKGROUND,
                                 seed=4200 + j * 31, profile=prof)
            sc = _raw(kor.score_events(ev))
            alerts = int((sc >= thr).sum())

            share = ev["bin_id"].value_counts(normalize=True)
            inc = build_incidents(ev, sc, thr)
            rows.append({
                "profile": name,
                "stream": j,
                "events": len(ev),
                "false_alerts": alerts,
                "false_alert_rate": round(alerts / len(ev), 5),
                "false_incidents": len(inc),
                "largest_false_incident": max((len(i.rows) for i in inc), default=0),
                "distinct_bins": int(ev["bin_id"].nunique()),
                "top_bin_share": round(float(share.iloc[0]), 4),
                "bins_over_link_cap": int((share > MAX_LINK_SHARE).sum()),
            })

    # Second half: the same profiles WITH a campaign, so the cost of BIN
    # washing out as a discriminator is measured rather than argued. An earlier
    # version of this took these numbers from a one-off script; feature_parity
    # already showed how a one-off can flip sign against the test protocol.
    det = []
    for name in ("base", "bin_dense"):
        for j in range(4):
            ev = _profile_stream(1200 + j * 13, BY_NAME[name], TEST_K, TEST_CAMO, 400)
            y = ev["label"].to_numpy()
            sc = _raw(kor.score_events(ev))
            det.append({
                "profile": name, "stream": j,
                "pr_auc": average_precision_score(y, sc),
                "recall": ((sc >= thr) & (y == 1)).sum() / max((y == 1).sum(), 1),
                "false_positives": int(((sc >= thr) & (y == 0)).sum()),
            })
    det_med = (pd.DataFrame(det).drop(columns=["stream"])
               .groupby("profile").median().round(4).reset_index())

    df = pd.DataFrame(rows)
    med = df.drop(columns=["stream"]).groupby("profile").median().round(5).reset_index()
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "bin_concentration_raw.csv", index=False)
    med.to_csv(RESULTS / "bin_concentration.csv", index=False)
    det_med.to_csv(RESULTS / "bin_concentration_detection.csv", index=False)

    print(f"\nlegitimate traffic only, no campaign injected, {n_streams} streams "
          "per profile, medians\n")
    print(med.to_string(index=False))
    print("\nwith a campaign injected into the same profiles\n")
    print(det_med.to_string(index=False))
    print("\nEvery alert in the first table is a false one. `bins_over_link_cap` is how many "
          "BIN\nvalues exceed the 2% link-share cap - those are refused as consolidation "
          "evidence,\nwhich is the second layer of defence and is separate from the score.")
    return df


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



def saturation() -> pd.DataFrame:
    """Where does *this* model stop working? - and why this sweep cannot say.

    The frontier draws the baseline's failure boundary and shows Koronis
    detecting across the whole grid. That is half a characterisation: a detector
    with no measured failure boundary has not been characterised. So `k` is
    pushed to `n`, where every attempt carries its own device, IP and BIN, the
    campaign shares nothing with itself, and there is no campaign subgraph left.

    The model still reports recall 1.0 there. That is not a result about the
    model - it is a result about the generator, and the diagnostic columns say
    so. At `k = n` every campaign event has degree **zero** while background
    events average ~46 and none are isolated, because the campaign draws its
    entities from a pool disjoint from the background's. "Has no neighbours at
    all" is then a perfect label proxy, available for free, and the model is
    reading that rather than any coordination.

    Real traffic is full of first-time customers on a fresh device, IP and BIN.
    A background where zero legitimate events are isolated cannot test the claim
    this sweep was built to test, so the sweep is published as an invalid
    measurement rather than as a favourable one. The limitation it was meant to
    probe - an attacker on genuinely fresh infrastructure leaves no graph signal
    - stands unmeasured, which is what the README already says.
    """
    import numpy as np

    from .data.schema import MODEL_RELATIONS
    from .eval.frontier import sweep
    from .graph.build import build_edges

    df = sweep(n_values=[400, 800], k_values=[2, 10, 50, 100, 200, 400, 800],
               fp_budget=FP_BUDGET, seed=0, window_s=WINDOW_S,
               n_background=N_BACKGROUND)
    df["k_over_n"] = (df["k"] / df["n"]).round(3)

    # The diagnostic that invalidates the sweep, measured rather than asserted.
    bg = load_background(path=None, n_rows=N_BACKGROUND, seed=0)
    diag = []
    for n, k in zip(df["n"], df["k"]):
        ev = inject(bg, [CampaignSpec(n_attempts=int(n), k_devices=int(k),
                                      k_ips=int(k), n_bins=int(k),
                                      duration_s=WINDOW_S,
                                      start_ts=float(bg["ts"].iloc[300]))], seed=0)
        y = ev["label"].to_numpy() == 1
        deg = np.zeros(len(ev))
        for arr in build_edges(ev, window_s=WINDOW_S,
                               relations=MODEL_RELATIONS).values():
            if arr.shape[1]:
                np.add.at(deg, arr[0], 1)
                np.add.at(deg, arr[1], 1)
        shared = sum(len(set(ev[r][y]) & set(ev[r][~y])) for r in MODEL_RELATIONS)
        diag.append({"campaign_isolated": round(float((deg[y] == 0).mean()), 4),
                     "background_isolated": round(float((deg[~y] == 0).mean()), 4),
                     "entity_values_shared_with_background": shared})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(diag)], axis=1)

    df = df[["n", "k", "k_over_n", "velocity_detected", "koronis_detected",
             "koronis_recall", "koronis_pr_auc", "campaign_isolated",
             "background_isolated", "entity_values_shared_with_background"]]
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "saturation.csv", index=False)

    print("\nspread pushed to one entity per attempt, frozen model and threshold\n")
    print(df.to_string(index=False))
    print("\nRead the last three columns before the recall column. Where "
          "campaign_isolated\nreaches 1.0 while background_isolated stays at 0.0 and no "
          "entity value is shared,\n'has no neighbours' is a perfect label proxy and the "
          "recall above measures the\ngenerator, not the detector. This sweep does not "
          "locate the model's failure\nboundary; it shows why this simulator cannot.")
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
            # Which prior attempts, and via which entity value. The demo
            # panel names them instead of showing a bare count.
            "evid": out["evidence_ids"],
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
     "relations": relations, "aperture": aperture,
     "architecture": architecture, "online": online,
     "resilience": resilience, "ceiling": ceiling,
     "feature_parity": feature_parity, "saturation": saturation,
     "bin_concentration": bin_concentration,
     "sharding": sharding, "select": select,
     "replicate": replicate, "capacity": capacity}[cmd]()
