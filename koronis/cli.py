"""Experiment entry points.

    python -m koronis.cli ablation    # the headline comparison
    python -m koronis.cli frontier    # predicted vs measured boundary
    python -m koronis.cli latency     # precision/recall/INR over time
    python -m koronis.cli seeds       # repeat across seeds, report intervals
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from .data.background import load_background
from .data.campaigns import inject
from .data.schema import CampaignSpec
from .eval.calibration import cost_optimal_threshold, expected_calibration_error
from .eval.cost import COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR
from .eval.latency import detection_times, exposure, latency_curve, money_prevented
from .models.gbdt import GBDTDetector
from .models.heuristic import DeclineBurstDetector, SharedEntityDetector
from .models.koronis import KoronisDetector
from .models.velocity import MultiEntityVelocityDetector, tune_velocity

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


def _dataset(seed: int, k: int, camouflage: float = 0.0) -> pd.DataFrame:
    bg = load_background(path=None, n_rows=N_BACKGROUND, seed=seed)
    spec = CampaignSpec(n_attempts=N_ATTEMPTS, k_devices=k, k_ips=k, n_bins=k,
                        duration_s=WINDOW_S, start_ts=float(bg["ts"].iloc[500]),
                        camouflage=camouflage)
    return inject(bg, [spec], seed=seed)


def _calibration_set() -> pd.DataFrame:
    """A third split, drawn from the TRAINING distribution, used only to pick
    the operating threshold.

    Choosing a threshold with test labels optimises detection time, false
    positives and rupees prevented against the answers - the operating-point
    results stop being held out. Calibration comes from the same distribution
    as training because that is what a deployed system would actually have:
    you tune on attacks you have already seen, then meet a new one.
    """
    return _train_set(seed=2)


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
    calib = _train_set(seed * 10 + 2)
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

    Intervals are reported as median with 2.5th/97.5th percentiles rather than
    mean +/- 1.96*sd, because several of these metrics are visibly skewed - a
    detector that collapses on some draws has a long left tail that a normal
    approximation misrepresents.
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

    print(f"\n{n_seeds} independent trials; median [2.5th, 97.5th percentile]\n")
    for _, r in out.iterrows():
        print(f"  {r['detector']:>15}"
              f"  PR-AUC {r['pr_auc_median']:.3f} [{r['pr_auc_lo95']:.3f}, {r['pr_auc_hi95']:.3f}]"
              f"  P {r['precision_median']:.3f} [{r['precision_lo95']:.3f}, {r['precision_hi95']:.3f}]"
              f"  R {r['recall_median']:.3f}"
              f"  FP {r['false_positives_median']:.0f} [{r['false_positives_lo95']:.0f}, {r['false_positives_hi95']:.0f}]"
              f"  detect {r['detected_rate']:.0%}")
    print("\nper-trial values: results/seeds_raw.csv")
    return out


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
    calib = _train_set(2)
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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    {"ablation": ablation, "frontier": frontier, "latency": latency,
     "seeds": seeds}[cmd]()
