"""Experiment entry points.

    python -m koronis.cli ablation    # the headline comparison
    python -m koronis.cli frontier    # predicted vs measured boundary
    python -m koronis.cli latency     # precision/recall/INR over time
    python -m koronis.cli seeds       # repeat across seeds, report intervals
    python -m koronis.cli replay      # causal event-by-event replay -> JSON
    python -m koronis.cli benchmark   # p50/p95 per-event inference latency
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
from .eval.calibration import cost_optimal_threshold, expected_calibration_error
from .eval.cost import COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR
from .eval.latency import detection_times, exposure, latency_curve, money_prevented
from .models.gbdt import GBDTDetector
from .models.heuristic import DeclineBurstDetector, SharedEntityDetector
from .models.koronis import KoronisDetector
from .models.velocity import MultiEntityVelocityDetector, tune_velocity
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


def _dataset(seed: int, k: int, camouflage: float = 0.0) -> pd.DataFrame:
    bg = load_background(path=None, n_rows=N_BACKGROUND, seed=seed)
    spec = CampaignSpec(n_attempts=N_ATTEMPTS, k_devices=k, k_ips=k, n_bins=k,
                        duration_s=WINDOW_S, start_ts=float(bg["ts"].iloc[500]),
                        camouflage=camouflage)
    return inject(bg, [spec], seed=seed)


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
     "seeds": seeds, "replay": replay, "benchmark": benchmark}[cmd]()
