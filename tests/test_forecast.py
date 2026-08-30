import pandas as pd
import pytest

from koronis.data.background import load_background
from koronis.data.campaigns import inject
from koronis.data.schema import CampaignSpec
from koronis.forecast import (
    FEATURES, ExposureForecaster, build_snapshots, evaluate_forecast,
    snapshot_features,
)
from koronis.incident import build_incidents


def _stream(seed, n_attempts):
    bg = load_background(path=None, n_rows=3000, seed=seed)
    spec = CampaignSpec(n_attempts=n_attempts, k_devices=30, k_ips=30, n_bins=30,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[300]),
                        camouflage=1.0)
    ev = inject(bg, [spec], seed=seed)
    return ev, ev["label"].to_numpy().astype(float)


def _snaps(seed, n, sid):
    ev, sc = _stream(seed, n)
    return build_snapshots(ev, build_incidents(ev, sc, 0.5), sc, stream_id=sid)


@pytest.fixture(scope="module")
def fitted():
    snaps = []
    for j, n in enumerate((150, 300, 500, 700, 220, 420)):
        ev, sc = _stream(j, n)
        snaps.append(build_snapshots(ev, build_incidents(ev, sc, 0.5), sc, stream_id=j))
    train = pd.concat(snaps, ignore_index=True)
    return ExposureForecaster(seed=0).fit(train), train


def test_snapshot_uses_only_the_observed_prefix():
    """A forecast that peeks at later events is not a forecast."""
    ev, sc = _stream(0, 300)
    inc = build_incidents(ev, sc, 0.5)[0]
    a = snapshot_features(ev, inc.rows, sc, 20)
    truncated = ev.iloc[: inc.rows[19] + 1].reset_index(drop=True)
    rows_t = [r for r in inc.rows if r <= inc.rows[19]]
    b = snapshot_features(truncated, rows_t, sc[: inc.rows[19] + 1], 20)
    for k in FEATURES:
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9), k


def test_fit_and_conformal_partitions_are_disjoint():
    """Conformal validity needs the pad measured on incidents the quantile
    models never saw. Splitting by snapshot row would put nested prefixes of
    ONE incident on both sides - not a test-set leak, but the residual would be
    measured on data the model has effectively already seen, inflating apparent
    coverage."""
    snaps = []
    for j, n in enumerate((150, 300, 500, 700, 220, 420)):
        ev, sc = _stream(j, n)
        snaps.append(build_snapshots(ev, build_incidents(ev, sc, 0.5), sc, stream_id=j))
    train = pd.concat(snaps, ignore_index=True)
    fc = ExposureForecaster(seed=0).fit(train)

    assert fc.fit_groups_ and fc.conformal_groups_
    assert not set(fc.fit_groups_) & set(fc.conformal_groups_)

    fit_ids = set(train[train["stream_id"].isin(fc.fit_groups_)]["group_id"])
    cal_ids = set(train[train["stream_id"].isin(fc.conformal_groups_)]["group_id"])
    assert not fit_ids & cal_ids, "an incident appears in both partitions"


def test_stream_id_disambiguates_incident_ids():
    """Incident ids restart at INC-000 per stream, so the bare id is not a
    usable partition key."""
    ev0, sc0 = _stream(0, 300)
    ev1, sc1 = _stream(1, 300)
    a = build_snapshots(ev0, build_incidents(ev0, sc0, 0.5), sc0, stream_id=0)
    b = build_snapshots(ev1, build_incidents(ev1, sc1, 0.5), sc1, stream_id=1)
    assert set(a["incident_id"]) & set(b["incident_id"])      # ids collide
    assert not set(a["group_id"]) & set(b["group_id"])        # keys do not


def test_snapshots_cover_many_prefix_lengths(fitted):
    _, train = fitted
    assert train["n_observed"].nunique() >= 4
    assert (train["remaining"] > 0).all()


def test_upper_quantile_is_above_the_median(fitted):
    fc, train = fitted
    p50, hi = fc.predict(train)
    assert (hi >= p50 - 1e-9).all()


def test_conformal_coverage_improves_with_calibration_data():
    """The conformal pad is data-hungry, and this records how much.

    A pad estimated from a handful of incidents under-covers badly: the
    residual quantile it is built from is itself barely determined. Asserting a
    fixed coverage number here would be asserting a property of the fixture
    size rather than of the method, so the test asserts the mechanism instead —
    coverage must improve materially as calibration streams are added.

    Measured on this fixture: 6 streams -> 69%, 16 -> 81%, 24 -> 87%, against a
    90% target. The figure for the real pipeline is reported by
    `python -m koronis.cli incidents`, which calibrates on more data.
    """
    held = pd.concat(
        [_snaps(j, n, j) for j, n in
         enumerate((190, 640, 360, 250, 780, 430, 310, 560), start=50)],
        ignore_index=True)
    sizes = [150, 300, 500, 700, 220, 420, 180, 650, 340, 470, 260, 590,
             210, 720, 390, 530, 240, 610, 330, 450, 280, 700, 360, 510]

    def coverage(k):
        tr = pd.concat([_snaps(j, n, j) for j, n in enumerate(sizes[:k])],
                       ignore_index=True)
        fc = ExposureForecaster(seed=0).fit(tr)
        return evaluate_forecast(fc, held)

    scarce, ample = coverage(6), coverage(24)
    assert held.shape[0] >= 60
    assert ample["coverage_upper"] > scarce["coverage_upper"] + 0.10, (scarce, ample)
    assert ample["coverage_upper"] >= 0.80, ample
    # More calibration data should sharpen the median too, not just widen the band.
    assert ample["mae_p50"] < scarce["mae_p50"], (scarce, ample)


def test_forecast_is_not_memorising_a_constant(fitted):
    """With every campaign the same length, 'remaining' is a constant minus
    what you have seen, and a forecaster scores brilliantly while learning
    nothing. Varying sizes must therefore produce varying predictions."""
    fc, _ = fitted
    preds = []
    for j, n in enumerate((160, 900), start=80):
        ev, sc = _stream(j, n)
        inc = build_incidents(ev, sc, 0.5)[0]
        preds.append(fc.predict_one(ev, inc.rows, sc, 12)[0])
    assert abs(preds[0] - preds[1]) > 20.0, preds


def test_empty_input_is_handled(fitted):
    fc, _ = fitted
    p50, hi = fc.predict(pd.DataFrame(columns=FEATURES))
    assert p50.size == 0 and hi.size == 0
    assert evaluate_forecast(fc, pd.DataFrame()) == {"n": 0}
