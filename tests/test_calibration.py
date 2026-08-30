import numpy as np

from koronis.eval.calibration import reliability, cost_optimal_threshold


def test_reliability_bins_sum_to_population():
    rng = np.random.default_rng(0)
    s, y = rng.random(500), rng.integers(0, 2, 500)
    assert reliability(s, y, bins=10)["count"].sum() == 500


def test_perfect_scores_give_perfect_reliability():
    y = np.array([0, 0, 1, 1])
    r = reliability(y.astype(float), y, bins=2).dropna()
    assert np.allclose(r["observed"].to_numpy(), r["predicted"].to_numpy())


def test_expensive_false_negatives_lower_the_threshold():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    t_cheap, _ = cost_optimal_threshold(s, y, c_fn=10.0, c_fp=10.0)
    t_dear, _ = cost_optimal_threshold(s, y, c_fn=500.0, c_fp=10.0)
    assert t_dear <= t_cheap


def test_cost_optimal_beats_naive_half_threshold():
    """The point of the exercise: 0.5 is not the right operating point when
    the two error types cost different amounts."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.5 + rng.normal(0.25, 0.25, 400), 0, 1)
    c_fn, c_fp = 200.0, 5.0
    t, cost = cost_optimal_threshold(s, y, c_fn, c_fp)
    naive = c_fn * ((y == 1) & (s < 0.5)).sum() + c_fp * ((y == 0) & (s >= 0.5)).sum()
    assert cost <= naive
