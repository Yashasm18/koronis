from koronis.eval.frontier import predicted_boundary_k, sweep


def test_boundary_matches_claim_one():
    """Claim 1: a threshold detector needs n/k > tau, so k = n/tau is the edge."""
    assert predicted_boundary_k(n=4000, tau=40) == 100.0
    assert predicted_boundary_k(n=400, tau=8) == 50.0


def test_velocity_fails_above_the_predicted_boundary():
    """The experiment Claim 1 stakes itself on.

    Spread must be applied to EVERY counted entity - a campaign that rotates
    devices and IPs but reuses two BIN ranges is caught by the BIN counter.
    """
    df = sweep(n_values=[400], k_values=[2, 400], fp_budget=0.01, seed=0)
    below = df[df["k"] == 2].iloc[0]
    above = df[df["k"] == 400].iloc[0]
    assert bool(below["velocity_detected"]) is True
    assert bool(above["velocity_detected"]) is False


def test_sweep_covers_the_grid_and_reports_the_boundary():
    df = sweep(n_values=[200, 400], k_values=[5, 50], fp_budget=0.01, seed=0)
    assert len(df) == 4
    for col in ("n", "k", "velocity_detected", "koronis_detected",
                "predicted_k_boundary", "velocity_blind_predicted"):
        assert col in df.columns


def test_prediction_agrees_with_observation():
    """The falsifiable bit: where theory says velocity is blind, it should be."""
    df = sweep(n_values=[400, 800], k_values=[2, 400], fp_budget=0.01, seed=0)
    agree = df["velocity_blind_predicted"] == ~df["velocity_detected"]
    assert agree.all(), df[~agree].to_string()
