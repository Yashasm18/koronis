import torch


def expected_cost_loss(logits: torch.Tensor, labels: torch.Tensor,
                       c_fn: float, c_fp: float) -> torch.Tensor:
    """Expected rupee cost of the decision, differentiable in the score.

    A missed campaign attempt costs `c_fn`; a blocked legitimate attempt costs
    `c_fp`. Minimising this trains directly against the business objective
    instead of optimising cross-entropy and then repairing the mismatch with a
    threshold chosen afterwards.

    It also handles the class imbalance implicitly: when misses are dearer than
    false alarms, the asymmetry is in the objective rather than in a resampling
    step that distorts the score distribution.
    """
    p = torch.sigmoid(logits)
    return ((1.0 - p) * labels * c_fn + p * (1.0 - labels) * c_fp).mean()
