import torch
import torch.nn as nn


class RelationalLayer(nn.Module):
    """Relational message passing, implemented directly.

    Per relation r:  m_v^r = sum_{u -> v} g(x_u, x_v) * W_r x_u  /  deg(v)
    then             h_v   = ReLU( W_self x_v + sum_r a_r * m_v^r )

    Three deliberate choices:

    `a_r` is a learned softmax over relations, so the model discovers which
    entity type carries the signal rather than being told. Reading it back
    after training says whether devices, IPs, BINs or email domains gave the
    campaign away.

    `g` is the heterophily gate. It scores each edge from the feature
    *difference* between endpoints, damping edges that join dissimilar nodes.
    Fraud rings deliberately attach to legitimate traffic as camouflage;
    without the gate those edges dilute the ring's signal. Vanilla GNNs assume
    homophily — connected nodes share labels — which is exactly the assumption
    an adversary is motivated to break.

    Aggregation is `index_add_` rather than a graph library, because the point
    of this project is to demonstrate the mechanism, not to import it.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int):
        super().__init__()
        self.n_relations = n_relations
        self.self_w = nn.Linear(in_dim, out_dim)
        self.rel_w = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)])
        self.rel_att = nn.Parameter(torch.zeros(n_relations))
        self.gate = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, edges: list[torch.Tensor]) -> torch.Tensor:
        if len(edges) != self.n_relations:
            raise ValueError(
                f"expected {self.n_relations} edge tensors, got {len(edges)}")

        out = self.self_w(x)
        att = torch.softmax(self.rel_att, dim=0)

        for r, ei in enumerate(edges):
            if ei.numel() == 0:
                continue
            src, dst = ei[0], ei[1]
            w = self.gate(torch.abs(x[src] - x[dst]))          # (E, 1)
            msg = self.rel_w[r](x)[src] * w
            agg = torch.zeros_like(out).index_add_(0, dst, msg)
            deg = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            deg = deg.index_add_(0, dst, w).clamp(min=1.0)
            out = out + att[r] * (agg / deg)

        return torch.relu(out)
