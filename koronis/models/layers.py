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
    The reasoning: fraud rings attach to legitimate traffic as camouflage, so
    those edges dilute the ring's signal, and vanilla GNNs assume homophily —
    connected nodes share labels — which is exactly the assumption an
    adversary is motivated to break.

    That reasoning did not survive being tested. `koronis.cli architecture`
    removes the gate and refits under the same protocol: PR-AUC improves and
    false positives fall from 24 to 17 at full camouflage, and the gap widens
    as camouflage rises, which is the inverse of what the rationale predicts.
    The gate is left in place rather than deleted, because selecting an
    architecture on test results is the leakage this project refuses
    elsewhere. On this data it is the second relational layer, not this gate,
    that carries camouflaged coordination.

    Aggregation is `index_add_` rather than a graph library, because the point
    of this project is to demonstrate the mechanism, not to import it.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int,
                 use_gate: bool = True, use_rel_attention: bool = True):
        super().__init__()
        self.n_relations = n_relations
        # Ablation switches for the two architectural claims. With `use_gate`
        # off every edge carries weight 1, which is ordinary mean aggregation.
        # With `use_rel_attention` off the relations are mixed uniformly at
        # 1/R instead of by a learned softmax. Both default on; they exist so
        # the claims can be measured rather than asserted.
        self.use_gate = use_gate
        self.use_rel_attention = use_rel_attention
        self.self_w = nn.Linear(in_dim, out_dim)
        self.rel_w = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)])
        self.rel_att = nn.Parameter(torch.zeros(n_relations))
        self.gate = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

    def _attention(self) -> torch.Tensor:
        """Relation mixing weights: learned softmax, or uniform when ablated."""
        if self.use_rel_attention:
            return torch.softmax(self.rel_att, dim=0)
        return torch.full_like(self.rel_att, 1.0 / self.n_relations)

    def forward(self, x: torch.Tensor, edges: list[torch.Tensor]) -> torch.Tensor:
        if len(edges) != self.n_relations:
            raise ValueError(
                f"expected {self.n_relations} edge tensors, got {len(edges)}")

        out = self.self_w(x)
        att = self._attention()

        for r, ei in enumerate(edges):
            if ei.numel() == 0:
                continue
            src, dst = ei[0], ei[1]
            w = (self.gate(torch.abs(x[src] - x[dst])) if self.use_gate
                 else torch.ones(src.numel(), 1, device=x.device, dtype=x.dtype))
            msg = self.rel_w[r](x)[src] * w
            agg = torch.zeros_like(out).index_add_(0, dst, msg)
            deg = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            deg = deg.index_add_(0, dst, w).clamp(min=1.0)
            out = out + att[r] * (agg / deg)

        return torch.relu(out)

    def forward_single(self, x_self: torch.Tensor,
                       neighbours: list[torch.Tensor]) -> torch.Tensor:
        """Compute one destination node's output from its in-neighbours.

        `x_self` is (in_dim,); `neighbours[r]` is (E_r, in_dim), the features of
        the nodes with an edge into this one under relation r.

        This is the streaming counterpart of `forward`, and it is arithmetically
        identical for a single destination: `forward` accumulates per relation
        with `index_add_`, which for one destination reduces to a plain sum over
        that relation's in-edges. Keeping the two in step is what lets the
        replay reproduce batch scores exactly rather than approximately, and a
        parity test holds them to it.
        """
        if len(neighbours) != self.n_relations:
            raise ValueError(
                f"expected {self.n_relations} neighbour blocks, got {len(neighbours)}")

        out = self.self_w(x_self)
        att = self._attention()

        for r, nb in enumerate(neighbours):
            if nb is None or nb.numel() == 0:
                continue
            w = (self.gate(torch.abs(nb - x_self.unsqueeze(0))) if self.use_gate
                 else torch.ones(nb.size(0), 1, device=nb.device, dtype=nb.dtype))
            msg = (self.rel_w[r](nb) * w).sum(dim=0)
            deg = w.sum().clamp(min=1.0)
            out = out + att[r] * (msg / deg)

        return torch.relu(out)
