import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.schema import MODEL_RELATIONS
from ..graph.build import build_edges
from .layers import RelationalLayer
from .loss import expected_cost_loss


# The authorisation outcome is observed only AFTER an attempt is submitted, so
# it can never prevent the attempt it describes. Its value is in what it says
# about the attempts that follow. Kept as an explicit switch so its
# contribution can be measured rather than assumed.
FEATURE_NAMES = ["log_amount", "is_micro", "approved", "hour", "email_free", "bias"]


def node_features(events: pd.DataFrame, use_approved: bool = True) -> np.ndarray:
    """Per-event features only.

    Deliberately identical in spirit to the GBDT baseline's features: all
    coordination signal must arrive through the graph, otherwise the ablation
    proves nothing about graph structure.
    """
    amt = events["amount"].to_numpy(dtype=np.float64)
    f = np.stack([
        np.log1p(amt),
        (amt < 25.0).astype(np.float64),
        events["approved"].to_numpy().astype(np.float64),
        ((events["ts"].to_numpy() // 3600) % 24) / 24.0,
        events["email_domain"].isin(["gmail.com", "outlook.com"])
            .to_numpy().astype(np.float64),
        np.ones(len(events)),
    ], axis=1)
    if not use_approved:
        f[:, 2] = 0.0          # zero the column, keeping the input width fixed
    return f.astype(np.float32)


class _Net(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, n_rel: int,
                 use_gate: bool = True, use_rel_attention: bool = True):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            [RelationalLayer(dims[i], dims[i + 1], n_rel,
                             use_gate=use_gate, use_rel_attention=use_rel_attention)
             for i in range(n_layers)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edges):
        for layer in self.layers:
            x = layer(x, edges)
        return self.head(x).squeeze(-1)


class KoronisDetector:
    """Inductive graph detector for coordinated card-testing campaigns.

    The default architecture is the one selected on the calibration split:
    three relations and no heterophily gate (`koronis.cli select`), at 32
    hidden units and 3 layers (`koronis.cli capacity`). Every one of those was
    arrived at by measurement rather than preference - see docs/evaluation.md.

    Inductive by construction: there are no per-entity embedding tables, only
    per-relation weights. Entity ids never enter the model — they only decide
    which events share an edge. That is what lets it score devices and IPs it
    has never seen, which is the only regime that matters in production.
    """

    def __init__(self, hidden: int = 32, layers: int = 3,
                 window_s: float = 3600.0, seed: int = 0,
                 use_approved: bool = True, use_edges: bool = True,
                 use_gate: bool = False, use_rel_attention: bool = True,
                 relations: list[str] | None = None):
        self.hidden, self.n_layers = hidden, layers
        self.window_s, self.seed = window_s, seed
        # Ablation switches. `use_edges=False` strips every graph edge, leaving
        # the self-transformation - which isolates how much of the result comes
        # from per-event features rather than coordination. `use_approved=False`
        # zeroes the authorisation outcome, isolating the reverse.
        self.use_approved, self.use_edges = use_approved, use_edges
        # Architecture switches, so the heterophily gate and the learned
        # relation attention can be ablated the same way the data sources are.
        self.use_gate, self.use_rel_attention = use_gate, use_rel_attention
        # Which relations this model consumes, which is narrower than what the
        # data contains. The default is the architecture `koronis.cli select`
        # chose on the calibration split; pass an explicit list to ablate.
        self.relations = list(relations if relations is not None else MODEL_RELATIONS)
        self.net: _Net | None = None

    def _tensors(self, events: pd.DataFrame):
        x = torch.from_numpy(node_features(events, self.use_approved))
        if not self.use_edges:
            empty = torch.zeros((2, 0), dtype=torch.int64)
            return x, [empty for _ in self.relations]
        edges = build_edges(events, window_s=self.window_s,
                            relations=self.relations)
        return x, [torch.from_numpy(edges[r]) for r in self.relations]

    def fit(self, events: pd.DataFrame, epochs: int = 40,
            c_fn: float = 73.0, c_fp: float = 40.0) -> None:
        torch.manual_seed(self.seed)
        x, edges = self._tensors(events)
        y = torch.from_numpy(events["label"].to_numpy(dtype=np.float32))
        self.net = _Net(x.shape[1], self.hidden, self.n_layers, len(self.relations),
                        use_gate=self.use_gate,
                        use_rel_attention=self.use_rel_attention)
        opt = torch.optim.Adam(self.net.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            loss = expected_cost_loss(self.net(x, edges), y, c_fn, c_fp)
            loss.backward()
            opt.step()

    def score_events(self, events: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("fit() must be called before score_events()")
        self.net.eval()
        with torch.no_grad():
            x, edges = self._tensors(events)
            return torch.sigmoid(self.net(x, edges)).numpy()

    def relation_attention(self) -> dict[str, float]:
        """Learned attention per relation, averaged over layers.

        Read this as *where the model looked*, not as what each relation was
        worth. A per-relation ablation (`python -m koronis.cli relations`)
        disagrees with these weights: they are near-uniform with email_domain
        highest, while removing email_domain slightly IMPROVES the model and
        removing bin_id costs a fifth of recall. Attention is not attribution,
        and only deletion measures contribution.
        """
        if self.net is None:
            raise RuntimeError("fit() must be called before relation_attention()")
        with torch.no_grad():
            att = torch.stack([torch.softmax(layer.rel_att, dim=0)
                               for layer in self.net.layers]).mean(0)
        return {rel: float(att[i]) for i, rel in enumerate(self.relations)}
