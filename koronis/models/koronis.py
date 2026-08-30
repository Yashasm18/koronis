import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.schema import RELATIONS
from ..graph.build import build_edges
from .layers import RelationalLayer
from .loss import expected_cost_loss


def node_features(events: pd.DataFrame) -> np.ndarray:
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
    return f.astype(np.float32)


class _Net(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, n_rel: int):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            [RelationalLayer(dims[i], dims[i + 1], n_rel) for i in range(n_layers)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edges):
        for layer in self.layers:
            x = layer(x, edges)
        return self.head(x).squeeze(-1)


class KoronisDetector:
    """Inductive graph detector for coordinated card-testing campaigns.

    Inductive by construction: there are no per-entity embedding tables, only
    per-relation weights. Entity ids never enter the model — they only decide
    which events share an edge. That is what lets it score devices and IPs it
    has never seen, which is the only regime that matters in production.
    """

    def __init__(self, hidden: int = 32, layers: int = 2,
                 window_s: float = 3600.0, seed: int = 0):
        self.hidden, self.n_layers = hidden, layers
        self.window_s, self.seed = window_s, seed
        self.net: _Net | None = None

    def _tensors(self, events: pd.DataFrame):
        x = torch.from_numpy(node_features(events))
        edges = build_edges(events, window_s=self.window_s)
        return x, [torch.from_numpy(edges[r]) for r in RELATIONS]

    def fit(self, events: pd.DataFrame, epochs: int = 40,
            c_fn: float = 73.0, c_fp: float = 40.0) -> None:
        torch.manual_seed(self.seed)
        x, edges = self._tensors(events)
        y = torch.from_numpy(events["label"].to_numpy(dtype=np.float32))
        self.net = _Net(x.shape[1], self.hidden, self.n_layers, len(RELATIONS))
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
        """Which entity type gave the campaign away, averaged over layers.

        A cheap stand-in for full subgraph explainability: it answers the first
        question an analyst asks, which is *what linked these attempts*.
        """
        if self.net is None:
            raise RuntimeError("fit() must be called before relation_attention()")
        with torch.no_grad():
            att = torch.stack([torch.softmax(layer.rel_att, dim=0)
                               for layer in self.net.layers]).mean(0)
        return {rel: float(att[i]) for i, rel in enumerate(RELATIONS)}
