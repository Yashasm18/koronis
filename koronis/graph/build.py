import numpy as np
import pandas as pd

from ..data.schema import RELATIONS


def build_edges(events: pd.DataFrame, window_s: float,
                max_degree: int = 32,
                relations: list[str] | None = None) -> dict[str, np.ndarray]:
    """Link events that share an entity value within a time window.

    Returns {relation: (2, E) array of (src, dst) row indices}. Edges point
    backwards in time: src is the earlier event, dst the later one, so a node
    only ever aggregates from its own past. That is what keeps the streaming
    evaluation honest — without it the model reads the future and every
    latency number is fiction.

    `max_degree` bounds the fan-in. A popular email domain would otherwise
    create a near-complete subgraph and exhaust memory.
    """
    ts = events["ts"].to_numpy()
    out: dict[str, np.ndarray] = {}

    for rel in (relations if relations is not None else RELATIONS):
        src_list, dst_list = [], []
        for idx in events.groupby(rel, sort=False).indices.values():
            idx = np.sort(idx)
            if len(idx) < 2:
                continue
            t = ts[idx]
            lo = np.searchsorted(t, t - window_s, side="left")
            for j in range(1, len(idx)):
                start = lo[j]
                if start >= j:
                    continue
                srcs = idx[start:j]
                if len(srcs) > max_degree:
                    srcs = srcs[-max_degree:]      # keep the most recent
                src_list.append(srcs)
                dst_list.append(np.full(len(srcs), idx[j]))

        if src_list:
            out[rel] = np.stack([np.concatenate(src_list),
                                 np.concatenate(dst_list)]).astype(np.int64)
        else:
            out[rel] = np.zeros((2, 0), dtype=np.int64)
    return out
