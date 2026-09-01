"""Does the graph survive being split across machines?

Throughput is not the hard part - per-event cost is already constant in stream
length, so more traffic is more processes. The hard part is that **partitioning
a graph deletes edges**. If one shard holds a device and another holds an IP
that co-occurs with it, that edge never forms, and the signal being detected is
silently thrown away. Sharding is therefore not an infrastructure detail; it is
a modelling decision.

Which suggests the routing key is not arbitrary. Events are routed by hashing
some field, and every event carrying the same value for that field lands
together - so **the relation you shard on is the one relation preserved
perfectly**, while every other relation survives only when both endpoints
happen to collide onto the same shard.

PREDICTION, from a result already in this repo. The per-relation ablation found
shared BIN ranges carry almost all of the signal: removing that relation costs a
fifth of recall, while removing device or email *improves* the model. So routing
by BIN should hold detection roughly flat as shards multiply, routing by device
should behave close to routing at random, and routing at random should decay
fastest. If BIN-routing is no better than random, the relation ablation does not
mean what it appears to mean.

Defense-only: in-memory partitioning of a synthetic stream.
"""
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..data.schema import EVENT_COLUMNS, RELATIONS
from ..eval.cost import COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR
from ..graph.build import build_edges

STRATEGIES = ("random", "bin_id", "device_id")


def assign_shards(events: pd.DataFrame, n_shards: int, strategy: str,
                  seed: int = 0) -> np.ndarray:
    """Route each event to a shard, the way a keyed stream partitioner would.

    `random` stands in for round-robin or event-id keying: nothing is
    co-located deliberately. The others key on an entity value, which is how a
    partitioned log keeps related records together.
    """
    if n_shards == 1:
        return np.zeros(len(events), dtype=int)
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, n_shards, len(events))
    if strategy not in RELATIONS:
        raise ValueError(f"unknown strategy {strategy!r}")
    # CRC32 rather than Python's hash(), which is randomised per process, and
    # rather than int.from_bytes(): interpreting the bytes as a little-endian
    # integer makes the value modulo a power of two depend only on the FIRST
    # character, so "b0" and "b17" and every other background BIN routed to one
    # shard while every campaign entity routed to another. The sweep then
    # reported entity routing as costless, because 94% of events were landing
    # on a single shard and nothing was being partitioned at all.
    return np.array([zlib.crc32(str(v).encode()) % n_shards
                     for v in events[strategy]], dtype=int)


def edges_preserved(events: pd.DataFrame, shard: np.ndarray,
                    window_s: float) -> dict[str, float]:
    """Share of each relation's edges that survive the partition."""
    whole = build_edges(events, window_s=window_s)
    kept = {rel: 0 for rel in RELATIONS}
    total = {rel: max(whole[rel].shape[1], 1) for rel in RELATIONS}
    for rel in RELATIONS:
        src, dst = whole[rel][0], whole[rel][1]
        kept[rel] = int((shard[src] == shard[dst]).sum())
    out = {rel: kept[rel] / total[rel] for rel in RELATIONS}
    out["all"] = sum(kept.values()) / max(sum(total.values()), 1)
    return out


def score_sharded(model, events: pd.DataFrame, shard: np.ndarray) -> np.ndarray:
    """Score each shard from its own graph only, as separate workers would."""
    out = np.zeros(len(events), dtype=float)
    for s in np.unique(shard):
        idx = np.flatnonzero(shard == s)
        part = events.iloc[idx][EVENT_COLUMNS].reset_index(drop=True)
        out[idx] = model.score_events(part)
    return out


def sweep(model, events: pd.DataFrame, thr: float, shard_counts: list[int],
          window_s: float, seed: int = 0) -> pd.DataFrame:
    """Detection quality against shard count, for each routing strategy.

    The model and the threshold are frozen; the only thing changing is how the
    stream is cut up. At one shard every strategy is the same undivided stream,
    which is the experiment's own control.
    """
    y = events["label"].to_numpy() == 1
    rows = []
    for n in shard_counts:
        for strat in STRATEGIES:
            shard = assign_shards(events, n, strat, seed=seed)
            sc = score_sharded(model, events, shard)
            fired = sc >= thr
            keep = edges_preserved(events, shard, window_s)
            tp = int((fired & y).sum()); fp = int((fired & ~y).sum())
            fn = int((~fired & y).sum())
            # PR-AUC hides which error a routing key trades for the other, and
            # the two are not worth the same. Priced with the project's own
            # declared constants: a missed attempt costs an authorisation fee,
            # a false alert costs checkout friction.
            cost = fn * COST_PER_ATTEMPT_INR + fp * COST_PER_FALSE_BLOCK_INR
            rows.append({
                "n_shards": n, "strategy": strat,
                "pr_auc": round(float(average_precision_score(y, sc)), 4),
                "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "recall": round(tp / int(y.sum()), 4),
                "false_positives": fp, "missed": fn,
                "decision_cost_inr": round(cost, 1),
                "edges_kept": round(keep["all"], 4),
                "bin_edges_kept": round(keep["bin_id"], 4),
                "device_edges_kept": round(keep["device_id"], 4),
                "largest_shard_share": round(
                    float(np.bincount(shard, minlength=n).max() / len(events)), 4),
            })
    return pd.DataFrame(rows)


# ── recovering what a partition deletes ─────────────────────────────────────
#
# Routing by BIN keeps every BIN edge and cuts most device and IP edges, which
# is why it holds precision and loses recall as shards multiply. The loss is
# not inherent to partitioning, only to insisting each event live in exactly
# one place: an edge is missing when its two endpoints are apart, so an event
# can be *copied* to the shard where its other relations would find company.
#
# PREDICTION, stated before the run. Only entities that actually recur can
# carry an edge, and background entity frequencies are heavy-tailed - most
# values appear once and can form nothing. Replicating only events whose
# device or IP is shared should therefore restore most of the lost edges while
# copying a minority of the traffic. Campaign entities are shared by
# construction, so campaign events should be replicated preferentially: the
# recall lost to BIN routing should come back at less than proportional cost.
# If duplication instead runs away, the traffic is not as heavy-tailed as the
# generator claims and the whole approach is uneconomic.


def assign_with_replication(events: pd.DataFrame, n_shards: int, primary: str,
                            replicate_on: tuple[str, ...] = ("device_id", "ip_id"),
                            min_degree: int = 2) -> list[list[int]]:
    """Place every event on its primary shard, and copy the ones that can link.

    Returns `n_shards` lists of row indices. An event appears on its primary
    shard always, and additionally on the shard owning a relation value it
    shares with at least `min_degree - 1` other events. `min_degree` is the
    knob: raising it copies less traffic and recovers fewer edges.

    Uses only observable traffic structure - how often a value occurs - and no
    labels, so it is a routing rule rather than a detector in disguise.
    """
    if n_shards == 1:
        return [list(range(len(events)))]
    buckets: list[set[int]] = [set() for _ in range(n_shards)]
    prim = assign_shards(events, n_shards, primary)
    for row, sh in enumerate(prim):
        buckets[sh].add(row)
    for rel in replicate_on:
        counts = events[rel].value_counts()
        shard_of = assign_shards(events, n_shards, rel)
        vals = events[rel].to_numpy()
        for row, sh in enumerate(shard_of):
            if counts.get(vals[row], 0) >= min_degree:
                buckets[sh].add(row)
    return [sorted(b) for b in buckets]


def score_replicated(model, events: pd.DataFrame,
                     buckets: list[list[int]]) -> tuple[np.ndarray, float]:
    """Score each shard from its own copy of the graph; keep the highest score.

    An event seen by several shards gets several opinions. The maximum is the
    conservative choice for a detector: a shard that can see a coordinated
    neighbourhood should not be overruled by one that cannot.

    Also returns the duplication factor - total placements over events - which
    is what the fidelity costs in compute.
    """
    best = np.zeros(len(events), dtype=float)
    placements = 0
    for rows in buckets:
        if not rows:
            continue
        placements += len(rows)
        part = events.iloc[rows][EVENT_COLUMNS].reset_index(drop=True)
        sc = model.score_events(part)
        idx = np.asarray(rows)
        best[idx] = np.maximum(best[idx], sc)
    return best, placements / max(len(events), 1)


def sweep_replication(model, events: pd.DataFrame, thr: float,
                      shard_counts: list[int], window_s: float,
                      min_degrees: tuple[int, ...] = (2, 4)) -> pd.DataFrame:
    """BIN routing, with and without replication, against shard count."""
    y = events["label"].to_numpy() == 1
    rows = []
    for n in shard_counts:
        variants = [("bin_id, no replication", None)]
        variants += [(f"bin_id + replicate d>={d}", d) for d in min_degrees]
        for label, d in variants:
            if d is None:
                shard = assign_shards(events, n, "bin_id")
                sc = score_sharded(model, events, shard)
                dup = 1.0
                keep = edges_preserved(events, shard, window_s)["all"]
            else:
                buckets = assign_with_replication(events, n, "bin_id", min_degree=d)
                sc, dup = score_replicated(model, events, buckets)
                # an edge survives if BOTH endpoints share at least one shard
                where = [set() for _ in range(len(events))]
                for si, b in enumerate(buckets):
                    for r in b:
                        where[r].add(si)
                whole = build_edges(events, window_s=window_s)
                kept = tot = 0
                for rel in RELATIONS:
                    src, dst = whole[rel][0], whole[rel][1]
                    tot += len(src)
                    kept += sum(1 for a, b2 in zip(src, dst) if where[a] & where[b2])
                keep = kept / max(tot, 1)
            fired = sc >= thr
            tp = int((fired & y).sum()); fp = int((fired & ~y).sum())
            fn = int((~fired & y).sum())
            rows.append({
                "n_shards": n, "routing": label,
                "pr_auc": round(float(average_precision_score(y, sc)), 4),
                "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "recall": round(tp / max(int(y.sum()), 1), 4),
                "false_positives": fp, "missed": fn,
                "decision_cost_inr": round(
                    fn * COST_PER_ATTEMPT_INR + fp * COST_PER_FALSE_BLOCK_INR, 1),
                "edges_kept": round(keep, 4),
                "duplication": round(dup, 3),
            })
    return pd.DataFrame(rows)
