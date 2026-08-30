# Ringfence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a detector that finds distributed card-testing campaigns in a payment stream far earlier than per-entity velocity rules or per-transaction models, and characterize the region where each approach provably can or cannot work.

**Architecture:** Real transaction data supplies background traffic; labeled card-testing campaigns are injected into it with controlled spread parameters `(n, k)`. Every detector implements one interface — events in, per-event scores out — so a shared replay harness converts scores into campaign-level detection latency and makes the ablation exactly fair. The learned detector builds an entity-sharing graph over a sliding window and runs relational message passing written from scratch, trained on an expected-rupee-cost objective rather than cross-entropy.

**Tech Stack:** Python 3.11+, PyTorch (autograd only — no DGL/PyG), pandas, numpy, scikit-learn, LightGBM, pytest, matplotlib.

**Spec:** `docs/superpowers/specs/2026-08-30-ringfence-design.md`

## Global Constraints

- **Defense-only.** No network calls anywhere in the codebase. The campaign injector operates on in-memory dataframes only, uses no real BIN ranges, and reproduces only characteristics documented publicly by Visa's anti-enumeration guidance. This gets its own README section.
- **No graph libraries.** Message passing is implemented with `torch.index_add_`. Importing `dgl`, `torch_geometric`, or `stellargraph` defeats the purpose of the project.
- **Inductive only.** No per-entity embedding tables. The model must score `device_id` / `ip_id` values absent from training. Any lookup keyed on a raw entity id is a bug.
- **Every detector implements `score_events(events: pd.DataFrame) -> np.ndarray`** returning one float per row, higher = more suspicious. This is the contract the whole evaluation rests on.
- **Currency is INR**; all cost figures are declared in `ringfence/eval/cost.py` as named constants with a source comment, never inlined.
- **Seeds are explicit.** Every function that samples takes a `seed: int`. No global `np.random` calls.

## Timeline

| Day | Tasks | Checkpoint |
|---|---|---|
| Sun 30 Aug | 1–2 | Labeled dataset exists |
| Mon 31 Aug | 3–5 | Baselines measured; frontier argument written |
| Tue 1 Sep | 6–7 | Learned model trains |
| Wed 2 Sep | 8–10 | **Thesis proven**; ablation + frontier charted |
| Thu 3 Sep | 11 | README, console, video, submit |

Fri 4 – Sat 5 Sep are buffer. If Task 7 is not done by Tuesday night, cut Task 10 and ship the ablation.

---

## File Structure

| File | Responsibility |
|---|---|
| `ringfence/data/schema.py` | Canonical event column contract, `CampaignSpec` |
| `ringfence/data/background.py` | Real transactions → canonical background events |
| `ringfence/data/campaigns.py` | Inject labeled campaigns with controlled `(n, k)` |
| `ringfence/graph/build.py` | Events → entity-sharing edge lists per relation |
| `ringfence/models/velocity.py` | Baseline: per-entity thresholds |
| `ringfence/models/gbdt.py` | Baseline: per-transaction LightGBM |
| `ringfence/models/layers.py` | From-scratch relational message passing |
| `ringfence/models/ringfence.py` | Inductive model, feature builder, training loop |
| `ringfence/models/loss.py` | Expected-rupee-cost objective |
| `ringfence/eval/cost.py` | Cost constants and exposure model |
| `ringfence/eval/latency.py` | Replay harness: scores → detection latency |
| `ringfence/eval/calibration.py` | Reliability diagram, cost-optimal threshold |
| `ringfence/eval/frontier.py` | `(n, k)` sweep and predicted boundary |
| `ringfence/cli.py` | Experiment entry points |

---

### Task 1: Event schema and background loader

**Files:**
- Create: `ringfence/data/schema.py`
- Create: `ringfence/data/background.py`
- Test: `tests/test_background.py`

**Interfaces:**
- Produces: `EVENT_COLUMNS: list[str]`, `CampaignSpec` dataclass, `load_background(path: Path | None, n_rows: int, seed: int) -> pd.DataFrame`

**Background data.** Primary source is the IEEE-CIS Fraud Detection dataset, which carries real device, card, email-domain and timestamp fields — real entity-reuse structure is exactly what we need for realistic false positives. Download once:

```bash
pip install kaggle
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip -o data/raw/ieee-fraud-detection.zip -d data/raw
```

This needs a Kaggle account and accepting the competition rules on the website first. If that is blocked, `load_background(path=None, ...)` produces a bootstrap sample with the same column contract so the rest of the pipeline runs unchanged — but the real file is strongly preferred, because synthetic negatives cannot produce honest false-positive costs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_background.py
import pandas as pd
from ringfence.data.background import load_background
from ringfence.data.schema import EVENT_COLUMNS


def test_background_has_contract_columns_and_no_labels():
    ev = load_background(path=None, n_rows=2000, seed=0)
    assert list(ev.columns) == EVENT_COLUMNS
    assert len(ev) == 2000
    assert (ev["label"] == 0).all()
    assert ev["campaign_id"].isna().all()
    assert ev["ts"].is_monotonic_increasing


def test_background_is_deterministic_under_seed():
    a = load_background(path=None, n_rows=500, seed=7)
    b = load_background(path=None, n_rows=500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_background_entities_are_reused_not_unique():
    ev = load_background(path=None, n_rows=5000, seed=1)
    # real traffic shares devices; unique-per-row would make the graph trivial
    assert ev["device_id"].nunique() < len(ev) * 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_background.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.data.background'`

- [ ] **Step 3: Write the schema**

```python
# ringfence/data/schema.py
from dataclasses import dataclass

EVENT_COLUMNS = [
    "event_id", "ts", "amount", "card_id", "bin_id",
    "device_id", "ip_id", "email_domain", "approved",
    "label", "campaign_id",
]

# Relations are the entity types two events can share. Order is fixed because
# model weights are indexed by relation position.
RELATIONS = ["device_id", "ip_id", "bin_id", "email_domain"]


@dataclass(frozen=True)
class CampaignSpec:
    """One injected card-testing campaign.

    n_attempts: total attempts (the `n` of the detectability frontier)
    k_devices:  distinct device fingerprints the attacker owns (the `k`)
    k_ips:      distinct IPs; independent of k_devices
    duration_s: wall-clock span of the campaign
    start_ts:   when it begins, in the background stream's time base
    n_bins:     how many BIN ranges are being enumerated
    """
    n_attempts: int
    k_devices: int
    k_ips: int
    duration_s: float
    start_ts: float
    n_bins: int = 2
```

- [ ] **Step 4: Write the background loader**

```python
# ringfence/data/background.py
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import EVENT_COLUMNS


def load_background(path: Path | None, n_rows: int, seed: int) -> pd.DataFrame:
    """Return `n_rows` canonical background events, all label=0.

    When `path` points at IEEE-CIS train_transaction.csv the entity columns are
    derived from real fields, preserving real reuse structure. Otherwise a
    bootstrap sampler produces the same contract with plausible reuse.
    """
    rng = np.random.default_rng(seed)
    if path is not None and Path(path).exists():
        df = _from_ieee(Path(path), n_rows, rng)
    else:
        df = _bootstrap(n_rows, rng)

    df["label"] = 0
    df["campaign_id"] = pd.Series([None] * len(df), dtype="object")
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["event_id"] = [f"bg_{i}" for i in range(len(df))]
    return df[EVENT_COLUMNS]


def _from_ieee(path: Path, n_rows: int, rng) -> pd.DataFrame:
    cols = ["TransactionID", "TransactionDT", "TransactionAmt", "card1",
            "card2", "DeviceInfo", "P_emaildomain", "addr1", "isFraud"]
    raw = pd.read_csv(path, usecols=lambda c: c in cols, nrows=n_rows * 3)
    raw = raw.head(n_rows).copy()
    return pd.DataFrame({
        "ts": raw["TransactionDT"].astype(float).to_numpy(),
        "amount": raw["TransactionAmt"].astype(float).to_numpy(),
        "card_id": raw["card1"].astype("string").fillna("unk").to_numpy(),
        "bin_id": raw["card2"].astype("string").fillna("unk").to_numpy(),
        "device_id": raw["DeviceInfo"].astype("string").fillna("unk").to_numpy(),
        "ip_id": raw["addr1"].astype("string").fillna("unk").to_numpy(),
        "email_domain": raw["P_emaildomain"].astype("string").fillna("unk").to_numpy(),
        # IEEE-CIS has no auth outcome; approval is modelled from its fraud flag
        "approved": (raw["isFraud"].to_numpy() == 0),
    })


def _bootstrap(n_rows: int, rng) -> pd.DataFrame:
    # Zipfian entity reuse: a few heavy sharers (offices, CGNAT), a long tail.
    def zipf_ids(prefix: str, pool: int) -> np.ndarray:
        idx = np.minimum(rng.zipf(1.6, n_rows), pool)
        return np.array([f"{prefix}{i}" for i in idx])

    ts = np.sort(rng.uniform(0, 30 * 86400, n_rows))
    return pd.DataFrame({
        "ts": ts,
        "amount": np.round(rng.lognormal(6.2, 1.1, n_rows), 2),
        "card_id": zipf_ids("c", n_rows),
        "bin_id": zipf_ids("b", 400),
        "device_id": zipf_ids("d", int(n_rows * 0.55)),
        "ip_id": zipf_ids("i", int(n_rows * 0.35)),
        "email_domain": rng.choice(
            ["gmail.com", "yahoo.com", "outlook.com", "rediff.com", "proton.me"],
            n_rows, p=[0.55, 0.15, 0.15, 0.10, 0.05]),
        "approved": rng.random(n_rows) > 0.08,
    })
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_background.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add ringfence/data/schema.py ringfence/data/background.py tests/test_background.py
git commit -m "feat: canonical event schema and background traffic loader"
```

---

### Task 2: Campaign injector

**Files:**
- Create: `ringfence/data/campaigns.py`
- Test: `tests/test_campaigns.py`

**Interfaces:**
- Consumes: `CampaignSpec`, `EVENT_COLUMNS`, `load_background`
- Produces: `inject(background: pd.DataFrame, specs: list[CampaignSpec], seed: int) -> pd.DataFrame` — returns background plus campaign rows, sorted by `ts`, with `label=1` and a non-null `campaign_id` on campaign rows.

The `(n, k)` parameters are the whole point: `k_devices` controls how far the attacker spreads, which is the axis the detectability frontier is measured on.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_campaigns.py
import numpy as np
from ringfence.data.background import load_background
from ringfence.data.campaigns import inject
from ringfence.data.schema import CampaignSpec, EVENT_COLUMNS


def _bg():
    return load_background(path=None, n_rows=4000, seed=0)


def test_injects_exact_attempt_count_and_spread():
    bg = _bg()
    spec = CampaignSpec(n_attempts=200, k_devices=25, k_ips=10,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[100]))
    ev = inject(bg, [spec], seed=3)
    camp = ev[ev["label"] == 1]
    assert len(camp) == 200
    assert camp["device_id"].nunique() == 25
    assert camp["ip_id"].nunique() == 10
    assert camp["campaign_id"].nunique() == 1


def test_campaign_rows_look_like_card_testing():
    bg = _bg()
    spec = CampaignSpec(n_attempts=300, k_devices=20, k_ips=8,
                        duration_s=1800.0, start_ts=float(bg["ts"].iloc[50]))
    camp = inject(bg, [spec], seed=1).query("label == 1")
    assert camp["amount"].max() <= 25.0            # micro-amounts
    assert camp["card_id"].nunique() > 250          # many cards, few devices
    assert camp["approved"].mean() < 0.15           # mostly declines


def test_output_is_sorted_and_contract_preserved():
    bg = _bg()
    spec = CampaignSpec(n_attempts=50, k_devices=5, k_ips=3,
                        duration_s=600.0, start_ts=float(bg["ts"].iloc[10]))
    ev = inject(bg, [spec], seed=0)
    assert list(ev.columns) == EVENT_COLUMNS
    assert ev["ts"].is_monotonic_increasing
    assert len(ev) == len(bg) + 50
    assert ev["event_id"].is_unique
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_campaigns.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.data.campaigns'`

- [ ] **Step 3: Write the injector**

```python
# ringfence/data/campaigns.py
import numpy as np
import pandas as pd
from .schema import CampaignSpec, EVENT_COLUMNS


def inject(background: pd.DataFrame, specs: list[CampaignSpec],
           seed: int) -> pd.DataFrame:
    """Add labeled card-testing campaigns to a background stream."""
    rng = np.random.default_rng(seed)
    frames = [background]
    for c, spec in enumerate(specs):
        frames.append(_one_campaign(spec, f"camp_{c}", rng))
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.sort_values("ts", kind="mergesort").reset_index(drop=True)
    ev["event_id"] = [f"e_{i}" for i in range(len(ev))]
    return ev[EVENT_COLUMNS]


def _one_campaign(spec: CampaignSpec, cid: str, rng) -> pd.DataFrame:
    n = spec.n_attempts
    devices = np.array([f"{cid}_d{i}" for i in range(spec.k_devices)])
    ips = np.array([f"{cid}_i{i}" for i in range(spec.k_ips)])
    bins = np.array([f"{cid}_b{i}" for i in range(spec.n_bins)])

    # Round-robin assignment guarantees exactly k distinct entities appear,
    # which is what makes the (n, k) frontier sweep well defined.
    dev = devices[np.arange(n) % spec.k_devices]
    ip = ips[np.arange(n) % spec.k_ips]

    return pd.DataFrame({
        "event_id": [f"{cid}_{i}" for i in range(n)],
        "ts": np.sort(rng.uniform(spec.start_ts, spec.start_ts + spec.duration_s, n)),
        "amount": np.round(rng.uniform(1.0, 20.0, n), 2),
        "card_id": [f"{cid}_c{i}" for i in range(n)],   # a fresh card each attempt
        "bin_id": rng.choice(bins, n),
        "device_id": dev,
        "ip_id": ip,
        "email_domain": rng.choice(["gmail.com", "outlook.com"], n),
        "approved": rng.random(n) < 0.04,               # ~96% decline
        "label": 1,
        "campaign_id": cid,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_campaigns.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ringfence/data/campaigns.py tests/test_campaigns.py
git commit -m "feat: labeled card-testing campaign injector with (n,k) spread control"
```

---

### Task 3: Velocity baseline

**Files:**
- Create: `ringfence/models/velocity.py`
- Test: `tests/test_velocity.py`

**Interfaces:**
- Produces: `VelocityDetector(tau: int, window_s: float, entity: str = "ip_id")` with `score_events(events: pd.DataFrame) -> np.ndarray`

This is the detector Claim 1 of the spec proves blind. Implementing it faithfully — not as a strawman — is what makes the ablation honest.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_velocity.py
import numpy as np
import pandas as pd
from ringfence.data.background import load_background
from ringfence.data.campaigns import inject
from ringfence.data.schema import CampaignSpec
from ringfence.models.velocity import VelocityDetector


def test_catches_concentrated_campaign():
    bg = load_background(path=None, n_rows=3000, seed=0)
    spec = CampaignSpec(n_attempts=300, k_devices=2, k_ips=2,
                        duration_s=1800.0, start_ts=float(bg["ts"].iloc[100]))
    ev = inject(bg, [spec], seed=0)
    s = VelocityDetector(tau=40, window_s=3600.0).score_events(ev)
    assert s[ev["label"].to_numpy() == 1].max() > 0


def test_blind_to_spread_campaign():
    """Claim 1: with k >= n/tau every entity stays under threshold."""
    bg = load_background(path=None, n_rows=3000, seed=0)
    spec = CampaignSpec(n_attempts=300, k_devices=200, k_ips=200,
                        duration_s=1800.0, start_ts=float(bg["ts"].iloc[100]))
    ev = inject(bg, [spec], seed=0)
    s = VelocityDetector(tau=40, window_s=3600.0).score_events(ev)
    assert s[ev["label"].to_numpy() == 1].max() == 0


def test_returns_one_score_per_row():
    ev = load_background(path=None, n_rows=500, seed=1)
    assert VelocityDetector(tau=10, window_s=600.0).score_events(ev).shape == (500,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_velocity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.models.velocity'`

- [ ] **Step 3: Write the detector**

```python
# ringfence/models/velocity.py
import numpy as np
import pandas as pd


class VelocityDetector:
    """Fires when one entity exceeds `tau` attempts inside `window_s`.

    Score is the entity's rolling count above threshold, so it is monotone and
    usable in a PR curve rather than only as a binary rule.
    """

    def __init__(self, tau: int, window_s: float, entity: str = "ip_id"):
        self.tau = tau
        self.window_s = window_s
        self.entity = entity

    def score_events(self, events: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(events), dtype=float)
        ts = events["ts"].to_numpy()
        for _, idx in events.groupby(self.entity, sort=False).indices.items():
            idx = np.sort(idx)
            t = ts[idx]
            # count of same-entity events in the preceding window
            start = np.searchsorted(t, t - self.window_s, side="left")
            counts = np.arange(len(t)) - start + 1
            out[idx] = np.maximum(counts - self.tau, 0)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_velocity.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ringfence/models/velocity.py tests/test_velocity.py
git commit -m "feat: per-entity velocity baseline"
```

---

### Task 4: Per-transaction GBDT baseline

**Files:**
- Create: `ringfence/models/gbdt.py`
- Test: `tests/test_gbdt.py`

**Interfaces:**
- Produces: `GBDTDetector(seed: int)` with `fit(events) -> None` and `score_events(events) -> np.ndarray`; module function `transaction_features(events: pd.DataFrame) -> pd.DataFrame`

This baseline must be genuinely strong — a weak one invalidates the whole comparison. Give it every per-transaction feature a real system would have.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gbdt.py
import numpy as np
from ringfence.data.background import load_background
from ringfence.data.campaigns import inject
from ringfence.data.schema import CampaignSpec
from ringfence.models.gbdt import GBDTDetector, transaction_features


def _data(seed=0):
    bg = load_background(path=None, n_rows=6000, seed=seed)
    specs = [CampaignSpec(n_attempts=250, k_devices=8, k_ips=4,
                          duration_s=3600.0, start_ts=float(bg["ts"].iloc[500]))]
    return inject(bg, specs, seed=seed)


def test_features_have_no_leakage_columns():
    f = transaction_features(_data())
    for banned in ("label", "campaign_id", "event_id"):
        assert banned not in f.columns


def test_learns_something_on_concentrated_campaign():
    tr, te = _data(0), _data(1)
    m = GBDTDetector(seed=0)
    m.fit(tr)
    s = m.score_events(te)
    y = te["label"].to_numpy()
    assert s[y == 1].mean() > s[y == 0].mean()


def test_returns_one_score_per_row():
    ev = _data()
    m = GBDTDetector(seed=0)
    m.fit(ev)
    assert m.score_events(ev).shape == (len(ev),)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gbdt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.models.gbdt'`

- [ ] **Step 3: Write the baseline**

```python
# ringfence/models/gbdt.py
import lightgbm as lgb
import numpy as np
import pandas as pd

_FEATURES = ["amount", "log_amount", "hour", "approved",
             "amount_is_micro", "email_is_free"]


def transaction_features(events: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction features only — deliberately no cross-event aggregation.

    That restriction IS the baseline's thesis: scoring attempts in isolation.
    """
    amt = events["amount"].to_numpy()
    return pd.DataFrame({
        "amount": amt,
        "log_amount": np.log1p(amt),
        "hour": (events["ts"].to_numpy() // 3600) % 24,
        "approved": events["approved"].to_numpy().astype(float),
        "amount_is_micro": (amt < 25.0).astype(float),
        "email_is_free": events["email_domain"]
            .isin(["gmail.com", "yahoo.com", "outlook.com"]).to_numpy().astype(float),
    })[_FEATURES]


class GBDTDetector:
    def __init__(self, seed: int):
        self.seed = seed
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, events: pd.DataFrame) -> None:
        self.model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=self.seed, verbose=-1)
        self.model.fit(transaction_features(events), events["label"].to_numpy())

    def score_events(self, events: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() must be called before score_events()")
        return self.model.predict_proba(transaction_features(events))[:, 1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pip install lightgbm && pytest tests/test_gbdt.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ringfence/models/gbdt.py tests/test_gbdt.py
git commit -m "feat: per-transaction GBDT baseline"
```

---

### Task 5: Graph builder

**Files:**
- Create: `ringfence/graph/build.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `RELATIONS`
- Produces: `build_edges(events: pd.DataFrame, window_s: float, max_degree: int = 32) -> dict[str, np.ndarray]` — maps each relation name to a `(2, E)` int array of `(src, dst)` row indices. Edges only join events within `window_s` of each other.

`max_degree` matters: a popular email domain would otherwise create a near-complete subgraph and blow up memory.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py
import numpy as np
import pandas as pd
from ringfence.data.background import load_background
from ringfence.graph.build import build_edges
from ringfence.data.schema import RELATIONS


def test_edges_only_join_same_entity_within_window():
    ev = load_background(path=None, n_rows=800, seed=0)
    edges = build_edges(ev, window_s=3600.0)
    assert set(edges.keys()) == set(RELATIONS)
    for rel, ei in edges.items():
        assert ei.shape[0] == 2
        src, dst = ei
        assert (ev[rel].to_numpy()[src] == ev[rel].to_numpy()[dst]).all()
        dt = np.abs(ev["ts"].to_numpy()[src] - ev["ts"].to_numpy()[dst])
        assert (dt <= 3600.0).all()


def test_no_self_loops():
    ev = load_background(path=None, n_rows=400, seed=1)
    for ei in build_edges(ev, window_s=3600.0).values():
        assert (ei[0] != ei[1]).all()


def test_degree_is_capped():
    ev = load_background(path=None, n_rows=1500, seed=2)
    edges = build_edges(ev, window_s=10**9, max_degree=4)
    for ei in edges.values():
        if ei.shape[1]:
            _, counts = np.unique(ei[1], return_counts=True)
            assert counts.max() <= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.graph.build'`

- [ ] **Step 3: Write the builder**

```python
# ringfence/graph/build.py
import numpy as np
import pandas as pd
from ..data.schema import RELATIONS


def build_edges(events: pd.DataFrame, window_s: float,
                max_degree: int = 32) -> dict[str, np.ndarray]:
    """Link events that share an entity value within a time window.

    Returns {relation: (2, E) array of (src, dst) row indices}. Edges point
    backwards in time: src is the earlier event, dst the later one, so a node
    only ever aggregates from its own past. That is what keeps the streaming
    evaluation honest — no lookahead.
    """
    ts = events["ts"].to_numpy()
    out: dict[str, np.ndarray] = {}

    for rel in RELATIONS:
        src_list, dst_list = [], []
        for _, idx in events.groupby(rel, sort=False).indices.items():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ringfence/graph/build.py tests/test_graph.py
git commit -m "feat: windowed entity-sharing graph builder with degree cap"
```

---

### Task 6: From-scratch relational message passing

**Files:**
- Create: `ringfence/models/layers.py`
- Test: `tests/test_layers.py`

**Interfaces:**
- Produces: `RelationalLayer(in_dim: int, out_dim: int, n_relations: int)`, a `torch.nn.Module` whose `forward(x: Tensor, edges: list[Tensor]) -> Tensor` takes node features `(N, in_dim)` and a list of `(2, E_r)` edge tensors, one per relation.

**This is the centerpiece.** No `dgl`, no `torch_geometric` — aggregation is `index_add_`. The heterophily gate is what handles camouflage: an edge to a node whose features look nothing like yours gets down-weighted, so a ring wiring itself to legitimate traffic gains less cover.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layers.py
import torch
from ringfence.models.layers import RelationalLayer


def test_output_shape_and_gradients():
    layer = RelationalLayer(in_dim=6, out_dim=8, n_relations=2)
    x = torch.randn(10, 6, requires_grad=True)
    edges = [torch.tensor([[0, 1], [2, 3]]), torch.tensor([[4], [5]])]
    out = layer(x, edges)
    assert out.shape == (10, 8)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_isolated_node_still_gets_self_transform():
    layer = RelationalLayer(in_dim=4, out_dim=4, n_relations=1)
    x = torch.randn(5, 4)
    out_no_edges = layer(x, [torch.zeros((2, 0), dtype=torch.long)])
    assert torch.isfinite(out_no_edges).all()
    assert not torch.allclose(out_no_edges, torch.zeros_like(out_no_edges))


def test_is_inductive_over_node_count():
    """Same weights must apply to a graph of any size — no per-node params."""
    layer = RelationalLayer(in_dim=3, out_dim=3, n_relations=1)
    e = torch.tensor([[0], [1]])
    assert layer(torch.randn(2, 3), [e]).shape == (2, 3)
    assert layer(torch.randn(50, 3), [e]).shape == (50, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.models.layers'`

- [ ] **Step 3: Write the layer**

```python
# ringfence/models/layers.py
import torch
import torch.nn as nn


class RelationalLayer(nn.Module):
    """Relational message passing, implemented directly.

    Per relation r:  m_v = sum_{u -> v} g(x_u, x_v) * W_r x_u  /  deg(v)
    then             h_v = ReLU( W_self x_v + sum_r a_r * m_v^r )

    `a_r` is a learned softmax over relations, so the model discovers which
    entity type carries the signal. `g` is the heterophily gate: it scores each
    edge from the feature *difference* between endpoints, damping edges that
    join dissimilar nodes. Fraud rings deliberately attach to legitimate nodes
    as camouflage; without the gate those edges dilute the ring's signal.
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
            raise ValueError(f"expected {self.n_relations} edge tensors, got {len(edges)}")

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pip install torch && pytest tests/test_layers.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ringfence/models/layers.py tests/test_layers.py
git commit -m "feat: from-scratch relational message passing with heterophily gate"
```

---

### Task 7: Cost-sensitive loss and the Ringfence model

**Files:**
- Create: `ringfence/models/loss.py`
- Create: `ringfence/models/ringfence.py`
- Test: `tests/test_ringfence.py`

**Interfaces:**
- Consumes: `RelationalLayer`, `build_edges`, `RELATIONS`
- Produces:
  - `expected_cost_loss(logits: Tensor, labels: Tensor, c_fn: float, c_fp: float) -> Tensor`
  - `node_features(events: pd.DataFrame) -> np.ndarray` — `(N, 6)` float32
  - `RingfenceDetector(hidden: int = 32, layers: int = 2, window_s: float = 3600.0, seed: int = 0)` with `fit(events, epochs=40) -> None` and `score_events(events) -> np.ndarray`

The loss is the point of difference: the network is trained on **expected rupees**, so the business objective is the training objective rather than a threshold chosen afterwards.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ringfence.py
import numpy as np
import torch
from ringfence.data.background import load_background
from ringfence.data.campaigns import inject
from ringfence.data.schema import CampaignSpec
from ringfence.models.loss import expected_cost_loss
from ringfence.models.ringfence import RingfenceDetector, node_features


def _data(seed):
    bg = load_background(path=None, n_rows=4000, seed=seed)
    spec = CampaignSpec(n_attempts=300, k_devices=30, k_ips=15,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[400]))
    return inject(bg, [spec], seed=seed)


def test_loss_penalises_confident_mistakes_most():
    y = torch.tensor([1.0, 1.0])
    confident_wrong = expected_cost_loss(torch.tensor([-6.0, -6.0]), y, 100.0, 10.0)
    confident_right = expected_cost_loss(torch.tensor([6.0, 6.0]), y, 100.0, 10.0)
    assert confident_wrong > confident_right


def test_loss_respects_cost_asymmetry():
    y = torch.tensor([0.0])
    logits = torch.tensor([3.0])                      # a false positive
    cheap = expected_cost_loss(logits, y, c_fn=100.0, c_fp=1.0)
    dear = expected_cost_loss(logits, y, c_fn=100.0, c_fp=50.0)
    assert dear > cheap


def test_node_features_shape_and_finiteness():
    f = node_features(_data(0))
    assert f.shape[1] == 6 and np.isfinite(f).all()


def test_ranks_campaign_above_background():
    tr, te = _data(0), _data(1)
    m = RingfenceDetector(seed=0)
    m.fit(tr, epochs=40)
    s = m.score_events(te)
    y = te["label"].to_numpy()
    assert s[y == 1].mean() > s[y == 0].mean()


def test_scores_unseen_entities_inductively():
    """Test entities share no ids with train — a transductive model would fail."""
    tr, te = _data(0), _data(1)
    assert not set(tr["device_id"]) & set(te.query("label == 1")["device_id"])
    m = RingfenceDetector(seed=0)
    m.fit(tr, epochs=20)
    assert np.isfinite(m.score_events(te)).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ringfence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.models.loss'`

- [ ] **Step 3: Write the loss**

```python
# ringfence/models/loss.py
import torch


def expected_cost_loss(logits: torch.Tensor, labels: torch.Tensor,
                       c_fn: float, c_fp: float) -> torch.Tensor:
    """Expected rupee cost of the decision, differentiable in the score.

    A missed campaign attempt costs `c_fn`; a blocked legitimate attempt costs
    `c_fp`. Minimising this trains directly against the business objective
    instead of optimising cross-entropy and repairing it with a threshold.
    """
    p = torch.sigmoid(logits)
    return ((1.0 - p) * labels * c_fn + p * (1.0 - labels) * c_fp).mean()
```

- [ ] **Step 4: Write the model**

```python
# ringfence/models/ringfence.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.schema import RELATIONS
from ..graph.build import build_edges
from .layers import RelationalLayer
from .loss import expected_cost_loss


def node_features(events: pd.DataFrame) -> np.ndarray:
    """Per-event features only. All coordination signal must arrive through
    the graph — otherwise the ablation against GBDT proves nothing."""
    amt = events["amount"].to_numpy(dtype=np.float64)
    f = np.stack([
        np.log1p(amt),
        (amt < 25.0).astype(np.float64),
        events["approved"].to_numpy().astype(np.float64),
        ((events["ts"].to_numpy() // 3600) % 24) / 24.0,
        events["email_domain"].isin(["gmail.com", "outlook.com"]).to_numpy().astype(np.float64),
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


class RingfenceDetector:
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ringfence.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add ringfence/models/loss.py ringfence/models/ringfence.py tests/test_ringfence.py
git commit -m "feat: inductive graph detector trained on expected rupee cost"
```

---

### Task 8: Cost model and replay harness

**Files:**
- Create: `ringfence/eval/cost.py`
- Create: `ringfence/eval/latency.py`
- Test: `tests/test_latency.py`

**Interfaces:**
- Produces:
  - `COST_PER_ATTEMPT_INR`, `COST_PER_FALSE_BLOCK_INR` constants
  - `detection_times(events, scores, threshold) -> dict[str, float | None]` — campaign id → seconds from campaign onset to first alert, `None` if never
  - `money_prevented(events, detect_s, campaign_id) -> float`

This is where the project's headline claim becomes a number.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_latency.py
import numpy as np
from ringfence.data.background import load_background
from ringfence.data.campaigns import inject
from ringfence.data.schema import CampaignSpec
from ringfence.eval.latency import detection_times, money_prevented


def _data():
    bg = load_background(path=None, n_rows=2000, seed=0)
    spec = CampaignSpec(n_attempts=200, k_devices=10, k_ips=5,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[200]))
    return inject(bg, [spec], seed=0)


def test_perfect_scores_detect_at_first_attempt():
    ev = _data()
    scores = ev["label"].to_numpy().astype(float)
    assert detection_times(ev, scores, threshold=0.5)["camp_0"] == 0.0


def test_never_detected_returns_none():
    ev = _data()
    assert detection_times(ev, np.zeros(len(ev)), threshold=0.5)["camp_0"] is None


def test_earlier_detection_prevents_more_money():
    ev = _data()
    early = money_prevented(ev, 60.0, "camp_0")
    late = money_prevented(ev, 3000.0, "camp_0")
    assert early > late >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_latency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.eval.latency'`

- [ ] **Step 3: Write the cost model**

```python
# ringfence/eval/cost.py
"""Cost constants. These are ASSUMPTIONS, not measurements, and the README
must say so. Each carries its reasoning so a reviewer can substitute their own.
"""

# Authorization fee charged on every attempt including declines, plus an
# amortized share of card-network enumeration penalties and the expected
# downstream chargeback cost on cards the attack validates.
COST_PER_ATTEMPT_INR = 73.0

# Blocking a legitimate checkout: lost order margin plus a churn proxy.
COST_PER_FALSE_BLOCK_INR = 40.0
```

- [ ] **Step 4: Write the harness**

```python
# ringfence/eval/latency.py
import numpy as np
import pandas as pd
from .cost import COST_PER_ATTEMPT_INR


def detection_times(events: pd.DataFrame, scores: np.ndarray,
                    threshold: float) -> dict[str, float | None]:
    """Seconds from each campaign's first attempt to its first alert."""
    out: dict[str, float | None] = {}
    fired = scores >= threshold
    for cid, idx in events[events["label"] == 1].groupby("campaign_id").indices.items():
        idx = np.sort(idx)
        onset = events["ts"].to_numpy()[idx[0]]
        hit = idx[fired[idx]]
        out[str(cid)] = float(events["ts"].to_numpy()[hit[0]] - onset) if len(hit) else None
    return out


def money_prevented(events: pd.DataFrame, detect_s: float | None,
                    campaign_id: str) -> float:
    """Exposure avoided by stopping the campaign `detect_s` after onset."""
    camp = events[events["campaign_id"] == campaign_id]
    if detect_s is None:
        return 0.0
    onset = camp["ts"].min()
    remaining = int((camp["ts"] > onset + detect_s).sum())
    return remaining * COST_PER_ATTEMPT_INR
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_latency.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add ringfence/eval/cost.py ringfence/eval/latency.py tests/test_latency.py
git commit -m "feat: rupee cost model and detection-latency replay harness"
```

---

### Task 9: Calibration and cost-optimal threshold

**Files:**
- Create: `ringfence/eval/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces: `reliability(scores, labels, bins=10) -> pd.DataFrame` with columns `["bin_mid", "predicted", "observed", "count"]`; `cost_optimal_threshold(scores, labels, c_fn, c_fp) -> tuple[float, float]` returning `(threshold, expected_cost)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration.py
import numpy as np
from ringfence.eval.calibration import reliability, cost_optimal_threshold


def test_reliability_bins_sum_to_population():
    rng = np.random.default_rng(0)
    s, y = rng.random(500), rng.integers(0, 2, 500)
    assert reliability(s, y, bins=10)["count"].sum() == 500


def test_perfect_scores_give_perfect_reliability():
    y = np.array([0, 0, 1, 1])
    r = reliability(y.astype(float), y, bins=2)
    assert np.allclose(r["observed"].to_numpy(), r["predicted"].to_numpy())


def test_expensive_false_negatives_lower_the_threshold():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    t_cheap, _ = cost_optimal_threshold(s, y, c_fn=10.0, c_fp=10.0)
    t_dear, _ = cost_optimal_threshold(s, y, c_fn=500.0, c_fp=10.0)
    assert t_dear <= t_cheap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.eval.calibration'`

- [ ] **Step 3: Write the module**

```python
# ringfence/eval/calibration.py
import numpy as np
import pandas as pd


def reliability(scores: np.ndarray, labels: np.ndarray,
                bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed positive rate per score bin.

    A model can rank well and still be badly calibrated; a threshold chosen on
    an uncalibrated score does not mean what its number implies.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({
            "bin_mid": (edges[b] + edges[b + 1]) / 2,
            "predicted": float(scores[m].mean()) if m.any() else np.nan,
            "observed": float(labels[m].mean()) if m.any() else np.nan,
            "count": int(m.sum()),
        })
    return pd.DataFrame(rows)


def cost_optimal_threshold(scores: np.ndarray, labels: np.ndarray,
                           c_fn: float, c_fp: float) -> tuple[float, float]:
    """Threshold minimising total rupee cost, not F1."""
    best_t, best_c = 0.5, float("inf")
    for t in np.unique(np.round(scores, 4)):
        pred = scores >= t
        cost = c_fn * ((labels == 1) & ~pred).sum() + c_fp * ((labels == 0) & pred).sum()
        if cost < best_c:
            best_t, best_c = float(t), float(cost)
    return best_t, best_c
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calibration.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ringfence/eval/calibration.py tests/test_calibration.py
git commit -m "feat: reliability diagram and cost-optimal thresholding"
```

---

### Task 10: The detectability frontier

**Files:**
- Create: `ringfence/eval/frontier.py`
- Test: `tests/test_frontier.py`

**Interfaces:**
- Produces: `predicted_boundary_k(n: int, tau: int) -> float`; `sweep(n_values, k_values, tau, seed) -> pd.DataFrame` with columns `["n", "k", "velocity_detected", "ringfence_detected"]`

This is the spec's §3.1 made empirical: Claim 1 predicts velocity fails once `k >= n/tau`, and the sweep is the experiment that either confirms it or does not.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frontier.py
from ringfence.eval.frontier import predicted_boundary_k, sweep


def test_boundary_matches_claim_one():
    assert predicted_boundary_k(n=4000, tau=40) == 100.0


def test_velocity_fails_above_the_predicted_boundary():
    """The experiment Claim 1 stakes itself on."""
    df = sweep(n_values=[400], k_values=[2, 400], tau=40, seed=0)
    below = df[df["k"] == 2].iloc[0]
    above = df[df["k"] == 400].iloc[0]
    assert bool(below["velocity_detected"]) is True
    assert bool(above["velocity_detected"]) is False


def test_sweep_covers_the_grid():
    df = sweep(n_values=[200, 400], k_values=[5, 50], tau=40, seed=0)
    assert len(df) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frontier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ringfence.eval.frontier'`

- [ ] **Step 3: Write the module**

```python
# ringfence/eval/frontier.py
import numpy as np
import pandas as pd

from ..data.background import load_background
from ..data.campaigns import inject
from ..data.schema import CampaignSpec
from ..models.velocity import VelocityDetector
from ..models.ringfence import RingfenceDetector


def predicted_boundary_k(n: int, tau: int) -> float:
    """Claim 1: a threshold detector needs n/k > tau, so it is blind at k >= n/tau."""
    return n / tau


def sweep(n_values: list[int], k_values: list[int], tau: int,
          seed: int, window_s: float = 3600.0) -> pd.DataFrame:
    """Run both detectors across the (n, k) grid and record who fired."""
    rows = []
    for n in n_values:
        for k in k_values:
            bg = load_background(path=None, n_rows=4000, seed=seed)
            spec = CampaignSpec(n_attempts=n, k_devices=k, k_ips=k,
                                duration_s=window_s,
                                start_ts=float(bg["ts"].iloc[300]))
            ev = inject(bg, [spec], seed=seed)
            y = ev["label"].to_numpy() == 1

            vel = VelocityDetector(tau=tau, window_s=window_s,
                                   entity="ip_id").score_events(ev)
            model = RingfenceDetector(seed=seed)
            model.fit(ev, epochs=30)
            rf = model.score_events(ev)

            rows.append({
                "n": n, "k": k,
                "velocity_detected": bool(vel[y].max() > 0),
                "ringfence_detected": bool(rf[y].mean() > rf[~y].mean()),
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_frontier.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ringfence/eval/frontier.py tests/test_frontier.py
git commit -m "feat: (n,k) detectability frontier sweep against predicted boundary"
```

---

### Task 11: Experiment CLI, results, and README

**Files:**
- Create: `ringfence/cli.py`
- Create: `README.md`
- Modify: `docs/console-mockup.html` (replace illustrative constants with measured output)

**Interfaces:**
- Consumes: everything above
- Produces: `python -m ringfence.cli ablation` and `python -m ringfence.cli frontier`, each writing a table to `results/`

- [ ] **Step 1: Write the ablation runner**

```python
# ringfence/cli.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data.background import load_background
from .data.campaigns import inject
from .data.schema import CampaignSpec
from .models.velocity import VelocityDetector
from .models.gbdt import GBDTDetector
from .models.ringfence import RingfenceDetector
from .eval.latency import detection_times, money_prevented
from .eval.calibration import cost_optimal_threshold
from .eval.cost import COST_PER_ATTEMPT_INR, COST_PER_FALSE_BLOCK_INR

RESULTS = Path("results")


def _split(seed: int, k_devices: int):
    """Train on loud campaigns, test on a spread one it has never seen.

    Holding out attack MORPHOLOGY rather than samples is what makes the result
    evidence rather than memorisation.
    """
    bg = load_background(path=None, n_rows=6000, seed=seed)
    spec = CampaignSpec(n_attempts=400, k_devices=k_devices, k_ips=k_devices,
                        duration_s=3600.0, start_ts=float(bg["ts"].iloc[500]))
    return inject(bg, [spec], seed=seed)


def ablation() -> pd.DataFrame:
    train = _split(seed=0, k_devices=4)      # loud, concentrated
    test = _split(seed=1, k_devices=120)     # low-and-slow, unseen morphology
    y = test["label"].to_numpy()

    gbdt = GBDTDetector(seed=0); gbdt.fit(train)
    rf = RingfenceDetector(seed=0); rf.fit(train, epochs=60)

    scored = {
        "velocity": VelocityDetector(tau=40, window_s=3600.0).score_events(test),
        "gbdt": gbdt.score_events(test),
        "ringfence": rf.score_events(test),
    }

    rows = []
    for name, s in scored.items():
        s = np.asarray(s, dtype=float)
        norm = s / s.max() if s.max() > 0 else s
        thr, _ = cost_optimal_threshold(norm, y, COST_PER_ATTEMPT_INR,
                                        COST_PER_FALSE_BLOCK_INR)
        dt = detection_times(test, norm, thr)["camp_0"]
        rows.append({
            "detector": name,
            "pr_auc": round(float(average_precision_score(y, norm)), 4),
            "threshold": round(thr, 4),
            "detect_s": dt,
            "inr_prevented": round(money_prevented(test, dt, "camp_0"), 2),
        })
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "ablation.csv", index=False)
    print(df.to_string(index=False))
    return df


def frontier() -> pd.DataFrame:
    from .eval.frontier import sweep
    df = sweep(n_values=[200, 400, 800, 1600], k_values=[2, 10, 50, 200],
               tau=40, seed=0)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "frontier.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    {"ablation": ablation, "frontier": frontier}[cmd]()
```

- [ ] **Step 2: Run the ablation**

Run: `python -m ringfence.cli ablation`
Expected: a three-row table printed and written to `results/ablation.csv`. The claim under test is that `ringfence` has the smallest `detect_s` and the largest `inr_prevented` on the held-out spread campaign.

**If it does not:** that is a result, not a failure. Report it, diagnose it in the README, and keep the ablation — the brief asks for honest metrics, and a measured negative with a correct protocol beats an unmeasured claim. Try, in order: more epochs, a longer `window_s`, and `k_ips` smaller than `k_devices` (IP pools are usually scarcer than devices, which is realistic).

- [ ] **Step 3: Run the frontier sweep**

Run: `python -m ringfence.cli frontier`
Expected: 16 rows in `results/frontier.csv`. Check that `velocity_detected` flips to `False` near `k = n/40`, as `predicted_boundary_k` says it should.

- [ ] **Step 4: Wire measured numbers into the console**

In `docs/console-mockup.html`, replace `FIRE_RF`, `FIRE_GBDT`, `RATE` and `COST` with the values from `results/ablation.csv`, and replace the `pts` array in the frontier block with the rows of `results/frontier.csv`. Delete the "Design target, not a build" note and replace it with the real cost-model assumptions.

- [ ] **Step 5: Write the README**

Required first-screen content, in this order:

1. One-line thesis: *a detector for distributed card-testing campaigns, reporting precision and recall on a held-out test set, and the rupee cost of detecting late.*
2. The ablation table from `results/ablation.csv`.
3. The frontier chart, with the predicted boundary drawn over the measured points.
4. `## Defense-only` — no network calls, synthetic campaigns only, no real BIN ranges, characteristics drawn from public Visa anti-enumeration guidance.
5. `## What is assumed, not measured` — the cost constants, and the fact that campaigns are injected rather than observed.
6. How to run.

The opening paragraph must contain the words **detector**, **precision and recall**, **held-out test set**, and **false-positive cost** — a judge scanning for track fit is looking for exactly those.

- [ ] **Step 6: Commit**

```bash
git add ringfence/cli.py README.md docs/console-mockup.html results/
git commit -m "feat: experiment CLI, measured results, and README"
```

---

## Self-Review

**Spec coverage.** §1–2 problem framing → README (Task 11). §3.1 frontier → Tasks 10, 11. §3.2 latency → Task 8. §4.1 graph → Task 5. §4.2 inductive + heterophily → Tasks 6, 7. §4.3 early classification → Task 8. §4.4 imbalance → the cost-sensitive loss in Task 7. §5 held-out morphology → `_split` in Task 11. §5 calibration → Task 9. §6 defense-only → Global Constraints and Task 11 Step 5. §9 engineering log → keep it as you go; it is the form field they read first.

**Not covered, deliberately:** subgraph explainability. It is presentation, not evidence, and 4 days does not fit it. If Wednesday goes well, add per-relation attention weights (`layer.rel_att` after softmax) as a cheap substitute — it says which entity type carried the signal, which is most of what an analyst wants.

**Type consistency check.** `score_events(events) -> np.ndarray` is identical across `VelocityDetector`, `GBDTDetector`, `RingfenceDetector`. `build_edges` returns a dict keyed by relation name; `RingfenceDetector._tensors` orders it by `RELATIONS`, matching `RelationalLayer(n_relations=len(RELATIONS))`. `detection_times` returns `dict[str, float | None]`, and `money_prevented` accepts that `None`. `CampaignSpec` field names are identical in Tasks 1, 2, 10, 11.
