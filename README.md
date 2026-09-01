# Koronis

> Detection of distributed card-testing campaigns that per-entity velocity rules cannot see at any threshold.

[![CI](https://github.com/Yashasm18/koronis/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashasm18/koronis/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-131%20passing-2ea44f)](tests/)
[![python](https://img.shields.io/badge/python-3.14-3776ab)](https://www.python.org/)
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![graph libs](https://img.shields.io/badge/graph%20libraries-none-8a3ffc)](koronis/models/layers.py)
[![track](https://img.shields.io/badge/Razorpay%20Buildathon-Track%2002%20·%20AI%20Risk%20Manager-0c2451)](https://razorpay.com/buildathon/)

![Koronis demo — held-out campaign replayed event by event](docs/assets/koronis-demo.gif)

*Detection, incident consolidation and the cost-optimal action ladder on a held-out
campaign. [Full screen recording (MP4)](docs/assets/koronis-demo.mp4) · run it live at
**[yashasm18.github.io/koronis](https://yashasm18.github.io/koronis/)**.*

## What it solves

**Card testing.** A fraudster pushes thousands of ₹1–₹20 charges through a merchant's
checkout to find which stolen cards are still live. The checkout becomes a free
card-validation service, and the merchant pays: an authorisation fee on every attempt
including declines, network penalties when the decline rate spikes, and chargebacks weeks
later on the cards their own site validated.

**Velocity rules cannot catch it — not because they are tuned badly, but by arithmetic.**
An attacker spreading `n` attempts across `k ≥ n/τ` entities keeps every counter under
threshold at *any* threshold. Koronis is a **defence-only detector and decision-support
prototype** built for exactly that region.

Its contribution is not that it beats its baselines. It is a **characterisation of when
detection is possible at all** — stated as arithmetic, then tested.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 131 tests, ~1-2 min
.venv/bin/python -m koronis.cli ablation      # reproduces the headline table below
```

## Key results

**Why the graph wins, in one line.** Per-entity signal dilutes as `n/k`, but the number of
attempt-pairs sharing an entity grows as `n²/k` — so the graph's advantage *increases* with
attack size, and the attacker's only escape is `k → n`: one fresh device, IP and BIN per
attempt, bounded by infrastructure cost.

**Predicted vs. measured detectability boundary**, `predicted_boundary_k(n, τ) = n/τ`, on
a 4×4 grid (`python -m koronis.cli frontier`):

| n | k=2 | k=10 | k=50 | k=200 | predicted boundary `n/τ` |
|---:|:---:|:---:|:---:|:---:|---:|
| 200 | fires | fires | **blind** | **blind** | 25 |
| 400 | fires | fires | **blind** | **blind** | 50 |
| 800 | fires | fires | fires | **blind** | 100 |
| 1600 | fires | fires | fires | **blind** | 200 |

The binding threshold is `τ = 8` (the device counter; `τ_ip = 61`, `τ_bin = 236`). Every
cell agrees with `k ≥ n/τ` (**16 / 16**). Koronis detects in all sixteen.

![Detectability frontier: 16 measured cells against the boundary k = n/τ](docs/assets/frontier.svg)

The dashed line is arithmetic, drawn before any run. Every measured cell falls on the side
it predicts, and Koronis detects across the whole grid — including the entire region above
the line, where no per-entity counter can trip at any threshold.

**Held-out detection**, median with the 2.5th / 97.5th percentiles observed across 10
independent trials (`python -m koronis.cli seeds`):

| detector | PR-AUC | precision | recall | false positives | detected |
|---|---:|---:|---:|---:|---:|
| `velocity_tuned` | 0.062 `[0.062, 0.062]` | 0.000 | 0.000 | 44 | 0 / 10 |
| `decline_burst` *(no graph, no learning)* | 0.222 `[0.205, 0.234]` | 0.000 | 0.000 | 0 | 3 / 10 |
| `shared_entity` *(graph, no learning)* | 0.051 `[0.050, 0.051]` | 0.000 | 0.000 | 40 | 0 / 10 |
| `gbdt_per_txn` | 0.332 `[0.311, 0.352]` | 0.410 | 0.836 | 476 | 10 / 10 |
| **`koronis_graph`** | **0.990** `[0.987, 0.995]` | **0.951** | **0.964** | **20** | **10 / 10** |

Threshold rules do not degrade here — they stop working: at `k = 60` against a binding
`τ = 8`, no counter can trip. The per-transaction model detects every time but at **476
false positives against Koronis's 20**. Recall was never the hard part; precision at a
usable alert volume is.

**Detection latency.** At one minute Koronis has already recalled every campaign attempt
so far, at 0.43 precision against the per-transaction model's 0.09; by ten minutes it is at
0.88 precision with recall still at 1.00. None of the three learning-free detectors ever
crosses its frozen threshold on this campaign.
→ [full latency curves](docs/evaluation.md#streaming-and-inference-latency)

**An alert is not a task.** Four hundred event alerts are one campaign, so Koronis
consolidates them and picks the intervention with the lowest expected rupee cost:

```
414 event alerts  ->  17 incidents  ->  1 action recommended
```

Median across 8 test streams. The policy sees only an incident's first 12 events plus a
forecast — never the true remaining count:

| policy | analyst minutes | merchant cost |
|---|---:|---:|
| always hold | 234.0 | ₹1,21,644 |
| event-by-event thresholding | 214.8 | ₹19,167 |
| **causal policy** *(forecast only)* | **12.0** | **₹16,240** |
| oracle *(upper bound, knows the future)* | 12.0 | ₹8,691 |

Event thresholding reaches the same decision and hands an analyst **eighteen times the
triage**. Consolidation, not detection, is the difference.

## Architecture

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 520, "nodeSpacing": 30, "rankSpacing": 40}}}%%
flowchart TB
    IN["<b>Attempt stream</b> — ts · amount · auth outcome · device · IP · BIN · email"]
    G["<b>Temporal graph</b>, strictly causal — backwards-in-time edges · window 3600 s · fan-in ≤ 32"]
    MP["<b>Relational message passing</b>, written from scratch — torch.index_add_ · per-relation weights · 2 layers · cost-sensitive loss trained on rupees"]
    SC{"score ≥ frozen threshold?"}
    CON["<b>Incident consolidation</b> — union-find on the alerted subgraph · links only via values &lt; 2% of stream"]
    FC["<b>Incident risk + exposure forecast</b> — separately recalibrated risk · P50/P90 from the first 12 events · conformal band"]
    POL["<b>Cost-optimal action</b> — monitor · rate-limit · step-up · hold + review"]
    DRIFT{"PSI > cut-off?"}
    REV["<b>Analyst review</b> — automation stood down"]
    OUT["<b>Recommendation + audit dossier</b><br/><i>simulated workflow — nothing blocks a live payment</i>"]

    IN --> G --> MP --> SC
    SC -->|"alert"| CON --> FC --> POL --> DRIFT
    SC -.->|"below"| IN
    DRIFT -->|"no"| OUT
    DRIFT -->|"yes"| REV --> OUT
```

Attempts are nodes; two are linked when they share a device, IP, BIN range or email domain
inside a window. Edges point **backwards in time**, so a streaming evaluation cannot read
the future — which is also what lets the replay reproduce batch scores exactly. Aggregation
is `torch.index_add_`: **no DGL, no PyTorch Geometric**.

The detector is a **temporal heterogeneous graph network**, 6,225 parameters, inductive, and
trained on expected rupee cost rather than cross-entropy — the business objective *is* the
training objective.

→ **[Full architecture, and which parts are learned](docs/architecture.md)**

## Reproducing everything

Every experiment writes to `results/`, and **every number in this repo and on the demo site
is read from there** — nothing is transcribed by hand.

<details>
<summary><b>All experiments</b></summary>

```bash
.venv/bin/python -m koronis.cli seeds           # 10 trials, median + across-run range
.venv/bin/python -m koronis.cli frontier        # predicted vs measured boundary
.venv/bin/python -m koronis.cli mechanism       # which mechanism carries the signal
.venv/bin/python -m koronis.cli relations       # which entity type carries the signal
.venv/bin/python -m koronis.cli architecture    # do the gate and the attention earn their place
.venv/bin/python -m koronis.cli online          # online consolidation vs the batch grouping
.venv/bin/python -m koronis.cli sharding        # does the graph survive being split across machines
.venv/bin/python -m koronis.cli select          # architecture selection on calibration, tested once
.venv/bin/python -m koronis.cli aperture        # merchant view vs gateway view
.venv/bin/python -m koronis.cli incidents       # alerts -> incidents -> forecast -> action
.venv/bin/python -m koronis.cli drift           # traffic-profile transfer stress test
.venv/bin/python -m koronis.cli latency         # precision / recall over time
.venv/bin/python -m koronis.cli replay          # causal event-by-event replay -> JSON
.venv/bin/python -m koronis.cli benchmark       # p50 / p95 per-event inference latency
python site/build.py                            # results/ -> docs/index.html
```

</details>

## What is measured, and where

| Question | Answer | Detail |
|---|---|---|
| Does it detect what velocity rules cannot? | 0.990 PR-AUC vs 0.062, on a hold-out spread past the boundary | [Evaluation → protocol](docs/evaluation.md#protocol) |
| What does a false positive cost? | costed in rupees; 20 FPs against a GBDT's 476 | [Evaluation → decision layer](docs/evaluation.md#decision-layer) |
| Which mechanism carries the signal? | outcome buys earliness, the graph buys precision | [Evaluation](docs/evaluation.md#which-mechanism-carries-the-signal) |
| Do the architectural claims hold? | one does not — the heterophily gate is **net-negative**, and it is retracted | [Evaluation](docs/evaluation.md#does-the-architecture-earn-its-place) |
| Can that be acted on without cheating? | yes — chosen on calibration, held up on test: cost ₹1,836 → ₹892 | [Evaluation](docs/evaluation.md#closing-the-loop-selecting-an-architecture-without-touching-test) |
| Is a gateway's wider view worth anything? | measured: the gap grows with the number of merchants | [Evaluation](docs/evaluation.md#vantage-point-one-merchant-or-the-whole-gateway) |
| Does it survive a different merchant? | flagged on all three shifted profiles; the guardrail is **experimental** | [Evaluation](docs/evaluation.md#traffic-profile-transfer-stress-test) |
| Can it run online? | 0.99 ms p50; streaming reproduces batch scores exactly | [Evaluation](docs/evaluation.md#streaming-and-inference-latency) |
| Is the *whole* pipeline causal? | yes — consolidation too, via a sliding count-min sketch in fixed memory | [Evaluation](docs/evaluation.md#making-consolidation-causal) |
| Does it survive being split across machines? | measured — and PR-AUC and rupees rank the routing keys **oppositely** | [Evaluation](docs/evaluation.md#does-the-graph-survive-being-split-across-machines) |
| What broke? | 8 defects, **4 retracted claims** | [Engineering log](docs/engineering-log.md) |

## Limitations

This is a **semi-synthetic proof of concept**, not production fraud detection.

- **Background traffic and campaigns are generated.** The IEEE-CIS loader exists and the
  real data was profiled, then deliberately not used — its native density would reintroduce
  a defect this project already fixed.
- **It will not catch** an attacker using genuinely fresh infrastructure for every attempt.
  That limit is real, and it is also the point: driving `k → n` costs one device, IP and BIN
  per attempt.
- **Cost figures are declared assumptions**, not measurements.
- **The drift guardrail is experimental** — its false-flag rate on held-out base traffic is
  too high to depend on, and the reason is measured.
- **Defence-only.** No network capability anywhere in the package, no live payment
  integration, no real card or BIN data. Every recommended action is a recommendation in a
  simulated workflow.

→ **[Full limitations, deployment gaps, and scope](docs/limitations.md)** ·
[Security policy](SECURITY.md)

## Documentation

| | |
|---|---|
| [Architecture](docs/architecture.md) | how it works, and where a learned model is and is not used |
| [Evaluation](docs/evaluation.md) | protocol, ablations, calibration, forecasting, drift, latency |
| [Limitations](docs/limitations.md) | what is assumed rather than measured, and what production would need |
| [Engineering log](docs/engineering-log.md) | repository map, and every defect that changed a result |
| [Contributing](CONTRIBUTING.md) | development setup and the conventions that keep results reproducible |

## License

[MIT](LICENSE) © 2026 Yashas.

## Acknowledgements

The Koronis asteroids share orbital elements because they are fragments of one shattered
parent body. Hirayama found them in 1918 not by looking at the sky, where they are
scattered and unremarkable, but by plotting them in *orbital-element space*, where they
clump unmistakably. Individually ordinary events, scattered in the obvious view; plotted in
shared-entity space, a common origin becomes undeniable.

Built for the Razorpay AI Buildathon 2026 · Track 02, AI Risk Manager.
