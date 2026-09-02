# Koronis

> Detection of distributed card-testing campaigns that per-entity velocity rules cannot see at any threshold.

[![CI](https://github.com/Yashasm18/koronis/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashasm18/koronis/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-251%20passing-2ea44f)](tests/)
[![python](https://img.shields.io/badge/python-3.14-3776ab)](https://www.python.org/)
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![graph libs](https://img.shields.io/badge/graph%20libraries-none-8a3ffc)](koronis/models/layers.py)
[![track](https://img.shields.io/badge/Razorpay%20Buildathon-Track%2002%20·%20AI%20Risk%20Manager-0c2451)](https://razorpay.com/buildathon/)

![Koronis demo — held-out campaign replayed event by event](docs/assets/koronis-demo.gif)

> **Defence-only.** No network capability anywhere in the package — no HTTP client, no
> socket, no subprocess, no process spawning — asserted per module by
> [`tests/test_defence_only.py`](tests/test_defence_only.py). No real card data, no real BIN
> ranges. Every recommended action is a recommendation in a simulated workflow; nothing
> blocks a payment, and no code path here can.

*Detection, incident consolidation and the cost-optimal action ladder on a held-out
campaign. [Full screen recording (MP4)](docs/assets/koronis-demo.mp4) · run it live at
**[yashasm18.github.io/koronis](https://yashasm18.github.io/koronis/)**.*

## What it solves

**Card testing.** A fraudster pushes thousands of ₹1–₹20 charges through a merchant's
checkout to find which stolen cards are still live. The checkout becomes a free
card-validation service, and the merchant pays: a per-authorisation fee on the attempts
(whether declines carry one depends on the pricing plan), network penalties once the
decline ratio crosses scheme thresholds, disputes on the test charges that succeed, and a
decline profile degraded while the attack runs.

**It is also upstream of a chargeback class.** The cards a campaign validates are resold
and spent elsewhere, so the large downstream fraud lands on *other* merchants — which is
why this repo does not count it against the merchant being tested. That is the scope
choice, stated rather than hidden: the track names fraud, returns and chargebacks, and this
covers **one loss class end to end with the money measured**, rather than three shallowly.
Card testing is the cause; the disputes are a symptom weeks later.

**Velocity rules cannot catch it — not because they are tuned badly, but by arithmetic.**
An attacker spreading `n` attempts across `k ≥ n/τ` entities keeps every **per-entity**
counter under threshold at *any* threshold. Koronis is a **defence-only detector and
decision-support prototype** built for exactly that region.

Its contribution is not that it beats its baselines. It is a **characterisation of where a
per-entity counter cannot work**, stated as arithmetic and implemented faithfully. The
matching characterisation of where *this* model stops working is
[attempted and published as unmeasurable on this simulator](docs/evaluation.md#where-does-this-model-stop-working--an-invalid-measurement-published) —
the honest half of the same question.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m playwright install chromium   # the site tests drive the demo page
.venv/bin/python site/build.py                    # they need docs/index.html to exist
.venv/bin/python -m pytest tests/ -q              # 251 tests, ~2 min
.venv/bin/python -m koronis.cli ablation          # reproduces the headline table below
```

The suite runs without `requirements-dev.txt` — the eight tests that drive the demo page
skip — but then 243 are collected rather than 251, and the test-count check says so.

## Key results

**What it produces is a decision, not a score.** A merchant does not lose money to a
probability; it loses money to attempts nobody acted on. So the headline is the funnel and
its price:

**An alert is not a task.** Hundreds of event alerts are one campaign, so Koronis
consolidates them and picks the intervention with the lowest expected rupee cost:

```
402 event alerts  ->  7 incidents  ->  2 actions recommended
```

One test stream, end to end. One of those two actions was wrong — a two-attempt incident
rate-limited when the oracle would have left it alone — and it is the *only* decision in
that stream the oracle would have made differently, so it accounts for the whole of that
stream's ₹520 regret. The policy sees only an incident's first 12 events plus a forecast,
never the true remaining count. The costs below are medians across all 8 test streams:

| policy | analyst minutes | merchant cost |
|---|---:|---:|
| always hold | 126.0 | ₹50,307 |
| event-by-event thresholding | 211.0 | ₹6,602 |
| **causal policy** *(forecast only)* | **12.0** | **₹3,405** |
| oracle *(upper bound, knows the future)* | 12.0 | ₹3,145 |

Event thresholding reaches the same decision and hands an analyst **eighteen times the
triage**. Consolidation, not detection, is the difference. Not knowing the future still
costs — ₹3,405 against the oracle's ₹3,145.

The detection work below is the evidence that the thing making those decisions is sound.

**Why the graph wins, in one line.** Per-entity signal dilutes as `n/k`, but the number of
attempt-pairs sharing an entity grows as `n²/k` — so the graph's advantage *increases* with
attack size, and the attacker's only escape is `k → n`: one fresh device, IP and BIN per
attempt, bounded by infrastructure cost.

**The per-entity blind region**, `predicted_boundary_k(n, τ) = n/τ`, on
a 4×4 grid (`python -m koronis.cli frontier`):

| n | k=2 | k=10 | k=50 | k=200 | predicted boundary `n/τ` |
|---:|:---:|:---:|:---:|:---:|---:|
| 200 | fires | fires | **blind** | **blind** | 25 |
| 400 | fires | fires | **blind** | **blind** | 50 |
| 800 | fires | fires | fires | **blind** | 100 |
| 1600 | fires | fires | fires | **blind** | 200 |

The binding threshold is `τ = 8` (the device counter; `τ_ip = 61`, `τ_bin = 236`). Every
cell agrees with `k ≥ n/τ` (**16 / 16**), and Koronis detects in all sixteen.

**What that 16/16 is and is not.** The campaign generator spreads attempts *uniformly*
across entities, so each one carries exactly `n/k` attempts and a counter trips precisely
when `k ≤ n/τ`. Under uniform spread the agreement is therefore **exact by construction**:
it confirms the pipeline implements the arithmetic, which is an implementation check worth
having, not a risky prediction that survived. Under realistic non-uniform spread the
busiest entity carries more than `n/k`, so a counter trips at a *higher* `k` than `n/τ`
predicts — the genuinely blind region is smaller than the dashed line, and a sloppy attacker
is caught earlier. This repo learned that the hard way: an earlier version disagreed on 25%
of the grid purely because BINs were sampled rather than spread evenly
([defect 4](docs/engineering-log.md)).

![Detectability frontier: 16 measured cells against the boundary k = n/τ](docs/assets/frontier.svg)

Koronis detects across the whole grid, including the entire region above the line, where
no per-entity counter can trip at any threshold. The line itself is arithmetic — see above
for why the grid agreeing with it is an implementation check rather than a result.

**Held-out detection**, median with the 2.5th / 97.5th percentiles observed across 10
independent trials (`python -m koronis.cli seeds`):

| detector | PR-AUC | precision | recall | false positives | **false-positive cost** | detected |
|---|---:|---:|---:|---:|---:|---:|
| `velocity_tuned` | 0.062 `[0.062, 0.062]` | 0.000 | 0.000 | 44 | ₹1,760 | 0 / 10 |
| `decline_burst` *(no graph, no learning)* | 0.222 `[0.205, 0.234]` | 0.000 | 0.000 | 0 | ₹0 | 3 / 10 |
| `shared_entity` *(graph, no learning)* | 0.051 `[0.050, 0.051]` | 0.000 | 0.000 | 40 | ₹1,600 | 0 / 10 |
| `gbdt_per_txn` | 0.332 `[0.311, 0.352]` | 0.410 | 0.836 | 476 | **₹19,040** | 10 / 10 |
| **`koronis_graph`** | **0.997** `[0.996, 0.999]` | **0.977** | **0.990** | **10** | **₹400** | **10 / 10** |

False-positive cost is the published count × **₹40**, the declared cost of turning away one
legitimate checkout ([`cost.py`](koronis/eval/cost.py)). Like every rupee figure here it is
a *declared assumption, not a measurement* — substitute your own and re-run. `decline_burst`
costs ₹0 because it never fires at all: it detects 3 campaigns in 10 and alerts on nothing
else. **BIN thresholds are optimistic for the baseline** — the tuned `τ_bin = 236` would sit
higher on real traffic, so the counters here trip more easily than they would in production
([full caveat](docs/limitations.md#what-is-assumed-not-measured)).

Threshold rules do not degrade here — they stop working: at `k = 60` against a binding
`τ = 8`, no counter can trip. The per-transaction model detects every time but at **476
false positives against Koronis's 10**. Recall was never the hard part; precision at a
usable alert volume is.

Every part of the architecture producing these numbers was **chosen on the calibration
split**, never by hand: which relations to keep and whether to gate
([eight candidates](docs/evaluation.md#closing-the-loop-selecting-an-architecture-without-touching-test)),
and how large to make it — 32 hidden units and 3 layers,
9,171 parameters, from a
[nine-point width × depth grid](docs/evaluation.md#was-the-model-sized-or-just-chosen).
Test was read once, after each choice was already made.

**Detection latency.** At one minute Koronis has already recalled every campaign attempt
so far, at 0.46 precision against the per-transaction model's 0.09; by ten minutes it is at
0.90 precision with 0.94 recall. None of the three learning-free detectors ever crosses its
frozen threshold on this campaign.
→ [full latency curves](docs/evaluation.md#streaming-and-inference-latency)

## Architecture

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 520, "nodeSpacing": 30, "rankSpacing": 40}}}%%
flowchart TB
    IN["<b>Attempt stream</b> — ts · amount · auth outcome · device · IP · BIN · email"]
    G["<b>Temporal graph</b>, strictly causal — backwards-in-time edges · window 3600 s · fan-in ≤ 32"]
    MP["<b>Relational message passing</b>, written from scratch — torch.index_add_ · per-relation weights · 3 layers · cost-sensitive loss trained on rupees"]
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

Attempts are nodes; two are linked when they share a **device, IP or BIN range** inside a
window. Email domain is deliberately not among them: the
[per-relation ablation](docs/evaluation.md#which-entity-type-carries-the-signal) found it
net-negative for the detector, and calibration dropped it — while incident consolidation
still links on it, where the frequency cap governs it. `schema.py` keeps the two sets
apart for exactly that reason. Edges point **backwards in time**, so a streaming evaluation cannot read
the future — which is also what lets the replay reproduce batch scores exactly. Aggregation
is `torch.index_add_`: **no DGL, no PyTorch Geometric**.

The detector is a **temporal heterogeneous graph network**, inductive, trained on expected
rupee cost rather than cross-entropy — the business objective *is* the training objective.
Its relation set and its gate were **selected on calibration**, not chosen by taste.

→ **[Full architecture, and which parts are learned](docs/architecture.md)**

## Where this would run

**This section is a design proposal, not something that has been built.** The numbers in it
are measured; the placement is reasoned. Nothing here talks to a payment system, and no
part of this repository can.

Koronis is **post-authorisation by construction** — it reads the authorisation outcome,
which exists only after an attempt is submitted. It therefore cannot block the attempt it
learns from, and the value is in the attempts that follow. That single fact decides where
it belongs: **the model never sits on the checkout's critical path.**

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 26, "rankSpacing": 34}}}%%
flowchart TB
    CO["Checkout"] --> PRE{"pre-auth risk check<br/>key-value lookup by device · IP · BIN<br/><b>no model call</b>"}
    PRE --> ACQ["Acquirer / card network"]
    ACQ -->|"authorisation outcome"| BUS["Event stream, partitioned by BIN"]
    BUS --> SC["<b>Koronis scorers</b> — 9 workers<br/>0.91 ms per event · 1 h window<br/>shared device / IP replicated across shards"]
    SC --> CON["<b>Consolidation</b> — online union-find<br/>count-min sketch, 4 MB fixed"]
    CON --> POL["<b>Incident risk → exposure forecast → cost-optimal action</b>"]
    POL --> KV["Decision store<br/>entity → action, expires with the window"]
    POL --> AQ["Analyst queue<br/>+ audit dossier"]
    KV -.->|"read by the next attempt"| PRE
```

Scoring consumes the authorisation stream asynchronously. Decisions are written to a
key-value store keyed by entity, and the pre-auth path performs a **lookup** — the same
shape as the velocity counters most gateways already run there.

**Sizing, from measured constants.** At 0.91 ms per event and 1,095 events/sec per worker,
nine workers cover ~9,900 events/sec. Memory is bounded by the window rather than by
traffic history — measured, because it was not always true: the scorer's caches peak at
**a median peak of 1,859.5 rows against 6,310 events seen**, and the frequency state is fixed at 4 MB however
many distinct entity values pass through. That last part matters, because a card-testing
campaign mints a fresh card id per attempt.

**When the stream is not clean.** Real authorisation streams carry nulls and malformed
rows, so the failure behaviour is injected and measured rather than assumed
([`koronis.cli resilience`](koronis/cli.py)). An event that cannot be scored honestly is
**quarantined and counted, never scored anyway** — 5% NaN amounts cost 314 events and drop
recall 0.952 → 0.884, loudly. A missing entity links to **nothing**: the same missing data
with a placeholder substituted upstream instead manufactures **958 device links at a 1%
rate and 19,792 at 10%**, to devices that do not exist. Full table:
[Failure behaviour](docs/evaluation.md#failure-behaviour).

**Is the gap a modelling gap?** No — it is an information gap, and that was tested rather
than argued. Scaling a per-transaction learner does not close it: the best per-event result
in a capacity sweep across two model families is the *smallest* GBDT (1,550 parameters,
PR-AUC 0.2891), and adding capacity makes it worse, down to 0.2233 at 1,020,000. The graph
model reaches **0.9915 with 9,171 parameters**.
[The per-event ceiling](docs/evaluation.md#the-per-event-ceiling) ·
[why there is no language model in here](docs/ai-decisions.md).

**Partitioning is a modelling decision, and it was measured, not assumed.** Splitting a
graph deletes edges. Routing by BIN preserves the relation that
[carries the signal](docs/evaluation.md#which-entity-type-carries-the-signal); replicating
the events whose device or IP actually recurs
[restores recall from 0.72 to 0.99 at eight shards](docs/evaluation.md#recovering-the-edges-a-partition-deletes),
for 2.421× the scoring work. Reported honestly: **every partitioned configuration is worse
than not partitioning at all**, so more workers is a cost to justify, not a free lever.

**What each recommendation maps to.** `monitor` is no action. `rate_limit` throttles the
linked entities. `step_up` is additional verification scoped to the implicated subgraph
rather than to all traffic. `hold_review` queues for a person, with the
[audit dossier](docs/evaluation.md#decision-layer) as the evidence. This is a signal
source for an existing risk stack, not a replacement for one — it covers a single class of
loss that per-transaction scoring provably cannot see above a spread.

**What would have to be true first**, stated plainly: the model is fitted once and frozen,
which is what makes the hold-out honest and is not what a deployment wants; the data is
semi-synthetic, and BIN — the relation carrying most of the signal — is the one whose real
behaviour differs most from the simulation; and any deployment would begin in shadow mode,
scoring and logging without acting, until its false-positive rate had been measured on real
traffic. [Full limitations.](docs/limitations.md)

## Reproducing everything

Every experiment writes to `results/`, and **every number in this repo and on the demo site
is read from there** — nothing is transcribed by hand.

<details>
<summary><b>All experiments</b></summary>

```bash
.venv/bin/python -m koronis.cli ablation        # headline detector comparison
.venv/bin/python -m koronis.cli seeds           # 10 trials, median + across-run range
.venv/bin/python -m koronis.cli frontier        # the per-entity blind region, k >= n/tau
.venv/bin/python -m koronis.cli saturation      # spread pushed to k = n; why it is not measurable here
.venv/bin/python -m koronis.cli mechanism       # which mechanism carries the signal
.venv/bin/python -m koronis.cli relations       # which entity type carries the signal
.venv/bin/python -m koronis.cli architecture    # do the gate and the attention earn their place
.venv/bin/python -m koronis.cli online          # online consolidation vs the batch grouping
.venv/bin/python -m koronis.cli sharding        # does the graph survive being split across machines
.venv/bin/python -m koronis.cli replicate       # can replication recover what sharding deletes
.venv/bin/python -m koronis.cli select          # architecture selection on calibration, tested once
.venv/bin/python -m koronis.cli capacity        # width x depth grid, selected on calibration
.venv/bin/python -m koronis.cli aperture        # merchant view vs gateway view
.venv/bin/python -m koronis.cli resilience      # fault injection: how it fails, measured
.venv/bin/python -m koronis.cli ceiling         # can any per-event model close the gap
.venv/bin/python -m koronis.cli feature_parity  # the one disclosed baseline asymmetry
.venv/bin/python -m koronis.cli incidents       # alerts -> incidents -> forecast -> action
.venv/bin/python -m koronis.cli drift           # traffic-profile transfer stress test
.venv/bin/python -m koronis.cli bin_concentration  # a legitimate BIN giant component
.venv/bin/python -m koronis.cli latency         # precision / recall over time
.venv/bin/python -m koronis.cli replay          # causal event-by-event replay -> JSON
.venv/bin/python -m koronis.cli benchmark       # p50 / p95 per-event inference latency
python site/build.py                            # results/ -> docs/index.html
```

</details>

## What is measured, and where

| Question | Answer | Detail |
|---|---|---|
| Does it detect what velocity rules cannot? | 0.997 PR-AUC vs 0.062, on a hold-out spread past the boundary | [Evaluation → protocol](docs/evaluation.md#protocol) |
| What does a false positive cost? | costed in rupees; 10 FPs against a GBDT's 476 | [Evaluation → decision layer](docs/evaluation.md#decision-layer) |
| Which mechanism carries the signal? | outcome buys earliness, the graph buys precision | [Evaluation](docs/evaluation.md#which-mechanism-carries-the-signal) |
| Do the architectural claims hold? | one did not — the heterophily gate was **net-negative at two layers, noise at three**, and selection removed it | [Evaluation](docs/evaluation.md#does-the-architecture-earn-its-place) |
| Was the model sized, or just chosen? | sized — 32×3 selected on calibration from a 9-point grid, and it held up on test | [Evaluation](docs/evaluation.md#closing-the-loop-selecting-an-architecture-without-touching-test) |
| Is a gateway's wider view worth anything? | measured: the gap grows with the number of merchants | [Evaluation](docs/evaluation.md#vantage-point-one-merchant-or-the-whole-gateway) |
| Does it survive a different merchant? | flagged on all four shifted profiles, including a legitimate BIN giant component; the guardrail is **experimental** | [Evaluation](docs/evaluation.md#traffic-profile-transfer-stress-test) |
| Can it run online? | 0.91 ms p50; streaming reproduces batch scores exactly | [Evaluation](docs/evaluation.md#streaming-and-inference-latency) |
| Is the *whole* pipeline causal? | yes — consolidation too, via a sliding count-min sketch in fixed memory | [Evaluation](docs/evaluation.md#making-consolidation-causal) |
| Does it survive being split across machines? | measured — and PR-AUC and rupees disagree about which routing is better | [Evaluation](docs/evaluation.md#does-the-graph-survive-being-split-across-machines) |
| Can the loss be recovered? | yes — replication restores recall 0.65 → 0.99 at sixteen shards, and costs less than not replicating at every shard count | [Evaluation](docs/evaluation.md#recovering-the-edges-a-partition-deletes) |
| What broke? | 22 defects, and **7 published claims withdrawn** | [Engineering log](docs/engineering-log.md#claims-withdrawn) |

## Limitations

This is a **semi-synthetic proof of concept**, not production fraud detection.

- **Background traffic and campaigns are generated.** The IEEE-CIS loader exists and the
  real data was profiled, then deliberately not used — its native density would reintroduce
  a defect this project already fixed.
- **It will not catch** an attacker using genuinely fresh infrastructure for every attempt.
  That limit is real, and it is also the point: driving `k → n` costs one device, IP and BIN
  per attempt.
- **Only attempts that reach authorisation are in scope.** The outcome is one of the two
  mechanisms, so a flow where a step-up challenge precedes authorisation produces no outcome
  to read, and a campaign confined to one is invisible — not detected poorly, not at all.
  Which flows those are varies by market and acquirer; this prototype models none of that.
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
| [AI decisions](docs/ai-decisions.md) | every model choice, the ones rejected, and the measurement behind each |
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
