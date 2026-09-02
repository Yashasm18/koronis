# Evaluation

Every number here is written by an experiment in [`koronis/cli.py`](../koronis/cli.py)
into `results/`, and read from there by the README and the demo site. Nothing is
transcribed by hand.

[← back to the README](../README.md)

### Protocol

Three splits, and the threshold never sees the test set:

| split | contents | used for |
|---|---|---|
| train | `k ∈ {4, 12, 30}` × `camouflage ∈ {0, 0.5, 1}` | fitting model weights |
| calibration | same distribution as train, different draw | choosing the operating threshold, then frozen |
| test | `k = 60`, `camouflage = 1.0`, unseen entities | reported numbers only |

All three derive from one run seed, so repeating the run resamples every split together.
Training contains only campaigns concentrated enough that a tuned velocity engine still
catches them (`k ≤ 30`, below the `n/τ = 50` boundary at `n = 400`); the test campaign is
spread past that
boundary, fully camouflaged, with unseen entities, so the hold-out is **extrapolation**,
not interpolation. Scores are used raw — rescaling each split by its own maximum would
let every split redefine what a score means.

### Decision layer

Event alerts are not tasks for a fraud team — they are one campaign. Koronis consolidates
them and recommends the intervention with the lowest expected cost, not the one matching
the highest risk score:

```
402 event alerts  →  7 incidents  →  2 actions recommended
```

One test stream, end to end — the same stream the demo replays. Of those two actions one was wrong: a two-attempt incident rate-limited where the oracle would have left it alone. The other five incidents are correctly left on `monitor`.

Two concurrent rings stay two incidents: alerts are linked only through entity values
covering under 2% of the whole stream, so sharing `gmail.com` links nothing. Action
figures are declared assumptions about a merchant workflow, in
[`koronis/incident.py`](../koronis/incident.py):

| action | friction (genuine) | harm (false) | stops | analyst |
|---|---:|---:|---:|---:|
| monitor | ₹0 | ₹0 | 0% | — |
| rate-limit | ₹120 | ₹400 | 55% | — |
| step-up verification | ₹350 | ₹1,800 | 85% | — |
| hold + review | ₹900 | ₹6,000 | 97% | 12 min |

The chosen action minimises
`friction + risk × forecast_exposure × (1 − stops) + (1 − risk) × false_harm`.

Each consolidated incident carries a plain-text **audit dossier** —
`koronis.incident.dossier()`, also printed by `koronis.cli incidents` and shown in the
console — that reformats the fields already computed (spread and per-entity load,
consolidation, recalibrated risk, the forecast, and the chosen action versus keeping
`monitor`). For the held-out campaign incident:

```
Spread          395 alerted attempts · 60 devices · 60 IPs · 60 BINs
                per-entity load 6.6/device, 6.6/IP, 6.6/BIN  (binding velocity τ = 8)
Consolidation   395 event alerts → 1 incident · link window 900 s
Incident risk   1.000  (recalibrated logistic on calibration incidents)
Forecast        decided after 12 events · remaining P50 309 [P90 557]
                exposure P50 ₹22,584 [P90 ₹40,658]
Recommendation  Hold + analyst review — hold matching attempts, queue for review
                expected ₹1,685  vs ₹22,557 to keep monitoring
                oracle action: hold_review  (matches)
```

### Exposure forecast

Choosing an action needs an estimate of what inaction would cost. Offline that can be read
off the campaign log; a live system cannot, so a policy built on the true remaining count
is an oracle upper bound, not a product. `causal_policy` sees only the first 12 events of
an incident plus a forecast — never the true remaining count or the ground-truth label.

At each incident snapshot a quantile model predicts, from observed signals only, how many
more alerted events will join the incident. That target is deliberately label-free; the
incident risk model answers whether the incident matters, and the policy multiplies the
two:

```
expected remaining exposure  =  P(genuine) × forecast(remaining attempts) × ₹73
```

The upper quantile carries a conformal pad fit on a held-out subset of calibration
incidents; coverage is then evaluated on held-out test incidents. That split is **by
stream, never by snapshot row**, because snapshots of one incident are nested prefixes and
splitting by row inflates apparent coverage (91.8% by row vs. the figure below by stream).

| | measured |
|---|---|
| P90 interval coverage | 95.3% (target 90%) |
| Median absolute error, P50 | 104.0 attempts |
| Mean true remaining | 382.1 attempts |
| Fit / conformal streams | 4 (campaigns 2, 3, 4, 6) / 4 (campaigns 0, 1, 5, 7) |
| Snapshots | 84 calibration / 85 held-out |

The interval over-covers: conservative, which is the safe direction for a policy that
escalates on uncertainty, but not well calibrated with only four conformal streams.
Campaign length is varied across streams for this evaluation — with a fixed length,
"remaining" collapses to a constant minus what you have seen and the forecaster scores a
6.7-attempt median error while learning nothing.

### Policy comparison

Median across 8 independent test streams:

| policy | incidents actioned | false incidents | analyst minutes | merchant cost |
|---|---:|---:|---:|---:|
| always allow | 0.0 | 0.0 | 0.0 | ₹53,144 |
| always hold | 10.5 | 6.0 | 126.0 | ₹50,307 |
| event-by-event thresholding | 1.0 | 0.0 | 211.0 | ₹6,602 |
| **causal policy** *(forecast only)* | 1.0 | 0.0 | 12.0 | ₹3,405 |
| oracle policy *(upper bound)* | 1.0 | 0.0 | 12.0 | ₹3,145 |

Fractional counts are medians across an even number of streams. Event thresholding reaches
the same decision but hands an analyst 211 minutes of triage instead of 12 —
consolidation, not detection, is the difference. When the forecast interval is wide
relative to its median, the policy escalates to analyst review rather than automating.

**On the ₹0 regret this project once retracted.** An earlier version reported zero action
regret and treated it as a result; it was retracted when the cause turned out to be a
generator defect. That defect stayed fixed. On the current demo stream regret is
**₹520** with 6 of
7 actions matching the oracle — non-zero, which is what an honest
forecast-only policy should look like. Across the eight streams the oracle still leads,
₹3,145 against
₹3,405: not knowing the future still costs.

### Incident-level calibration

An event model with ECE 0.0025 does not give a calibrated incident probability for free —
events inside an incident are strongly dependent, and that dependence is the signal.
Incident risk is a separate model, fitted on 70 calibration incidents pooled across 8
streams and measured on 93 held-out incidents:

| predicted | observed | incidents |
|---:|---:|---:|
| 0.077 | 0.234 | 77 |
| 0.261 | 0.667 | 6 |
| 0.660 | 1.000 | 1 |
| 0.999 | 1.000 | 9 |

Separation is clean at the top (0.999 → 1.000 on 9 incidents). The bottom bin is
under-confident by a factor of three (0.077 → 0.234), and it holds 77 of the 93 held-out
incidents, so that is the bin that matters: an incident this model calls quiet is roughly
three times more likely to be real than it says. The middle is barely determined at all —
two bins holding 6 and 1 incidents, and one holding none — which is the honest state of a
model fitted on 70 incidents, and is reported rather than smoothed over.

### Which entity type carries the signal

Dropping each relation in turn and re-fitting under the same protocol (5 trials, medians;
`python -m koronis.cli relations`):

| variant | PR-AUC | precision | recall | false positives | PR-AUC change |
|---|---:|---:|---:|---:|---:|
| all relations | 0.995 | 0.971 | 0.988 | 12 | — |
| no `device_id` | 0.996 | 0.966 | 0.993 | 14 | +0.0012 |
| no `ip_id` | 0.995 | 0.964 | 0.980 | 15 | +0.0004 |
| no `bin_id` | 0.967 | 0.961 | 0.868 | 14 | -0.0281 |
| no `email_domain` | 0.998 | 0.977 | 0.998 | 9 | +0.0033 |

Shared BIN ranges carry almost all of it — remove that relation and recall collapses from
0.988 to 0.868, the only change here large enough to matter. Removing `email_domain`
*improves* PR-AUC and cuts false positives from 12 to 9: it contributes noise, not
evidence, and it is not among the relations the detector consumes. Removing `device_id` or
`ip_id` moves PR-AUC by +0.0012 and +0.0004 — within trial-to-trial noise — while raising
false positives slightly, so neither is carrying weight the BIN relation is not. This retires an earlier claim based on the
model's per-relation attention weights — attention says where a model looked, not what it
gained.

Acting on that finding here would have meant selecting an architecture on test results,
which is the leakage this project refuses. It was therefore reported and left alone until
it could be done properly, and then it was:
[calibration-based selection](#closing-the-loop-selecting-an-architecture-without-touching-test)
removed `email_domain` and the gate, and that leaner model is now the default. This table
keeps all four relations and the gate on, so it measures the question it was built to
measure rather than the model that shipped.

### Which mechanism carries the signal

Removing each mechanism in turn under the same protocol (5 trials, medians;
`python -m koronis.cli mechanism`):

| variant | PR-AUC | precision | recall | false positives | first alert |
|---|---:|---:|---:|---:|---:|
| **`koronis_full`** | 0.998 | 0.976 | 0.988 | 10 | 0.0 s |
| `no_edges` — event features only | 0.380 | 0.451 | 0.955 | 466 | 0.0 s |
| `no_approved` — graph only | 0.916 | 0.903 | 0.812 | 35 | 66.6 s |
| `no_edges` + `no_approved` | 0.061 | 0.000 | 0.000 | 11 | never |

The authorisation outcome buys earliness (alert at t = 0, but 0.452 precision and 464
false positives). The graph buys precision (first alert moves to 66.6 s and recall drops
to 0.695, but false positives fall to 53 and precision rises to 0.840). Together: 15 false
positives at 0.964 precision, a 31× reduction over event-features-alone; with both removed
the model never fires, so no third signal source is hiding in the features.
`tests/test_first_event.py` pins the structural claim that the opening attempt has zero
campaign-derived links and cannot acquire any.

### Streaming and inference latency

The detector runs as a strictly causal stream: `StreamingKoronis.push(event)` scores one
event at a time and **reproduces batch scores exactly** (asserted to `1e-5` in
`tests/test_stream.py`), which falls out of the backwards-in-time edge rule. Measured over
6,200 events after 200 warm-up, timing only `push`:

| p50 | p95 | p99 | mean | throughput |
|---:|---:|---:|---:|---:|
| 0.91 ms | 1.14 ms | 1.36 ms | 0.91 ms | ~1,095 events/sec |

**Per-event cost is flat in stream length**, by construction:

- **Time** — `O(R · D_max · L · d)` per `push`: `R = 4` relations, `D_max = 32` (the
  `max_degree` fan-in cap in [`graph/build.py`](../koronis/graph/build.py), which keeps the
  most recent neighbours), `L = 2` message-passing layers, `d = 32`
  hidden units. None of these depends on the number of events already seen, so measured
  latency holds at ~0.91 ms p50 regardless of stream length.
- **Space** — `O(W · λ · d)`: the `window_s = 3600 s` span times the arrival rate `λ`.
  `bucket.popleft()` evicts events once `t − t_event > W`, so memory tracks active window
  occupancy, not cumulative volume. `test_stream.py::test_window_bounds_memory` asserts
  the buffered total stays well below an unbounded stream's.

On the held-out stream Koronis alerts on the campaign's opening attempt — but that alert
is a declined authorisation with no campaign neighbours yet, weak on its own; the graph is
what makes the following attempts actionable.

### Closing the loop: selecting an architecture without touching test

Three components had been measured as net-negative or neutral — the device relation and the
email relation in the [per-relation ablation](#which-entity-type-carries-the-signal), and
the heterophily gate in the [architecture ablation](#does-the-architecture-earn-its-place).
Every one of those measurements used the **test** split, so acting on them directly would
have been selecting an architecture on test results: the leakage this project refuses
elsewhere. Each was therefore reported and left in place.

This closes it properly. Eight candidates are re-scored on **calibration only**, the winner
is chosen there, and test is read once at the end to report what the choice was worth.

Two independent calibration draws are used, not one: the threshold is fitted on the first
and the selection score measured on the second. Scoring a candidate at a threshold fitted
on the same events flatters whichever variant suits that draw — the same mistake as tuning
on test, one level down.

`python -m koronis.cli select`, 5 trials, medians. **Only the first column may inform the
choice:**

| candidate | calibration cost *(selects)* | test cost | test PR-AUC | precision | recall | FPs |
|---|---:|---:|---:|---:|---:|---:|
| `full` | ₹779 | ₹685 | 0.9950 | 0.971 | 0.988 | 12 |
| `no_device` | ₹786 | ₹852 | 0.9962 | 0.966 | 0.993 | 14 |
| `no_email` | ₹520 | ₹713 | 0.9983 | 0.977 | 0.998 | 9 |
| `no_device_no_email` | ₹586 | ₹513 | 0.9970 | 0.975 | 0.993 | 10 |
| `no_gate` | ₹979 | ₹979 | 0.9950 | 0.954 | 0.993 | 19 |
| `no_device_no_gate` | ₹819 | ₹739 | 0.9963 | 0.970 | 0.993 | 12 |
| `no_email_no_gate` | ₹466 | ₹645 | 0.9978 | 0.976 | 0.988 | 10 |
| `lean` | ₹619 | ₹633 | 0.9962 | 0.975 | 0.993 | 10 |

**Chosen on calibration: `no_email_no_gate`** — drop the email relation and
the heterophily gate. Calibration cost falls from ₹779 to
₹466.

**It held up.** On test, cost falls from ₹685 to ₹645, PR-AUC rises 0.9950 → 0.9978,
precision 0.971 → 0.976, recall 0.988 → 0.988, and false positives fall
12 → 10. Every candidate that
removes something beats the full model, so the earlier ablations were reading a real signal
rather than noise.

**Selection is not free, and the table shows it.** The variant with the lowest *test* cost
is `no_device_no_email` (₹513), not the one calibration chose. Picking on
held-out data costs something against picking with hindsight — that gap is the honest price
of not cheating, and reporting the calibration-chosen variant rather than the test-optimal
one is the whole point of the exercise.

### Does the graph survive being split across machines?

Throughput is not the hard part — per-event cost is constant in stream length, so more
traffic is more workers. The hard part is that **partitioning a graph deletes edges**: route
a device to one shard and a co-occurring IP to another and that edge never forms. Sharding
is a modelling decision, not an infrastructure detail.

Events are routed by hashing a field, so **the field you route on is the one relation
preserved perfectly** and every other survives only by collision. The prediction stated in
[`eval/sharding.py`](../koronis/eval/sharding.py) before the run came from the
[per-relation ablation](#which-entity-type-carries-the-signal): BIN carries the signal, so
BIN-routing should hold up best and random should decay fastest.

`python -m koronis.cli sharding`, frozen model and threshold, only the partition varying:

| shards | PR-AUC random | PR-AUC bin | PR-AUC device | ₹ random | ₹ bin | ₹ device |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.999 | 0.999 | 0.999 | ₹645 | ₹645 | ₹645 |
| 2 | 0.993 | 0.996 | 0.997 | ₹1,676 | ₹1,740 | ₹1,372 |
| 4 | 0.958 | 0.991 | 0.988 | ₹3,592 | ₹4,262 | ₹2,558 |
| 8 | 0.895 | 0.973 | 0.978 | ₹6,628 | ₹8,423 | ₹4,711 |
| 16 | 0.800 | 0.962 | 0.962 | ₹9,198 | ₹10,686 | ₹7,551 |

**Entity routing beats random decisively, and the specific prediction was still wrong.**
At 16 shards both entity keys hold 0.962 against random's
0.800 — but BIN does not beat device, so the reason given for
preferring BIN does not survive.

**The two entity keys fail in opposite directions, which PR-AUC hides.** At 16 shards:

| routing | precision | recall | false positives |
|---|---:|---:|---:|
| BIN | **0.970** | 0.645 | **8** |
| device | 0.691 | **0.983** | 176 |

BIN-routing cuts the campaign into fragments, so recall falls to 0.645 —
but every fragment that fires is genuinely coordinated, and false positives stay at
8, barely above the undivided run. Device routing
co-locates the heavy legitimate device reuse in background traffic, so recall survives and
precision collapses.

**And the two metrics disagree about which is better.** BIN has far the better PR-AUC
(0.962 against 0.800) and is the more
expensive of the two (₹10,686 against
₹9,198), because at these prices a missed attempt
costs more than a false alert and BIN routing misses the most. A ranking metric cannot see
that. Reported as PR-AUC alone, BIN would look like the clear choice; priced, it is the
worst of the three.

**What is unambiguous: sharding costs accuracy whichever key you pick.**
₹645 undivided against
₹7,551 for the best 16-shard option.

**A defect this surfaced.** The first run reported entity routing as completely free — flat
PR-AUC at every shard count. The tell was `largest_shard_share = 0.9375` no matter how many
shards were requested. Routing hashed ids with `int.from_bytes(..., "little")`, and that
value modulo a power of two depends only on the *first character*: every background BIN
began `b`, every campaign entity `c`, so they landed on two shards and nothing was
partitioned. With CRC32 the result above appeared. A test asserts no strategy puts a
disproportionate share on one shard.

### Recovering the edges a partition deletes

BIN routing keeps precision and bleeds recall as shards multiply, because device and IP
edges are cut when their endpoints land apart. That loss is not inherent to partitioning —
only to insisting every event live in exactly one place. An event can be **copied** to the
shard where its other relations would find company.

The prediction, stated in [`eval/sharding.py`](../koronis/eval/sharding.py) before the run:
entity frequencies are heavy-tailed, so most values appear once and can form no edge at all.
Replicating only events whose device or IP actually recurs should restore most lost edges
while copying a minority of traffic — and campaign entities, shared by construction, should
be copied preferentially.

`python -m koronis.cli replicate`, frozen model and threshold. `d` is the degree bar: an
event is copied only if the value it shares occurs at least `d` times.

| shards | recall, BIN only | recall, + repl | ₹ BIN only | ₹ + repl (d≥2) | ₹ + repl (d≥4) | events scored |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.950 | **0.988** | ₹1,740 | ₹725 | ₹805 | 1.675× / 1.468× |
| 4 | 0.865 | **0.988** | ₹4,262 | ₹1,445 | ₹1,365 | 2.147× / 1.747× |
| 8 | 0.723 | **0.990** | ₹8,423 | ₹2,772 | ₹2,412 | 2.421× / 1.898× |
| 16 | 0.645 | **0.990** | ₹10,686 | ₹4,532 | ₹3,212 | 2.564× / 1.976× |

**The mechanism works, and works completely.** At 16 shards recall goes from
0.645 to
**0.990** — the campaign is fully
recovered — for 2.564× the scoring work.
That the factor is under three rather than sixteen is the heavy tail doing exactly what was
predicted: only shared entities are worth copying, and most are not.

**Replication is the cheapest routing at every shard count**, beating BIN alone, random and
device routing alike. It costs precision — copying high-degree entities co-locates the
legitimate dense clusters too — but at these prices recovering the missed attempts is worth
more than the extra alerts.

**The stricter degree bar is better at scale, which was not predicted.** `d≥4` copies less
(1.976× against
2.564× at 16 shards) *and* costs less
(₹3,212 against
₹4,532). Copying
entities shared by only two or three events buys edges that are mostly noise, and pays for
them twice — in compute and in false positives.

**And nothing beats not partitioning.** The undivided stream costs
₹645; every routing strategy at every shard count is
worse. That is the honest summary of this line of work: sharding a coordination detector
always costs something, replication buys back most of it, and the right first move is a
bigger window per worker rather than more workers.

### Making consolidation causal

The detector was always strictly online — `StreamingKoronis.push` reproduces batch scores
exactly. Consolidation was not. `build_incidents` decides whether an entity value is too
common to link on by counting it with `value_counts()` **across the whole frame**, which
includes traffic that had not happened when the alert fired. Half the pipeline was causal
and half was not.

`StreamingIncidents` closes that. Frequencies come from a **sliding count-min sketch** fed
by every event as it passes, so the cap at time `t` reflects only what was known at `t`.
Memory is fixed by the sketch dimensions rather than by how many distinct entity values the
stream contains — which matters, because a card-testing campaign mints a fresh card id per
attempt, so cardinality is exactly the thing that grows without bound.

**The sketch's error is one-sided, and it points the safe way.** A count-min sketch never
underestimates. An inflated count makes a value look *more* common, and a value that looks
common is excluded from linking — so the failure mode is a missed link, fragmenting one
campaign, rather than a false link merging two rings into a single incident with one action
for two different attackers. Fragmentation an analyst can recover from; a wrong merge
silently hides an attack.

`python -m koronis.cli online`, 6 held-out streams, batch / online:

| stream | campaign attempts | incidents | purity of largest | campaign recall |
|---:|---:|---:|---:|---:|
| 0 | 150 | 41 / 41 | 1.000 / 1.000 | 0.700 / 0.700 |
| 1 | 240 | 10 / 10 | 1.000 / 1.000 | 0.925 / 0.925 |
| 2 | 380 | 7 / 7 | 1.000 / 1.000 | 0.987 / 0.987 |
| 3 | 520 | 10 / 10 | 1.000 / 1.000 | 0.983 / 0.983 |
| 4 | 700 | 7 / 7 | 1.000 / 1.000 | 0.993 / 0.993 |
| 5 | 900 | 5 / 5 | 1.000 / 1.000 | 0.994 / 0.994 |

**Median: 8.5 incidents either way, purity
1.000 / 1.000, recall 0.9848 /
0.9848.** Making the decision causal costs nothing measurable here, in
**4 MB** of sketch that does not grow with the stream.

The two are not expected to agree exactly, and where they differ the batch version is not
the ground truth — it is the one using the future.

**A defect this surfaced.** The first implementation fed all four relations into a single
shared sketch. The share denominator then counted one add *per relation per event*, so
every share came out four times too small: a domain covering ~6% of the stream measured
0.015 against a 0.02 cap, slipped through, and bridged two unrelated rings into one
incident — defect 7 reappearing through a new mechanism. Each relation now has its own
sketch, which also stops a device id and an email domain colliding in one counter, where
the collision is noise between incomparable namespaces. A test asserts two concurrent rings
stay apart online.

### Does the architecture earn its place?

Variants are departures from the selected architecture, so the heterophily gate appears
here as something to **add back**. 5 trials, medians:

| camouflage | selected | + gate | uniform relation attention | two layers |
|---:|---:|---:|---:|---:|
| 0.0 | 0.9999 | 1.0000 | 0.9999 | 0.9999 |
| 0.5 | 0.9996 | 0.9996 | 0.9996 | 0.9993 |
| 1.0 | 0.9978 | 0.9983 | 0.9978 | 0.9968 |

**The gate no longer changes anything either way, and that is a correction to what this
section used to say.** At two layers it was measurably net-negative — better without it on
12 of 15 seed × camouflage cells, false positives falling 24 to 17. At three layers the
difference is noise: adding it back wins on 2 of 5 seeds at full camouflage, a median of
+0.0005 against a seed-to-seed spread running 0.992 to 0.999. Whatever the gate was doing,
a third round of aggregation absorbs it.

The honest statement is therefore narrower than before. The gate was harmful at the depth
it was first measured at, is neither here nor there at the depth finally selected, and was
removed by [calibration-based selection](#closing-the-loop-selecting-an-architecture-without-touching-test)
rather than by this table. **Relation attention remains a wash**, consistent with the
[per-relation ablation](#which-entity-type-carries-the-signal). **Depth is the one
component that still earns its place**, and it is measured properly below.

### Was the model sized, or just chosen?

Width, depth, epochs and learning rate were defaults for most of this project's life, and a
default is not a decision. This sweeps the two that govern capacity, on the selected
relation set, under the same protocol as every other selection here: scored on calibration,
test read once at the end.

The prediction, stated in [`cli.py`](../koronis/cli.py) before the run: the input is six
features and the signal is structural rather than a rich per-event representation, so
**width should saturate almost immediately**; **depth should matter more**, since a second
hop is what reaches coordination that survives camouflage; and **a third layer should not
help and may hurt**, because repeated neighbourhood averaging drives representations
together — over-smoothing — and dense legitimate traffic is a good place for that to bite.

`python -m koronis.cli capacity`, 5 trials, medians. **Only the calibration column may
inform the choice:**

| hidden | layers | params | calibration cost *(selects)* | test cost | test PR-AUC | FPs |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 427 | ₹4,544 | ₹7,278 | 0.8998 | 35 |
| 16 | 2 | 1,487 | ₹1,617 | ₹1,848 | 0.9904 | 18 |
| 16 | 3 | 2,547 | ₹786 | ₹1,106 | 0.9968 | 17 |
| 32 | 1 | 843 | ₹3,816 | ₹5,674 | 0.9254 | 32 |
| 32 | 2 | 5,007 | ₹673 | ₹892 | 0.9968 | 15 |
| 32 | 3 | 9,171 | ₹466 | ₹645 | 0.9978 | 10 |
| 64 | 1 | 1,675 | ₹3,244 | ₹4,398 | 0.9505 | 24 |
| 64 | 2 | 18,191 | ₹772 | ₹706 | 0.9973 | 12 |
| 64 | 3 | 34,707 | ₹673 | ₹732 | 0.9988 | 11 |

**Two of three predictions held; the third was wrong.**

*Width saturates.* 32 beats 64 at every depth, and 64 never wins anything — doubling the
width doubles the parameters and buys nothing, which is what a structural signal over six
features should look like.

*Depth dominates width.* The spread in calibration cost across depth is ₹3,143; across
width it is ₹845. One layer to two is worth more than every width change combined.

*A third layer helps, and over-smoothing does not appear.* **It beats two layers on 15 of
15 seed × width cells.** Coordination surviving camouflage evidently reaches further than
two hops in this graph. The prediction was reasonable; the measurement disagrees.

**Chosen on calibration: 32 hidden units, 3 layers,
9,171 parameters.** It held up on test — cost
₹892 → ₹645 against the
previous default — and is now the default. The model is still small; it is small on purpose
now rather than by inheritance.

**One thing this broke underneath.** The streaming replay cached only layer-1 outputs, so it
reproduced batch scores exactly for a two-layer model and silently stopped at three. Parity
is a property of the backwards-in-time edge rule, not of any particular depth, so the cache
now holds every intermediate layer and a test asserts parity at one, two, three and four
layers.

### Vantage point: one merchant or the whole gateway

A merchant sees its own checkout. A gateway sees thousands at once, and a
card-testing ring does not confine itself to one of them — spreading across
merchants is simply another axis on which per-merchant counters stay quiet.

The prediction was stated in [`eval/aperture.py`](../koronis/eval/aperture.py) before the
run. A campaign of `n` attempts split evenly over `M` merchants leaves `n/M` attempts in
any one merchant's view. Co-occurrence goes as attempts squared over spread, so a single
merchant sees `(n/M)²/k` pairs where the pooled stream sees `n²/k` — about **M times less
signal**. The merchant-scoped view should therefore decay as `M` grows, while the pooled
view sees the stream it would have seen anyway.

Same campaign, same frozen model, same frozen threshold; the only variable is how much of
the stream the detector may see at once (`python -m koronis.cli aperture`):

| merchants | gateway PR-AUC | merchant PR-AUC | gap | largest merchant's share of the campaign |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | **0.000** | 100% |
| 2 | 1.000 | 0.999 | 0.001 | 50% |
| 4 | 0.998 | 0.993 | 0.006 | 27% |
| 8 | 0.996 | 0.971 | 0.025 | 14% |
| 16 | 0.993 | 0.924 | **0.070** | 8% |

**At `M = 1` the two views are the same stream by construction, and they score
identically** — that is the experiment's control, and a test asserts it. From there the
gap opens monotonically, as predicted — and faster than linearly in `M`, widening from
0.001 at two merchants to 0.070 at sixteen. Recall tells the same story: the gateway view
stays between 0.985 and 0.995 across the sweep while the merchant view falls from 0.995 to
0.938.

Both views degrade somewhat as `M` grows, because more merchants means more legitimate
traffic and more chances to false-positive; the merchant view degrades about three times
faster. **Honest limit: even at `M = 16` the merchant-scoped view still detects this
campaign.** This measures a widening gap, not a blindness boundary — it says the wider
aperture is worth something and quantifies how much, not that a merchant alone is
helpless.

Entity ids are namespaced per merchant. Without that, two merchants would reuse the same
`d17` and the pooled graph would link strangers — the gateway view would then win on an
artefact. A test asserts no background entity is shared across merchants.

### The per-event ceiling

Every headline comparison here is against per-transaction models, so the obvious objection
is that the baselines were simply too small — that a bigger learner, or a more fashionable
one, would close the gap. That is testable rather than arguable
(`python -m koronis.cli ceiling`, 3 trials, medians). The per-event feature set is held
fixed while capacity is scaled across two families with different inductive biases, on the
**same 60-epoch budget as every published number**.

| family | capacity | parameters | PR-AUC |
|---|---|---:|---:|
| per-event GBDT | 50 trees × 31 leaves | 1,550 | **0.2891** |
| per-event GBDT | 300 trees × 31 leaves | 9,300 | 0.2572 |
| per-event GBDT | 1500 trees × 63 leaves | 94,500 | 0.2244 |
| per-event GBDT | 4000 trees × 255 leaves | 1,020,000 | 0.2233 |
| per-event net | 8 wide × 1 deep | 219 | 0.1621 |
| per-event net | 32 wide × 2 deep | 5,007 | 0.2370 |
| per-event net | 32 wide × 3 deep | 9,171 | 0.3100 |
| per-event net | 128 wide × 3 deep | 134,931 | **0.3279** |
| graph net | 8 wide × 1 deep | 219 | 0.6775 |
| graph net | 32 wide × 2 deep | 5,007 | 0.9666 |
| graph net | 32 wide × 3 deep | 9,171 | **0.9915** |
| graph net | 128 wide × 3 deep | 134,931 | 0.9894 |

**Capacity does not buy the gap.** The best per-transaction result in the whole sweep is
the *smallest* GBDT — 1,550 parameters at 0.2891 — and adding capacity makes it steadily
worse, down to 0.2233 at 1,020,000. The per-event network plateaus around 0.33 and stays
there through a 15× parameter increase. Meanwhile the graph model reaches **0.9915 with
9,171 parameters**, and 14× more parameters (0.9894 at 134,931) does not improve it either.
The selected size is the best size in both directions.

So the gap is an **information gap, not a modelling gap**. What a single authorisation
contains does not identify a card-testing attempt, because the attacker controls everything
in it except whether the card authorises — and one decline is unremarkable. The signal is
the relation between attempts, and a model that never looks across attempts cannot reach it
however large it is.

This is also the measured answer to "why is there no language model in here". A transformer
reading one transaction is another per-event model, and the ceiling above is a property of
the row, not of who is reading it. A model given the *neighbourhood* is no longer a
per-event model — it is doing the graph's job, in a per-event budget measured at 0.909 ms.
See [AI decisions](ai-decisions.md).

**Two runs were discarded before this table.** The first used the 40-epoch default rather
than the 60 epochs `_fit_all` gives every published model; an unfair training budget would
have flattered the conclusion in exactly the direction the conclusion points. The 256-wide
row is also omitted above: both families collapse there (0.0596 and 0.0478), which is
divergence under the shared budget rather than a ceiling, and it is left in
`results/ceiling.csv` rather than quietly dropped.

### Failure behaviour

Everything above assumes a clean stream. Real authorisation traffic carries nulls,
malformed rows and fields that go missing at a boundary, so the failure behaviour is
injected and measured rather than asserted (`python -m koronis.cli resilience`, medians
over 4 held-out streams, model and threshold frozen throughout).

Three defects were found this way, by probing the live streaming path rather than reading
it. All three were **silent** — the stream kept running and kept returning answers:

| fault | rate | quarantined | NaN scores | invented device links | campaign recall | peak cache rows |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 0% | 0 | 0 | 0 | 0.952 | 1,859 |
| `null_device` | 1% | 0 | 0 | **0** | 0.952 | 1,859 |
| `placeholder_device` | 1% | 0 | 0 | **958** | 0.952 | 1,859 |
| `null_device` | 10% | 0 | 0 | **0** | 0.936 | 1,859 |
| `placeholder_device` | 10% | 0 | 0 | **19,792** | 0.948 | 1,859 |
| `nan_amount` | 5% | 314 | 0 | 0 | 0.884 | 1,773 |
| `dropped_approved` | 5% | 314 | 0 | 0 | 0.884 | 1,773 |
| `entity_explosion` | 100% | 0 | 0 | 0 | 0.577 | 1,859 |

**An event that cannot be scored is quarantined, not scored anyway.** A non-finite feature
used to produce a NaN score, and `NaN >= threshold` is `False` in IEEE arithmetic — so the
event reported itself as *no alert*. A missing field made the detector quietly stop
detecting. Now 5% NaN amounts cost 314 events and drop recall from 0.952 to 0.884: a real
loss, reported as a count with a reason rather than absorbed. Losing an event loudly is
recoverable; a fraud detector that says "no alert" when it means "I could not read this"
is not.

**Missing data cannot become evidence.** Entity values were interned with `str(value)`, so
a null became the key `"None"` and every event without a device fingerprint linked to every
other one. Null device IDs are ordinary in production — a browser blocking the fingerprint
— so this manufactures a ring out of absent data.

`placeholder_device` is the control that makes the point, and the reason the two rates are
swept either side of the link-share cap. It is the *same* missing data, except a
well-meaning upstream step replaced the null with a constant before the detector saw it.
At 10% the frequency cap already refuses to link on a value that common, so recall barely
moves — but **19,792 links were still drawn and still appear in the audit dossier**, telling
an analyst these attempts share a device when nothing shared anything. At 1% the cap is
silent, because the share is below its threshold, and `entity_key` is the only thing
standing between missing data and an invented ring. The null column is 0 at both rates.

**Memory tracks the window, not the traffic.** Nothing was ever evicted from the scoring
caches: 3,120 events through a 60-second window left 3,120 rows in every layer while 12
events were actually in scope, and the entity index kept a key for every distinct value
ever seen — which is precisely what a campaign minting a fresh entity per attempt inflates.
Both are evicted against the window now: 1,859 peak rows against 6,310 events seen.

**`entity_explosion` is the honest floor.** Give every attempt its own device and recall
falls to 0.577, because a third of the graph has been destroyed. The purity of the largest
incident stays 1.000 throughout: the detector loses signal, it does not invent any. That is
the same limit stated in [Limitations](limitations.md) — an attacker with genuinely fresh
infrastructure per attempt leaves no graph signal — arrived at from the other direction.

### Traffic-profile transfer stress test

These are synthetic merchant shapes, not real merchants. Surviving this is evidence the
detector is not tuned to one traffic profile; it is not evidence of production
cross-merchant transfer. Everything is fitted on the **base** profile and frozen before
any shifted traffic is scored. The three shifted profiles are declared in
[`koronis/profiles.py`](../koronis/profiles.py) before being run:

| profile | what it breaks |
|---|---|
| `subscription` | legitimate device and card reuse is high — dense co-occurrence is normal |
| `marketplace` | entities are diffuse — the graph is sparse and thresholds sit wrong |
| `flash_sale` | a legitimate burst — high volume and elevated declines, no attack |

Drift is measured by Population Stability Index, with the cut-off set to the 95th
percentile of PSI between disjoint base samples:

| profile | median PSI | flagged | largest shift | what changed |
|---|---:|---:|---|---|
| base | 0.141 | 0 / 3 | reuse_bin | — |
| `subscription` | 0.584 | 3 / 3 | reuse_device | ✓ device reuse |
| `marketplace` | 0.924 | 3 / 3 | reuse_ip | ✓ entity diffusion |
| `flash_sale` | 0.400 | 3 / 3 | log_interarrival | ✓ the burst |

Cut-off 0.162. When PSI exceeds it, the policy stands down from automated intervention to
analyst review. That is a trade, not a free win:

| profile | false auto-actions avoided | true responses downgraded | analyst minutes added |
|---|---:|---:|---:|
| `subscription` | 3 | 10 | 108 |
| `marketplace` | 1 | 10 | 84 |
| `flash_sale` | 6 | 10 | 144 |

Across the shifted profiles the guardrail prevents 10 false automated interventions and
downgrades 30 genuine responses, adding 336 analyst minutes.

**Status: experimental decision support, not a safety control.** The cut-off is fitted on
16 base streams and the false-flag rate then measured on 12 disjoint base streams comes
out at 33.3% — too high to run as a default. The reason is a confound, measurable by
holding the merchant fixed at base and varying only the campaign:

| base traffic, merchant held fixed | false-flag rate |
|---|---:|
| campaign matches calibration morphology (`k=30`) | 8.3% |
| background only, no campaign | 16.7% |
| campaign of unseen morphology (`k=60`) | 33.3% |

The signal is substantially detecting the attack, not the merchant. That is an
identification problem, not a tuning bug: live, "different merchant" and "under attack"
cannot be separated before deciding, and the standard fix — monitoring drift on a much
slower timescale than detection — is not implemented here.
