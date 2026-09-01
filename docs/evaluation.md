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
412 event alerts  →  18 incidents  →  1 action recommended
```

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
| P90 interval coverage | 95.5% (target 90%) |
| Median absolute error, P50 | 107.3 attempts |
| Mean true remaining | 367.6 attempts |
| Fit / conformal streams | 4 (campaigns 2, 3, 4, 6) / 4 (campaigns 0, 1, 5, 7) |
| Snapshots | 84 calibration / 88 held-out |

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
| always hold | 17.0 | 13.5 | 204.0 | ₹1,04,722 |
| event-by-event thresholding | 1.0 | 0.0 | 213.0 | ₹7,693 |
| **causal policy** *(forecast only)* | 1.0 | 0.0 | 12.0 | ₹5,158 |
| oracle policy *(upper bound)* | 1.0 | 0.0 | 12.0 | ₹3,145 |

Fractional counts are medians across an even number of streams. Event thresholding reaches
the same decision but hands an analyst 213 minutes of triage instead of 12 —
consolidation, not detection, is the difference. When the forecast interval is wide
relative to its median, the policy escalates to analyst review rather than automating.

**A retracted claim has come back, and it needs saying carefully.** On the demo stream the
causal policy now matches the oracle on **18 / 18** incidents, for a regret of **₹0**. An
earlier version of this project reported exactly that — ₹0 on 11 / 11 — and
[retracted it](engineering-log.md) when the cause turned out to be a generator defect that
fragmented campaigns into unrealistically clean pieces. That defect is fixed and stayed
fixed; the reason the number returned is different, and duller: the
[selected architecture](#closing-the-loop-selecting-an-architecture-without-touching-test)
raises event precision enough that the incidents reaching the policy on this one stream are
unambiguous, so there is nothing left for hindsight to improve on.

It is one stream, and it is not the claim to lean on. **Across the eight streams the oracle
is still ahead — ₹3,145 against the causal policy's ₹5,158.** Not knowing the future still
costs about 64% more; the demo stream simply is not where that shows.

### Incident-level calibration

An event model with ECE 0.0025 does not give a calibrated incident probability for free —
events inside an incident are strongly dependent, and that dependence is the signal.
Incident risk is a separate model, fitted on 129 calibration incidents pooled across 8
streams and measured on 145 held-out incidents:

| predicted | observed | incidents |
|---:|---:|---:|
| 0.095 | 0.180 | 128 |
| 0.343 | 0.000 | 1 |
| 0.432 | 0.500 | 4 |
| 0.613 | 0.000 | 1 |
| 0.994 | 1.000 | 11 |

Separation is clean at the top (0.994 → 1.000 on 11 incidents). The bottom bin is
under-confident (0.095 → 0.180 on 128). The middle is barely determined at all — three
bins holding 1, 4 and 1 incidents — which is the honest state of a model fitted on 129
incidents, and is reported rather than smoothed over.

### Which entity type carries the signal

Dropping each relation in turn and re-fitting under the same protocol (5 trials, medians;
`python -m koronis.cli relations`):

| variant | PR-AUC | precision | recall | false positives | PR-AUC change |
|---|---:|---:|---:|---:|---:|
| all relations | 0.989 | 0.942 | 0.968 | 24 | — |
| no `device_id` | 0.994 | 0.963 | 0.978 | 15 | +0.0049 |
| no `ip_id` | 0.983 | 0.927 | 0.960 | 29 | -0.0059 |
| no `bin_id` | 0.942 | 0.951 | 0.812 | 17 | -0.0468 |
| no `email_domain` | 0.993 | 0.956 | 0.983 | 18 | +0.0042 |

Shared BIN ranges carry almost all of it — remove that relation and recall collapses from
0.968 to 0.813. Dropping `device_id` or `email_domain` *improves* PR-AUC and cuts false
positives: they contribute noise, not evidence. This retires an earlier claim based on the
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
| **`koronis_full`** | 0.997 | 0.964 | 0.990 | 15 | 0.0 s |
| `no_edges` — event features only | 0.348 | 0.452 | 0.953 | 464 | 0.0 s |
| `no_approved` — graph only | 0.817 | 0.840 | 0.695 | 53 | 66.6 s |
| `no_edges` + `no_approved` | 0.061 | 0.000 | 0.000 | 0 | never |

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
| 0.83 ms | 1.05 ms | 1.27 ms | 0.84 ms | ~1,195 events/sec |

**Per-event cost is flat in stream length**, by construction:

- **Time** — `O(R · D_max · L · d)` per `push`: `R = 4` relations, `D_max = 32` (the
  `max_degree` fan-in cap in [`graph/build.py`](../koronis/graph/build.py), which keeps the
  most recent neighbours), `L = 2` message-passing layers, `d = 32`
  hidden units. None of these depends on the number of events already seen, so measured
  latency holds at ~0.83 ms p50 regardless of stream length.
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
| `full` | ₹1,398 | ₹1,836 | 0.9892 | 0.942 | 0.968 | 24 |
| `no_device` | ₹1,038 | ₹1,111 | 0.9941 | 0.963 | 0.978 | 15 |
| `no_email` | ₹1,125 | ₹1,511 | 0.9934 | 0.956 | 0.983 | 18 |
| `no_device_no_email` | ₹885 | ₹918 | 0.9959 | 0.970 | 0.985 | 12 |
| `no_gate` | ₹833 | ₹1,278 | 0.9943 | 0.958 | 0.980 | 17 |
| `no_device_no_gate` | ₹833 | ₹958 | 0.9951 | 0.965 | 0.990 | 14 |
| `no_email_no_gate` | ₹673 | ₹892 | 0.9968 | 0.964 | 0.990 | 15 |
| `lean` | ₹819 | ₹812 | 0.9957 | 0.968 | 0.990 | 13 |

**Chosen on calibration: `no_email_no_gate`** — drop the email relation and
the heterophily gate. Calibration cost falls from ₹1,398 to
₹673.

**It held up.** On test, cost falls from ₹1,836 to
₹892, PR-AUC rises 0.9892 → 0.9968,
precision 0.942 → 0.964, recall
0.968 → 0.990, and false positives fall
24 → 15. Every candidate that
removes something beats the full model, so the earlier ablations were reading a real signal
rather than noise.

**Selection is not free, and the table shows it.** The variant with the lowest *test* cost
is `lean` (₹812), not the one calibration chose. Picking on
held-out data costs something against picking with hindsight — that gap is the honest price
of not cheating, and reporting the calibration-chosen variant rather than the test-optimal
one is the whole point of the exercise.

### Does the graph survive being split across machines?

Throughput is not the hard part — per-event cost is already constant in stream length, so
more traffic is more processes. The hard part is that **partitioning a graph deletes
edges**: if one shard holds a device and another holds an IP that co-occurs with it, that
edge never forms. Sharding is a modelling decision, not an infrastructure detail.

Events are routed by hashing a field, so **the field you route on is the one relation
preserved perfectly** and every other survives only by collision. The prediction stated in
[`eval/sharding.py`](../koronis/eval/sharding.py) before the run came from the
[per-relation ablation](#which-entity-type-carries-the-signal): BIN carries the signal, so
BIN-routing should hold up best and random should decay fastest.

`python -m koronis.cli sharding`, frozen model and threshold, only the partition varying:

| shards | PR-AUC random | PR-AUC bin | PR-AUC device | ₹ random | ₹ bin | ₹ device |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.993 | 0.993 | 0.993 | ₹1,125 | ₹1,125 | ₹1,125 |
| 2 | 0.967 | 0.979 | 0.983 | ₹3,135 | ₹3,056 | ₹2,805 |
| 4 | 0.913 | 0.956 | 0.971 | ₹5,072 | ₹6,560 | ₹5,278 |
| 8 | 0.834 | 0.933 | 0.955 | ₹7,159 | ₹10,754 | ₹9,852 |
| 16 | 0.720 | 0.921 | 0.941 | ₹8,973 | ₹13,594 | ₹14,379 |

**Half right.** Entity routing does beat random decisively on PR-AUC — 0.921
and 0.941 against 0.720 at 16 shards. But BIN did
*not* beat device, so the specific prediction is wrong.

**The two entity keys fail in opposite directions, which PR-AUC hides.** At 16 shards:

| routing | precision | recall | false positives | missed |
|---|---:|---:|---:|---:|
| BIN | **0.937** | 0.555 | **15** | 178 |
| device | 0.529 | **0.993** | 354 | **3** |

BIN-routing cuts the campaign into fragments — device edges drop to
9% — so recall falls, but every fragment that fires is
genuinely coordinated and false positives actually go *below* the unsharded run. Device
routing co-locates the heavy legitimate device reuse in background traffic, so recall
survives and precision collapses to 0.529.

**Priced in rupees, the ranking inverts.** Under this project's own declared constants — a
missed attempt costs an authorisation fee, a false alert costs checkout friction —
**random routing is the cheapest option at 4, 8 and 16 shards despite having by far the
worst PR-AUC.** Keeping recall is worth more than keeping precision at these prices, and a
ranking metric cannot see that.

That is the thesis of this repo landing on its own experiment: *decide in the currency you
actually pay in.* Had this been reported as PR-AUC only, the recommendation would have been
the opposite of the priced one.

**What it does not license.** The pricing is per event, while the deployed path consolidates
alerts into incidents before acting — so it charges device-routing for 354
separate false alerts that consolidation would collapse into a handful of incidents. Read
the rupee column as evidence that the metrics disagree, not as a settled recommendation.
The one unambiguous finding is that **sharding costs accuracy whichever key you pick**:
₹1,125 undivided against ₹8,973 at the best 16-shard
option.

**A defect this surfaced.** The first run reported entity routing as completely free —
flat PR-AUC at every shard count. The tell was `largest_shard_share = 0.9375` no matter how
many shards were requested. Routing hashed ids with `int.from_bytes(..., "little")`, and
that value modulo a power of two depends only on the *first character*: every background
BIN began `b`, every campaign entity `c`, so they landed on two shards and nothing was
partitioned. With CRC32 the result above appeared. A test asserts no strategy puts a
disproportionate share on one shard.

### Recovering the edges a partition deletes

BIN routing keeps precision and bleeds recall as shards multiply, because device and IP
edges are cut when their endpoints land apart. That loss is not inherent to partitioning —
only to insisting every event live in exactly one place. An event can be **copied** to the
shard where its other relations would find company.

The prediction, stated in [`eval/sharding.py`](../koronis/eval/sharding.py) before the run:
entity frequencies are heavy-tailed, so most values appear once and can form no edge at
all. Replicating only events whose device or IP actually recurs should restore most lost
edges while copying a minority of traffic — and campaign entities, shared by construction,
should be copied preferentially. If duplication instead ran away, the approach would be
uneconomic and the generator's traffic would not be as heavy-tailed as it claims.

`python -m koronis.cli replicate`, frozen model and threshold:

| shards | recall, BIN only | recall, + replication | precision, BIN only | precision, + replication | events scored |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.927 | **0.985** | 0.966 | 0.938 | 1.68× |
| 4 | 0.812 | **0.995** | 0.964 | 0.858 | 2.15× |
| 8 | 0.585 | **1.000** | 0.955 | 0.730 | 2.42× |
| 16 | 0.453 | **0.995** | 0.948 | 0.583 | 2.56× |

**The mechanism works, and works completely.** At 16 shards recall goes from **0.453 to
0.995** — the campaign is fully recovered — for **2.56× the scoring work**. That the factor
is 2.56 and not 16 is the heavy tail doing exactly what was predicted: only shared entities
are worth copying, and most are not shared.

**It is not free, and the cost is precision.** Replicating high-degree entities co-locates
the *legitimate* dense clusters too, so false positives rise the same way they did under
device routing. Priced with the project's own constants:

| shards | BIN only | BIN + replication | random |
|---:|---:|---:|---:|
| 2 | ₹2,637 | ₹1,478 | ₹3,213 |
| 4 | ₹5,955 | ₹2,786 | ₹5,529 |
| 8 | ₹12,558 | ₹5,920 | ₹8,393 |
| 16 | ₹16,387 | ₹11,546 | ₹10,487 |

**Replication is the cheapest option at 2, 4 and 8 shards** — and 8 shards is about 9,500
events/sec at the measured single-process rate, which is the range a large gateway's peak
would actually sit in. **At 16 it stops winning:** plain random routing costs ₹10,487
against replication's ₹11,546, and does it at 1× compute instead of 2.56×.

**And nothing beats not partitioning.** The undivided stream costs ₹1,045. Every routing
strategy at every shard count is worse than that, which is the honest summary of this whole
line of work: sharding a coordination detector always costs something, replication buys back
most of it in the range that matters, and the right first move is a bigger window per shard
rather than more shards.

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
| 0 | 150 | 43 / 43 | 1.000 / 1.000 | 0.767 / 0.767 |
| 1 | 240 | 25 / 25 | 1.000 / 1.000 | 0.912 / 0.912 |
| 2 | 380 | 14 / 14 | 1.000 / 1.000 | 0.982 / 0.982 |
| 3 | 520 | 20 / 20 | 1.000 / 1.000 | 0.979 / 0.979 |
| 4 | 700 | 14 / 14 | 1.000 / 1.000 | 0.987 / 0.987 |
| 5 | 900 | 16 / 16 | 1.000 / 1.000 | 0.994 / 0.994 |

**Median: 18 incidents either way, purity
1.000 / 1.000, recall 0.9802 /
0.9802.** Making the decision causal costs nothing measurable here, in
**4096 KB** of sketch that does not grow with the stream.

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

The mechanism and relation ablations test *data sources*. This one tests the two design
decisions in [`layers.py`](../koronis/models/layers.py) — the heterophily gate and the
learned relation attention — plus depth.

The gate's justification was specific enough to make a **conditional** prediction. It damps
edges joining dissimilar nodes, and camouflage is exactly what creates those: a camouflaged
attempt draws its amount and email domain from the background, so it links to legitimate
traffic that looks nothing like the rest of the ring. The gate should therefore buy *more*
as camouflage rises, and little at camouflage 0.

**It does the opposite.** Variants are expressed as departures from the selected
architecture, so the gate appears here as something to *add back*:

| camouflage | selected | + gate | uniform relation attention | one layer |
|---:|---:|---:|---:|---:|
| 0.0 | 0.9999 | 1.0000 | 0.9999 | 0.9993 |
| 0.5 | 0.9993 | 0.9983 | 0.9993 | 0.9774 |
| 1.0 | 0.9968 | 0.9934 | 0.9965 | 0.9254 |

PR-AUC cost of departing from the selected architecture (positive = the selected choice was
better):

| camouflage | adding the gate | uniform attention | one layer |
|---:|---:|---:|---:|
| 0.0 | −0.0001 | 0.0000 | +0.0006 |
| 0.5 | **+0.0010** | 0.0000 | +0.0219 |
| 1.0 | **+0.0034** | +0.0003 | **+0.0714** |

**Adding the gate back costs more as camouflage rises** — the inverse of the prediction that
motivated it. **Relation attention is a wash**, consistent with the
[per-relation ablation](#which-entity-type-carries-the-signal), which already found the
learned weights disagree with what the relations are actually worth. **Depth is the piece
that earns its place**, and it is the one whose benefit shows the conditional shape the gate
was supposed to have: negligible at camouflage 0, +0.071 PR-AUC at camouflage 1.
Coordination that survives camouflage is visible two hops out, not one.

**What this retracted.** Earlier versions argued the gate was load-bearing because fraud
rings camouflage into legitimate traffic. The reasoning was plausible and the measurement
does not support it. It was reported and left in the model at the time, because acting on a
test-set finding is the leakage this project refuses — and then acted on properly, through
[calibration-based selection](#closing-the-loop-selecting-an-architecture-without-touching-test).
The gate is now off by default because a held-out procedure chose to remove it, not because
this table did.

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
| 1 | 0.999 | 0.999 | **0.000** | 100% |
| 2 | 0.997 | 0.991 | 0.006 | 50% |
| 4 | 0.994 | 0.980 | 0.014 | 27% |
| 8 | 0.988 | 0.947 | 0.041 | 14% |
| 16 | 0.975 | 0.902 | **0.073** | 8% |

**At `M = 1` the two views are the same stream by construction, and they score
identically** — that is the experiment's control, and a test asserts it. From there the
gap opens monotonically and roughly in proportion to `M`, as predicted. Recall tells the
same story: the gateway view holds 0.99 throughout while the merchant view falls to 0.95.

Both views degrade somewhat as `M` grows, because more merchants means more legitimate
traffic and more chances to false-positive; the merchant view degrades about three times
faster. **Honest limit: even at `M = 16` the merchant-scoped view still detects this
campaign.** This measures a widening gap, not a blindness boundary — it says the wider
aperture is worth something and quantifies how much, not that a merchant alone is
helpless.

Entity ids are namespaced per merchant. Without that, two merchants would reuse the same
`d17` and the pooled graph would link strangers — the gateway view would then win on an
artefact. A test asserts no background entity is shared across merchants.

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
| `subscription` | 20 | 11 | 264 |
| `marketplace` | 3 | 11 | 72 |
| `flash_sale` | 16 | 11 | 156 |

Across the shifted profiles the guardrail prevents 39 false automated interventions and
downgrades 33 genuine responses, adding 492 analyst minutes.

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
