# Koronis

**Detecting distributed card-testing campaigns that per-entity velocity rules cannot see — at any threshold.**

[![tests](https://img.shields.io/badge/tests-89%20passing-2ea44f)](tests/)
[![python](https://img.shields.io/badge/python-3.11%2B-3776ab)](https://www.python.org/)
[![graph libs](https://img.shields.io/badge/graph%20libraries-none-8a3ffc)](koronis/models/layers.py)
[![track](https://img.shields.io/badge/Razorpay%20Buildathon-Track%2002%20·%20AI%20Risk%20Manager-0c2451)](https://razorpay.com/buildathon/)

Koronis is a **defence-only detector and decision-support prototype** for one class of merchant loss: card testing, where a fraudster
runs thousands of micro-transactions through a checkout to find which stolen cards are
live. It reports **precision and recall on a held-out test set**, **false-positive cost**
in rupees, and one metric published work rarely does — *detection latency*, because the
damage accrues from the moment a campaign starts.

Its central claim is not that the model beats its baselines. It is a **characterisation of
when detection is possible at all**, stated as arithmetic and then tested: on a 16-point
grid, the measured blindness boundary matched the predicted one **16 / 16**.

---

> **Why "Koronis"?** The Koronis family is a group of asteroids that share orbital
> elements because they are fragments of a single shattered parent body. Hirayama found
> them in 1918 — not by looking at the sky, where they are scattered and unremarkable, but
> by plotting them in *orbital-element space*, where they clump unmistakably.
>
> That is this project. Individually ordinary events, scattered in the obvious view;
> plot them in shared-entity space and a common origin becomes undeniable.

---

## The problem

A fraudster buys stolen card numbers. Most are dead, so to make money they must learn
which still work. They find a checkout and run ₹1–₹20 charges against each card. Approved
means live.

**The merchant's checkout has become a free card-validation service**, and the merchant
pays for it — an authorisation fee on every attempt including declines, card-network
penalties when the decline rate spikes, and chargebacks weeks later on the very cards
their site validated.

The defences that exist miss the attacks that matter:

- **Velocity rules** were designed for one attacker on one IP. Spread a campaign across
  fifty IPs and forty devices and every counter stays under threshold.
- **Per-transaction models** score each attempt alone, where a real card charged a
  plausible amount looks entirely ordinary.
- **Chargeback-supervised systems** are structurally late: most of the damage lands
  before a single chargeback is filed.

## The claim

Let an attacker make `n` attempts spread across `k` entities, against a counter firing at
threshold `τ`.

**Claim 1 — threshold detectors are blind above a spread.**
Attempts per entity is `n/k`. Firing requires `n/k > τ`, so an attacker with `k ≥ n/τ`
escapes **for any n**. With counters on several entity types the engine fires if *any*
trips, so blindness needs `k ≥ n / min(τ)` — the binding constraint is the most sensitive
counter, not the least. And `τ` cannot simply be lowered: legitimate heavy users — offices,
shared CGNAT addresses — put a floor under it. In this repo that floor is *measured*, not
assumed: `tune_velocity` picks the most sensitive threshold per entity that still respects
a stated false-positive budget on clean traffic.

**Claim 2 — graph signal scales differently.**
The number of attempt-pairs sharing an entity is `≈ n²/2k`. Co-occurrence signal grows as
**n²/k** where the per-entity signal is only **n/k**. The graph's advantage *increases*
with attack size.

**Consequence.** The attacker's only escape from the graph is `k → n` — one fresh device,
IP and BIN per attempt — which is bounded by what infrastructure costs.

### The prediction, tested

`predicted_boundary_k(n, τ) = n/τ`, compared against measured detection on a 4×4 grid:

| n | k=2 | k=10 | k=50 | k=200 | predicted boundary |
|---:|:---:|:---:|:---:|:---:|---:|
| 200 | fires | fires | **blind** | **blind** | 22.2 |
| 400 | fires | fires | **blind** | **blind** | 44.4 |
| 800 | fires | fires | fires | **blind** | 88.9 |
| 1600 | fires | fires | fires | **blind** | 177.8 |

Every cell agrees with `k ≥ n/τ`. **Agreement: 16 / 16 (100%).**
Koronis detects in all sixteen. Reproduce with `python -m koronis.cli frontier`.

## Results

**Three splits, and the threshold never sees the test set.**

| split | contents | used for |
|---|---|---|
| train | `k ∈ {4, 12, 30}` × `camouflage ∈ {0, 0.5, 1}` | fitting model weights |
| calibration | same distribution as train, different draw | **choosing the operating threshold, then frozen** |
| test | `k = 60`, `camouflage = 1.0`, unseen entities | reported numbers only |

All three are derived from one run seed, so repeating the run resamples every split
together rather than holding any of them fixed.

Training contains only campaigns concentrated enough that a tuned velocity engine still
catches them (`k ≤ 30`, below the boundary of ~44). The test campaign is spread **past**
that boundary, fully camouflaged, with entirely unseen entities — so the hold-out is
**extrapolation**, not interpolation.

**The whole protocol is repeated across 10 independent trials** — each redrawing the
background traffic, the campaign entities, the calibration stream and the model
initialisation, while holding the held-out morphology fixed. Intervals are the median with the 2.5th and 97.5th percentiles **observed across the ten
runs**. They describe the spread actually seen, not a population confidence interval — ten
draws is good evidence of stability, not statistical certainty.

Scores are used **raw**. Rescaling each split by its own maximum would let every split
redefine what a score means, hollowing out the claim that the threshold was frozen.

| detector | PR-AUC | precision | recall | false positives | **detected** |
|---|---:|---:|---:|---:|---:|
| `velocity_tuned` | 0.062 `[0.062, 0.062]` | 0.000 | 0.000 | 44 | **0 / 10** |
| `decline_burst` *(no graph, no learning)* | 0.221 `[0.212, 0.234]` | 0.234 | 0.013 | 16 | 6 / 10 |
| `shared_entity` *(graph, no learning)* | 0.055 `[0.054, 0.056]` | 0.000 | 0.000 | 35 | **0 / 10** |
| `gbdt_per_txn` | 0.319 `[0.289, 0.339]` | 0.398 | 0.811 | **490** `[456, 540]` | 10 / 10 |
| **`koronis_graph`** | **0.988** `[0.984, 0.992]` | **0.947** `[0.931, 0.963]` | **0.970** | **22** `[15, 29]` | **10 / 10** |

Campaign exposure if never stopped: ₹29,200. Per-trial values in `results/seeds_raw.csv`.
Reproduce with `python -m koronis.cli seeds`.

**What this table actually says.**

*Threshold rules do not degrade — they stop working.* Velocity precision is 0.000 on all ten
trials and it never once detects the campaign. That is not an empirical tendency, it is
Claim 1: at `k = 60` against a binding `τ = 8`, no counter can trip.

*The per-transaction model finds the campaign and drowns the analyst.* It detects every
time and recalls 81% of attempts — at **490 false positives** against Koronis's **22**, a
**22× difference**. Recall was never the hard part of this problem; precision at a usable
alert volume is.

*Raw graph counting is defeated by dense traffic.* Plain inverse-frequency co-occurrence
(`shared_entity`) reaches only **0.055** PR-AUC and never fires. An earlier version of this
README reported it at 0.894 and called it a strong baseline — that was measured on
background traffic of 9 events/hour, where a campaign was 97% of everything in its own
window. At a realistic 1,600 events/hour, legitimate devices, IPs and BINs co-occur
constantly and counting alone has nothing to lock onto. The finding is retracted; the test
suite now records both sides, since counting still works on a concentrated burst.

*So the network is doing the work, and it is worth being precise about which work.* The
structure carries the signal only once you weight relations, gate camouflage edges, and
learn what a suspicious neighbourhood looks like against a noisy background. Counting the
neighbourhood is not enough — and neither is the graph alone, as the mechanism ablation
below shows.

### Detection latency

Precision and recall if you had only seen the stream up to `t` seconds after onset:

Each detector uses **one threshold, derived from calibration and held fixed at every
checkpoint** — so this measures a single deployed detector over time, not a sequence of
differently tuned ones.

| detector | t=60s | t=300s | t=600s |
|---|---|---|---|
| `velocity_tuned` | — not detected — | — not detected — | — not detected — |
| `decline_burst` | — not detected — | — not detected — | — not detected — |
| `shared_entity` | — not detected — | — not detected — | — not detected — |
| `gbdt_per_txn` | P 0.19 / R 0.80 | P 0.39 / R 0.71 | P 0.48 / R 0.70 |
| **`koronis_graph`** | **P 0.56 / R 1.00** | **P 0.82 / R 0.97** | **P 0.88 / R 0.97** |

At one minute Koronis has recalled **every** campaign attempt so far at 0.56 precision,
while the per-transaction model is at 0.19 — four in five of its alerts are wrong. None of
the three learning-free detectors ever crosses its frozen threshold on this campaign. When
you detect determines how much you save; how precisely you detect determines whether anyone
can act on it.

## Streaming replay

The detector runs as a **strictly causal stream**: `StreamingKoronis.push(event)` scores one
event at a time, and no future event is reachable from it. Per event it emits the raw score,
the frozen calibration threshold, the alert decision, how many prior events it linked to,
which relations supplied those links, and a rolling ring summary.

**Streaming reproduces batch scores exactly**, not approximately. That falls out of the
backwards-in-time edge rule: a node's layer-1 representation depends only on events that
preceded it, so layer-1 outputs can be cached as the stream advances and layer-2 computed
from neighbours' cached values. `tests/test_stream.py` asserts the two agree to `1e-5`, and
that scoring a prefix matches scoring the full stream — which is what makes the latency
numbers meaningful rather than a batch model dressed up as a stream.

### Per-event inference latency

Measured over 6,200 events after 200 warm-up, timing only `push` — neighbour lookup plus two
message-passing steps. Dataset construction and model fitting are excluded, since neither
happens per event in deployment.

| p50 | p95 | p99 | mean | throughput |
|---:|---:|---:|---:|---:|
| **0.99 ms** | **1.22 ms** | 1.40 ms | 1.00 ms | ~998 events/sec |

An earlier run of this benchmark reported p50 1.78 ms and p95 3.85 ms. That measurement was
taken while a ten-trial experiment was running in the background on the same machine and is
discarded: a latency benchmark competing with a training job measures the machine, not the
code. The figures above come from an otherwise idle run.

Entity buckets expire on the window, so memory is bounded by window occupancy rather than
stream length — a test holds that too. Reproduce with `python -m koronis.cli benchmark`.

On the held-out stream, Koronis alerts on the campaign's **opening attempt**, while the
tuned velocity engine never alerts on it at all. `results/replay.json` carries the full
per-event trace.

**That opening alert is not coordination, and it prevents nothing.** The first attempt has
no prior campaign neighbours — there is no campaign yet — so the evidence at that instant is
per-event, and it is the authorisation outcome. An outcome is observed only *after* an
attempt is submitted, so no detector can prevent the attempt it learns from. What an early
alert buys is the ability to stop everything that follows. The next section measures exactly
how much of the result comes from each mechanism.

## From alerts to a decision

Four hundred event alerts are not four hundred things for a fraud team to do — they are
one campaign. Koronis consolidates them, then recommends the intervention with the lowest
**expected cost**, not the one matching the highest risk score.

```
419 event alerts  →  11 incidents  →  1 action recommended
```

On the held-out stream, the genuine campaign becomes a single incident of **408 attempts
across 72 devices, 72 IPs and 72 BIN ranges**, at risk 1.000. The other ten incidents are
isolated background alerts at risk ≈ 0.07 — correctly left on `monitor`.

### The action assumptions

Every figure below is a declared assumption about a merchant workflow, not a measurement.
They live in [`koronis/incident.py`](koronis/incident.py) so a reviewer can substitute
their own and re-run.

| action | friction (genuine) | harm (false) | stops | analyst |
|---|---:|---:|---:|---:|
| monitor | ₹0 | ₹0 | 0% | — |
| rate-limit | ₹120 | ₹400 | 55% | — |
| step-up verification | ₹350 | ₹1,800 | 85% | — |
| hold + review | ₹900 | ₹6,000 | 97% | 12 min |

The chosen action minimises `friction + risk × forecast_exposure × (1 − stops) +
(1 − risk) × false_harm`. A high risk score with nothing left to prevent correctly gets
`monitor`; a test asserts exactly that.

### The policy cannot be allowed to know the future

Choosing an action needs an estimate of what inaction would cost — the attempts still to
come. Offline that can be read off the campaign log. **A live system cannot**, so a policy
built on the true remaining count is an *oracle upper bound*, not a product. Both are
reported, and the difference between them is the price of not knowing.

`causal_policy` sees only the first **12 events** of an incident plus a forecast. It never
sees the true remaining count or the ground-truth label.

### Forecasting what is still ahead

At each incident snapshot, using only observed signals — attempts so far, attempt-rate
trend, score trend, decline ratio, entity spread, incident age, relationship density — a
quantile model predicts **how many more alerted events will join this incident**.

That target is deliberately **label-free**: it is a structural quantity, observable in
hindsight without anyone deciding whether the incident was genuine. Whether it *matters* is
the separate question the incident risk model answers, and the policy multiplies the two:

```
expected remaining exposure  =  P(genuine) × forecast(remaining attempts) × ₹73
```

Raw quantile regression is routinely over-confident, so the upper quantile carries a
**conformal pad fit on a held-out subset of calibration incidents; coverage is then
evaluated on held-out test incidents.**

That split is **by stream, never by snapshot row**. Snapshots of one incident are nested
prefixes of the same sequence and are highly dependent — putting one prefix in the fit set
and another in the conformal set measures the residual on data the model has effectively
already seen. It is not a test-set leak, but it inflates apparent coverage. Splitting by
row reported **91.8%**; splitting by stream reports the figure below. A test asserts no
incident appears in both partitions, and that stream-qualified keys are used, since
incident ids restart at `INC-000` for every stream.

| | measured |
|---|---|
| P90 interval coverage | **99.0%** (target 90%) |
| Median absolute error, P50 | 221.6 attempts |
| Mean true remaining | 347.6 attempts |
| Fit / conformal / evaluation | 4 streams / 4 streams / 97 held-out snapshots |

**The interval over-covers.** 99% against a 90% target means it is wider than it needs to
be — conservative, which is the safe direction for a policy that escalates on uncertainty,
but not well calibrated. With only four conformal streams the pad quantile is coarsely
estimated. Reported rather than tuned toward the target.

**Campaign length is varied across streams** for this evaluation, and that is not a detail.
With every campaign the same length, "remaining" collapses to a constant minus what you have
seen — the forecaster scored a median error of **6.7 attempts** while learning nothing but
`N_ATTEMPTS`. Varying the size raised the error to 157.5 and made the problem real. A test
records the difference so it cannot regress.

### Policy comparison

Median across 8 independent test streams:

| policy | incidents actioned | false incidents | analyst minutes | merchant cost |
|---|---:|---:|---:|---:|
| always allow | 0 | 0 | 0 | ₹29,200 |
| always hold | 6 | **5** | 72 | ₹36,924 |
| event-by-event thresholding | 1 | 0 | **205.8** | ₹3,736 |
| **causal policy** *(forecast only)* | **1** | **0** | **12** | **₹1,884** |
| oracle policy *(upper bound)* | 1 | 0 | 12 | ₹1,884 |

**Action regret vs the oracle: ₹0 — the causal policy chooses the same action on 11 / 11
incidents.** Despite a median forecast error of 222 attempts, the decision is unchanged,
because the cost gaps between actions are large relative to that error.

That is the honest reading, and it comes with a caveat: on this distribution incidents are
either clearly large or clearly singleton, so the decision is not close. A mix with more
mid-sized incidents would expose non-zero regret, and the regret metric is reported so that
would show rather than hide.

When the forecast interval is wide relative to its own median, the policy **escalates to
analyst review rather than automating** on a number the model does not stand behind.

Event thresholding reaches the same decision, but hands an analyst **205.8 minutes** of
triage instead of 12 — consolidation, not detection, is the difference. Always-hold escalates
five false incidents per stream and costs more than doing nothing at all.

### Incident-level calibration is measured, not inherited

An event model with ECE 0.0025 does **not** give a calibrated incident probability for free:
the events inside an incident are strongly dependent — that dependence is the entire signal —
so no independence-flavoured combination of their scores is a probability of anything.

Incident risk is therefore a separate model, fitted on **47 calibration incidents pooled
across 8 streams**, and its reliability measured on 47 held-out incidents:

| predicted | observed | incidents |
|---:|---:|---:|
| 0.075 | 0.00 | 34 |
| 0.213 | 0.00 | 3 |
| 0.466 | 0.00 | 2 |
| 1.000 | 1.00 | 8 |

Separation is clean at both ends. **The middle is weakly determined** — five incidents across
two bins — and slightly over-confident. Reported rather than smoothed over.

A single calibration stream yields only two incidents, which cannot determine an eight-feature
model at all; a test records that failure so the pooling requirement cannot be quietly dropped.

## Traffic-profile transfer stress test

**These are synthetic merchant shapes, not real merchants.** Surviving this is evidence the
detector is not tuned to one traffic profile. It is *not* evidence of production
cross-merchant transfer, and nothing here should be read as such.

Everything is fitted on the **base** profile and frozen before any shifted traffic is
scored: detector weights, the alert threshold, the incident risk model, the exposure
forecaster, and the drift cut-off. The three shifted profiles are declared in
[`koronis/profiles.py`](koronis/profiles.py) — defined before being run, because choosing a
shift after seeing which one the model survives turns a stress test into a demonstration.

| profile | what it breaks |
|---|---|
| `subscription` | legitimate device and card reuse is **high** — dense co-occurrence is normal |
| `marketplace` | entities are **diffuse** — the graph is sparse and thresholds sit wrong |
| `flash_sale` | a **legitimate burst** — high volume and elevated declines, no attack |

### Does it notice?

Drift is measured by **Population Stability Index**, the standard statistic in payments
risk, so a reviewer can read it and its per-feature contributions say *which* aspect moved.
The cut-off is the 95th percentile of PSI between disjoint **base** samples — how much base
traffic varies against itself — so nothing about the shifted profiles informs it.

| profile | median PSI | flagged | largest shift | what actually changed |
|---|---:|---:|---|---|
| base | 0.141 | 0 / 3 | reuse_bin | — |
| `subscription` | **0.584** | 3 / 3 | **reuse_device** | ✓ device reuse |
| `marketplace` | **0.924** | 3 / 3 | **reuse_ip** | ✓ entity diffusion |
| `flash_sale` | **0.399** | 3 / 3 | **log_interarrival** | ✓ the burst |

Cut-off: **0.162**. Every shifted profile is flagged, and in each case the feature it names
is one the profile genuinely alters — it reports *how* the traffic differs, not merely that
it does.

### The guardrail, and what it costs

When PSI exceeds the cut-off, the policy **stands down from automated intervention to
analyst review**. `review_only` stops nothing on its own — that is the honest price of not
trusting yourself — and it is deliberately excluded from the cost-minimising choice, since
in an argmin it would be selected whenever doing nothing while billing an analyst looked
cheap. A test asserts it is unreachable that way.

| profile | false automated actions avoided | true responses downgraded | analyst minutes added |
|---|---:|---:|---:|
| `subscription` | 1 | 3 | 12 |
| `marketplace` | **5** | 9 | 72 |
| `flash_sale` | 2 | 7 | 36 |

**This is a trade, not a free win.** Across the shifted profiles the guardrail prevents 8
false automated interventions and downgrades 19 genuine responses to review-only, adding 120
analyst minutes. Whether that is worth it depends on the merchant's tolerance for wrongly
throttling real customers, which is exactly the judgement a person should make.

### Status: experimental decision support, not a safety control

The cut-off is fitted on **16 independent base streams**; the false-flag rate is then
measured on **12 disjoint base streams**. It comes out at **33.3%** — far too high to run as
a default safety control, and that verdict stands regardless of how well it catches the
shifted profiles.

**Why it is that high is measurable, and it is the confound rather than merchant variation.**
Holding the merchant fixed at base and varying only the campaign:

| base traffic, merchant held fixed | false-flag rate |
|---|---:|
| campaign matches calibration morphology (`k=30`) | **8.3%** |
| background only, no campaign | 16.7% |
| campaign of unseen morphology (`k=60`) | **33.3%** |

Against a 5% nominal target, the monitor behaves acceptably when the campaign resembles what
calibration contained, and degrades badly when it does not. **The signal is substantially
detecting the attack, not the merchant** — a campaign of several hundred events with
distinctive reuse moves the very statistics being watched.

That is an identification problem, not a tuning bug: live, you cannot separate "different
merchant" from "under attack" before deciding. The standard fix is to monitor drift on a
much slower timescale than detection, so no single campaign can move the baseline. That is
not implemented here.

**So it is labelled what it is:** experimental decision support that can route an incident to
a human, not a control anything should depend on. A false drift alarm is not free either —
on the base streams it downgraded genuine responses for nothing.

### Wording that follows from this

The demo says:

> *"This live traffic profile is outside Koronis's calibration distribution. Automation is
> lowered and the incident is routed to review."*

Not *"this merchant's traffic doesn't resemble…"*, which asserts a cause the measurement
cannot support. Where a reuse ratio is quoted it is the **observed** mean events-per-entity
divided by the base mean, recorded in `results/drift.json` — a PSI value says how much a
distribution moved, not by what factor.

## Which mechanism carries the signal

The full model alerting on the opening attempt demanded an explanation rather than a
victory lap, so each mechanism was removed in turn under the same three-split protocol
(5 trials, medians). `python -m koronis.cli mechanism`.

| variant | PR-AUC | precision | recall | false positives | first alert |
|---|---:|---:|---:|---:|---:|
| **`koronis_full`** | **0.987** | **0.936** | 0.978 | **27** | 0.0 s |
| `no_edges` — event features only | 0.334 | 0.454 | 0.968 | 464 | 0.0 s |
| `no_approved` — graph only | 0.712 | 0.734 | 0.850 | 106 | **25.8 s** |
| `no_edges` + `no_approved` | 0.061 | 0.000 | 0.000 | 2 | **never** |

**Both mechanisms matter, and they do different jobs.**

*The authorisation outcome buys earliness.* Strip the edges and the model still alerts at
t = 0 — but at 0.454 precision and **464 false positives**. Early and unusable.

*The graph buys precision.* Strip `approved` and the first alert moves from 0.0 s to
**25.8 s**, because with no per-event signal the model must wait for coordination to
accumulate. But false positives fall from 464 to 106, and precision rises to 0.734.

*Together they are worth more than either.* 27 false positives at 0.936 precision — a **17×
reduction** over event-features-alone. And with both removed the model has nothing left:
PR-AUC 0.061, never fires. There is no third source of signal hiding in the features.

So the honest statement is: **transaction outcome gives early, weak evidence; temporal graph
structure converts it into a high-precision campaign alert.** Neither half is the product.

`tests/test_first_event.py` pins the structural claim: the opening attempt has zero
campaign-derived links in every relation, and cannot acquire any, because devices, IPs and
BINs are minted fresh per campaign. It *can* link to legitimate history through a shared
email domain — evidence that carries no campaign information — and campaign links only
begin to accumulate after it.

## How it works

**1 · Build a graph, not a list.** Every attempt is a node; two attempts are linked when
they share a device, IP, BIN range or email domain, within a time window. Real customers
are strangers, so legitimate traffic is sparse and scattered. An attack — however
distributed — reuses infrastructure somewhere, because infrastructure costs money.

Edges point **backwards in time**: a node only ever aggregates from its own past. Without
that, the model reads the future during a streaming evaluation and every latency figure is
fiction.

**2 · Score groups, not transactions.** The question stops being *"is this payment
fraudulent?"* — unanswerable, since each attempt genuinely looks fine — and becomes *"are
these attempts one coordinated campaign?"*, which is visible in the structure.

**3 · Relational message passing, written from scratch.** No DGL, no PyTorch Geometric —
aggregation is `torch.index_add_` in [`layers.py`](koronis/models/layers.py):

```
m_v^r = Σ_{u→v} g(x_u, x_v) · W_r x_u / deg(v)
h_v   = ReLU( W_self x_v + Σ_r a_r · m_v^r )
```

- `a_r` is a learned softmax over relations, so the model *discovers* which entity type
  carries the signal instead of being told.
- `g` is a **heterophily gate**, scoring each edge from the feature difference between its
  endpoints. Fraud rings deliberately attach themselves to legitimate traffic as
  camouflage; vanilla GNNs assume connected nodes share labels, which is precisely the
  assumption an adversary is motivated to break.

**4 · Inductive by construction.** There are no per-entity embedding tables — entity ids
only decide which events share an edge. The model scores devices and IPs it has never
seen, which is the only regime that exists in production.

**5 · Trained on rupees, not cross-entropy.**

```python
p = torch.sigmoid(logits)
loss = ((1 - p) * labels * c_fn + p * (1 - labels) * c_fp).mean()
```

The business objective *is* the training objective, rather than a threshold repaired
afterwards.

## What this does not claim

Koronis does not identify every fraudulent payment, and it is not a general fraud model.
It detects **coordinated card-testing campaigns where individual events remain ambiguous
but shared infrastructure creates measurable temporal structure**.

An attacker who uses genuinely fresh infrastructure for every single attempt — a new
device, a new IP and a new BIN each time — leaves no graph signal, and Koronis will not
find them. That is a real limit, and it is also the point of Claim 2: driving `k → n`
costs the attacker one unit of infrastructure per attempt, which is the economic bound the
detector pushes them against.

This is a **semi-synthetic proof of concept**, not production fraud detection. See below.

## What is assumed, not measured

Stated plainly, because a result is only as good as its caveats.

- **Background traffic is synthetic.** The loader supports the IEEE-CIS dataset for real
  entity-reuse structure, but the reported numbers use the bootstrap sampler. Real
  negatives would make the false-positive costs more trustworthy.
- **Campaigns are injected, not observed.** Ground truth exists because it was
  constructed. The mitigation is that the hold-out extrapolates to a spread never seen in
  training, but this is not the same as detecting a campaign in the wild.
- **Cost constants are estimates.** ₹73 per attempt and ₹40 per false block are declared
  in [`cost.py`](koronis/eval/cost.py) with reasoning. Substitute your own and rerun.
- **`money_prevented` assumes detection halts the campaign instantly.** It is therefore
  *estimated avoidable exposure under stated assumptions*, not guaranteed savings. A real
  merchant workflow is a ladder — flag, then rate-limit or step-up verification, then a
  temporary hold, then analyst review — and each rung takes time and lets more attempts
  through. Read the figure as the **cost of latency**: how much more is lost per minute of
  delay.
- **BIN thresholds are optimistic for the baseline.** Real BIN ranges carry heavy
  legitimate volume, so `τ_bin` would sit far above the 9 measured here. That makes the
  baseline stronger than reality, which is the safe direction.

## Defense-only

Koronis is a detector. It identifies coordinated activity; it does not generate it against
any live system. The campaign injector exists solely to produce labelled test data, as it
does in every fraud-ML paper, and is constrained accordingly:

- **every recommended action is simulated.** Nothing blocks a payment, calls a gateway, or
  touches live merchant infrastructure. The action ladder is decision support in a modelled
  workflow, with its cost and effectiveness figures declared as assumptions
- operates on in-memory dataframes; **no network capability anywhere in the codebase**
- no real BIN ranges, no real card numbers, no live endpoints
- reproduces only attack characteristics already documented publicly in Visa's own
  anti-enumeration guidance

Verifiable in one command — the package imports nothing that can reach a network or spawn
a process:

```bash
grep -rnE "^(import|from) (requests|urllib|socket|http|aiohttp|subprocess)" koronis/
# no matches
```

The complete third-party surface is `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `torch`.

## Live demo

**[yashasm18.github.io/koronis](https://yashasm18.github.io/koronis/)** — replay a held-out
campaign event by event, watch the evidence accumulate, and read the evaluation and
limitations alongside it.

The page is generated from the result files, not written by hand:

```bash
python site/build.py       # results/*.json|csv  ->  docs/index.html
```

Every figure it displays is embedded from `results/` at build time, so it cannot drift out
of step with the experiment.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 49 tests
.venv/bin/python -m koronis.cli ablation      # the headline table
.venv/bin/python -m koronis.cli frontier      # predicted vs measured boundary
.venv/bin/python -m koronis.cli latency       # precision/recall over time
.venv/bin/python -m koronis.cli seeds         # 10 trials, median + across-run range
.venv/bin/python -m koronis.cli replay        # causal event-by-event replay -> JSON
.venv/bin/python -m koronis.cli benchmark     # p50/p95 per-event inference latency
.venv/bin/python -m koronis.cli mechanism     # which mechanism carries the signal
.venv/bin/python -m koronis.cli incidents     # alerts -> incidents -> forecast -> action
.venv/bin/python -m koronis.cli drift         # traffic-profile transfer stress test
```

Results are written to `results/*.csv`. Every number in this README comes from those files.

Optional — real background traffic (needs a Kaggle account and accepting the competition
rules):

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw
```

## Repository

| Path | Responsibility |
|---|---|
| [`koronis/data/`](koronis/data/) | Event schema, background loader, campaign injector with `(n, k, camouflage)` control |
| [`koronis/graph/build.py`](koronis/graph/build.py) | Windowed entity-sharing graph, backwards-in-time edges, degree cap |
| [`koronis/models/layers.py`](koronis/models/layers.py) | **From-scratch relational message passing** |
| [`koronis/models/koronis.py`](koronis/models/koronis.py) | Inductive detector, relation attention |
| [`koronis/models/loss.py`](koronis/models/loss.py) | Expected-rupee-cost objective |
| [`koronis/models/velocity.py`](koronis/models/velocity.py) | FP-budget-tuned multi-entity baseline |
| [`koronis/eval/`](koronis/eval/) | Cost model, latency harness, calibration, frontier sweep |

## What broke

Four defects, all surfaced by running experiments rather than by reading code.

**1 · A leak that would have invalidated the headline claim.** Campaign entity ids were
named per campaign index, so datasets built with different seeds reused the same device
fingerprints — train and test shared entity identity. Every test passed and the metrics
looked excellent, but the model could have keyed on entity names rather than structure,
making the inductive claim false while appearing perfect. It surfaced only because a test
*asserted* the hold-out's entities were disjoint, rather than the code merely intending it.

**2 · A segfault from import order.** LightGBM and PyTorch each ship an OpenMP runtime. On
macOS, loading both deadlocks the test suite; loading them in the wrong order crashes the
interpreter outright. The package now fixes the order before either is reachable.

**3 · A model that scored 0.998 on train and 0.000 on test.** Trained on a single loud
campaign, it memorised micro-amounts instead of coordination, and transferred nothing to a
camouflaged attack. The fix was not architectural — the training distribution now spans
both spread and camouflage, so coordination is the only invariant available to learn.

**4 · The frontier disagreed with its own prediction on 25% of the grid.** The theory
looked wrong. It wasn't: devices and IPs were assigned round-robin while BINs used random
sampling, and the multinomial maximum reached 18 attempts where the mean was 8 — tripping
a threshold of 9 for reasons unrelated to the hypothesis under test. With uniform spread
across every entity, agreement went to **100%**.

**5 · A rescaling that quietly changed a published conclusion.** Calibration and test
scores were each divided by their own maximum before thresholding. No test labels were
touched, so it was not leakage in the usual sense — but it let each split redefine what a
score of `1.0` meant, so a "frozen" threshold referred to a different absolute quantity on
each one. The conclusion drawn from it was wrong, in the direction that flattered this
project.

**6 · A simulation that made the problem easy, and the three bugs hiding behind it.** The
background ran at **9 events per hour** across thirty days. An injected campaign was then
97% of the traffic in its own window, and 96% of a campaign node's graph neighbours were
other campaign events — so separating the cluster was close to trivial, for reasons that
had nothing to do with the detector. Fixing the density to a realistic 1,600 events/hour
uncovered three further defects that the thin traffic had masked:

- every entity type shared one Zipf draw with a clamp, which piled the entire tail onto a
  single id — **one device carried 42% of all traffic**, forcing the velocity threshold so
  high that no campaign of any shape could trip it;
- the frontier used `max(τ)` as its binding constraint, when a multi-entity engine fires if
  *any* counter trips, so blindness actually needs `k ≥ n / min(τ)`;
- the calibration split carried nine campaigns — a 37% positive rate, under which "alert on
  every event" is genuinely cost-optimal, so the threshold search returned a detector that
  fired on the entire stream.

And it reversed a headline finding: raw co-occurrence counting, previously reported at
0.894 PR-AUC and described as a strong baseline, collapses to **0.055** once legitimate
traffic is dense enough to co-occur constantly.

Three lessons, none of them about neural networks. From (1) and (4): *a property you assert
in a test gets checked; a property you merely intend gets silently violated the moment an
unrelated helper changes.* From (5): *a preprocessing step that seems neutral can decide
your conclusion, and the dangerous ones are those whose bias points your way.* From (6):
*when a result looks too clean, suspect the simulation before congratulating the model — an
easy benchmark hides not one bug but a nest of them.*

---

<sub>Built for the Razorpay AI Buildathon 2026 · Track 02, AI Risk Manager</sub>
