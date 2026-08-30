# Koronis

**Detecting distributed card-testing campaigns that per-entity velocity rules cannot see — at any threshold.**

[![tests](https://img.shields.io/badge/tests-56%20passing-2ea44f)](tests/)
[![python](https://img.shields.io/badge/python-3.11%2B-3776ab)](https://www.python.org/)
[![graph libs](https://img.shields.io/badge/graph%20libraries-none-8a3ffc)](koronis/models/layers.py)
[![track](https://img.shields.io/badge/Razorpay%20Buildathon-Track%2002%20·%20AI%20Risk%20Manager-0c2451)](https://razorpay.com/buildathon/)

Koronis is a **detector** for one class of merchant loss: card testing, where a fraudster
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
neighbourhood is not enough.

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
| **1.78 ms** | **3.85 ms** | 6.73 ms | 2.10 ms | ~476 events/sec |

Entity buckets expire on the window, so memory is bounded by window occupancy rather than
stream length — a test holds that too. Reproduce with `python -m koronis.cli benchmark`.

On the held-out stream, Koronis raises its first campaign alert at **t = 0.0 s** — the campaign's opening attempt already crosses the frozen threshold — while the
tuned velocity engine never alerts on the campaign at all. `results/replay.json` carries the
full per-event trace.

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
