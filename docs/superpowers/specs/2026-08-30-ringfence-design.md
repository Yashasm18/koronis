# Ringfence — Design Spec

**Razorpay AI Buildathon 2026 · Track 02: AI Risk Manager**
Spec dated 2026-08-25 · Build starts 2026-08-29 · Deadline 2026-09-05 (7 build days)

> Early detection of distributed card-testing rings on a temporal graph,
> evaluated on how *fast* it detects — not just how well.

---

## 1. The class of loss

**Card testing / BIN enumeration.** Fraudsters buy a BIN range (~$10–50 on dark
web markets) and push thousands of micro-transactions through a merchant's
checkout to discover which stolen cards are live. A typical attack runs
**1,000–5,000 transactions in under 60 hours**.

The merchant pays authorization fees on every attempt, absorbs network penalties
(Visa's anti-enumeration programs), carries the operational load, and later eats
the chargebacks on cards that got validated and resold.

## 2. Why existing defenses fail — the three facts this project is built on

**Fact 1 — Per-entity velocity rules are blind by construction.**
> *"IP-rate velocity rules were designed for one lazy fraudster and one static IP.
> Distributed card testing spreads across rotating IPs and small amounts, so every
> individual attempt sits below your threshold, and the transaction ML tools score
> them one by one."*

**Fact 2 — Per-transaction ML is blind too.** Each attempt is individually
unremarkable: a real card, a tiny amount, slow enough to clear every threshold.
Stripe Radar's documented weakness on card testing is specifically *"timing and
distribution."* Scoring transactions independently cannot see a coordinated set.

**Fact 3 — and this is what makes it a research problem, not a feature.**
> *"Most BIN attack damage — authorization fees, network penalties, operational
> overhead — occurs **before a single chargeback is filed**."*

Damage accrues *before the labels exist*. Chargeback-supervised systems are
structurally late.

**The consequence:** the ring is invisible per-entity and visible only as
**coincidence structure** across entities and time. And because damage is
cumulative from attack onset, **when** you detect determines **how much** you save.

## 3. The contribution

### 3.1 A detectability frontier, not just a detector

The primary claim is not "my model beats the baselines" — every submission claims
that. It is a characterization of **when detection is possible at all**.

Let an attacker make `n` attempts spread across `k` entities of some type (IPs,
devices). A per-entity velocity detector fires when one entity exceeds `τ` in a
window.

**Claim 1 — threshold detectors are provably blind above a spread.**
Attempts per entity ≈ `n/k`. Detection requires `n/k > τ`, i.e. `k < n/τ`. An
attacker with `k ≥ n/τ` escapes **for any n**. And `τ` cannot be lowered freely:
legitimate heavy users (offices, CGNAT, shared devices) have their own count
distribution, so `τ` is bounded below by the false-positive budget. There is
therefore a region undetectable at *any* acceptable FP rate. This is a proof, not
an experimental finding.

**Claim 2 — graph signal scales differently.**
Attempt-pairs sharing an entity ≈ `k · C(n/k, 2) ≈ n²/2k`. Co-occurrence signal
grows as **n²/k** where the per-entity signal is only **n/k**. The graph's
advantage *increases* with attack size.

**Consequence.** The attacker's only escape from the graph is `k → n` — one fresh
entity per attempt, zero reuse — which is bounded by infrastructure cost. The
`(n, k)` plane therefore splits into three regions: velocity suffices, graph-only,
and uneconomic-to-attack.

**The experiment that makes it science:** plot measured campaigns on those axes and
show the empirical boundary lands where the theory predicts. This converts the
baselines' failure from an anecdote ("maybe you tuned them badly") into a
consequence of Claim 1 — removing the strongest available objection to the result.

### 3.2 Latency as the reported metric

Published work reports precision/recall at end state. Ringfence reports
**precision, recall, and rupees prevented as a function of detection latency** —
how good is the model at hour 1, hour 3, hour 12 after attack onset?

That reframing does two things. It is the metric that actually maps to money.
And it forces the hard version of the ML problem: **be confident early, on partial
evidence**, rather than accurate eventually.

**The ablation is the argument:**

| Method | Expected blind spot |
|---|---|
| Per-entity velocity rules | Distributed attacks sit under every threshold |
| Per-transaction GBDT | Scores attempts independently; cannot see coordination |
| Static relational graph | Sees structure, but only after it has accumulated |
| **Temporal relational graph (ours)** | Target: same structure, far earlier |

Each row's blind spot is quantified, not asserted.

---

## 4. Method

### 4.1 Graph construction
Attempt-level nodes joined through shared entities: **card / BIN, device
fingerprint, IP / ASN, email, merchant**. Two attempts are related when they share
an entity; edges are typed by which entity, and timestamped.

### 4.2 Model
Relational message passing **implemented from scratch** — per-relation weight
matrices, attention over relations, temporal decay on edge weights. Written
directly in PyTorch rather than imported from DGL/PyG, because the point is to
demonstrate the mechanism is understood.

Two non-negotiable design properties:

**Inductive, not transductive.** The model must score entities it has never seen —
production sees new IPs and device fingerprints hourly. Neighborhood aggregation
(GraphSAGE-style) over learned *entity-type* embeddings rather than per-entity
lookup tables, and evaluated explicitly on held-out unseen entities. A transductive
model is useless in this domain; the evaluation should demonstrate we know that.

**Heterophily-aware.** Vanilla GNNs assume connected nodes share labels. Fraud
graphs violate this deliberately — rings wire themselves to legitimate nodes as
camouflage, and this is the named open problem in the 2026 graph-fraud literature.
Edge reweighting to down-weight camouflage edges, **with an ablation** showing what
it bought.

Two-level output:
- **attempt-level** risk score
- **ring-level** cluster score — this is what triggers action, because you defend
  against a campaign, not a transaction

### 4.3 Early classification
Scored at sliding windows over the replayed stream. A ring is declared when its
cluster score crosses threshold; **time-to-detection from attack onset is
recorded** for every campaign.

### 4.4 Known hard parts (deliberately kept, not avoided)
- **Heterophily / camouflage** — rings deliberately wire to legitimate nodes; the
  documented failure mode of vanilla GNNs and an open problem in the 2026 literature
- **Extreme class imbalance** — focal loss, negative sampling, calibration
- **Streaming budget** — the graph cannot be rebuilt per transaction; windowed
  subgraph extraction under a latency budget
- **Explainability** — subgraph attribution, because you cannot block a customer
  on an unexplainable score

---

## 5. Evaluation — and the trap we must not fall into

**The trap:** train and test on the same synthetic generator and the model learns
the generator, not fraud. Any serious reviewer asks this within two minutes.

**The design that answers it: hold out attack *morphology*, not just samples.**
Train on loud, concentrated campaigns. Test on unseen low-and-slow distributed
ones. Generalizing to an attack *shape* never seen in training is evidence the
graph structure carries real signal.

### Metrics reported
- Attempt-level precision / recall / PR-AUC on a held-out test set *(the brief's
  explicit requirement)*
- Ring-level detection rate; **time-to-detection: median and p90**
- **₹ prevented as a function of latency**, via an explicit cost model
- **False-positive cost in ₹** *(the brief's explicit requirement)* — a blocked
  legitimate checkout is a lost order plus a churn proxy, and at ring level a false
  positive blocks a whole cluster, so FP cost is superlinear. Stated, not buried.
- The full 4-way ablation table, **plus** the heterophily-reweighting ablation
- **Calibration** — a reliability diagram, and the operating threshold chosen by
  minimising *expected rupee cost* rather than by maximising F1. Almost no
  hackathon project calibrates; it is the clearest available signal that this is a
  system rather than a notebook.
- **Inductive generalisation** — performance on entities absent from training
- **The detectability frontier** — measured campaigns plotted on the `(n, k)` plane
  against the predicted `k = n/τ` boundary (§3.1)

**Cost model assumptions** (auth fee per attempt, chargeback cost, penalty
thresholds, average order value) are declared in one place, sourced where
sourceable, and marked as assumptions where not. No number is presented as
measured when it is assumed.

---

## 6. Defense-only compliance

The brief: *"Strictly defense-only: anything offense-capable is disqualified."*

Ringfence is a detector. The synthetic attack generator exists solely to produce
labeled training and test data — as it does in every fraud-ML paper — and is
constrained accordingly:

- operates on synthetic logs only; **no network capability whatsoever**
- no real BIN ranges, no real card numbers, no live endpoints
- reproduces only attack characteristics already publicly documented by Visa's
  own anti-enumeration guidance; publishes nothing novel or operational

This gets **its own README section**, stated plainly, so a judge never has to
wonder.

---

## 7. Scope for 14 days

**In scope (core):** synthetic stream generator with labelled campaigns of varying
morphology; graph construction; from-scratch relational message passing; the three
baselines; the latency evaluation harness; the cost model; subgraph explanations;
a report page; the full metrics set above.

**Simplification taken deliberately:** sliding-window graph snapshots rather than a
full continuous-time dynamic GNN (TGN-style). Continuous-time is the better answer
and does not fit the calendar. This is documented as a limitation, not hidden.

**Label-efficiency curve (kept, cheap):** train on 100/50/25/10% of labels and
chart the degradation. Makes the label-scarcity point — labels arrive after the
damage — with measured evidence, for roughly the cost of a few re-runs.

**Cut for time:** self-supervised contrastive pretraining. It was the deepest
upgrade available and it does not fit 7 days without threatening the core. The
label-efficiency curve carries the same insight at a fraction of the risk.

**Out of scope:** production serving, real payment integration, multi-tenancy, auth.

### Schedule — build starts Sat 29 Aug, submit Sat 5 Sep (7 build days)

| Day | Date | Work | Submittable? |
|---|---|---|---|
| 1 | Sat 29 | Stream generator + campaign morphologies + graph construction + **inductive split design** | — |
| 2 | Sun 30 | Baselines: velocity rules + per-transaction GBDT · **write the formal boundary argument (§3.1)** | ✅ |
| 3–4 | Mon 31 – Tue 1 | From-scratch inductive message passing + heterophily reweighting | ✅ |
| 5 | Wed 2 | Streaming inference + latency harness | ✅ **thesis proven** |
| 6 | Thu 3 | Cost model + calibration + all ablations + **frontier chart** | ✅ |
| 7 | Fri 4 | Console wired to real output + README + **record video** | ✅ |
| — | Sat 5 | Buffer + **submit** | — |

The boundary argument on day 2 is deliberate: it is thinking work, it needs no
code, and it can be done while baselines train. It is also the highest
value-per-hour item in the entire plan.

**Day 5 is the real deadline** — that is when the core claim is proven. Days 6–7
are presentation, and presentation is what gets sacrificed if anything slips.

**The #1 schedule risk is the synthetic generator, not the model.** It has no
natural stopping point and will expand to fill the calendar. Time-box it to day 1
and accept that it is crude: 70% realistic with correct labels proves the thesis;
a beautiful generator delivered on day 3 has already cost the ablation.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| **Model fails to beat baselines** | The velocity baseline is weak by construction (Fact 1), so the margin is likely. If it still loses, report it honestly with the diagnosis — the brief rewards honest metrics, and a measured negative result with a correct evaluation design beats an unmeasured claim |
| Learns the generator, not fraud | Held-out attack morphology (§5) |
| Training doesn't converge in time | Baselines land by day 5, so there is always a working submission |
| Graph blows up in memory | Window-bounded subgraphs; cap node degree |
| Defense-only misread | §6, its own README section |
| Synthetic data looks toy | Model real artifacts: legitimate bursts (sales, paydays), shared IPs (offices, CGNAT), device-fingerprint collisions — the things that create genuine false positives |

---

## 9. Deliverables

- Public GitHub repo, clear README, honest commit history
- 5-minute pitch video
- **"What broke, and how you got out"** — the form says *"the last one is the one we
  read first."* Keep an engineering log from day one. Likely candidates: class
  imbalance collapsing the model to all-negative, or the streaming window making
  early detection impossible because the ring hasn't accumulated enough edges yet.

### README opening must contain
The words **detector**, **precision and recall**, **held-out test set**, and
**false-positive cost** — a judge scanning for track fit is looking for exactly
those.
