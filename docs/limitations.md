# Limitations, deployment gaps, and scope

[← back to the README](../README.md)

## What is assumed, not measured

Koronis does not identify every fraudulent payment and is not a general fraud model. It
detects coordinated card-testing campaigns where individual events remain ambiguous but
shared infrastructure creates measurable temporal structure. An attacker who uses
genuinely fresh infrastructure for every attempt leaves no graph signal — a real limit,
and also the point of Claim 2: driving `k → n` costs one unit of infrastructure per
attempt.

**That limit is stated, not measured, and the attempt to measure it failed.** Pushing the
sweep to `k = n` ([saturation](evaluation.md#where-does-this-model-stop-working--an-invalid-measurement-published))
returns recall 1.0, which is a fact about the generator rather than the model: campaign
entities come from a pool disjoint from the background's, so at `k = n` every campaign event
has degree zero while essentially no legitimate event does, and "has no neighbours" becomes
a free label. Real traffic is full of first-time customers on fresh infrastructure. Testing
this properly needs a background with a realistic first-time-customer rate — a change to the
data generator, not the model — and it has not been made.

This is a **semi-synthetic proof of concept**, not production fraud detection.

- **Background traffic is synthetic, and keeping it that way was a measured call.** The
  loader has an IEEE-CIS path ([`background.py`](../koronis/data/background.py)) and the real
  dataset was pulled and profiled. In a contiguous slice its native density is ~210
  events/hour (the first 6,000 rows span 28 h); the bootstrap sampler runs at ~1,500
  events/hour *by design*, because thin background was a corrected defect — at low density
  an injected campaign becomes the majority of the traffic in its own window and
  separating it is trivial. Feeding IEEE-CIS in directly reintroduces that regime;
  compressing its timeline to match the density would discard the real inter-arrival
  structure that is the only reason to prefer it. `DeviceInfo` also lives in
  `train_identity.csv` (≈24% row coverage), not `train_transaction.csv`.
- **Campaigns are injected, not observed.** Ground truth exists because it was
  constructed. The hold-out extrapolates to an unseen spread, but this is not the same as
  detecting a campaign in the wild.
- **Cost constants are estimates.** ₹73 per attempt and ₹40 per false block are declared
  in [`cost.py`](../koronis/eval/cost.py) with reasoning. Substitute your own and rerun.
- **`money_prevented` assumes detection halts the campaign instantly.** Read it as the
  *cost of latency* — how much more is lost per minute of delay — not guaranteed savings.
- **BIN thresholds are optimistic for the baseline.** The tuned `τ_bin` here is **236**
  (`τ_device = 8` is the binding one). Real BIN ranges carry far heavier legitimate volume,
  so holding a real false-positive budget would push `τ_bin` higher still — a counter that
  trips less easily than the one measured here. The baseline is therefore given better
  conditions than reality, which is the conservative direction for this comparison.
- **The drift guardrail is experimental** — [as measured](evaluation.md#traffic-profile-transfer-stress-test),
  not a safety control.

### What a production deployment would still need

Stated because the gap between this and a deployed system is itself a design question,
and a reviewer should not have to guess where the seams are.

- **Where it sits.** Koronis is **post-authorisation by construction**: the authorisation
  outcome is one of its two mechanisms, and an outcome exists only after an attempt is
  submitted. It cannot prevent the attempt it learns from — the value is in the attempts
  that follow. A deployment would score inline after auth, where ~1 ms is affordable, and
  feed the decision layer asynchronously.
- **The whole pipeline is causal, including consolidation.** `StreamingKoronis.push`
  scores online and reproduces batch scores exactly. `StreamingIncidents` groups online:
  the link-share cap comes from a sliding count-min sketch fed event by event, so a
  decision at time `t` uses only what was known at `t`. `build_incidents` remains as the
  batch reference, and it is the one using the future — see
  [Evaluation → online consolidation](evaluation.md#making-consolidation-causal) for what
  the difference costs.
- **Sharding costs accuracy, and no routing makes it free.** Measured in
  [Evaluation](evaluation.md#does-the-graph-survive-being-split-across-machines): a
  partitioned graph loses edges, and the routing key decides whether you lose recall or
  precision. [Replication recovers it](evaluation.md#recovering-the-edges-a-partition-deletes) at
  under 3× the scoring work and costs less than not replicating at every shard count — but the
  undivided stream still beats every partitioned configuration
  (₹645 against
  ₹3,212 at sixteen
  shards), so horizontal scale is a cost to be justified rather than a free lever.
- **No adaptation loop.** The model is fitted once and frozen, which is what makes the
  hold-out meaningful here and is *not* what a deployment wants. Attacks move. A
  production version needs a retraining cadence, analyst dispositions fed back as labels,
  and drift monitored on a far slower timescale than detection — the last of which is
  exactly the confound measured above.
- **Single-merchant scope by default.** The
  [aperture experiment](evaluation.md#vantage-point-one-merchant-or-the-whole-gateway)
  quantifies what a gateway-wide view is worth; running it that way raises data-governance questions this
  prototype does not address.

### Defence-only

Koronis is a detector; the campaign injector exists solely to produce labelled test data.
Every recommended action is decision support in a modelled workflow — nothing blocks a
payment, calls a gateway or touches live infrastructure. There are no real BIN ranges, no
real card numbers, no live endpoints, and no network capability anywhere in the codebase:

```bash
grep -rnE "^(import|from) (requests|urllib|socket|http|aiohttp|subprocess)" koronis/
# no matches
```

That command is the reader-facing check; the enforced one is
[`tests/test_defence_only.py`](../tests/test_defence_only.py), which walks the AST of every
module and so also catches an import inside a function, an aliased import, and
`__import__` / `eval` / `exec`.

It reproduces only attack characteristics already documented publicly in Visa's
anti-enumeration guidance.
