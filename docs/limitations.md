# Limitations, deployment gaps, and scope

[← back to the README](../README.md)

## What is assumed, not measured

Koronis does not identify every fraudulent payment and is not a general fraud model. It
detects coordinated card-testing campaigns where individual events remain ambiguous but
shared infrastructure creates measurable temporal structure. An attacker who uses
genuinely fresh infrastructure for every attempt leaves no graph signal — a real limit,
and also the point of Claim 2: driving `k → n` costs one unit of infrastructure per
attempt.

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
- **BIN thresholds are optimistic for the baseline.** Real BIN ranges carry heavy
  legitimate volume, so `τ_bin` would sit far above the 9 measured here, which makes the
  baseline stronger than reality.
- **The drift guardrail is experimental**, as measured above — not a safety control.

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
- **No adaptation loop.** The model is fitted once and frozen, which is what makes the
  hold-out meaningful here and is *not* what a deployment wants. Attacks move. A
  production version needs a retraining cadence, analyst dispositions fed back as labels,
  and drift monitored on a far slower timescale than detection — the last of which is
  exactly the confound measured above.
- **Single-merchant scope by default.** The aperture experiment above quantifies what a
  gateway-wide view is worth; running it that way raises data-governance questions this
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

It reproduces only attack characteristics already documented publicly in Visa's
anti-enumeration guidance.
