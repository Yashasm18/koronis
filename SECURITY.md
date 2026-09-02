# Security Policy

## Scope

Koronis is a **defence-only research prototype**. It is not a production system and
handles no sensitive data:

- No real cardholder data, no real BIN ranges, no live payment integration.
- No network capability anywhere in the package — it operates only on in-memory
  dataframes. Asserted per module in
  [`tests/test_defence_only.py`](tests/test_defence_only.py), which walks the AST and so
  also catches an import inside a function, an aliased import, and `__import__` /
  `eval` / `exec`. The same property is reproducible in one command, and that command is
  itself checked against this file so the two cannot drift:

  ```bash
  grep -rnE "^(import|from) (requests|urllib|socket|http|aiohttp|subprocess)" koronis/
  ```

- The campaign generator exists solely to produce labelled test data. It reproduces only
  attack characteristics already documented publicly in Visa's anti-enumeration guidance
  and cannot be pointed at any external system.

## Integrating this with a payment stack

Koronis is a research prototype, and nothing in this repository talks to a payment system.
This section exists because the question a payment gateway, PSP or acquirer would ask
first is not "how accurate is it" but **"what would this touch, and what could it do?"**
Those answers are properties of the code, so they are stated here with the checks that
hold them true. Where the components would sit is a separate question, answered in the
README's [Where this would run](README.md#where-this-would-run).

### What it needs to see

Five observed values per event, and three identifiers:

| | |
|---|---|
| Per-event features | amount (log), a micro-amount flag, the **authorisation outcome**, hour of day, a free-mail flag |
| Identifiers | `device_id`, `ip_id`, `bin_id` |

That is the whole input — the model's sixth feature is a bias constant, not something an
integrator sends. **No card number is read anywhere.** `card_id` exists in the
event schema and is deliberately not a model input — [`velocity.py`](koronis/models/velocity.py)
excludes it explicitly, because a card-testing attempt uses a fresh card every time, which
is what makes per-card counting useless against this attack in the first place.

### The identifiers can be tokenised, and it costs nothing

[`build_edges`](koronis/graph/build.py) links two events when they **share** a value. It
groups by the column and never reads the value itself, so any bijective relabelling —
a salted hash, a network token, an opaque surrogate key — has to produce the same graph.

It does, bit for bit. Salted-SHA-256 `device_id`, `ip_id` and `bin_id` and every score is
**numerically identical**, asserted in
[`test_pseudonymisation_is_lossless.py`](tests/test_pseudonymisation_is_lossless.py).
An integrator never has to send a raw identifier to get the same answer.

**One exception, and it is checked too.** `email_domain` cannot be tokenised: a node
feature asks whether the domain is a free-mail provider, so that column carries meaning
beyond equality. Hashing it moves scores by up to 1.7e-2. Either send the domain, or
compute that flag upstream and send it. The test asserts the exception as well as the
rule, so a second value-reading feature cannot be added quietly.

### What it cannot do

- **It cannot reach anything.** There is no network capability in the package — no HTTP
  client, no socket, no subprocess, no process spawning. It operates on in-memory
  dataframes. Asserted per module in
  [`tests/test_defence_only.py`](tests/test_defence_only.py); the same test checks that no
  generated identifier could be mistaken for real card data.
- **It cannot act.** Every output is a recommendation. `hold_review` queues an incident
  for a person with an audit dossier; it does not block a payment, and no code path in
  this repository can. The rupee figures attached to each rung are declared assumptions,
  not measurements.
- **It cannot sit on the checkout path.** Koronis is post-authorisation by construction: it
  reads the authorisation outcome, which exists only after an attempt is submitted. It
  consumes the outcome stream asynchronously.

### Data it would hold, and for how long

State is bounded by the window rather than by traffic history: a one-hour scoring window,
and frequency state fixed at **4 MB** regardless of how many distinct values pass through.
Decisions expire with the window. Nothing accumulates a profile of an entity across days,
so there is no long-lived store to protect — a consequence of the design, not a policy
layered on top.

That bound is **measured, not assumed**, because an earlier version of this section
asserted it while the streaming scorer was in fact retaining one row per event ever seen.
The caches and the entity index are both evicted against the window now, and
[`test_runtime_resilience.py`](tests/test_runtime_resilience.py) fails if they start
tracking total traffic again.

### What would have to be true before real traffic

Stated plainly, because the honest answer is "more than has been done":

- The model is fitted once and frozen. That is what makes the hold-out honest; it is not
  what a deployment wants.
- The data is semi-synthetic, and BIN — the relation carrying most of the signal — is the
  one whose real behaviour differs most from the simulation.
- **The drift guardrail is experimental and should not be relied on.** It flags base
  traffic too often, and the reason is measured: the signal partly tracks campaign shape
  rather than merchant shift.
- Any deployment would begin in **shadow mode** — scoring and logging without acting —
  until its false-positive rate had been measured on real traffic.

Full accounting in [docs/limitations.md](docs/limitations.md).

## Supported versions

Only the `main` branch is maintained. There are no tagged releases.

## Reporting a vulnerability

If you find a security issue in the code or its dependencies:

1. **Preferred:** open a private report via GitHub's
   [Security Advisories](https://github.com/Yashasm18/koronis/security/advisories/new)
   ("Report a vulnerability").
2. **Alternative:** email `sssyashu850@gmail.com` with the details and, if possible, a
   minimal reproduction.

Please do not open a public issue for anything that could be exploited.

This is a student project maintained by one person, so responses are best-effort. Expect
an acknowledgement within about a week.

## Out of scope

- The synthetic campaign generator producing attack-shaped data is by design and clearly
  labelled; it is not a vulnerability.
- Model evasion (an attacker spreading a campaign across enough fresh infrastructure to
  leave no graph signal) is a documented limitation and an open research question, not a
  defect. See the "Limitations" section of the README.
