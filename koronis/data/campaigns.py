"""Labelled card-testing campaigns, generated for evaluation. **Defence-only.**

This is the file in the repository whose name looks most alarming, so what it is
and is not belongs at the top of it rather than in a docstring further down.

It operates on **in-memory dataframes only**. It makes no network call and can
make none - `tests/test_defence_only.py` walks the AST of every module in the
package and fails on any import that could reach outside the process. It uses no
real BIN ranges, no real card numbers and no real cardholder data; every
identifier it emits is a synthetic label such as `camp_0_3c91727f_d24`, and
`tests/test_defence_only.py` asserts that none of them could be mistaken for
card data. It cannot be pointed at an external system, because there is nothing
here that reaches one.

It reproduces only characteristics already published in Visa's anti-enumeration
guidance: many attempts, small amounts, spread across entities, mostly declined.
Nothing here is a technique a defender does not already have to handle.

**Why a detection project generates attacks at all.** Without labelled campaigns
there is no ground truth, and without ground truth there is no measured precision,
no measured recall, and no false-positive cost - the numbers the rest of this
repository is built to report. The generator is what makes the honest metrics
possible; it is a requirement of the evaluation, not a capability of the product.
"""
import numpy as np
import pandas as pd

from .schema import CampaignSpec, EVENT_COLUMNS


def inject(background: pd.DataFrame, specs: list[CampaignSpec],
           seed: int) -> pd.DataFrame:
    """Add labelled card-testing campaigns to a background stream.

    Defence-only; see the module docstring above for the full statement and the
    test that enforces it.
    """
    rng = np.random.default_rng(seed)
    frames = [background]
    # Entity ids are namespaced per generated dataset. Without this, two
    # datasets built with different seeds reuse the same device/IP names, and
    # a train/test split would leak entity identity — quietly permitting a
    # transductive shortcut and invalidating the inductive claim.
    ns = f"{int(rng.integers(0, 2**31)):08x}"
    for c, spec in enumerate(specs):
        frames.append(_one_campaign(spec, f"camp_{c}", ns, background, rng))
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.sort_values("ts", kind="mergesort").reset_index(drop=True)
    ev["event_id"] = [f"e_{i}" for i in range(len(ev))]
    return ev[EVENT_COLUMNS]


def _blend(spec: CampaignSpec, n: int, background: pd.DataFrame, rng):
    """Interpolate the campaign's per-transaction marginals toward the
    background's, according to `camouflage`.

    Each attempt independently either looks naive or is bootstrapped from real
    background traffic, with the mix set by `camouflage`. Bootstrapping rather
    than fitting a distribution keeps the camouflaged rows exactly as realistic
    as the traffic they hide in.
    """
    c = float(np.clip(spec.camouflage, 0.0, 1.0))
    hide = rng.random(n) < c

    naive_amt = np.round(rng.uniform(1.0, 20.0, n), 2)
    bg_amt = rng.choice(background["amount"].to_numpy(), n)
    amount = np.where(hide, bg_amt, naive_amt)

    naive_dom = rng.choice(["gmail.com", "outlook.com"], n)
    bg_dom = rng.choice(background["email_domain"].to_numpy(), n)
    email = np.where(hide, bg_dom, naive_dom)

    return amount, email


def _one_campaign(spec: CampaignSpec, cid: str, ns: str,
                  background: pd.DataFrame, rng) -> pd.DataFrame:
    n = spec.n_attempts
    tag = f"{cid}_{ns}"
    amount, email = _blend(spec, n, background, rng)
    devices = np.array([f"{tag}_d{i}" for i in range(spec.k_devices)])
    ips = np.array([f"{tag}_i{i}" for i in range(spec.k_ips)])
    bins = np.array([f"{tag}_b{i}" for i in range(spec.n_bins)])

    # Round-robin assignment guarantees exactly k distinct entities appear AND
    # that load is uniform across them, which is what makes the (n, k) frontier
    # sweep well defined.
    #
    # Each entity type gets an INDEPENDENT shuffle of that assignment. Using
    # the same order for all three made device, IP and BIN partition the
    # campaign identically - device 0, IP 0 and BIN 0 covered the very same
    # attempts - so the three relations produced one grouping and never
    # cross-linked. The campaign was 60 disjoint cliques with no bridge between
    # them, and only a shared email domain held it together. A real attacker
    # does not rotate devices, IPs and BIN ranges in lockstep; independent
    # assignment lets the relations cross-cut, which is what makes the campaign
    # one connected component for the right reason.
    #
    # Every entity type must use it. Assigning one of them randomly instead -
    # BINs originally used rng.choice - looks equivalent but is not: the
    # multinomial maximum runs far above the mean, so at n=400, k=50 the
    # busiest BIN drew 18 attempts against an average of 8 and tripped a
    # threshold of 9. That made the measured frontier disagree with the
    # predicted one on a quarter of the grid, for reasons that had nothing to
    # do with the theory being tested.
    #
    # Uniform spread is also the attacker's best play, so this measures the
    # boundary against a maximally evasive adversary rather than a sloppy one.
    def _assign(pool, k):
        return pool[rng.permutation(n) % k]

    dev = _assign(devices, spec.k_devices)
    ip = _assign(ips, spec.k_ips)
    binseq = _assign(bins, spec.n_bins)

    return pd.DataFrame({
        "event_id": [f"{tag}_{i}" for i in range(n)],
        "ts": np.sort(rng.uniform(spec.start_ts, spec.start_ts + spec.duration_s, n)),
        "amount": amount,
        "card_id": [f"{tag}_c{i}" for i in range(n)],   # a fresh card each attempt
        "bin_id": binseq,
        "device_id": dev,
        "ip_id": ip,
        "email_domain": email,
        "approved": rng.random(n) < 0.04,               # ~96% decline
        "label": 1,
        "campaign_id": cid,
    })
