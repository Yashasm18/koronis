"""Traffic profiles for a transfer stress test.

These are **synthetic merchant shapes, not real merchants**. Passing this test
is evidence that the detector is not tuned to one traffic profile; it is not
evidence of production cross-merchant transfer, and the README says so.

All three shifted profiles are defined here, before any of them is evaluated.
Choosing a shift after seeing which one the model survives is how a stress test
becomes a demonstration.

Each profile breaks a different assumption the base profile satisfies:

  subscription  legitimate device and card reuse is HIGH, because customers
                come back. Dense legitimate co-occurrence is normal here, which
                is precisely the signal the graph relies on.

  marketplace   entities are diffuse: many sellers, little reuse. The graph is
                sparse, and thresholds calibrated on denser traffic sit in the
                wrong place.

  flash_sale    a legitimate burst. Traffic compresses into a short window and
                the decline rate rises under load — structurally similar to a
                campaign, without being one. This is the profile most likely to
                produce a false escalation, which is why it is included.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    entity_shape: dict          # entity -> (pool fraction, power-law alpha)
    span_s: float               # wall-clock span the traffic occupies
    decline_rate: float         # legitimate decline rate
    blurb: str


# Base: the profile everything is trained and calibrated on. Nothing else may
# be used for fitting, thresholds, or the drift cut-off.
BASE = Profile(
    name="base",
    entity_shape={"device": (0.80, 0.45), "ip": (0.25, 0.70), "bin": (0.02, 0.90)},
    span_s=4 * 3600.0,
    decline_rate=0.08,
    blurb="mixed e-commerce, ~1,600 events/hour",
)

SHIFTED = [
    Profile(
        name="subscription",
        # Small device and IP pools, heavily concentrated: the same customers
        # return on the same devices.
        entity_shape={"device": (0.12, 0.85), "ip": (0.10, 0.85), "bin": (0.02, 0.90)},
        span_s=4 * 3600.0,
        decline_rate=0.11,
        blurb="high-repeat subscription merchant; legitimate reuse is dense",
    ),
    Profile(
        name="marketplace",
        # Large pools, flat: many one-off buyers, little shared infrastructure.
        entity_shape={"device": (0.95, 0.25), "ip": (0.70, 0.35), "bin": (0.06, 0.60)},
        span_s=4 * 3600.0,
        decline_rate=0.06,
        blurb="diffuse marketplace; little legitimate reuse",
    ),
    Profile(
        name="flash_sale",
        # Same entity mix as base, but four hours of traffic arrives in one,
        # and declines rise under load.
        entity_shape=dict(BASE.entity_shape),
        span_s=1 * 3600.0,
        decline_rate=0.22,
        blurb="legitimate burst; high volume and elevated declines, no attack",
    ),
]

BY_NAME = {p.name: p for p in [BASE, *SHIFTED]}
