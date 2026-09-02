"""What to do with an event that cannot be scored honestly.

Three runtime failures were found by injecting faults into the streaming path,
and all three were silent - the stream kept running and kept returning answers:

1. A non-finite feature (`amount` NaN, `approved` None) produced a NaN score.
   `NaN >= threshold` is False in IEEE arithmetic, so the event was reported as
   "no alert". A missing field made the detector quietly stop detecting.

2. A missing entity became an entity. Values were interned with `str(value)`,
   so `None` turned into the key `"None"` and every event missing a device
   fingerprint linked to every other one. Null device IDs are ordinary in
   production - a browser blocking the fingerprint - so this manufactures a
   ring out of absent data.

3. Nothing was ever evicted from the scoring caches, so memory tracked total
   traffic rather than the window.

The rule adopted here: **an event that cannot be scored honestly is escalated,
never scored anyway.** A quarantined event is counted and surfaced with its
reason; it is not silently given a passing score. Losing an event loudly is
recoverable, and a fraud detector that says "no alert" when it means "I could
not read this" is not.

Missing entity values are treated one-sidedly, the same argument the sketch
uses: refusing to link on an absent value can only fragment an incident, which
an analyst can still see. Linking on it invents coordination that is not there.
"""
import math

from .data.schema import RELATIONS

#: Values that mean "absent", including the ways upstream systems spell it after
#: a null has been through a string column. Matching is case-insensitive.
NULL_MARKERS = frozenset({"", "none", "nan", "null", "na", "n/a", "unknown", "unk", "-"})

#: Fields an event must carry before it can be scored at all.
REQUIRED_FIELDS = ("ts", "amount", "approved", "email_domain", *RELATIONS)


def entity_key(value) -> str | None:
    """The key an entity value is interned under, or None if it is absent.

    Returning None means "this event shares no edge on this relation", which is
    the safe direction: a fragmented incident is visible, an invented one is not.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    key = str(value).strip()
    return None if key.lower() in NULL_MARKERS else key


def _finite(value) -> bool:
    if value is None or isinstance(value, bool):
        return value is not None
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def problems(event) -> list[str]:
    """Why this event cannot be scored. Empty list means it can be."""
    found = []

    for field in REQUIRED_FIELDS:
        if field not in event:
            found.append(f"missing field `{field}`")

    if "ts" in event and not _finite(event["ts"]):
        found.append("`ts` is not a finite number")
    if "amount" in event and not _finite(event["amount"]):
        found.append("`amount` is not a finite number")
    if "approved" in event and event["approved"] is None:
        found.append("`approved` is null")

    if not any(entity_key(event.get(rel)) for rel in RELATIONS):
        # Every relation absent means there is no graph to reason over. The
        # per-event features alone measure PR-AUC 0.380 (mechanism ablation),
        # so a score here would be one the evaluation does not support.
        found.append("no usable entity on any relation")

    return found
