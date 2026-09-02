"""Cost constants for the rupee model.

These are ASSUMPTIONS, not measurements, and the README says so. Each carries
its reasoning so a reviewer can substitute their own numbers and rerun.
"""

# What one card-testing attempt costs the merchant being tested. Components:
#   - a per-authorisation fee on the attempt. Whether declines carry this
#     depends on the pricing plan: interchange-plus and per-transaction plans
#     generally charge on the authorisation attempt, blended
#     percentage-of-settled-volume plans generally do not. Stated as a
#     dependency rather than a fact, because it is one.
#   - an amortised share of card-network enumeration penalties, which trigger
#     once the decline ratio crosses scheme thresholds
#   - the operational cost of the elevated decline rate itself, which degrades
#     the merchant's own approval profile while the attack runs
#
# Deliberately NOT included: the chargebacks on cards the attack validates.
# Those land on whichever merchant the card is later spent at, which is usually
# somebody else - counting them here would charge this merchant for a loss it
# does not bear. An earlier version of this comment did exactly that.
#
# Published per-component figures vary by acquirer, scheme and plan, so this is
# a deliberately round central estimate, not a sourced constant. Substitute your
# own and re-run; every rupee figure in the repo derives from it.
COST_PER_ATTEMPT_INR = 73.0

# What blocking one legitimate checkout costs: lost order margin plus a churn
# proxy for the customer who was turned away at payment.
COST_PER_FALSE_BLOCK_INR = 40.0

# Merchant scale used only for the illustrative monthly-impact figure in the
# README. Chosen to represent a mid-size Indian D2C merchant.
ILLUSTRATIVE_ATTEMPTS_PER_DAY = 40_000
