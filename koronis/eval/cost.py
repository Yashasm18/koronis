"""Cost constants for the rupee model.

These are ASSUMPTIONS, not measurements, and the README says so. Each carries
its reasoning so a reviewer can substitute their own numbers and rerun.
"""

# What one card-testing attempt costs the merchant. Three components:
#   - the authorization fee, charged on every attempt including declines
#   - an amortized share of card-network enumeration penalties, which trigger
#     once the decline ratio crosses scheme thresholds
#   - the expected downstream chargeback cost on cards the attack validates
# Published per-component figures vary by acquirer and scheme, so this is a
# deliberately round central estimate rather than a sourced constant.
COST_PER_ATTEMPT_INR = 73.0

# What blocking one legitimate checkout costs: lost order margin plus a churn
# proxy for the customer who was turned away at payment.
COST_PER_FALSE_BLOCK_INR = 40.0

# Merchant scale used only for the illustrative monthly-impact figure in the
# README. Chosen to represent a mid-size Indian D2C merchant.
ILLUSTRATIVE_ATTEMPTS_PER_DAY = 40_000
