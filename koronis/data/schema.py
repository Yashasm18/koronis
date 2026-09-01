from dataclasses import dataclass

EVENT_COLUMNS = [
    "event_id", "ts", "amount", "card_id", "bin_id",
    "device_id", "ip_id", "email_domain", "approved",
    "label", "campaign_id",
]

# Relations are the entity types two events can share. Order is fixed because
# model weights are indexed by relation position.
#
# RELATIONS describes what the DATA contains. Incident linking, the sharding
# sweep and the aperture experiment all legitimately use every one of them.
RELATIONS = ["device_id", "ip_id", "bin_id", "email_domain"]

# MODEL_RELATIONS is what the DETECTOR consumes, which is a separate question
# and a narrower answer. `koronis.cli select` scored eight candidate
# architectures on the calibration split and chose this one; email_domain was
# dropped there. Keeping the two lists apart matters: email is weak evidence
# for the model and still perfectly good evidence for whether two alerts belong
# to the same incident, where the link-share cap already governs it.
MODEL_RELATIONS = ["device_id", "ip_id", "bin_id"]


@dataclass(frozen=True)
class CampaignSpec:
    """One injected card-testing campaign.

    n_attempts: total attempts (the `n` of the detectability frontier)
    k_devices:  distinct device fingerprints the attacker owns (the `k`)
    k_ips:      distinct IPs; independent of k_devices
    duration_s: wall-clock span of the campaign
    start_ts:   when it begins, in the background stream's time base
    n_bins:     how many BIN ranges are being enumerated
    camouflage: how hard the attacker works to look ordinary per transaction,
                in [0, 1]. At 0 the attempts are naive - micro-amounts, a
                giveaway email set - and any row-level model catches them. At
                1 amounts and email domains are drawn from the background's
                own distribution, so no single attempt is distinguishable and
                the only remaining signal is coordination across attempts.

                This is the axis that separates the attacks existing tools
                already handle from the ones that cost real money. An attacker
                fully controls amount and email domain; they cannot control
                whether a stolen card authorises, but a single decline is
                unremarkable - 8% of legitimate traffic declines too - so the
                decline *rate* is only visible in aggregate, which is a
                grouping signal rather than a per-transaction one.
    """
    n_attempts: int
    k_devices: int
    k_ips: int
    duration_s: float
    start_ts: float
    n_bins: int = 2
    camouflage: float = 0.0
