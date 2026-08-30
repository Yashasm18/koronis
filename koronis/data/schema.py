from dataclasses import dataclass

EVENT_COLUMNS = [
    "event_id", "ts", "amount", "card_id", "bin_id",
    "device_id", "ip_id", "email_domain", "approved",
    "label", "campaign_id",
]

# Relations are the entity types two events can share. Order is fixed because
# model weights are indexed by relation position.
RELATIONS = ["device_id", "ip_id", "bin_id", "email_domain"]


@dataclass(frozen=True)
class CampaignSpec:
    """One injected card-testing campaign.

    n_attempts: total attempts (the `n` of the detectability frontier)
    k_devices:  distinct device fingerprints the attacker owns (the `k`)
    k_ips:      distinct IPs; independent of k_devices
    duration_s: wall-clock span of the campaign
    start_ts:   when it begins, in the background stream's time base
    n_bins:     how many BIN ranges are being enumerated
    """
    n_attempts: int
    k_devices: int
    k_ips: int
    duration_s: float
    start_ts: float
    n_bins: int = 2
