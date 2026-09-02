import lightgbm as lgb
import numpy as np
import pandas as pd

_FEATURES = ["amount", "log_amount", "hour", "approved",
             "amount_is_micro", "email_is_free"]


def transaction_features(events: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction features only — deliberately no cross-event aggregation.

    That restriction IS the baseline's thesis: scoring attempts in isolation.
    A card-testing attempt is individually unremarkable, so a model that never
    looks across events has nothing to find, however strong the learner.
    """
    amt = events["amount"].to_numpy()
    return pd.DataFrame({
        "amount": amt,
        "log_amount": np.log1p(amt),
        "hour": (events["ts"].to_numpy() // 3600) % 24,
        "approved": events["approved"].to_numpy().astype(float),
        "amount_is_micro": (amt < 25.0).astype(float),
        "email_is_free": events["email_domain"]
            .isin(["gmail.com", "yahoo.com", "outlook.com"]).to_numpy().astype(float),
    })[_FEATURES]


class GBDTDetector:
    def __init__(self, seed: int, n_estimators: int = 300, num_leaves: int = 31):
        self.seed = seed
        # Capacity is a parameter so `koronis.cli ceiling` can sweep it. The
        # defaults are the headline baseline's and are not changed by that sweep.
        self.n_estimators, self.num_leaves = n_estimators, num_leaves
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, events: pd.DataFrame) -> None:
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators, learning_rate=0.05,
            num_leaves=self.num_leaves,
            class_weight="balanced", random_state=self.seed, verbose=-1)
        self.model.fit(transaction_features(events), events["label"].to_numpy())

    def score_events(self, events: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() must be called before score_events()")
        return self.model.predict_proba(transaction_features(events))[:, 1]
