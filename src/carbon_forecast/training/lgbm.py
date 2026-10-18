"""LightGBM forecaster.

fit:     learns from the precomputed feature table, using only rows whose target
         was already published in the history the harness provides.
predict: builds features live with the same build_features used for the table,
         so training and serving are identical.
"""
import lightgbm as lgb
import pandas as pd

from carbon_forecast.features.build import FEATURE_COLUMNS, build_features

DEFAULT_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
OBJECTIVES = {"l2": "regression", "l1": "regression_l1"}


class LightGBMForecaster:
    def __init__(self, feature_table: pd.DataFrame, objective: str = "l2",
                 train_from: str | None = None, params: dict | None = None):
        self.table = feature_table.sort_values(["issue_time", "horizon"]).reset_index(drop=True)
        self.objective = objective
        self.train_from = pd.Timestamp(train_from, tz="UTC") if train_from else None
        self.params = {**DEFAULT_PARAMS, **(params or {}), "objective": OBJECTIVES[objective]}
        self.name = f"lgbm_{objective}_" + (f"from{self.train_from.year}" if self.train_from else "all")
        self.model = None
        self.n_train = 0

    def _training_mask(self, history: pd.DataFrame) -> pd.Series:
        # Latest period visible in the harness's history: nothing after it may be used as a target
        last_known = history["period_start"].max()
        mask = self.table["is_valid_target"] & (self.table["target_time"] <= last_known)
        if self.train_from is not None:
            mask &= self.table["target_time"] >= self.train_from
        return mask

    def fit(self, history: pd.DataFrame, weather: pd.DataFrame) -> None:
        mask = self._training_mask(history)
        self.n_train = int(mask.sum())
        if self.n_train == 0:
            raise ValueError("No training rows available before the cutoff")
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(self.table.loc[mask, FEATURE_COLUMNS], self.table.loc[mask, "y"])

    def predict(self, history, weather, issue_time, target_times):
        feats = build_features(history, weather, issue_time, target_times)
        return self.model.predict(feats[FEATURE_COLUMNS]).astype("float64")

    def feature_importance(self) -> pd.Series:
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=FEATURE_COLUMNS, name="gain_share").sort_values(ascending=False).round(4)

class BoosterForecaster:
    """Serves an already-trained LightGBM model. fit() deliberately does nothing,
    so the model is scored (or served) exactly as it was registered."""

    def __init__(self, booster: lgb.Booster, name: str = "champion"):
        self.booster = booster
        self.name = name

    def fit(self, history: pd.DataFrame, weather: pd.DataFrame) -> None:
        pass

    def predict(self, history, weather, issue_time, target_times):
        feats = build_features(history, weather, issue_time, target_times)
        return self.booster.predict(feats[FEATURE_COLUMNS]).astype("float64")