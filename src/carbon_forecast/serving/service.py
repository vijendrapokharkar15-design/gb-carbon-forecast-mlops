"""Forecasting service: issues the official 12:00 UTC forecast from live data with the champion model."""
import logging
from typing import Callable

import pandas as pd
from carbon_forecast.serving.timing import latest_issue_time
from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.evaluation.metrics import best_window_start
from carbon_forecast.features.build import FEATURE_COLUMNS
from carbon_forecast.processing.clean import clean_neso, clean_weather
from carbon_forecast.serving.live_data import fetch_live_neso, fetch_live_weather
from carbon_forecast.training.lgbm import BoosterForecaster
from carbon_forecast.training.registry import ModelRegistry
from carbon_forecast.validation.schemas import (
    RAW_NESO_SCHEMA,
    DataValidationError,
    build_processed_weather_schema,
    build_raw_weather_schema,
    validate,
)

logger = logging.getLogger(__name__)




def greenest_window(forecast: pd.DataFrame, hours: float, not_before: pd.Timestamp | None = None) -> dict | None:
    """Lowest-mean window of `hours` within the forecast, optionally only from `not_before` onwards.
    Returns None if there aren't enough periods left."""
    steps = int(round(hours * 2))
    candidates = forecast
    if not_before is not None:
        candidates = forecast[forecast["period_start"] >= not_before.floor("30min")]
    candidates = candidates.reset_index(drop=True)
    if steps < 1 or len(candidates) < steps:
        return None
    start = best_window_start(candidates["intensity"].to_numpy(), steps)
    window = candidates.iloc[start : start + steps]
    return {
        "start": window["period_start"].iloc[0].isoformat(),
        "end": window["period_end"].iloc[-1].isoformat(),
        "hours": hours,
        "mean_intensity": round(float(window["intensity"].mean()), 1),
    }


class ForecastService:
    def __init__(
        self,
        booster,
        metadata: dict,
        cfg: dict,
        neso_fetcher: Callable = fetch_live_neso,
        weather_fetcher: Callable = fetch_live_weather,
        clock: Callable[[], pd.Timestamp] | None = None,
    ):
        self.booster = booster
        self.metadata = metadata
        self.ev = cfg["evaluation"]
        self.variables = cfg["weather"]["variables"]
        self.locations = cfg["weather"]["locations"]
        self.neso_fetcher = neso_fetcher
        self.weather_fetcher = weather_fetcher
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self._cache: dict = {}

    @classmethod
    def from_registry(cls, cfg: dict | None = None) -> "ForecastService":
        cfg = cfg or load_config()
        champ = ModelRegistry(resolve_path(cfg["promotion"]["registry_dir"])).load_champion(FEATURE_COLUMNS)
        if champ is None:
            raise RuntimeError("No champion model in the registry; run pipelines/train_and_promote.py first")
        booster, metadata = champ
        return cls(booster, metadata, cfg)

    @property
    def model_version(self) -> str | None:
        return self.metadata.get("version")

    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self.clock()).tz_convert("UTC")

    def forecast(self) -> tuple[pd.Timestamp, pd.DataFrame]:
        """(issue_time, forecast table). Computed once per issue time, then served from memory."""
        issue = latest_issue_time(self.now(), self.ev["issue_hour_utc"])
        if issue not in self._cache:
            self._cache = {issue: self._compute(issue)}  # keep only the latest issue
        return issue, self._cache[issue]

    def _compute(self, issue_time: pd.Timestamp) -> pd.DataFrame:
        logger.info("Computing forecast for issue time %s", issue_time)

        # NESO: validate raw, clean, keep only what was published by the cutoff
        raw_neso = self.neso_fetcher(issue_time)
        validate(raw_neso, RAW_NESO_SCHEMA, "live NESO")
        neso = clean_neso(raw_neso)
        cutoff = issue_time - pd.Timedelta(minutes=self.ev["publication_lag_minutes"])
        history = neso[neso["period_start"] < cutoff].reset_index(drop=True)
        if history["actual"].notna().sum() == 0:
            raise DataValidationError("live NESO: no published actuals before the cutoff")

        # Weather: validate raw, reshape to wide hourly, and require a complete grid with no gaps
        names = [loc["name"] for loc in self.locations]
        raw_wx = self.weather_fetcher(self.locations, self.variables)
        validate(raw_wx, build_raw_weather_schema(self.variables), "live weather")
        weather = clean_weather(raw_wx, self.variables, names)
        validate(weather, build_processed_weather_schema(self.variables, names), "live weather (processed)")

        target_times = pd.date_range(issue_time, periods=self.ev["horizon_steps"], freq="30min")
        if weather["time"].min() > target_times[0] or weather["time"].max() < target_times[-1].floor("h"):
            raise DataValidationError("live weather does not cover the forecast horizon")

        preds = BoosterForecaster(self.booster).predict(history, weather, issue_time, target_times)
        return pd.DataFrame({
            "period_start": target_times,
            "period_end": target_times + pd.Timedelta("30min"),
            "intensity": preds,
        })