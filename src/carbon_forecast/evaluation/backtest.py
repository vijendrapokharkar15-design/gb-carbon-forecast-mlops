"""Walk-forward backtest: simulate issuing a forecast every day at a fixed time.

Leakage guard: the harness itself slices history to data known at issue time,
so no forecaster can see future actuals, however its features are written.
"""
import logging
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Forecaster(Protocol):
    """Interface every baseline and model implements."""

    name: str

    def fit(self, history: pd.DataFrame, weather: pd.DataFrame) -> None:
        """Train on processed NESO history and weather known before the cutoff."""

    def predict(
        self,
        history: pd.DataFrame,
        weather: pd.DataFrame,
        issue_time: pd.Timestamp,
        target_times: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Return one prediction per target time."""


def make_issue_times(test_start: str, test_end: str, issue_hour_utc: int) -> pd.DatetimeIndex:
    """One issue time per day in [test_start, test_end)."""
    days = pd.date_range(
        pd.Timestamp(test_start, tz="UTC"), pd.Timestamp(test_end, tz="UTC"), freq="D", inclusive="left"
    )
    return days + pd.Timedelta(hours=issue_hour_utc)


def run_backtest(
    forecaster: Forecaster,
    neso: pd.DataFrame,
    weather: pd.DataFrame,
    test_start: str,
    test_end: str,
    issue_hour_utc: int = 12,
    horizon_steps: int = 48,
    publication_lag_minutes: int = 30,
    refit_monthly: bool = True,
) -> pd.DataFrame:
    """Run a daily walk-forward backtest and return one row per (issue time, target time).

    neso: processed NESO table (clean_neso output)
    weather: processed wide hourly weather (clean_weather output)
    """
    neso = neso.sort_values("period_start").reset_index(drop=True)
    weather = weather.sort_values("time").reset_index(drop=True)
    truth = neso.set_index("period_start")[["actual", "is_valid_target", "neso_forecast_latest"]]
    lag = pd.Timedelta(minutes=publication_lag_minutes)

    fitted_month = None
    records = []

    for issue_time in make_issue_times(test_start, test_end, issue_hour_utc):
        cutoff = issue_time - lag

        # Only periods that had ended and been published by the cutoff
        history = neso.iloc[: neso["period_start"].searchsorted(cutoff, side="left")]

        month = (issue_time.year, issue_time.month)
        if fitted_month is None or (refit_monthly and month != fitted_month):
            fit_weather = weather.iloc[: weather["time"].searchsorted(cutoff, side="left")]
            forecaster.fit(history, fit_weather)
            fitted_month = month
            logger.info("%s refit at %s on %d rows", forecaster.name, issue_time, len(history))

        target_times = pd.date_range(issue_time, periods=horizon_steps, freq="30min").as_unit(truth.index.unit)

        # Weather up to the end of the horizon (observed weather as a stand-in for weather forecasts)
        visible_weather = weather.iloc[: weather["time"].searchsorted(target_times[-1], side="right")]

        y_pred = np.asarray(
            forecaster.predict(history, visible_weather, issue_time, target_times), dtype="float64"
        )
        if y_pred.shape != (horizon_steps,) or not np.isfinite(y_pred).all():
            raise ValueError(
                f"{forecaster.name} returned invalid predictions at {issue_time}: shape {y_pred.shape}"
            )

        block = truth.reindex(target_times)
        records.append(
            pd.DataFrame({
                "model": forecaster.name,
                "issue_time": issue_time,
                "target_time": target_times,
                "horizon": np.arange(1, horizon_steps + 1),
                "y_true": block["actual"].to_numpy(dtype="float64"),
                "is_valid_target": block["is_valid_target"].fillna(False).astype(bool).to_numpy(),
                "neso_forecast_latest": block["neso_forecast_latest"].to_numpy(dtype="float64"),
                "y_pred": y_pred,
            })
        )

    return pd.concat(records, ignore_index=True)