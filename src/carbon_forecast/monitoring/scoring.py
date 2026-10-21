"""Live monitoring: score frozen forecasts against actuals and against NESO's day-ahead snapshot.

All functions are pure (no network, no files) so every rule is easy to test.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from carbon_forecast.evaluation.metrics import daily_regret
from carbon_forecast.ingestion.neso import parse_forecast_rows


def target_periods(issue_time: pd.Timestamp, horizon_steps: int = 48) -> pd.DatetimeIndex:
    return pd.date_range(issue_time, periods=horizon_steps, freq="30min").as_unit("ns")


def as_series(times, values) -> pd.Series:
    """A float series indexed by UTC timestamps at ns precision, so reindexing always lines up."""
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns")
    return pd.Series(np.asarray(values, dtype="float64"), index=index)


def score(y_pred: pd.Series, y_true: pd.Series, window_steps: int = 8, good_threshold: float = 10.0) -> dict:
    """Score one forecast. Both series must share the same ordered target index.
    Zero or missing actuals (and missing predictions) are excluded, as in the backtest."""
    frame = pd.DataFrame({
        "model": "live",
        "issue_time": y_true.index[0],
        "horizon": np.arange(1, len(y_true) + 1),
        "y_true": y_true.to_numpy(),
        "y_pred": y_pred.to_numpy(),
    })
    frame["is_valid_target"] = frame["y_true"].notna() & (frame["y_true"] > 0) & frame["y_pred"].notna()
    n_valid = int(frame["is_valid_target"].sum())
    if n_valid == 0:
        return {"n_valid": 0, "mae": np.nan, "bias": np.nan, "regret": np.nan, "good_window": None}

    valid = frame[frame["is_valid_target"]]
    err = valid["y_pred"] - valid["y_true"]
    reg = daily_regret(frame, window_steps)          # needs every period valid, else empty
    regret = float(reg["regret"].iloc[0]) if len(reg) else np.nan
    return {
        "n_valid": n_valid,
        "mae": round(float(err.abs().mean()), 2),
        "bias": round(float(err.mean()), 2),          # positive = forecast too high
        "regret": round(regret, 2) if np.isfinite(regret) else np.nan,
        "good_window": bool(regret <= good_threshold) if np.isfinite(regret) else None,
    }


def snapshot_time(name: str) -> pd.Timestamp:
    """'fw24h_20261021T1138Z.json' -> 2026-10-21 11:38 UTC"""
    stamp = name.split("_")[1].split(".")[0]
    return pd.Timestamp(datetime.strptime(stamp, "%Y%m%dT%H%MZ")).tz_localize("UTC")


def select_on_time_snapshot(names: list[str], issue_time: pd.Timestamp) -> str | None:
    """The latest snapshot taken on the issue day BEFORE the issue time, or None.
    Late snapshots never qualify: they would have seen information our forecast didn't have."""
    candidates = [
        n for n in names
        if snapshot_time(n).date() == issue_time.date() and snapshot_time(n) < issue_time
    ]
    return max(candidates, key=snapshot_time) if candidates else None


def snapshot_series(snapshot: dict, issue_time: pd.Timestamp, horizon_steps: int = 48) -> pd.Series:
    """NESO's day-ahead values for our target periods only (earlier periods dropped)."""
    rows = parse_forecast_rows(snapshot["data"])
    series = as_series(rows["period_start"], rows["forecast"])
    return series.reindex(target_periods(issue_time, horizon_steps))


def evaluate_alerts(
    scores: pd.DataFrame,
    as_of: pd.Timestamp,
    rolling_days: int,
    mae_alert: float,
    bias_alert: float,
) -> list[str]:
    """Alerts from the rolling window of ON-TIME scored days ending at as_of."""
    if scores.empty:
        return []
    dates = pd.to_datetime(scores["issue_date"]).dt.tz_localize("UTC")
    window = scores[(dates > as_of - pd.Timedelta(days=rolling_days)) & (~scores["record_late"].astype(bool))]
    if window.empty:
        return []

    alerts = []
    mae = window["mae"].mean()
    if mae > mae_alert:
        alerts.append(f"Rolling {rolling_days}-day MAE {mae:.1f} exceeds {mae_alert} ({len(window)} days)")
    bias = window["bias"].mean()
    if abs(bias) > bias_alert:
        direction = "too high" if bias > 0 else "too low"
        alerts.append(f"Rolling {rolling_days}-day bias {bias:+.1f}: forecasts systematically {direction}")
    return alerts