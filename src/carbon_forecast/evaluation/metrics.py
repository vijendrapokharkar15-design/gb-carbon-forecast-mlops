"""Technical and business metrics for day-ahead forecasts."""
import numpy as np
import pandas as pd


def mae(preds: pd.DataFrame) -> float:
    """Mean absolute error over valid targets only."""
    valid = preds[preds["is_valid_target"]]
    return float((valid["y_pred"] - valid["y_true"]).abs().mean())


def mae_by(preds: pd.DataFrame, by: str | pd.Series) -> pd.Series:
    """MAE grouped by a column (e.g. 'horizon') or a Series (e.g. hour of day)."""
    valid = preds[preds["is_valid_target"]]
    if isinstance(by, pd.Series):
        by = by.loc[valid.index]
    return (valid["y_pred"] - valid["y_true"]).abs().groupby(by).mean()


def best_window_start(values, window_steps: int) -> int:
    """Index where the window of `window_steps` consecutive values has the lowest mean."""
    values = np.asarray(values, dtype="float64")
    if len(values) < window_steps:
        raise ValueError("window longer than series")
    means = np.convolve(values, np.ones(window_steps) / window_steps, mode="valid")
    return int(np.argmin(means))


def daily_regret(preds: pd.DataFrame, window_steps: int = 8) -> pd.DataFrame:
    """Carbon regret per issue day: actual intensity of the window we recommended
    minus actual intensity of the truly greenest window. Days with any invalid
    target are skipped, since the true best window can't be known."""
    rows = []
    for issue_time, day in preds.sort_values(["issue_time", "horizon"]).groupby("issue_time"):
        if not day["is_valid_target"].all():
            continue
        y_true = day["y_true"].to_numpy()
        chosen = best_window_start(day["y_pred"].to_numpy(), window_steps)
        best = best_window_start(y_true, window_steps)
        regret = y_true[chosen : chosen + window_steps].mean() - y_true[best : best + window_steps].mean()
        rows.append({"issue_time": issue_time, "chosen_start": chosen, "best_start": best, "regret": regret})
    return pd.DataFrame(rows, columns=["issue_time", "chosen_start", "best_start", "regret"])


def summarise(preds: pd.DataFrame, window_steps: int = 8, good_window_threshold: float = 10.0) -> dict:
    """Headline metrics for one model's backtest."""
    regret = daily_regret(preds, window_steps)
    return {
        "model": preds["model"].iloc[0],
        "mae": round(mae(preds), 2),
        "regret_mean": round(float(regret["regret"].mean()), 2),
        "good_window_rate": round(float((regret["regret"] <= good_window_threshold).mean()), 3),
        "n_issue_days": int(preds["issue_time"].nunique()),
        "n_regret_days": len(regret),
        "n_valid_targets": int(preds["is_valid_target"].sum()),
    }