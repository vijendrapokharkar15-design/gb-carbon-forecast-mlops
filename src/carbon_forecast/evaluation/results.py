"""Tracked results table: one row per (model, period), committed to Git."""
from datetime import datetime, timezone

import pandas as pd

from carbon_forecast.config import resolve_path

RESULTS_PATH = "docs/results.csv"


def record_result(summary: dict, period: str, path: str = RESULTS_PATH) -> None:
    """Insert or replace the row for this model and period."""
    out = resolve_path(path)
    row = pd.DataFrame([{**summary, "period": period, "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}])
    if out.exists():
        existing = pd.read_csv(out)
        existing = existing[~((existing["model"] == summary["model"]) & (existing["period"] == period))]
        row = pd.concat([existing, row], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    row.sort_values(["period", "mae"]).to_csv(out, index=False)