"""Turn validated raw data into clean, model-ready tables.

Rules implemented here come from the Phase 1 data quality findings:
- NESO outages become explicit rows on a complete half-hourly grid
- actual == 0 is a faulty reading and is set to missing
- every row records whether it can be used as a training target, and why not
"""
import numpy as np
import pandas as pd

INVALID_REASONS = ["outage", "null_actual", "zero_actual"]


def clean_neso(raw: pd.DataFrame) -> pd.DataFrame:
    """Return NESO data on a complete half-hourly grid with target-validity flags.

    Output columns:
        period_start, period_end, actual, neso_forecast_latest,
        is_valid_target, invalid_reason
    """
    df = raw.drop_duplicates("period_start").sort_values("period_start").set_index("period_start")

    grid = pd.date_range(
        df.index.min(), df.index.max(), freq="30min", name="period_start"
    ).as_unit(df.index.unit)
    present = grid.isin(df.index)
    df = df.reindex(grid)

    actual = df["actual"].to_numpy(dtype="float64")
    null_mask = present & np.isnan(actual)
    zero_mask = present & (actual == 0)

    reason = np.full(len(grid), None, dtype=object)
    reason[~present] = "outage"
    reason[null_mask] = "null_actual"
    reason[zero_mask] = "zero_actual"

    out = pd.DataFrame(index=grid)
    out["period_end"] = grid + pd.Timedelta("30min")
    out["actual"] = np.where(zero_mask, np.nan, actual)
    # Renamed on purpose: this is NESO's latest (near-real-time) forecast, not a day-ahead one
    out["neso_forecast_latest"] = df["forecast"].to_numpy(dtype="float64")
    out["is_valid_target"] = ~(~present | null_mask | zero_mask)
    out["invalid_reason"] = pd.Series(reason, index=grid, dtype="string")

    return out.reset_index()


def clean_weather(raw: pd.DataFrame, variables: list[str], locations: list[str]) -> pd.DataFrame:
    """Reshape long weather data to wide format on a complete hourly grid.

    One row per hour; one column per variable and location, named
    '<variable>__<location>' (e.g. wind_speed_100m__north_sea).
    Missing hours or locations appear as NaN, which the processed schema rejects.
    """
    wide = raw.pivot(index="time", columns="location", values=variables)
    wide.columns = [f"{var}__{loc}" for var, loc in wide.columns]
    wide = wide.reindex(columns=[f"{v}__{loc}" for v in variables for loc in locations])

    grid = pd.date_range(
        wide.index.min(), wide.index.max(), freq="h", name="time"
    ).as_unit(wide.index.unit)
    return wide.reindex(grid).reset_index()