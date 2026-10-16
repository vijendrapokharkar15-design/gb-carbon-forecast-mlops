"""Validation schemas for raw data.

Raw checks answer one question: is what the source sent us sane?
Known issues (outage gaps, null or zero `actual`) are allowed here; they are
handled by the cleaning step, whose output has its own stricter schema.
"""
import logging

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

logger = logging.getLogger(__name__)

INTENSITY_INDEX_VALUES = ["very low", "low", "moderate", "high", "very high"]

# Physically plausible ranges for each weather variable
WEATHER_RANGES = {
    "wind_speed_100m": (0, 60),        # m/s
    "shortwave_radiation": (0, 1400),  # W/m2
    "cloud_cover": (0, 100),           # %
    "temperature_2m": (-30, 45),       # degrees C
}


class DataValidationError(Exception):
    """Raised when a dataset fails its schema."""


# ---------- reusable checks ----------

def _is_utc(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.DatetimeTZDtype) and str(s.dtype.tz) == "UTC"


def _on_half_hour(s: pd.Series) -> pd.Series:
    return s.dt.minute.isin([0, 30]) & (s.dt.second == 0) & (s.dt.microsecond == 0)


def _on_hour(s: pd.Series) -> pd.Series:
    return (s.dt.minute == 0) & (s.dt.second == 0) & (s.dt.microsecond == 0)


IS_UTC = pa.Check(_is_utc, error="must be timezone-aware UTC")


# ---------- raw NESO ----------

RAW_NESO_SCHEMA = pa.DataFrameSchema(
    columns={
        "period_start": pa.Column(
            nullable=False,
            unique=True,
            checks=[IS_UTC, pa.Check(_on_half_hour, error="must be on a half-hour boundary")],
        ),
        "period_end": pa.Column(nullable=False, checks=[IS_UTC]),
        "forecast": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, 1000)),
        "actual": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, 1000)),
        "intensity_index": pa.Column(nullable=True, checks=pa.Check.isin(INTENSITY_INDEX_VALUES)),
    },
    checks=[
        pa.Check(
            lambda df: (df["period_end"] - df["period_start"]) == pd.Timedelta("30min"),
            error="period_end must be exactly 30 minutes after period_start",
        )
    ],
    strict=True,  # unexpected or missing columns fail: catches API format changes
)


# ---------- raw weather ----------

def build_raw_weather_schema(variables: list[str]) -> pa.DataFrameSchema:
    """Schema for long-format weather (one row per time and location).
    Built from the config's variable list so the two can't drift apart."""
    unknown = set(variables) - set(WEATHER_RANGES)
    if unknown:
        raise ValueError(f"No validation range defined for: {sorted(unknown)}")

    columns = {
        "time": pa.Column(
            nullable=False,
            checks=[IS_UTC, pa.Check(_on_hour, error="must be on the hour")],
        ),
        "location": pa.Column(nullable=False),
    }
    for var in variables:
        low, high = WEATHER_RANGES[var]
        columns[var] = pa.Column(float, nullable=False, checks=pa.Check.in_range(low, high))

    return pa.DataFrameSchema(columns=columns, unique=["time", "location"], strict=True)

# ---------- processed NESO ----------

PROCESSED_NESO_SCHEMA = pa.DataFrameSchema(
    columns={
        "period_start": pa.Column(
            nullable=False,
            unique=True,
            checks=[IS_UTC, pa.Check(_on_half_hour, error="must be on a half-hour boundary")],
        ),
        "period_end": pa.Column(nullable=False, checks=[IS_UTC]),
        "actual": pa.Column(
            float,
            nullable=True,
            checks=[pa.Check.greater_than(0), pa.Check.less_than_or_equal_to(1000)],
        ),
        "neso_forecast_latest": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, 1000)),
        "is_valid_target": pa.Column(bool, nullable=False),
        "invalid_reason": pa.Column(
            nullable=True, checks=pa.Check.isin(["outage", "null_actual", "zero_actual"])
        ),
    },
    checks=[
        pa.Check(
            lambda df: df["period_start"].diff().dropna().eq(pd.Timedelta("30min")).all(),
            error="period_start must form a complete half-hourly grid",
        ),
        pa.Check(
            lambda df: df["is_valid_target"] == df["invalid_reason"].isna(),
            error="is_valid_target must be True exactly when invalid_reason is empty",
        ),
        pa.Check(
            lambda df: ~df["is_valid_target"] | df["actual"].notna(),
            error="valid targets must have an actual value",
        ),
    ],
    strict=True,
)


# ---------- processed weather ----------

def build_processed_weather_schema(variables: list[str], locations: list[str]) -> pa.DataFrameSchema:
    """Wide hourly weather: no gaps, no nulls, every variable and location present."""
    columns = {
        "time": pa.Column(
            nullable=False,
            unique=True,
            checks=[IS_UTC, pa.Check(_on_hour, error="must be on the hour")],
        )
    }
    for var in variables:
        low, high = WEATHER_RANGES[var]
        for loc in locations:
            columns[f"{var}__{loc}"] = pa.Column(float, nullable=False, checks=pa.Check.in_range(low, high))

    return pa.DataFrameSchema(
        columns=columns,
        checks=[
            pa.Check(
                lambda df: df["time"].diff().dropna().eq(pd.Timedelta("1h")).all(),
                error="time must form a complete hourly grid",
            )
        ],
        strict=True,
    )


# ---------- runner ----------

def validate(df: pd.DataFrame, schema: pa.DataFrameSchema, name: str) -> pd.DataFrame:
    """Validate with lazy=True so every failure is reported at once, not just the first."""
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        logger.error("%s failed validation with %d failures:\n%s", name, len(cases), cases.head(20).to_string())
        raise DataValidationError(f"{name} failed validation ({len(cases)} failures)") from exc