import numpy as np
import pandas as pd
import pytest

from carbon_forecast.processing.clean import clean_neso, clean_weather
from carbon_forecast.validation.schemas import (
    PROCESSED_NESO_SCHEMA,
    DataValidationError,
    build_processed_weather_schema,
    validate,
)

VARIABLES = ["wind_speed_100m", "shortwave_radiation", "cloud_cover", "temperature_2m"]
LOCATIONS = ["north_sea", "london"]


def make_raw_neso():
    """6 half-hours: index 2 missing (outage), 3 is zero, 4 is null."""
    start = pd.date_range("2026-01-01 00:00", periods=6, freq="30min", tz="UTC")
    df = pd.DataFrame({
        "period_start": start,
        "period_end": start + pd.Timedelta("30min"),
        "forecast": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "actual": [98.0, 99.0, 100.0, 0.0, np.nan, 103.0],
        "intensity_index": ["moderate"] * 6,
    })
    return df.drop(index=2).reset_index(drop=True)


def make_raw_weather(hours=3):
    times = pd.date_range("2026-01-01 00:00", periods=hours, freq="h", tz="UTC")
    rows = []
    for loc in LOCATIONS:
        for t in times:
            rows.append({"time": t, "location": loc, "wind_speed_100m": 8.0,
                         "shortwave_radiation": 0.0, "cloud_cover": 70.0, "temperature_2m": 6.0})
    return pd.DataFrame(rows)


def test_clean_neso_fills_outage_as_explicit_row():
    out = clean_neso(make_raw_neso())
    assert len(out) == 6
    row = out.iloc[2]
    assert row["invalid_reason"] == "outage"
    assert not row["is_valid_target"]
    assert np.isnan(row["actual"])


def test_clean_neso_zero_becomes_missing_and_flagged():
    row = clean_neso(make_raw_neso()).iloc[3]
    assert np.isnan(row["actual"])
    assert row["invalid_reason"] == "zero_actual"
    assert not row["is_valid_target"]


def test_clean_neso_null_is_flagged():
    row = clean_neso(make_raw_neso()).iloc[4]
    assert row["invalid_reason"] == "null_actual"
    assert not row["is_valid_target"]


def test_clean_neso_valid_rows_unchanged_and_forecast_renamed():
    out = clean_neso(make_raw_neso())
    assert out.loc[0, "actual"] == 98.0
    assert out.loc[0, "neso_forecast_latest"] == 100.0
    assert bool(out.loc[0, "is_valid_target"])
    assert "forecast" not in out.columns
    assert "intensity_index" not in out.columns


def test_clean_neso_output_passes_processed_schema():
    validate(clean_neso(make_raw_neso()), PROCESSED_NESO_SCHEMA, "processed NESO")


def test_processed_schema_rejects_zero_actual():
    out = clean_neso(make_raw_neso())
    out.loc[0, "actual"] = 0.0
    with pytest.raises(DataValidationError):
        validate(out, PROCESSED_NESO_SCHEMA, "processed NESO")


def test_clean_weather_wide_format():
    out = clean_weather(make_raw_weather(), VARIABLES, LOCATIONS)
    assert len(out) == 3
    assert "wind_speed_100m__north_sea" in out.columns
    assert out.shape[1] == 1 + len(VARIABLES) * len(LOCATIONS)
    validate(out, build_processed_weather_schema(VARIABLES, LOCATIONS), "processed weather")


def test_clean_weather_missing_hour_fails_processed_schema():
    raw = make_raw_weather(hours=4)
    raw = raw[~((raw["location"] == "london") & (raw["time"] == raw["time"].min() + pd.Timedelta("2h")))]
    out = clean_weather(raw, VARIABLES, LOCATIONS)
    with pytest.raises(DataValidationError):
        validate(out, build_processed_weather_schema(VARIABLES, LOCATIONS), "processed weather")