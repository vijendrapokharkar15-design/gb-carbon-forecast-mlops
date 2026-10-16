import numpy as np
import pandas as pd
import pytest

from carbon_forecast.validation.schemas import (
    RAW_NESO_SCHEMA,
    DataValidationError,
    build_raw_weather_schema,
    validate,
)

VARIABLES = ["wind_speed_100m", "shortwave_radiation", "cloud_cover", "temperature_2m"]


def make_neso(n=4):
    start = pd.date_range("2026-01-01 00:00", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame({
        "period_start": start,
        "period_end": start + pd.Timedelta("30min"),
        "forecast": np.full(n, 120.0),
        "actual": np.full(n, 118.0),
        "intensity_index": ["moderate"] * n,
    })


def make_weather(n=3):
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 00:00", periods=n, freq="h", tz="UTC"),
        "location": ["north_sea"] * n,
        "wind_speed_100m": [8.0] * n,
        "shortwave_radiation": [0.0] * n,
        "cloud_cover": [75.0] * n,
        "temperature_2m": [6.5] * n,
    })


def test_valid_neso_passes():
    validate(make_neso(), RAW_NESO_SCHEMA, "neso")


def test_neso_allows_known_issues_null_and_zero_actual():
    df = make_neso()
    df.loc[0, "actual"] = np.nan
    df.loc[1, "actual"] = 0.0
    validate(df, RAW_NESO_SCHEMA, "neso")


def test_neso_duplicate_timestamp_fails():
    df = pd.concat([make_neso(), make_neso().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError):
        validate(df, RAW_NESO_SCHEMA, "neso")


def test_neso_misaligned_timestamp_fails():
    df = make_neso()
    df.loc[0, "period_start"] += pd.Timedelta("15min")
    df.loc[0, "period_end"] += pd.Timedelta("15min")
    with pytest.raises(DataValidationError):
        validate(df, RAW_NESO_SCHEMA, "neso")


def test_neso_naive_timestamps_fail():
    df = make_neso()
    df["period_start"] = df["period_start"].dt.tz_localize(None)
    with pytest.raises(DataValidationError):
        validate(df, RAW_NESO_SCHEMA, "neso")


def test_neso_out_of_range_actual_fails():
    df = make_neso()
    df.loc[0, "actual"] = -5.0
    with pytest.raises(DataValidationError):
        validate(df, RAW_NESO_SCHEMA, "neso")


def test_neso_unexpected_column_fails():
    df = make_neso()
    df["surprise"] = 1
    with pytest.raises(DataValidationError):
        validate(df, RAW_NESO_SCHEMA, "neso")


def test_valid_weather_passes():
    validate(make_weather(), build_raw_weather_schema(VARIABLES), "weather")


def test_weather_out_of_range_fails():
    df = make_weather()
    df.loc[0, "wind_speed_100m"] = 99.0
    with pytest.raises(DataValidationError):
        validate(df, build_raw_weather_schema(VARIABLES), "weather")


def test_weather_duplicate_time_and_location_fails():
    df = pd.concat([make_weather(), make_weather().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError):
        validate(df, build_raw_weather_schema(VARIABLES), "weather")


def test_weather_schema_rejects_variable_without_range():
    with pytest.raises(ValueError):
        build_raw_weather_schema(VARIABLES + ["snowfall"])