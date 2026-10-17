import numpy as np
import pandas as pd
import pytest

from carbon_forecast.features.build import (
    FEATURE_COLUMNS,
    build_training_set,
    calendar_features,
    solar_elevation,
)


def make_neso(start="2026-01-01", days=40):
    times = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=days * 48, freq="30min")
    actual = 100 + 50 * np.sin(np.arange(len(times)) * 2 * np.pi / 48)
    return pd.DataFrame({
        "period_start": times,
        "period_end": times + pd.Timedelta("30min"),
        "actual": actual,
        "neso_forecast_latest": actual,
        "is_valid_target": True,
        "invalid_reason": pd.Series([None] * len(times), dtype="string"),
    })


def make_weather(start="2026-01-01", days=40):
    times = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=days * 24, freq="h")
    df = pd.DataFrame({"time": times})
    for loc, speed in {"north_sea": 6.0, "east_anglia": 9.0, "irish_sea": 12.0, "moray_firth": 20.0}.items():
        df[f"wind_speed_100m__{loc}"] = speed
    df["temperature_2m__london"] = 8.0
    df["shortwave_radiation__south_england_solar"] = 100.0
    return df


ISSUE = pd.Timestamp("2026-02-05 12:00", tz="UTC")


def test_feature_table_has_expected_columns_and_rows():
    table = build_training_set(make_neso(), make_weather(), pd.DatetimeIndex([ISSUE]))
    assert len(table) == 48
    for col in FEATURE_COLUMNS:
        assert col in table.columns


def test_poisoned_future_does_not_change_features():
    """Make every actual after the cutoff absurd. If any feature peeks at the future, it changes."""
    neso = make_neso()
    clean = build_training_set(neso, make_weather(), pd.DatetimeIndex([ISSUE]))

    poisoned = neso.copy()
    poisoned.loc[poisoned["period_start"] >= ISSUE - pd.Timedelta("30min"), "actual"] = 9999.0
    dirty = build_training_set(poisoned, make_weather(), pd.DatetimeIndex([ISSUE]))

    pd.testing.assert_frame_equal(clean[FEATURE_COLUMNS], dirty[FEATURE_COLUMNS])
    assert (dirty["y"] == 9999.0).all()  # the poison really was applied to the targets


def test_unpublished_lag_is_missing():
    table = build_training_set(make_neso(), make_weather(), pd.DatetimeIndex([ISSUE]))
    assert not np.isnan(table["lag_24h"].iloc[0])   # first target's lag (12:00 yesterday) is published
    assert np.isnan(table["lag_24h"].iloc[-1])      # last target's lag (11:30 today) is not yet published


def test_wind_index_uses_three_offshore_points():
    table = build_training_set(make_neso(), make_weather(), pd.DatetimeIndex([ISSUE]))
    assert table["wind_index"].iloc[0] == pytest.approx(9.0)    # mean of 6, 9, 12
    assert table["wind_moray"].iloc[0] == pytest.approx(20.0)   # Moray kept separate


def test_solar_elevation_sensible():
    summer = solar_elevation(pd.DatetimeIndex([pd.Timestamp("2026-06-21 11:45", tz="UTC")]))[0]
    winter = solar_elevation(pd.DatetimeIndex([pd.Timestamp("2026-12-21 11:45", tz="UTC")]))[0]
    midnight = solar_elevation(pd.DatetimeIndex([pd.Timestamp("2026-06-21 23:45", tz="UTC")]))[0]
    assert 55 < summer < 65
    assert 9 < winter < 17
    assert midnight < 0


def test_bank_holiday_and_christmas_flags():
    times = pd.DatetimeIndex([pd.Timestamp("2025-12-25 12:00", tz="UTC"), pd.Timestamp("2025-12-10 12:00", tz="UTC")])
    cal = calendar_features(times)
    assert list(cal["is_bank_holiday"]) == [1, 0]
    assert list(cal["is_christmas_period"]) == [1, 0]