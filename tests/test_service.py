import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from carbon_forecast.config import load_config
from carbon_forecast.features.build import FEATURE_COLUMNS
from carbon_forecast.serving.service import ForecastService, greenest_window, latest_issue_time
from carbon_forecast.validation.schemas import DataValidationError

ISSUE = pd.Timestamp("2026-10-19 12:00", tz="UTC")


def tiny_booster():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = 150 + 20 * rng.normal(size=200)
    return lgb.train({"verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(X, y), num_boost_round=5)


def fake_neso(issue_time):
    times = pd.date_range(issue_time - pd.Timedelta(days=29), issue_time - pd.Timedelta("30min"), freq="30min")
    actual = 150 + 50 * np.sin(np.arange(len(times)) * 2 * np.pi / 48)
    return pd.DataFrame({
        "period_start": times,
        "period_end": times + pd.Timedelta("30min"),
        "forecast": actual,
        "actual": actual,
        "intensity_index": "moderate",
    })


def fake_weather(locations, variables, wind=8.0):
    times = pd.date_range(ISSUE - pd.Timedelta(days=2), ISSUE + pd.Timedelta(days=3), freq="h")
    frames = []
    for loc in locations:
        df = pd.DataFrame({"time": times, "location": loc["name"]})
        df["wind_speed_100m"], df["shortwave_radiation"] = wind, 100.0
        df["cloud_cover"], df["temperature_2m"] = 50.0, 10.0
        frames.append(df[["time", "location", *variables]])
    return pd.concat(frames, ignore_index=True)


def make_service(clock_time="2026-10-19 15:00", weather=fake_weather):
    counter = {"neso": 0}

    def counting_neso(issue_time):
        counter["neso"] += 1
        return fake_neso(issue_time)

    clock = {"now": pd.Timestamp(clock_time, tz="UTC")}
    service = ForecastService(
        tiny_booster(), {"version": "vtest"}, load_config(),
        neso_fetcher=counting_neso, weather_fetcher=weather, clock=lambda: clock["now"],
    )
    return service, counter, clock


# ---------- issue time ----------

def test_issue_time_after_noon_is_today():
    assert latest_issue_time(pd.Timestamp("2026-10-19 15:00", tz="UTC")) == ISSUE


def test_issue_time_before_noon_is_yesterday():
    assert latest_issue_time(pd.Timestamp("2026-10-19 09:00", tz="UTC")) == ISSUE - pd.Timedelta(days=1)


def test_issue_time_exactly_noon_is_today():
    assert latest_issue_time(ISSUE) == ISSUE


# ---------- service ----------

def test_forecast_has_48_finite_periods_from_issue_time():
    service, _, _ = make_service()
    issue, fc = service.forecast()
    assert issue == ISSUE
    assert len(fc) == 48
    assert fc["period_start"].iloc[0] == ISSUE
    assert np.isfinite(fc["intensity"]).all()


def test_forecast_is_cached_per_issue_time():
    service, counter, clock = make_service()
    service.forecast()
    clock["now"] += pd.Timedelta(hours=3)        # same issue (still after today's 12:00)
    service.forecast()
    assert counter["neso"] == 1


def test_bad_live_weather_fails_validation():
    def broken_weather(locations, variables):
        return fake_weather(locations, variables, wind=999.0)   # physically impossible

    service, _, _ = make_service(weather=broken_weather)
    with pytest.raises(DataValidationError):
        service.forecast()


# ---------- greenest window ----------

def _frame(values):
    times = pd.date_range(ISSUE, periods=len(values), freq="30min")
    return pd.DataFrame({"period_start": times, "period_end": times + pd.Timedelta("30min"), "intensity": values})


def test_greenest_window_finds_lowest_mean():
    w = greenest_window(_frame([100, 100, 20, 20, 100, 100]), hours=1)
    assert w["start"] == (ISSUE + pd.Timedelta("1h")).isoformat()
    assert w["mean_intensity"] == 20.0


def test_greenest_window_respects_not_before():
    fc = _frame([10, 10, 100, 100, 50, 50])
    w = greenest_window(fc, hours=1, not_before=ISSUE + pd.Timedelta("1h"))
    assert w["start"] == (ISSUE + pd.Timedelta("2h")).isoformat()  # the cheap early window is in the past


def test_greenest_window_none_when_not_enough_time_left():
    assert greenest_window(_frame([10, 20]), hours=4) is None