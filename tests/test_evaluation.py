import numpy as np
import pandas as pd
import pytest

from carbon_forecast.evaluation.backtest import make_issue_times, run_backtest
from carbon_forecast.evaluation.metrics import best_window_start, daily_regret, mae


def make_processed_neso(start="2026-01-01", days=10):
    times = pd.date_range(start, periods=days * 48, freq="30min", tz="UTC")
    actual = 100 + 50 * np.sin(np.arange(len(times)) * 2 * np.pi / 48)
    return pd.DataFrame({
        "period_start": times,
        "period_end": times + pd.Timedelta("30min"),
        "actual": actual,
        "neso_forecast_latest": actual,
        "is_valid_target": True,
        "invalid_reason": pd.Series([None] * len(times), dtype="string"),
    })


def make_processed_weather(start="2026-01-01", days=10):
    times = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    return pd.DataFrame({"time": times, "wind_speed_100m__north_sea": 8.0})


class SpyForecaster:
    """Records exactly what the harness hands it, and predicts the last known value."""
    name = "spy"

    def __init__(self):
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, history, weather):
        self.fit_calls.append(history["period_start"].max())

    def predict(self, history, weather, issue_time, target_times):
        self.predict_calls.append({
            "issue_time": issue_time,
            "history_max": history["period_start"].max(),
            "weather_max": weather["time"].max(),
            "last_target": target_times[-1],
        })
        return np.full(len(target_times), history["actual"].iloc[-1])


# ---------- harness ----------

def test_issue_times_daily_at_issue_hour():
    times = make_issue_times("2026-01-05", "2026-01-08", 12)
    assert len(times) == 3
    assert all(t.hour == 12 and t.minute == 0 for t in times)


def test_backtest_shape_and_horizon():
    preds = run_backtest(SpyForecaster(), make_processed_neso(), make_processed_weather(),
                         "2026-01-05", "2026-01-08")
    assert len(preds) == 3 * 48
    assert preds["horizon"].min() == 1 and preds["horizon"].max() == 48
    first = preds[preds["issue_time"] == preds["issue_time"].min()]
    assert first["target_time"].iloc[0] == first["issue_time"].iloc[0]


def test_forecaster_never_sees_future_actuals():
    spy = SpyForecaster()
    run_backtest(spy, make_processed_neso(), make_processed_weather(), "2026-01-05", "2026-01-08",
                 publication_lag_minutes=30)
    for call in spy.predict_calls:
        cutoff = call["issue_time"] - pd.Timedelta("30min")
        # latest period seen must have ENDED by the cutoff
        assert call["history_max"] + pd.Timedelta("30min") <= cutoff
        assert call["weather_max"] <= call["last_target"]


def test_refits_once_per_month():
    spy = SpyForecaster()
    neso = make_processed_neso(start="2026-01-20", days=20)
    weather = make_processed_weather(start="2026-01-20", days=20)
    run_backtest(spy, neso, weather, "2026-01-28", "2026-02-05")
    assert len(spy.fit_calls) == 2  # January, then February


def test_backtest_rejects_wrong_length_predictions():
    class Broken(SpyForecaster):
        def predict(self, history, weather, issue_time, target_times):
            return np.zeros(10)

    with pytest.raises(ValueError):
        run_backtest(Broken(), make_processed_neso(), make_processed_weather(), "2026-01-05", "2026-01-06")


# ---------- metrics ----------

def test_best_window_start():
    assert best_window_start([5, 5, 1, 1, 5, 5], window_steps=2) == 2


def _day(y_true, y_pred, valid=True):
    n = len(y_true)
    return pd.DataFrame({
        "model": "m",
        "issue_time": pd.Timestamp("2026-01-01 12:00", tz="UTC"),
        "horizon": np.arange(1, n + 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "is_valid_target": valid,
    })


def test_regret_hand_calculated():
    # True best 2-step window is [1, 1] (mean 1). Prediction picks the first window,
    # whose true mean is 10. Regret = 10 - 1 = 9.
    preds = _day([10, 10, 1, 1, 10, 10], [0, 0, 5, 5, 9, 9])
    regret = daily_regret(preds, window_steps=2)
    assert regret["regret"].iloc[0] == pytest.approx(9.0)


def test_regret_zero_when_window_picked_correctly():
    preds = _day([10, 10, 1, 1, 10, 10], [50, 50, 20, 20, 50, 50])  # wrong values, right ranking
    assert daily_regret(preds, window_steps=2)["regret"].iloc[0] == pytest.approx(0.0)


def test_regret_skips_days_with_invalid_targets():
    preds = _day([10, 10, 1, 1, 10, 10], [0] * 6)
    preds.loc[3, "is_valid_target"] = False
    assert daily_regret(preds, window_steps=2).empty


def test_mae_ignores_invalid_targets():
    preds = _day([10.0, 20.0], [12.0, 999.0])
    preds.loc[1, "is_valid_target"] = False
    assert mae(preds) == pytest.approx(2.0)

def test_serving_weather_used_for_predict_observed_for_fit():
    """Training must see observed weather; predicting must see the serving (forecast) weather."""
    class WeatherSpy(SpyForecaster):
        def fit(self, history, weather):
            self.fit_wind = set(weather["wind_speed_100m__north_sea"].unique())

        def predict(self, history, weather, issue_time, target_times):
            self.predict_wind = set(weather["wind_speed_100m__north_sea"].unique())
            return np.zeros(len(target_times))

    observed = make_processed_weather()                                 # wind = 8.0 everywhere
    forecast = observed.assign(wind_speed_100m__north_sea=99.0)         # deliberately different
    spy = WeatherSpy()
    run_backtest(spy, make_processed_neso(), observed, "2026-01-05", "2026-01-06", serve_weather=forecast)

    assert spy.fit_wind == {8.0}
    assert spy.predict_wind == {99.0}