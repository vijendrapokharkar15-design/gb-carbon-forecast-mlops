import numpy as np
import pandas as pd
import pytest

from carbon_forecast.training.baselines import NaiveForecaster, ProfileForecaster


def history_until(cutoff, start="2026-01-01", value_fn=None):
    """Processed-style history with every period ending by the cutoff."""
    times = pd.date_range(
        pd.Timestamp(start, tz="UTC"), pd.Timestamp(cutoff) - pd.Timedelta("30min"), freq="30min"
    )
    actual = value_fn(times) if value_fn else np.arange(len(times), dtype="float64")
    return pd.DataFrame({"period_start": times, "actual": actual})


ISSUE = pd.Timestamp("2026-01-05 12:00", tz="UTC")
CUTOFF = ISSUE - pd.Timedelta("30min")
TARGETS = pd.date_range(ISSUE, periods=48, freq="30min")


def test_naive_uses_same_time_yesterday():
    hist = history_until(CUTOFF)
    known = hist.set_index("period_start")["actual"]
    pred = NaiveForecaster().predict(hist, None, ISSUE, TARGETS)
    assert pred[0] == known[TARGETS[0] - pd.Timedelta(days=1)]


def test_naive_falls_back_to_48h_for_unpublished_last_step():
    hist = history_until(CUTOFF)
    known = hist.set_index("period_start")["actual"]
    pred = NaiveForecaster().predict(hist, None, ISSUE, TARGETS)
    # 24h before the last target (11:30 today) isn't published at a 12:00 issue
    assert TARGETS[-1] - pd.Timedelta(days=1) not in known.index
    assert pred[-1] == known[TARGETS[-1] - pd.Timedelta(days=2)]


def test_naive_skips_invalid_values():
    hist = history_until(CUTOFF)
    yesterday = TARGETS[0] - pd.Timedelta(days=1)
    hist.loc[hist["period_start"] == yesterday, "actual"] = np.nan
    known = hist.set_index("period_start")["actual"]
    pred = NaiveForecaster().predict(hist, None, ISSUE, TARGETS)
    assert pred[0] == known[TARGETS[0] - pd.Timedelta(days=2)]


def test_profile_separates_weekdays_and_weekends():
    def weekday_100_weekend_50(times):
        local = times.tz_convert("Europe/London")
        return np.where(local.dayofweek >= 5, 50.0, 100.0)

    issue = pd.Timestamp("2026-01-30 12:00", tz="UTC")   # Friday
    hist = history_until(issue - pd.Timedelta("30min"), value_fn=weekday_100_weekend_50)
    targets = pd.date_range(issue, periods=48, freq="30min")
    pred = ProfileForecaster(weeks=4).predict(hist, None, issue, targets)

    local = targets.tz_convert("Europe/London")
    assert np.allclose(pred[local.dayofweek >= 5], 50.0)  # Saturday morning targets
    assert np.allclose(pred[local.dayofweek < 5], 100.0)  # Friday afternoon targets


@pytest.mark.parametrize("forecaster", [NaiveForecaster(), ProfileForecaster()])
def test_baselines_return_48_finite_values(forecaster):
    hist = history_until(CUTOFF)
    pred = forecaster.predict(hist, None, ISSUE, TARGETS)
    assert pred.shape == (48,)
    assert np.isfinite(pred).all()