import numpy as np
import pandas as pd

from carbon_forecast.features.build import build_training_set
from carbon_forecast.training.lgbm import LightGBMForecaster
from tests.test_features import make_neso, make_weather

SMALL = {"n_estimators": 20, "min_child_samples": 5}


def _setup():
    neso, weather = make_neso(days=40), make_weather(days=40)
    issues = pd.date_range(pd.Timestamp("2026-01-30 12:00", tz="UTC"), periods=6, freq="D")
    return neso, weather, build_training_set(neso, weather, issues)


def _history_at(neso, issue):
    return neso[neso["period_start"] < issue - pd.Timedelta("30min")]


def test_fit_uses_only_targets_known_at_cutoff():
    neso, weather, table = _setup()
    history = _history_at(neso, pd.Timestamp("2026-02-02 12:00", tz="UTC"))
    f = LightGBMForecaster(table, params=SMALL)
    f.fit(history, weather)

    expected = int((table["is_valid_target"] & (table["target_time"] <= history["period_start"].max())).sum())
    assert f.n_train == expected
    assert 0 < f.n_train < len(table)  # later issue days' targets were excluded


def test_train_from_reduces_training_rows():
    neso, weather, table = _setup()
    history = _history_at(neso, pd.Timestamp("2026-02-06 12:00", tz="UTC"))
    full = LightGBMForecaster(table, params=SMALL)
    recent = LightGBMForecaster(table, params=SMALL, train_from="2026-02-02")
    full.fit(history, weather)
    recent.fit(history, weather)
    assert recent.n_train < full.n_train


def test_predict_returns_48_finite_values():
    neso, weather, table = _setup()
    issue = pd.Timestamp("2026-02-06 12:00", tz="UTC")
    history = _history_at(neso, issue)
    f = LightGBMForecaster(table, params=SMALL)
    f.fit(history, weather)
    pred = f.predict(history, weather, issue, pd.date_range(issue, periods=48, freq="30min"))
    assert pred.shape == (48,)
    assert np.isfinite(pred).all()