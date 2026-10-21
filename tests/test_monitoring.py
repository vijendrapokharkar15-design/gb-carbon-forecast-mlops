import numpy as np
import pandas as pd

from carbon_forecast.monitoring.scoring import (
    as_series,
    evaluate_alerts,
    score,
    select_on_time_snapshot,
    snapshot_series,
    target_periods,
)

ISSUE = pd.Timestamp("2026-10-21 12:00", tz="UTC")
T = target_periods(ISSUE)


def truth(values=None):
    return as_series(T, values if values is not None else 100 + 50 * np.sin(np.arange(48) / 8))


# ---------- score ----------

def test_perfect_forecast_scores_zero():
    y = truth()
    s = score(y.copy(), y)
    assert s["n_valid"] == 48 and s["mae"] == 0 and s["bias"] == 0
    assert s["regret"] == 0 and s["good_window"] is True


def test_constant_offset_shows_as_bias():
    y = truth()
    s = score(y + 5, y)
    assert s["mae"] == 5.0 and s["bias"] == 5.0
    assert s["regret"] == 0     # same shape, so the right window is still chosen


def test_zero_and_missing_actuals_excluded():
    values = truth().to_numpy(copy=True)
    values[3], values[10] = 0.0, np.nan
    s = score(truth(), as_series(T, values))
    assert s["n_valid"] == 46
    assert np.isnan(s["regret"])   # regret needs a complete day


# ---------- snapshot selection ----------

NAMES = ["fw24h_20261021T1138Z.json", "fw24h_20261021T1200Z.json", "fw24h_20261020T1551Z.json"]


def test_on_time_snapshot_selected():
    assert select_on_time_snapshot(NAMES, ISSUE) == "fw24h_20261021T1138Z.json"


def test_late_snapshot_never_selected():
    assert select_on_time_snapshot(NAMES, ISSUE - pd.Timedelta(days=1)) is None   # 20 Oct: only 15:51


def test_snapshot_trimmed_to_target_periods():
    # A real 11:05 snapshot: 49 rows starting 10:30, so it ends at 10:30 the next day
    times = pd.date_range("2026-10-21 10:30", periods=49, freq="30min", tz="UTC")
    snap = {"data": [{"from": t.strftime("%Y-%m-%dT%H:%MZ"), "to": "", "intensity": {"forecast": 100}} for t in times]}
    s = snapshot_series(snap, ISSUE)
    assert s.index[0] == ISSUE and len(s) == 48     # 10:30 and 11:00 / 11:30 dropped: before issue
    assert s.notna().sum() == 46                    # covers 12:00 to 10:30 next day; 11:00 and 11:30 missing


# ---------- alerts ----------

def scores_frame(maes, biases, late=None):
    dates = pd.date_range("2026-10-15", periods=len(maes), freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "issue_date": dates, "mae": maes, "bias": biases,
        "record_late": late if late is not None else [False] * len(maes),
    })


AS_OF = pd.Timestamp("2026-10-21 12:30", tz="UTC")


def test_no_alert_when_healthy():
    assert evaluate_alerts(scores_frame([18] * 7, [1] * 7), AS_OF, 7, 30.2, 10) == []


def test_mae_alert_fires():
    alerts = evaluate_alerts(scores_frame([35] * 7, [0] * 7), AS_OF, 7, 30.2, 10)
    assert any("MAE" in a for a in alerts)


def test_bias_alert_fires():
    alerts = evaluate_alerts(scores_frame([18] * 7, [-12] * 7), AS_OF, 7, 30.2, 10)
    assert any("too low" in a for a in alerts)


def test_late_days_excluded_from_alerts():
    late = [True] * 7
    assert evaluate_alerts(scores_frame([99] * 7, [50] * 7, late), AS_OF, 7, 30.2, 10) == []