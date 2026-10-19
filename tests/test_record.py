import pandas as pd
import pytest

from carbon_forecast.serving.record import build_record, validate_forecast_payload

ISSUE = pd.Timestamp("2026-10-19 12:00", tz="UTC")


def payload(n=48, issue="2026-10-19T12:00:00+00:00", value=120.0, version="v1"):
    times = pd.date_range(ISSUE, periods=n, freq="30min")
    return {
        "issue_time": issue,
        "model_version": version,
        "forecast": [{"period_start": t.isoformat(), "intensity": value} for t in times],
    }


def test_valid_payload_passes():
    validate_forecast_payload(payload(), ISSUE, 48)


def test_wrong_issue_rejected():
    with pytest.raises(ValueError, match="expected"):
        validate_forecast_payload(payload(issue="2026-10-18T12:00:00+00:00"), ISSUE, 48)


def test_wrong_length_rejected():
    with pytest.raises(ValueError, match="periods"):
        validate_forecast_payload(payload(n=47), ISSUE, 48)


def test_implausible_value_rejected():
    with pytest.raises(ValueError, match="Implausible"):
        validate_forecast_payload(payload(value=-5.0), ISSUE, 48)


def test_missing_model_version_rejected():
    with pytest.raises(ValueError, match="model_version"):
        validate_forecast_payload(payload(version=""), ISSUE, 48)


def test_record_flags_late_fetches():
    on_time = build_record(payload(), ISSUE + pd.Timedelta("10min"), ISSUE, "https://x", 60)
    late = build_record(payload(), ISSUE + pd.Timedelta("3h"), ISSUE, "https://x", 60)
    assert on_time["late"] is False and on_time["minutes_after_issue"] == 10.0
    assert late["late"] is True