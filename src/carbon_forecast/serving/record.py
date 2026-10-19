"""Freeze the official daily forecast: validate what the API served and build the record."""
import math

import pandas as pd


def validate_forecast_payload(payload: dict, expected_issue: pd.Timestamp, horizon_steps: int) -> None:
    """Raise ValueError if the API response isn't a complete, sane forecast for the expected issue."""
    if not payload.get("model_version"):
        raise ValueError("Response has no model_version")

    issue = pd.Timestamp(payload.get("issue_time"))
    if issue != expected_issue:
        raise ValueError(f"API served issue {issue}, expected {expected_issue}")

    periods = payload.get("forecast") or []
    if len(periods) != horizon_steps:
        raise ValueError(f"Expected {horizon_steps} periods, got {len(periods)}")

    for p in periods:
        value = p.get("intensity")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1000:
            raise ValueError(f"Implausible intensity {value!r} at {p.get('period_start')}")


def build_record(
    payload: dict,
    fetched_at: pd.Timestamp,
    expected_issue: pd.Timestamp,
    api_url: str,
    late_after_minutes: int,
) -> dict:
    """Wrap the API response with when and where it was captured, and whether it was late."""
    minutes_after_issue = (fetched_at - expected_issue).total_seconds() / 60
    return {
        "fetched_at_utc": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue_time_utc": expected_issue.strftime("%Y-%m-%dT%H:%MZ"),
        "minutes_after_issue": round(minutes_after_issue, 1),
        "late": minutes_after_issue > late_after_minutes,
        "api_url": api_url,
        "response": payload,
    }