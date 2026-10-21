from datetime import datetime, timezone

from carbon_forecast.ingestion.neso import parse_forecast_rows, snapshot_skip_reason


def test_parse_forecast_rows_sorts_and_types():
    rows = [
        {"from": "2026-10-16T12:30Z", "to": "2026-10-16T13:00Z", "intensity": {"forecast": 70, "actual": None, "index": "low"}},
        {"from": "2026-10-16T12:00Z", "to": "2026-10-16T12:30Z", "intensity": {"forecast": 69, "actual": None, "index": "low"}},
    ]
    df = parse_forecast_rows(rows)

    assert list(df.columns) == ["period_start", "period_end", "forecast"]
    assert df["period_start"].is_monotonic_increasing
    assert str(df["period_start"].dt.tz) == "UTC"
    assert df["forecast"].dtype == "float64"
    assert df["forecast"].tolist() == [69.0, 70.0]


def test_parse_forecast_rows_handles_empty_input():
    df = parse_forecast_rows([])
    assert df.empty
    assert list(df.columns) == ["period_start", "period_end", "forecast"]



def _at(hour, minute=0):
    return datetime(2026, 10, 21, hour, minute, tzinfo=timezone.utc)


def test_snapshot_taken_when_none_exists_before_issue():
    assert snapshot_skip_reason([], _at(11, 5)) is None


def test_skip_when_on_time_snapshot_exists():
    reason = snapshot_skip_reason(["fw24h_20261021T1105Z.json"], _at(11, 41))
    assert reason and "already exists" in reason


def test_skip_after_issue_time_even_with_no_snapshot():
    reason = snapshot_skip_reason([], _at(15, 51))   # GitHub's late backup run
    assert reason and "past" in reason


def test_late_snapshots_do_not_count_as_on_time():
    # A 12:00 file exists, but it's after the issue: still skip (past issue time), never "already exists"
    reason = snapshot_skip_reason(["fw24h_20261021T1200Z.json"], _at(12, 30))
    assert "past" in reason