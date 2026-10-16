from carbon_forecast.ingestion.neso import parse_forecast_rows


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