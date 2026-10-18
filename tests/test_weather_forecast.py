from datetime import date

from carbon_forecast.ingestion import weather as wx


def test_forecast_columns_renamed_to_plain_names(monkeypatch):
    def fake_get(session, url, params=None, **kwargs):
        assert url == wx.PREVIOUS_RUNS_URL
        assert "wind_speed_100m_previous_day1" in params["hourly"]
        return {"hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "wind_speed_100m_previous_day1": [5, 6],
            "temperature_2m_previous_day1": [3.0, 3.1],
        }}

    monkeypatch.setattr(wx, "get_json_with_retry", fake_get)
    monkeypatch.setattr(wx.time, "sleep", lambda s: None)

    df = wx.fetch_weather_forecast_history(
        "north_sea", 54.0, 1.0, date(2025, 1, 1), date(2025, 1, 1), ["wind_speed_100m", "temperature_2m"]
    )
    assert list(df.columns) == ["time", "location", "wind_speed_100m", "temperature_2m"]
    assert str(df["time"].dt.tz) == "UTC"
    assert df["wind_speed_100m"].dtype == "float64"   # integers from the API become floats