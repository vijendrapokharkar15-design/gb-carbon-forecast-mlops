import pandas as pd

from carbon_forecast.serving import live_data

VARIABLES = ["wind_speed_100m", "shortwave_radiation", "cloud_cover", "temperature_2m"]
LOCATIONS = [{"name": "north_sea", "latitude": 54.0, "longitude": 1.0},
             {"name": "london", "latitude": 51.51, "longitude": -0.13}]


def test_live_weather_returns_raw_long_format(monkeypatch):
    calls = []

    def fake_get(session, url, params=None, **kwargs):
        calls.append(params)
        assert url == live_data.FORECAST_URL
        return {"hourly": {
            "time": ["2026-10-19T12:00", "2026-10-19T13:00"],
            "wind_speed_100m": [7, 8],
            "shortwave_radiation": [300, 250],
            "cloud_cover": [40, 60],
            "temperature_2m": [12.5, 12.9],
        }}

    monkeypatch.setattr(live_data, "get_json_with_retry", fake_get)
    df = live_data.fetch_live_weather(LOCATIONS, VARIABLES)

    assert len(calls) == 2                                   # one request per location
    assert list(df.columns) == ["time", "location", *VARIABLES]
    assert set(df["location"]) == {"north_sea", "london"}
    assert str(df["time"].dt.tz) == "UTC"
    assert (df[VARIABLES].dtypes == "float64").all()         # integers become floats


def test_live_neso_requests_29_days_before_issue(monkeypatch):
    seen = {}

    def fake_fetch(start, end, chunk_days, pause_seconds):
        seen["start"], seen["end"] = start, end
        return pd.DataFrame()

    monkeypatch.setattr(live_data, "fetch_national_intensity", fake_fetch)
    issue = pd.Timestamp("2026-10-19 12:00", tz="UTC")
    live_data.fetch_live_neso(issue)

    assert seen["end"] == issue.to_pydatetime()
    assert seen["start"] == (issue - pd.Timedelta(days=29)).to_pydatetime()