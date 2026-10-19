"""Fetch the live inputs needed to issue a forecast.

Both functions return data in the same RAW format as the backfill,
so the existing schemas, cleaning and feature code apply unchanged.
"""
import pandas as pd
import requests

from carbon_forecast.ingestion.http import get_json_with_retry
from carbon_forecast.ingestion.neso import fetch_national_intensity

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_live_neso(issue_time: pd.Timestamp, history_days: int = 29) -> pd.DataFrame:
    """Recent national intensity before the issue time.
    29 days covers the 4-week profile and 7-day lag features, within NESO's 31-day request limit."""
    start = issue_time - pd.Timedelta(days=history_days)
    return fetch_national_intensity(
        start.to_pydatetime(), issue_time.to_pydatetime(), chunk_days=30, pause_seconds=0
    )


def fetch_live_weather(
    locations: list[dict],
    variables: list[str],
    past_days: int = 2,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """Hourly weather forecasts for each location, long format (time, location, variables).
    Uses Open-Meteo's default model, the same one behind the archived forecasts we evaluated with."""
    frames = []
    with requests.Session() as session:
        for loc in locations:
            params = {
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "hourly": ",".join(variables),
                "timezone": "UTC",
                "wind_speed_unit": "ms",
                "past_days": past_days,
                "forecast_days": forecast_days,
            }
            payload = get_json_with_retry(session, FORECAST_URL, params=params)
            df = pd.DataFrame(payload.get("hourly", {}))
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df[variables] = df[variables].apply(pd.to_numeric, errors="coerce").astype("float64")
            df.insert(1, "location", loc["name"])
            frames.append(df)
    return pd.concat(frames, ignore_index=True)