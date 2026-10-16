"""Fetch historical hourly weather from the Open-Meteo archive API."""
import logging
import time
from datetime import date

import pandas as pd
import requests

from carbon_forecast.ingestion.http import get_json_with_retry

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_history(
    name: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    variables: list[str],
    pause_seconds: float = 1.0,
) -> pd.DataFrame:
    """Return hourly weather for one location, requested one year at a time.
    Output columns: time (UTC), location, then one column per variable."""
    frames = []

    with requests.Session() as session:
        for year in range(start_date.year, end_date.year + 1):
            y_start = max(start_date, date(year, 1, 1))
            y_end = min(end_date, date(year, 12, 31))
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": y_start.isoformat(),
                "end_date": y_end.isoformat(),
                "hourly": ",".join(variables),
                "timezone": "UTC",
                "wind_speed_unit": "ms",
            }
            payload = get_json_with_retry(session, ARCHIVE_URL, params=params)
            year_df = pd.DataFrame(payload.get("hourly", {}))
            logger.info("Weather %s %d: %d rows", name, year, len(year_df))
            frames.append(year_df)
            time.sleep(pause_seconds)

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df[variables] = df[variables].apply(pd.to_numeric, errors="coerce").astype("float64")
    df.insert(1, "location", name)
    return df