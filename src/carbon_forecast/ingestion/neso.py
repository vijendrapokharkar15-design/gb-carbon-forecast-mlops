"""Fetch national GB carbon intensity history from the NESO Carbon Intensity API."""
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from carbon_forecast.ingestion.http import get_json_with_retry

logger = logging.getLogger(__name__)

BASE_URL = "https://api.carbonintensity.org.uk"
COLUMNS = ["period_start", "period_end", "forecast", "actual", "intensity_index"]


def make_chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into consecutive windows of at most chunk_days."""
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    chunks = []
    cursor = start
    step = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def fetch_national_intensity(
    start: datetime,
    end: datetime,
    chunk_days: int = 30,
    pause_seconds: float = 1.0,
) -> pd.DataFrame:
    """Return half-hourly national intensity for periods starting in [start, end).
    start and end must be timezone-aware UTC datetimes."""
    chunks = make_chunks(start, end, chunk_days)
    frames = []

    with requests.Session() as session:
        for i, (c_start, c_end) in enumerate(chunks, start=1):
            url = f"{BASE_URL}/intensity/{_iso(c_start)}/{_iso(c_end)}"
            payload = get_json_with_retry(session, url)
            rows = payload.get("data", [])
            logger.info("NESO chunk %d/%d (%s to %s): %d rows", i, len(chunks), _iso(c_start), _iso(c_end), len(rows))
            if rows:
                frames.append(pd.json_normalize(rows))
            time.sleep(pause_seconds)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.concat(frames, ignore_index=True).rename(
        columns={
            "from": "period_start",
            "to": "period_end",
            "intensity.forecast": "forecast",
            "intensity.actual": "actual",
            "intensity.index": "intensity_index",
        }
    )
    df = df.reindex(columns=COLUMNS)
    df["period_start"] = pd.to_datetime(df["period_start"], utc=True)
    df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
    # Always float, so every monthly file has the same schema
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce").astype("float64")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce").astype("float64")

    # Chunks overlap at boundaries: keep requested window and deduplicate
    mask = (df["period_start"] >= pd.Timestamp(start)) & (df["period_start"] < pd.Timestamp(end))
    df = (
        df[mask]
        .drop_duplicates(subset="period_start", keep="last")
        .sort_values("period_start")
        .reset_index(drop=True)
    )
    return df

def floor_to_half_hour(dt: datetime) -> datetime:
    """Round a datetime down to the start of its half-hour period."""
    return dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)


def fetch_forward_24h(from_dt: datetime) -> list[dict]:
    """Return NESO's forecast for the 24 hours after from_dt, as raw API rows."""
    url = f"{BASE_URL}/intensity/{_iso(from_dt)}/fw24h"
    with requests.Session() as session:
        payload = get_json_with_retry(session, url)
    return payload.get("data", [])


def parse_forecast_rows(rows: list[dict]) -> pd.DataFrame:
    """Turn raw forecast rows into a clean table: period_start, period_end, forecast."""
    df = pd.json_normalize(rows).rename(
        columns={"from": "period_start", "to": "period_end", "intensity.forecast": "forecast"}
    )
    df = df.reindex(columns=["period_start", "period_end", "forecast"])
    df["period_start"] = pd.to_datetime(df["period_start"], utc=True)
    df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce").astype("float64")
    return df.sort_values("period_start").reset_index(drop=True)