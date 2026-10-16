"""Backfill raw carbon intensity and weather history.

Usage:
    python pipelines/backfill.py              # both sources
    python pipelines/backfill.py --only neso
    python pipelines/backfill.py --only weather
"""
import argparse
import logging
from datetime import date, datetime, timedelta, timezone

from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.ingestion.neso import fetch_national_intensity
from carbon_forecast.ingestion.storage import save_monthly_partitions
from carbon_forecast.ingestion.weather import fetch_weather_history

logger = logging.getLogger("backfill")


def floor_to_half_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)


def backfill_neso(cfg: dict) -> None:
    start = datetime.fromisoformat(cfg["start_date"]).replace(tzinfo=timezone.utc)
    end = floor_to_half_hour(datetime.now(timezone.utc))
    df = fetch_national_intensity(start, end, cfg["neso"]["chunk_days"], cfg["neso"]["pause_seconds"])
    paths = save_monthly_partitions(
        df, "period_start", resolve_path(cfg["neso"]["out_dir"]), "neso_intensity", ["period_start"]
    )
    logger.info("NESO done: %d rows in %d monthly files", len(df), len(paths))


def backfill_weather(cfg: dict) -> None:
    wcfg = cfg["weather"]
    start = date.fromisoformat(cfg["start_date"])
    end = date.today() - timedelta(days=wcfg["archive_lag_days"])
    for loc in wcfg["locations"]:
        df = fetch_weather_history(
            loc["name"], loc["latitude"], loc["longitude"], start, end, wcfg["variables"], wcfg["pause_seconds"]
        )
        paths = save_monthly_partitions(
            df, "time", resolve_path(wcfg["out_dir"]), f"weather_{loc['name']}", ["time", "location"]
        )
        logger.info("Weather %s done: %d rows in %d monthly files", loc["name"], len(df), len(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill raw data")
    parser.add_argument("--only", choices=["neso", "weather"], help="Run one source only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()

    if args.only in (None, "neso"):
        backfill_neso(cfg)
    if args.only in (None, "weather"):
        backfill_weather(cfg)


if __name__ == "__main__":
    main()