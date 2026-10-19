"""Call the deployed API and save today's official forecast as a frozen record.

Usage:
    python pipelines/record_forecast.py           # skip if today's record already exists
    python pipelines/record_forecast.py --force   # overwrite
"""
import argparse
import json
import logging

import pandas as pd
import requests

from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.ingestion.http import get_json_with_retry
from carbon_forecast.serving.record import build_record, validate_forecast_payload
from carbon_forecast.serving.timing import latest_issue_time

logger = logging.getLogger("record_forecast")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite today's record if it exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    fr, ev = cfg["forecast_record"], cfg["evaluation"]

    issue = latest_issue_time(pd.Timestamp.now(tz="UTC"), ev["issue_hour_utc"])
    out = (
        resolve_path(fr["out_dir"]) / issue.strftime("%Y") / issue.strftime("%m")
        / f"forecast_{issue.strftime('%Y%m%d')}.json"
    )
    if out.exists() and not args.force:
        logger.info("Record for issue %s already exists (%s); skipping", issue, out.name)
        return

    # Retries with growing waits cover a cold start (the container may be scaled to zero)
    with requests.Session() as session:
        payload = get_json_with_retry(
            session, f"{fr['api_url']}/forecast",
            params={"window_hours": 4, "upcoming_only": "false"},
            retries=5, backoff_seconds=5, timeout=fr["timeout_seconds"],
        )
    fetched_at = pd.Timestamp.now(tz="UTC")

    validate_forecast_payload(payload, issue, ev["horizon_steps"])
    record = build_record(payload, fetched_at, issue, fr["api_url"], fr["late_after_minutes"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info(
        "Saved forecast for issue %s (model %s, %s min after issue%s) to %s",
        issue, payload["model_version"], record["minutes_after_issue"],
        ", LATE" if record["late"] else "", out,
    )


if __name__ == "__main__":
    main()