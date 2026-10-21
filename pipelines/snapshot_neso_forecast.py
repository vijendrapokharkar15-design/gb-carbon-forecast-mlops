"""Save NESO's 24h-ahead carbon intensity forecast as issued, before the 12:00 UTC issue time.

The NESO API only keeps the latest version of each forecast, so the day-ahead version
is lost unless captured on the day. These snapshots are the fair benchmark for our model.

Usage:
    python pipelines/snapshot_neso_forecast.py           # skips if not useful (see snapshot_skip_reason)
    python pipelines/snapshot_neso_forecast.py --force   # take one anyway
"""
import argparse
import json
import logging
from datetime import datetime, timezone

from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.ingestion.neso import (
    BASE_URL,
    fetch_forward_24h,
    floor_to_half_hour,
    parse_forecast_rows,
    snapshot_skip_reason,
)

logger = logging.getLogger("snapshot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Take a snapshot even if the skip rule says not to")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    bcfg = cfg["benchmark"]
    issue_hour = cfg["evaluation"]["issue_hour_utc"]

    fetched_at = datetime.now(timezone.utc)
    out_dir = resolve_path(bcfg["snapshot_dir"]) / fetched_at.strftime("%Y") / fetched_at.strftime("%m")

    if not args.force:
        existing = [p.name for p in out_dir.glob(f"fw24h_{fetched_at.strftime('%Y%m%d')}T*.json")]
        reason = snapshot_skip_reason(existing, fetched_at, issue_hour)
        if reason:
            logger.info("Skipping: %s", reason)
            return

    forecast_from = floor_to_half_hour(fetched_at)
    rows = fetch_forward_24h(forecast_from)

    # Fail loudly: a red workflow run is our alert that something is wrong
    if len(rows) < bcfg["min_periods"]:
        raise RuntimeError(f"Expected at least {bcfg['min_periods']} periods, got {len(rows)}")

    parsed = parse_forecast_rows(rows)
    if parsed["forecast"].isna().any():
        raise RuntimeError("Snapshot contains null forecast values")

    snapshot = {
        "fetched_at_utc": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_from_utc": forecast_from.strftime("%Y-%m-%dT%H:%MZ"),
        "source": f"{BASE_URL}/intensity/{forecast_from.strftime('%Y-%m-%dT%H:%MZ')}/fw24h",
        "n_periods": len(rows),
        "data": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fw24h_{fetched_at.strftime('%Y%m%dT%H%MZ')}.json"
    out_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    logger.info(
        "Saved %d periods (%s to %s) to %s",
        len(parsed), parsed["period_start"].min(), parsed["period_end"].max(), out_file,
    )


if __name__ == "__main__":
    main()