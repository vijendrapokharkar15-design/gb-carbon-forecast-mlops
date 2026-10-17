"""Build the full feature table for every historical issue time.

Usage:
    python pipelines/build_features.py
"""
import logging

import pandas as pd

from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.evaluation.backtest import make_issue_times
from carbon_forecast.features.build import FEATURE_COLUMNS, build_training_set

logger = logging.getLogger("build_features")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    ev = cfg["evaluation"]

    neso = pd.read_parquet(resolve_path(cfg["processed"]["neso_path"]))
    weather = pd.read_parquet(resolve_path(cfg["processed"]["weather_path"]))

    # Last issue day whose 24h horizon is fully covered by both NESO and weather
    data_end = min(neso["period_start"].max(), weather["time"].max()).floor("D")
    issue_times = make_issue_times(cfg["features"]["train_start"], data_end.strftime("%Y-%m-%d"), ev["issue_hour_utc"])
    logger.info("Building features for %d issue times (%s to %s)", len(issue_times), issue_times[0], issue_times[-1])

    table = build_training_set(
        neso, weather, issue_times,
        horizon_steps=ev["horizon_steps"],
        publication_lag_minutes=ev["publication_lag_minutes"],
    )

    out = resolve_path(cfg["features"]["path"])
    table.to_parquet(out, index=False)

    logger.info("Saved %s rows x %d features to %s", f"{len(table):,}", len(FEATURE_COLUMNS), out)
    logger.info("Valid targets: %s", f"{int(table['is_valid_target'].sum()):,}")
    nulls = table[FEATURE_COLUMNS].isna().mean().mul(100).round(2)
    logger.info("Share of missing values per feature (%%), non-zero only:\n%s", nulls[nulls > 0].to_string())


if __name__ == "__main__":
    main()