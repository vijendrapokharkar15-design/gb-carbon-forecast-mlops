"""Run a forecaster through the walk-forward backtest and record its scores.

Usage:
    python pipelines/run_backtest.py --model naive
    python pipelines/run_backtest.py --model all
    python pipelines/run_backtest.py --model profile --period tuning
"""
import argparse
import logging

import pandas as pd

from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.evaluation.backtest import run_backtest
from carbon_forecast.evaluation.metrics import mae_by, summarise
from carbon_forecast.evaluation.results import record_result
from carbon_forecast.training.baselines import NaiveForecaster, ProfileForecaster

logger = logging.getLogger("run_backtest")

MODELS = {
    "naive": NaiveForecaster,
    "profile": ProfileForecaster,
}



def period_bounds(ev: dict, period: str) -> tuple[str, str]:
    if period == "test":
        return ev["test_start"], ev["test_end"]
    return ev["tuning_start"], ev["test_start"]


def neso_reference(preds: pd.DataFrame) -> pd.DataFrame:
    """Score NESO's stored (near-real-time) forecast on the same targets. Reference only."""
    ref = preds.copy()
    ref["model"] = "neso_latest_reference"
    ref["y_pred"] = ref["neso_forecast_latest"]
    ref["is_valid_target"] = ref["is_valid_target"] & ref["y_pred"].notna()
    return ref



def run_one(key: str, period: str, cfg: dict, neso: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    ev = cfg["evaluation"]
    start, end = period_bounds(ev, period)
    forecaster = MODELS[key]()

    preds = run_backtest(
        forecaster, neso, weather, start, end,
        issue_hour_utc=ev["issue_hour_utc"],
        horizon_steps=ev["horizon_steps"],
        publication_lag_minutes=ev["publication_lag_minutes"],
    )

    out_dir = resolve_path(ev["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / f"{forecaster.name}_{period}.parquet", index=False)

    summary = summarise(preds, ev["window_steps"], ev["good_window_threshold"])
    record_result(summary, period)
    logger.info("%s", summary)

    buckets = pd.cut(preds["horizon"], bins=[0, 12, 24, 36, 48], labels=["0-6h", "6-12h", "12-18h", "18-24h"])
    logger.info("MAE by horizon:\n%s", mae_by(preds, buckets).round(1).to_string())
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODELS, "all"], required=True)
    parser.add_argument("--period", choices=["test", "tuning"], default="test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("carbon_forecast.evaluation.backtest").setLevel(logging.WARNING)  # quiet monthly refit logs

    cfg = load_config()
    neso = pd.read_parquet(resolve_path(cfg["processed"]["neso_path"]))
    weather = pd.read_parquet(resolve_path(cfg["processed"]["weather_path"]))

    keys = list(MODELS) if args.model == "all" else [args.model]
    preds = None
    for key in keys:
        preds = run_one(key, args.period, cfg, neso, weather)

    # NESO reference uses the same targets, so any model's predictions will do
    ev = cfg["evaluation"]
    ref_summary = summarise(neso_reference(preds), ev["window_steps"], ev["good_window_threshold"])
    record_result(ref_summary, args.period)
    logger.info("NESO latest (reference only, not day-ahead): %s", ref_summary)


if __name__ == "__main__":
    main()