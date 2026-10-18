"""Run one LightGBM experiment through the walk-forward backtest and log it to MLflow.

Usage:
    python pipelines/run_experiment.py --objective l2
    python pipelines/run_experiment.py --objective l1 --train-from 2024-01-01
    python pipelines/run_experiment.py --objective l1 --period test
"""
import argparse
import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.evaluation.backtest import run_backtest
from carbon_forecast.evaluation.metrics import mae_by, summarise
from carbon_forecast.evaluation.results import record_result
from carbon_forecast.training.lgbm import LightGBMForecaster

logger = logging.getLogger("run_experiment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["l2", "l1"], default="l2")
    parser.add_argument("--train-from", default=None, help="e.g. 2024-01-01; default uses all history")
    parser.add_argument("--period", choices=["tuning", "test"], default="tuning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("carbon_forecast.evaluation.backtest").setLevel(logging.WARNING)

    cfg = load_config()
    ev = cfg["evaluation"]
    neso = pd.read_parquet(resolve_path(cfg["processed"]["neso_path"]))
    weather = pd.read_parquet(resolve_path(cfg["processed"]["weather_path"]))
    table = pd.read_parquet(resolve_path(cfg["features"]["path"]))

    start, end = (ev["tuning_start"], ev["test_start"]) if args.period == "tuning" else (ev["test_start"], ev["test_end"])
    forecaster = LightGBMForecaster(table, objective=args.objective, train_from=args.train_from)

    mlflow.set_tracking_uri(f"sqlite:///{resolve_path('mlflow.db').as_posix()}")
    mlflow.set_experiment("carbon-forecast-backtest")

    with mlflow.start_run(run_name=f"{forecaster.name}_{args.period}"):
        mlflow.log_params({
            "model": forecaster.name,
            "period": args.period,
            "loss": args.objective,
            "train_from": args.train_from or "all",
            **forecaster.params,
        })

        preds = run_backtest(
            forecaster, neso, weather, start, end,
            issue_hour_utc=ev["issue_hour_utc"],
            horizon_steps=ev["horizon_steps"],
            publication_lag_minutes=ev["publication_lag_minutes"],
        )

        summary = summarise(preds, ev["window_steps"], ev["good_window_threshold"])
        mlflow.log_metrics({k: summary[k] for k in ["mae", "regret_mean", "good_window_rate"]})

        buckets = pd.cut(preds["horizon"], bins=[0, 12, 24, 36, 48], labels=["0-6h", "6-12h", "12-18h", "18-24h"])
        by_horizon = mae_by(preds, buckets)
        mlflow.log_metrics({f"mae_{label}": float(v) for label, v in by_horizon.items()})

        importance = forecaster.feature_importance()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature_importance.csv"
            importance.to_csv(path)
            mlflow.log_artifact(str(path))

        out_dir = resolve_path(ev["predictions_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out_dir / f"{forecaster.name}_{args.period}.parquet", index=False)
        record_result(summary, args.period)

    logger.info("%s", summary)
    logger.info("MAE by horizon:\n%s", by_horizon.round(1).to_string())
    logger.info("Top 10 features (share of gain, final refit):\n%s", importance.head(10).to_string())


if __name__ == "__main__":
    main()