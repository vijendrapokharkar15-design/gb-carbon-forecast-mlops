"""Train a challenger, compare it with the champion on recent unseen data, then promote or reject.

Usage:
    python pipelines/train_and_promote.py                     # as of the latest data
    python pipelines/train_and_promote.py --as-of 2026-08-01  # simulate running on an earlier date

On Day 3 this becomes the scheduled retraining job.
"""
import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.evaluation.backtest import run_backtest
from carbon_forecast.evaluation.metrics import summarise
from carbon_forecast.features.build import FEATURE_COLUMNS
from carbon_forecast.training.baselines import NaiveForecaster
from carbon_forecast.training.lgbm import BoosterForecaster, LightGBMForecaster
from carbon_forecast.training.promotion import decide
from carbon_forecast.training.registry import ModelRegistry

logger = logging.getLogger("train_and_promote")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None, help="Simulate running on this date (YYYY-MM-DD)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("carbon_forecast.evaluation.backtest").setLevel(logging.WARNING)

    cfg = load_config()
    ev, pr = cfg["evaluation"], cfg["promotion"]
    neso = pd.read_parquet(resolve_path(cfg["processed"]["neso_path"]))
    weather = pd.read_parquet(resolve_path(cfg["processed"]["weather_path"]))
    table = pd.read_parquet(resolve_path(cfg["features"]["path"]))

    # Latest day with complete data, or the simulated date
    data_end = min(neso["period_start"].max(), weather["time"].max()).floor("D")
    if args.as_of:
        data_end = min(data_end, pd.Timestamp(args.as_of, tz="UTC"))

    registry = ModelRegistry(resolve_path(pr["registry_dir"]))
    champ = registry.load_champion(FEATURE_COLUMNS)

    # Evaluation window: recent days that NEITHER model has trained on
    eval_start = data_end - pd.Timedelta(days=pr["eval_days"])
    if champ:
        trained_until = pd.Timestamp(champ[1]["trained_until"])
        eval_start = max(eval_start, trained_until)
        if data_end - eval_start < pd.Timedelta(days=pr["min_eval_days"]):
            logger.info(
                "Champion %s is trained until %s; fewer than %d days of new data. Nothing to do.",
                champ[1]["version"], trained_until.date(), pr["min_eval_days"],
            )
            return

    window = (eval_start.strftime("%Y-%m-%d"), data_end.strftime("%Y-%m-%d"))
    logger.info("Evaluation window: %s to %s", *window)

    def score(forecaster) -> dict:
        preds = run_backtest(
            forecaster, neso, weather, *window,
            issue_hour_utc=ev["issue_hour_utc"],
            horizon_steps=ev["horizon_steps"],
            publication_lag_minutes=ev["publication_lag_minutes"],
            refit_monthly=False,  # train once, at the window start
        )
        return summarise(preds, ev["window_steps"], ev["good_window_threshold"])

    challenger = score(LightGBMForecaster(table, objective=pr["objective"]))
    baseline = score(NaiveForecaster())
    champion = score(BoosterForecaster(champ[0])) if champ else None

    metrics = {"challenger": challenger, "champion": champion, "naive": baseline}
    shown = {k: v for k, v in metrics.items() if v is not None}
    logger.info(
        "Scores on the evaluation window:\n%s",
        pd.DataFrame(shown).T[["mae", "regret_mean", "good_window_rate"]].to_string(),
    )

    decision = decide(challenger, baseline, champion, pr["mae_tolerance"], pr["regret_tolerance"])

    if not decision.promote:
        registry.record_rejection(decision.reason, metrics)
        logger.info("REJECTED: %s. Champion %s stays.", decision.reason, registry.champion_version())
        return

    # Promoted: refit on ALL data up to data_end (including the evaluation window), then register
    final = LightGBMForecaster(table, objective=pr["objective"])
    final.fit(neso[neso["period_start"] < data_end], weather)

    version = registry.save_and_promote(final.model.booster_, {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trained_until": data_end.isoformat(),
        "n_train": final.n_train,
        "objective": pr["objective"],
        "params": final.params,
        "feature_columns": FEATURE_COLUMNS,
        "eval_window": list(window),
        "eval_metrics": metrics,
        "decision_reason": decision.reason,
    })
    logger.info("PROMOTED %s: %s (trained on %s rows until %s)", version, decision.reason, f"{final.n_train:,}", data_end.date())


if __name__ == "__main__":
    main()