"""Score finished forecasts against actuals and NESO's day-ahead forecast, then check for alerts.

Usage:
    python pipelines/monitor.py

Exits with code 1 if an alert fires: in GitHub Actions that makes the run red,
and GitHub emails the repository owner automatically.
"""
import json
import logging
import sys

import pandas as pd

from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.ingestion.neso import fetch_national_intensity
from carbon_forecast.monitoring.scoring import (
    as_series,
    evaluate_alerts,
    score,
    select_on_time_snapshot,
    snapshot_series,
    target_periods,
)
from carbon_forecast.serving.timing import latest_issue_time

logger = logging.getLogger("monitor")


def score_record(path, cfg: dict, now: pd.Timestamp) -> dict | None:
    """Score one frozen forecast record. Returns None if too few actuals are published yet."""
    ev, mon = cfg["evaluation"], cfg["monitoring"]
    rec = json.loads(path.read_text(encoding="utf-8"))
    issue = pd.Timestamp(rec["issue_time_utc"])
    targets = target_periods(issue, ev["horizon_steps"])
    resp = rec["response"]

    ours = as_series(
        [p["period_start"] for p in resp["forecast"]], [p["intensity"] for p in resp["forecast"]]
    ).reindex(targets)

    raw = fetch_national_intensity(
        issue.to_pydatetime(), (issue + pd.Timedelta(hours=24)).to_pydatetime(), chunk_days=2, pause_seconds=0
    )
    actual = as_series(raw["period_start"], raw["actual"]).reindex(targets)

    full = score(ours, actual, ev["window_steps"], ev["good_window_threshold"])
    if full["n_valid"] < mon["min_valid_targets"]:
        logger.info("%s: only %d actuals published so far; will retry next run", issue.date(), full["n_valid"])
        return None

    row = {
        "issue_date": issue.strftime("%Y-%m-%d"),
        "model_version": resp.get("model_version"),
        "record_late": bool(rec.get("late", False)),
        "minutes_after_issue": rec.get("minutes_after_issue"),
        "n_valid": full["n_valid"],
        "mae": full["mae"],
        "bias": full["bias"],
        "regret": full["regret"],
        "good_window": full["good_window"],
    }

    # Head-to-head with NESO: only an on-time snapshot, only the periods both forecasts cover
    snap_dir = resolve_path(cfg["benchmark"]["snapshot_dir"]) / issue.strftime("%Y") / issue.strftime("%m")
    names = [p.name for p in snap_dir.glob(f"fw24h_{issue.strftime('%Y%m%d')}T*.json")]
    chosen = select_on_time_snapshot(names, issue)
    row["neso_snapshot"] = chosen
    if chosen:
        neso = snapshot_series(json.loads((snap_dir / chosen).read_text(encoding="utf-8")), issue, ev["horizon_steps"])
        common = targets[neso.notna().to_numpy()]
        ours_c = score(ours.reindex(common), actual.reindex(common), ev["window_steps"], ev["good_window_threshold"])
        neso_c = score(neso.reindex(common), actual.reindex(common), ev["window_steps"], ev["good_window_threshold"])
        row.update({
            "common_periods": len(common),
            "ours_mae_common": ours_c["mae"],
            "ours_regret_common": ours_c["regret"],
            "neso_mae_common": neso_c["mae"],
            "neso_regret_common": neso_c["regret"],
        })

    row["scored_at_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    ev, mon = cfg["evaluation"], cfg["monitoring"]
    now = pd.Timestamp.now(tz="UTC")

    scores_path = resolve_path(mon["scores_path"])
    scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    complete = set() if scores.empty else set(scores.loc[scores["n_valid"] >= ev["horizon_steps"], "issue_date"])

    # Score every record whose 24h window has finished and isn't already fully scored
    records_dir = resolve_path(cfg["forecast_record"]["out_dir"])
    new_rows = []
    for path in sorted(records_dir.glob("*/*/forecast_*.json")):
        issue = pd.Timestamp(json.loads(path.read_text(encoding="utf-8"))["issue_time_utc"])
        if issue.strftime("%Y-%m-%d") in complete or now < issue + pd.Timedelta(hours=24):
            continue
        row = score_record(path, cfg, now)
        if row:
            new_rows.append(row)
            logger.info(
                "Scored %s: MAE %.1f, bias %+.1f, regret %s%s%s",
                row["issue_date"], row["mae"], row["bias"], row["regret"],
                " (LATE record)" if row["record_late"] else "",
                f" | vs NESO on {row['common_periods']} shared periods: ours {row['ours_mae_common']} "
                f"vs NESO {row['neso_mae_common']}" if row.get("neso_snapshot") else " | no on-time NESO snapshot",
            )

    if new_rows:
        new = pd.DataFrame(new_rows)
        scores = new if scores.empty else pd.concat(
            [scores[~scores["issue_date"].isin(new["issue_date"])], new], ignore_index=True
        )
        scores = scores.sort_values("issue_date").reset_index(drop=True)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(scores_path, index=False)
        logger.info("Saved %d scored days to %s", len(scores), scores_path)
    else:
        logger.info("Nothing new to score")

    # Alerts: performance and bias over the rolling window, plus pipeline health
    alerts = evaluate_alerts(scores, now, mon["rolling_days"], mon["mae_alert"], mon["bias_alert"])
    latest = latest_issue_time(now, ev["issue_hour_utc"])
    expected = records_dir / latest.strftime("%Y") / latest.strftime("%m") / f"forecast_{latest.strftime('%Y%m%d')}.json"
    if now >= latest + pd.Timedelta(hours=1) and not expected.exists():
        alerts.append(f"No forecast record for issue {latest}: the daily pipeline may have stopped")

    if alerts:
        for a in alerts:
            logger.error("ALERT: %s", a)
        sys.exit(1)
    logger.info("No alerts")


if __name__ == "__main__":
    main()