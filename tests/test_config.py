"""Checks the real config file has every setting the pipelines rely on.
Catches accidental deletions locally, before they break a cloud run."""
from carbon_forecast.config import load_config

REQUIRED = {
    "neso": ["chunk_days", "pause_seconds", "out_dir"],
    "weather": ["out_dir", "pause_seconds", "archive_lag_days", "variables", "locations"],
    "benchmark": ["snapshot_dir", "min_periods"],
    "features": ["train_start", "path"],
    "promotion": ["objective", "eval_days", "mae_tolerance", "regret_tolerance", "registry_dir"],
    "processed": ["neso_path", "weather_path"],
    "evaluation": [
        "issue_hour_utc", "horizon_steps", "publication_lag_minutes", "window_steps",
        "good_window_threshold", "test_start", "test_end", "tuning_start", "predictions_dir",
    ],
}


def test_config_has_required_settings():
    cfg = load_config()
    assert "start_date" in cfg
    missing = [
        f"{section}.{key}"
        for section, keys in REQUIRED.items()
        for key in keys
        if key not in (cfg.get(section) or {})
    ]
    assert not missing, f"Missing config settings: {missing}"