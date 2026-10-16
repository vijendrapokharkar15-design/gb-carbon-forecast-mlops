"""Build processed datasets: validate raw -> clean -> validate processed -> save.

Usage:
    python pipelines/build_processed.py
"""
import logging

from carbon_forecast.config import load_config, resolve_path
from carbon_forecast.ingestion.storage import load_partitions
from carbon_forecast.processing.clean import clean_neso, clean_weather
from carbon_forecast.validation.schemas import (
    PROCESSED_NESO_SCHEMA,
    RAW_NESO_SCHEMA,
    build_processed_weather_schema,
    build_raw_weather_schema,
    validate,
)

logger = logging.getLogger("build_processed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    variables = cfg["weather"]["variables"]
    locations = [loc["name"] for loc in cfg["weather"]["locations"]]

    # NESO
    raw_neso = load_partitions(resolve_path(cfg["neso"]["out_dir"]))
    validate(raw_neso, RAW_NESO_SCHEMA, "raw NESO")
    neso = clean_neso(raw_neso)
    validate(neso, PROCESSED_NESO_SCHEMA, "processed NESO")

    neso_path = resolve_path(cfg["processed"]["neso_path"])
    neso_path.parent.mkdir(parents=True, exist_ok=True)
    neso.to_parquet(neso_path, index=False)
    logger.info(
        "Processed NESO: %d rows, %d valid targets, invalid: %s",
        len(neso),
        int(neso["is_valid_target"].sum()),
        neso["invalid_reason"].value_counts().to_dict(),
    )

    # Weather (only locations currently in the config)
    raw_weather = load_partitions(resolve_path(cfg["weather"]["out_dir"]))
    raw_weather = raw_weather[raw_weather["location"].isin(locations)]
    validate(raw_weather, build_raw_weather_schema(variables), "raw weather")
    weather = clean_weather(raw_weather, variables, locations)
    validate(weather, build_processed_weather_schema(variables, locations), "processed weather")

    weather_path = resolve_path(cfg["processed"]["weather_path"])
    weather.to_parquet(weather_path, index=False)
    logger.info(
        "Processed weather: %d hours x %d columns (%s to %s)",
        len(weather),
        weather.shape[1] - 1,
        weather["time"].min(),
        weather["time"].max(),
    )


if __name__ == "__main__":
    main()