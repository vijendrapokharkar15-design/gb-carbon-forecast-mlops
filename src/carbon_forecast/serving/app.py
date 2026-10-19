"""FastAPI app serving the day-ahead GB carbon intensity forecast.

Run locally:
    uvicorn carbon_forecast.serving.app:create_app --factory --port 8000
"""
import logging

import requests
from fastapi import FastAPI, HTTPException, Query

from carbon_forecast.serving.service import ForecastService, greenest_window
from carbon_forecast.validation.schemas import DataValidationError


def create_app(service: ForecastService | None = None) -> FastAPI:
    """Build the app. Tests pass in a service with fake data; in production it loads the champion."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = service or ForecastService.from_registry()

    app = FastAPI(
        title="GB Carbon Intensity Forecast",
        description=(
            "Day-ahead forecast of Great Britain's electricity carbon intensity (gCO2/kWh), "
            "issued daily at 12:00 UTC for the next 24 hours, with the greenest window "
            "for shifting flexible electricity use."
        ),
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        """Liveness check, plus which model version is loaded."""
        return {"status": "ok", "model_version": service.model_version}

    @app.get("/model")
    def model():
        """Champion model metadata: training cutoff, evaluation scores, promotion reason."""
        return service.metadata

    @app.get("/forecast")
    def forecast(
        window_hours: float = Query(4, gt=0, le=12, description="Length of the flexible load, in hours"),
        upcoming_only: bool = Query(True, description="Only recommend windows that haven't started yet"),
    ):
        """The latest 12:00 UTC forecast and the recommended greenest window."""
        try:
            issue, fc = service.forecast()
        except DataValidationError as exc:
            raise HTTPException(status_code=503, detail=f"Input data failed validation: {exc}") from exc
        except (RuntimeError, requests.RequestException) as exc:
            raise HTTPException(status_code=503, detail=f"Upstream data source unavailable: {exc}") from exc

        not_before = service.now() if upcoming_only else None
        return {
            "issue_time": issue.isoformat(),
            "model_version": service.model_version,
            "unit": "gCO2/kWh",
            "greenest_window": greenest_window(fc, window_hours, not_before),
            "forecast": [
                {
                    "period_start": row.period_start.isoformat(),
                    "period_end": row.period_end.isoformat(),
                    "intensity": round(float(row.intensity), 1),
                }
                for row in fc.itertuples()
            ],
        }

    return app