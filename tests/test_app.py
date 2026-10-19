from fastapi.testclient import TestClient

from carbon_forecast.serving.app import create_app
from tests.test_service import fake_weather, make_service


def client(**kwargs):
    service, _, _ = make_service(**kwargs)
    return TestClient(create_app(service))


def test_health_reports_model_version():
    r = client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "vtest"}


def test_forecast_returns_48_periods_and_a_window():
    r = client().get("/forecast")
    assert r.status_code == 200
    body = r.json()
    assert body["issue_time"].startswith("2026-10-19T12:00")
    assert len(body["forecast"]) == 48
    assert body["greenest_window"] is not None


def test_invalid_window_rejected():
    assert client().get("/forecast", params={"window_hours": 0}).status_code == 422
    assert client().get("/forecast", params={"window_hours": 20}).status_code == 422


def test_bad_data_returns_503_not_a_forecast():
    def broken_weather(locations, variables):
        return fake_weather(locations, variables, wind=999.0)

    r = client(weather=broken_weather).get("/forecast")
    assert r.status_code == 503
    assert "validation" in r.json()["detail"]