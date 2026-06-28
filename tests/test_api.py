"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Tests for the FastAPI service using FastAPI's TestClient - health
         probe, successful predictions and input-validation error handling.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """GET /health returns 200 with the expected payload shape."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "model_path" in body


def test_root_endpoint() -> None:
    """GET / returns a small service descriptor."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Iris MLOps API"


def test_predict_returns_valid_response() -> None:
    """POST /predict returns prediction, class and confidence for valid input."""
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "class" in body  # serialised via alias
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["class"] == "setosa"


def test_predict_rejects_missing_field() -> None:
    """A missing feature triggers FastAPI's 422 validation error."""
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_value() -> None:
    """A negative measurement is rejected by schema validation (422)."""
    payload = {"sepal_length": -5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
