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
    """GET / returns a JSON service descriptor for non-browser clients."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Iris MLOps API"


def test_root_endpoint_json_explicit() -> None:
    """An explicit JSON Accept header still returns the descriptor."""
    response = client.get("/", headers={"Accept": "application/json"})
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"
    assert "sample_request" in response.json()


def test_root_endpoint_html_for_browsers() -> None:
    """A browser-style Accept header returns the dark HTML dashboard."""
    response = client.get("/", headers={"Accept": "text/html"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "<!DOCTYPE html>" in body
    assert "Iris MLOps Dashboard" in body
    assert "/predict" in body


def test_demo_page() -> None:
    """GET /demo returns the architecture walkthrough HTML page."""
    response = client.get("/demo")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "Full system architecture" in body
    # A few required components must be present on the page.
    for component in ["GitHub Actions", "Azure ML", "Evidently Drift", "OpenRouter", "AKS Production"]:
        assert component in body


def test_demo_flow_page() -> None:
    """GET /demo/flow returns the clickable flow explorer with all steps."""
    response = client.get("/demo/flow")
    assert response.status_code == 200
    body = response.text
    assert "Flow Explorer" in body
    assert "Load Iris dataset" in body
    assert "OpenRouter AI report" in body
    assert "Next line" in body


def test_drift_report_route() -> None:
    """GET /reports/drift returns HTML (the report or a friendly placeholder)."""
    response = client.get("/reports/drift")
    assert response.status_code in (200, 404)
    assert response.headers["content-type"].startswith("text/html")


def test_openrouter_report_route() -> None:
    """GET /reports/openrouter returns HTML (the report or a placeholder)."""
    response = client.get("/reports/openrouter")
    assert response.status_code in (200, 404)
    assert response.headers["content-type"].startswith("text/html")


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


def test_predict_includes_probabilities() -> None:
    """The additive probabilities field exposes one entry per class summing to ~1."""
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    body = client.post("/predict", json=payload).json()
    probs = body.get("probabilities")
    assert probs is not None
    for cls in ("setosa", "versicolor", "virginica"):
        assert cls in probs
    assert abs(sum(probs.values()) - 1.0) < 0.01


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
