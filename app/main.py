"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: FastAPI application exposing the Iris classifier. Provides a health
         probe (``GET /health``) and a prediction endpoint (``POST /predict``).
         Logging is wired so that, when an Application Insights connection
         string is present, telemetry is shipped to Azure Monitor.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app import ui
from app.inference import ModelService, get_model_service
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

# Repo-root-relative directories used by the dashboard to surface artifacts.
# Resolved from this file's location so it works regardless of the CWD.
_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = _ROOT / "reports"
MODELS_DIR = _ROOT / "models"
DATA_DIR = _ROOT / "data"

# Sample request body advertised on the landing page and reused as the JSON
# descriptor example, so the documentation and UI never drift apart.
SAMPLE_PREDICTION_REQUEST = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}
SAMPLE_PREDICTION_RESPONSE = {
    "prediction": 0,
    "class": "setosa",
    "confidence": 0.99,
}

# ---------------------------------------------------------------------------
# Logging configuration. Logs always go to stdout (captured by Docker / ACI /
# AKS). If APPINSIGHTS_CONNECTION_STRING is set and the Azure log handler is
# installed, telemetry is additionally exported to Application Insights.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("mlops.api")


def _configure_app_insights() -> None:
    """Attach the Azure Application Insights log handler when configured.

    The import is done lazily and wrapped in a broad ``except`` so the API
    starts cleanly even when the optional ``opencensus-ext-azure`` package or
    the connection string is unavailable (e.g. local development).
    """
    connection_string = os.getenv("APPINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.info("Application Insights not configured; skipping telemetry export.")
        return
    try:
        from opencensus.ext.azure.log_exporter import AzureLogHandler

        logger.addHandler(AzureLogHandler(connection_string=connection_string))
        logger.info("Application Insights telemetry enabled.")
    except Exception as exc:  # noqa: BLE001 - never let telemetry break startup
        logger.warning("Could not enable Application Insights: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: configure telemetry and warm-load the model."""
    _configure_app_insights()
    service = get_model_service()
    try:
        service.load()
        logger.info("Model loaded from '%s'.", service.model_path)
    except FileNotFoundError as exc:
        # Do not crash on startup - /health will report the model is missing and
        # /predict will return a clear 503 until a model is available.
        logger.warning("Model not loaded at startup: %s", exc)
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Iris MLOps API",
    description="Production-grade Iris classification service for the Azure MLOps pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """Liveness/readiness probe used by ACI, AKS and smoke tests.

    Returns:
        HealthResponse: Service status plus whether the model is loaded.
    """
    return HealthResponse(
        status="ok",
        model_loaded=service.is_loaded,
        model_path=service.model_path,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    payload: PredictionRequest,
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Classify a single Iris sample.

    Args:
        payload: Validated request body with the four measurements.
        service: Injected model service singleton.

    Returns:
        PredictionResponse: prediction label, class name and confidence.

    Raises:
        HTTPException: 503 if the model is unavailable, 400 for invalid input.
    """
    start = time.perf_counter()
    try:
        result = service.predict(payload.as_feature_list())
    except FileNotFoundError as exc:
        logger.error("Prediction failed - model missing: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available.") from exc
    except ValueError as exc:
        logger.warning("Rejected invalid prediction request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000.0
    # Structured log line; picked up by Application Insights when configured.
    logger.info(
        "prediction served",
        extra={
            "custom_dimensions": {
                "prediction": result["prediction"],
                "class": result["class"],
                "confidence": result["confidence"],
                "latency_ms": round(latency_ms, 3),
            }
        },
    )
    return PredictionResponse(
        prediction=result["prediction"],
        class_name=result["class"],
        confidence=result["confidence"],
    )


def _service_descriptor() -> dict:
    """Return the machine-readable service descriptor served as JSON."""
    return {
        "service": "Iris MLOps API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "dashboard": "/",
        "demo": "/demo",
        "flow": "/demo/flow",
        "sample_request": SAMPLE_PREDICTION_REQUEST,
    }


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning ``{}`` when missing or invalid."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - dashboard must never crash on bad reports
        return {}


def _dashboard_context() -> dict:
    """Assemble runtime context for the dashboard from on-disk artifacts.

    Reads training metrics and the quality-gate result when present, and derives
    a status for each pipeline stage from which artifacts exist. Everything
    degrades gracefully so the dashboard renders even on a fresh checkout or
    inside a container that only ships the model.

    Returns:
        dict: Context consumed by :func:`app.ui.build_dashboard`.
    """
    metrics = _read_json(REPORTS_DIR / "metrics.json")
    gate = _read_json(REPORTS_DIR / "quality_gate.json")
    registration = _read_json(REPORTS_DIR / "model_registration.json")

    model_exists = (MODELS_DIR / "model.joblib").exists() or get_model_service().is_loaded
    stages = [
        {"name": "Data Ingestion", "status": "ok" if (DATA_DIR / "raw" / "iris.csv").exists() else "ready",
         "label": "Complete" if (DATA_DIR / "raw" / "iris.csv").exists() else "Ready"},
        {"name": "Model Training", "status": "ok" if metrics else "ready",
         "label": "Complete" if metrics else "Ready"},
        {"name": "Quality Gate",
         "status": "ok" if gate.get("passed") else ("off" if gate.get("passed") is False else "ready"),
         "label": "Passed" if gate.get("passed") else ("Failed" if gate.get("passed") is False else "Ready")},
        {"name": "Model Registry",
         "status": "ok" if registration.get("registered") else "skip",
         "label": "Registered" if registration.get("registered") else "Local skip"},
        {"name": "Container", "status": "ready", "label": "Ready"},
        {"name": "Live API", "status": "ok" if model_exists else "off",
         "label": "Online" if model_exists else "No model"},
    ]

    return {
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "dataset_hash": metrics.get("dataset_hash"),
        "gate_passed": gate.get("passed"),
        "threshold": gate.get("threshold", float(os.getenv("ACCURACY_THRESHOLD", "0.90"))),
        "stages": stages,
    }


@app.get("/", tags=["ui"], response_class=HTMLResponse)
def root(request: Request):
    """Root route.

    Serves the dark MLOps dashboard to browsers (clients whose ``Accept`` header
    asks for ``text/html``) and a machine-readable JSON descriptor to everything
    else (curl, programmatic clients, health tooling). This keeps the API
    contract intact while giving humans a professional dashboard.

    Args:
        request: Incoming request, used to inspect the ``Accept`` header.

    Returns:
        HTMLResponse | JSONResponse: dashboard for browsers, JSON otherwise.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept.lower():
        return HTMLResponse(content=ui.build_dashboard(_dashboard_context()))
    return JSONResponse(_service_descriptor())


@app.get("/demo", tags=["ui"], response_class=HTMLResponse)
def demo() -> HTMLResponse:
    """Serve the full system-architecture walkthrough page."""
    return HTMLResponse(content=ui.build_demo())


@app.get("/demo/flow", tags=["ui"], response_class=HTMLResponse)
def demo_flow() -> HTMLResponse:
    """Serve the clickable, line-by-line pipeline flow explorer page."""
    return HTMLResponse(content=ui.build_flow())


@app.get("/reports/drift", tags=["reports"], response_class=HTMLResponse)
def drift_report() -> HTMLResponse:
    """Serve the generated Evidently/PSI drift report when available."""
    path = REPORTS_DIR / "drift_report.html"
    if path.exists():
        return HTMLResponse(content=path.read_text(encoding="utf-8"))
    return HTMLResponse(content=_missing_report_html(
        "Drift report not generated yet",
        "Run <code>python mlops/drift_detection.py</code> to produce <code>reports/drift_report.html</code>.",
    ), status_code=404)


@app.get("/reports/openrouter", tags=["reports"], response_class=HTMLResponse)
def openrouter_report() -> HTMLResponse:
    """Serve the OpenRouter AI report (Markdown rendered in a simple page)."""
    path = REPORTS_DIR / "ai_report.md"
    if path.exists():
        content = path.read_text(encoding="utf-8")
        return HTMLResponse(content=(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>OpenRouter AI Report</title>"
            "<style>body{background:#070b16;color:#e6edf7;font-family:Segoe UI,Arial,sans-serif;"
            "max-width:820px;margin:0 auto;padding:40px 24px;line-height:1.6}"
            "pre{white-space:pre-wrap;background:#121c34;border:1px solid #23314f;border-radius:12px;padding:18px}"
            "a{color:#4f8cff}</style></head><body>"
            "<p><a href='/'>&larr; Back to dashboard</a></p>"
            f"<h1>OpenRouter AI Report</h1><pre>{content}</pre></body></html>"
        ))
    return HTMLResponse(content=_missing_report_html(
        "AI report not generated yet",
        "Run <code>python mlops/openrouter_report.py</code> to produce <code>reports/ai_report.md</code>.",
    ), status_code=404)


def _missing_report_html(title: str, hint: str) -> str:
    """Render a friendly dark 'report not available yet' placeholder page."""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{background:#070b16;color:#e6edf7;font-family:Segoe UI,Arial,sans-serif;"
        "display:grid;place-items:center;min-height:100vh;margin:0;text-align:center}"
        ".box{max-width:520px;padding:32px;border:1px solid #23314f;border-radius:16px;background:#121c34}"
        "a{color:#4f8cff}code{background:#1a2848;padding:2px 6px;border-radius:6px}</style></head><body>"
        f"<div class='box'><h2>{title}</h2><p style='color:#93a4c4'>{hint}</p>"
        "<p><a href='/'>&larr; Back to dashboard</a></p></div></body></html>"
    )


if __name__ == "__main__":  # pragma: no cover - manual local run convenience
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
