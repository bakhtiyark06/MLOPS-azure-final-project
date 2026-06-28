"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: FastAPI application exposing the Iris classifier. Provides a health
         probe (``GET /health``) and a prediction endpoint (``POST /predict``).
         Logging is wired so that, when an Application Insights connection
         string is present, telemetry is shipped to Azure Monitor.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.inference import ModelService, get_model_service
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

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


@app.get("/", tags=["system"])
def root() -> JSONResponse:
    """Friendly landing route pointing callers at the docs."""
    return JSONResponse(
        {
            "service": "Iris MLOps API",
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict",
        }
    )


if __name__ == "__main__":  # pragma: no cover - manual local run convenience
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
