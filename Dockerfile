# ---------------------------------------------------------------------------
# Author:  Bakhtiyar Khan
# Date:    2026-06-27
# Purpose: Production image for the Iris FastAPI service. Uses a slim Python
#          3.11 base, installs only the lean serving dependencies, runs as a
#          non-root user and serves the API with uvicorn on port 8000.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# --- Runtime hygiene --------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=models/model.joblib \
    PORT=8000

WORKDIR /app

# --- Dependencies (separate layer for better build caching) -----------------
COPY requirements-api.txt ./
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# --- Application code + model artifact --------------------------------------
COPY app ./app
COPY models ./models

# --- Run as a non-root user for security ------------------------------------
RUN useradd --create-home --uid 1001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# --- Container-level health check against the API ---------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
