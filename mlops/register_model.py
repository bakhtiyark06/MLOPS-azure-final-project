"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Phase 4 - Azure ML model registry. Registers the trained model in the
         Azure Machine Learning workspace with rich governance tags (accuracy,
         dataset hash, git SHA, author, project, version). Skips gracefully when
         Azure credentials are unavailable so local runs still succeed.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Allow direct execution (e.g. ``python mlops/register_model.py``) by ensuring the
# repo root is on sys.path so the top-level ``mlops`` package can be imported.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

from mlops.azure_clients import azure_is_configured, get_ml_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops.register")

MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("reports/metrics.json")
REGISTRATION_RECORD = Path("reports/model_registration.json")


def _git_sha() -> str:
    """Return the current git commit SHA (env first, then ``git``, else ``unknown``)."""
    for env_var in ("GIT_SHA", "GITHUB_SHA"):
        if os.getenv(env_var):
            return os.environ[env_var]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def build_tags(metrics: Dict[str, object]) -> Dict[str, str]:
    """Construct the governance tags attached to the registered model.

    Args:
        metrics: Parsed metrics dict (provides accuracy + dataset hash).

    Returns:
        dict[str, str]: Tag key/value pairs (all stringified for Azure ML).
    """
    return {
        "accuracy": str(metrics.get("accuracy", "")),
        "dataset_hash": str(metrics.get("dataset_hash", "")),
        "git_sha": _git_sha(),
        "created_by": os.getenv("CREATED_BY", os.getenv("GITHUB_ACTOR", "bakhtiyar-khan")),
        "project_name": os.getenv("PROJECT_NAME", "mlops-azure-final-project"),
    }


def load_metrics() -> Dict[str, object]:
    """Load metrics, returning an empty dict if they are unavailable."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    logger.warning("Metrics file missing; tags will have empty accuracy/hash.")
    return {}


def register_model(model_name: str = "iris-classifier") -> Optional[str]:
    """Register the local model artifact in Azure ML.

    Args:
        model_name: Logical name for the model in the registry.

    Returns:
        str | None: The registered model version, or ``None`` if skipped.
    """
    logger.info("=== Phase 4: Model registration ===")

    if not MODEL_PATH.exists():
        logger.error("Model artifact not found at %s; run training first.", MODEL_PATH)
        return None

    metrics = load_metrics()
    tags = build_tags(metrics)

    if not azure_is_configured():
        logger.warning(
            "Azure ML not configured; skipping registration. Would register '%s' with tags: %s",
            model_name,
            tags,
        )
        _write_record(model_name, version=None, tags=tags, registered=False)
        return None

    client = get_ml_client()
    if client is None:
        logger.warning("Could not build MLClient; skipping registration.")
        _write_record(model_name, version=None, tags=tags, registered=False)
        return None

    try:
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Model

        model = Model(
            path=str(MODEL_PATH),
            name=model_name,
            description="Iris classifier registered by the MLOps CI/CD pipeline.",
            type=AssetTypes.CUSTOM_MODEL,
            tags=tags,
        )
        registered = client.models.create_or_update(model)
        version = str(registered.version)
        tags["version"] = version
        logger.info("Registered model '%s' version %s.", model_name, version)
        _write_record(model_name, version=version, tags=tags, registered=True)
        return version
    except Exception as exc:  # noqa: BLE001
        logger.error("Model registration failed: %s", exc)
        _write_record(model_name, version=None, tags=tags, registered=False)
        return None


def _write_record(model_name: str, version: Optional[str], tags: Dict[str, str], registered: bool) -> None:
    """Persist a record of the registration attempt for auditability."""
    REGISTRATION_RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model_name": model_name,
        "version": version,
        "registered": registered,
        "tags": tags,
    }
    REGISTRATION_RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Wrote registration record -> %s", REGISTRATION_RECORD)


def main() -> None:
    """CLI entry point for ``python mlops/register_model.py``."""
    parser = argparse.ArgumentParser(description="Register the model in Azure ML.")
    parser.add_argument("--name", default="iris-classifier", help="Model name in the registry.")
    args = parser.parse_args()
    register_model(model_name=args.name)


if __name__ == "__main__":
    main()
