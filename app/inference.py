"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Model loading and inference helpers for the Iris classifier. Provides a
         small ``ModelService`` wrapper that lazily loads the trained model from
         ``MODEL_PATH`` and exposes a typed ``predict`` method used by the API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from app.schemas import CLASS_NAMES, FEATURE_COLUMNS

# Default location of the serialised model. Overridable via the MODEL_PATH env
# var so the same image can serve different models without a rebuild.
DEFAULT_MODEL_PATH = "models/model.joblib"


def get_model_path() -> str:
    """Resolve the model path from the environment (falling back to the default).

    Returns:
        str: Path to the ``.joblib`` model artifact.
    """
    return os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)


def validate_features(features: List[float]) -> List[float]:
    """Validate a raw feature vector before it reaches the model.

    Args:
        features: Ordered list of the four Iris measurements.

    Returns:
        list[float]: The validated, float-cast feature vector.

    Raises:
        ValueError: If the wrong number of features is supplied or any value is
            negative / not a finite number.
    """
    if len(features) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features {FEATURE_COLUMNS}, "
            f"got {len(features)}."
        )

    cast: List[float] = []
    for name, value in zip(FEATURE_COLUMNS, features):
        fvalue = float(value)
        if not np.isfinite(fvalue):
            raise ValueError(f"Feature '{name}' must be a finite number.")
        if fvalue < 0:
            raise ValueError(f"Feature '{name}' must be non-negative.")
        cast.append(fvalue)
    return cast


class ModelService:
    """Thin wrapper around a scikit-learn model that handles lazy loading.

    The model is loaded the first time it is needed (or eagerly via ``load``)
    and cached for the lifetime of the process so each request is fast.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialise the service.

        Args:
            model_path: Optional explicit path; defaults to ``get_model_path()``.
        """
        self.model_path: str = model_path or get_model_path()
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        """bool: Whether the underlying model is currently loaded in memory."""
        return self._model is not None

    def load(self) -> Any:
        """Load and cache the model from disk.

        Returns:
            The deserialised scikit-learn estimator.

        Raises:
            FileNotFoundError: If no model artifact exists at ``model_path``.
        """
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{path}'. Run 'python mlops/train.py' "
                f"first or set MODEL_PATH to a valid model."
            )
        self._model = joblib.load(path)
        return self._model

    def _ensure_loaded(self) -> Any:
        """Return the cached model, loading it on first use."""
        if self._model is None:
            self.load()
        return self._model

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Run a single prediction and return a structured result.

        Args:
            features: Ordered list ``[sepal_length, sepal_width, petal_length,
                petal_width]``.

        Returns:
            dict: ``{"prediction": int, "class": str, "confidence": float}``.
        """
        model = self._ensure_loaded()
        clean = validate_features(features)
        sample = np.asarray(clean, dtype=float).reshape(1, -1)

        label = int(model.predict(sample)[0])

        # Derive a confidence score. Prefer calibrated probabilities when the
        # estimator exposes them; otherwise fall back to a neutral 1.0.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0

        class_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)
        return {
            "prediction": label,
            "class": class_name,
            "confidence": round(confidence, 6),
        }


# Module-level singleton reused across requests within a single process.
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Return a process-wide ``ModelService`` singleton (FastAPI dependency).

    Returns:
        ModelService: The shared model service instance.
    """
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
