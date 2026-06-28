"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Pydantic request/response schemas for the Iris classification API.
         Centralising the schemas keeps validation logic in one place and gives
         FastAPI automatic OpenAPI documentation and input validation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Canonical ordering of the four Iris features. The model is trained on columns
# in exactly this order, so inference must build feature vectors the same way.
FEATURE_COLUMNS: List[str] = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

# Human readable class names indexed by the integer label the model predicts.
CLASS_NAMES: List[str] = ["setosa", "versicolor", "virginica"]


class PredictionRequest(BaseModel):
    """Single Iris sample submitted to the ``POST /predict`` endpoint.

    All four measurements are required and must be non-negative floats
    (centimetres). ``ge=0`` rejects physically impossible measurements early.
    """

    sepal_length: float = Field(..., ge=0, description="Sepal length in cm", examples=[5.1])
    sepal_width: float = Field(..., ge=0, description="Sepal width in cm", examples=[3.5])
    petal_length: float = Field(..., ge=0, description="Petal length in cm", examples=[1.4])
    petal_width: float = Field(..., ge=0, description="Petal width in cm", examples=[0.2])

    def as_feature_list(self) -> List[float]:
        """Return the measurements ordered to match the training feature order.

        Returns:
            list[float]: ``[sepal_length, sepal_width, petal_length, petal_width]``.
        """
        return [
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]


class PredictionResponse(BaseModel):
    """Response returned by the ``POST /predict`` endpoint."""

    prediction: int = Field(..., description="Predicted integer class label", examples=[0])
    class_name: str = Field(..., alias="class", description="Predicted class name", examples=["setosa"])
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (0-1)", examples=[0.99])
    # Optional per-class probabilities. Additive field used by the dashboard to
    # render confidence bars; omitted (None) leaves the legacy contract intact.
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-class probabilities keyed by class name",
        examples=[{"setosa": 0.99, "versicolor": 0.01, "virginica": 0.0}],
    )

    # ``populate_by_name`` lets us build the object using ``class_name=...`` in
    # Python while still serialising the field as ``class`` in JSON responses.
    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    """Response returned by the ``GET /health`` endpoint."""

    status: str = Field(..., description="Service status", examples=["ok"])
    model_loaded: bool = Field(..., description="Whether a model is loaded in memory")
    model_path: str = Field(..., description="Path the model was loaded from")
