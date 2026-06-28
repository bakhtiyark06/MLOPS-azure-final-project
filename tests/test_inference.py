"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Tests for the inference layer - feature validation, model loading and
         the structure / correctness of prediction results.
"""

from __future__ import annotations

import pytest

from app.inference import ModelService, get_model_service, validate_features
from app.schemas import CLASS_NAMES, FEATURE_COLUMNS


def test_validate_features_accepts_valid_vector() -> None:
    """A correct-length, non-negative vector is returned as floats."""
    result = validate_features([5.1, 3.5, 1.4, 0.2])
    assert result == [5.1, 3.5, 1.4, 0.2]
    assert all(isinstance(x, float) for x in result)


def test_validate_features_rejects_wrong_length() -> None:
    """A vector of the wrong length raises ValueError."""
    with pytest.raises(ValueError):
        validate_features([5.1, 3.5, 1.4])


def test_validate_features_rejects_negative() -> None:
    """A negative measurement raises ValueError."""
    with pytest.raises(ValueError):
        validate_features([-1.0, 3.5, 1.4, 0.2])


def test_model_service_predicts_setosa() -> None:
    """A canonical setosa sample is classified as setosa with valid confidence."""
    service = ModelService()
    result = service.predict([5.1, 3.5, 1.4, 0.2])
    assert result["prediction"] in range(len(CLASS_NAMES))
    assert result["class"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    # Setosa is linearly separable and should be predicted reliably.
    assert result["class"] == "setosa"


def test_get_model_service_is_singleton() -> None:
    """The dependency factory returns a cached singleton instance."""
    assert get_model_service() is get_model_service()


def test_feature_columns_match_class_count() -> None:
    """Sanity check on the shared schema constants."""
    assert len(FEATURE_COLUMNS) == 4
    assert len(CLASS_NAMES) == 3
