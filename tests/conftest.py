"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Shared pytest fixtures. Ensures a trained model and ingested datasets
         exist before the test session so the API and inference tests can run
         deterministically in any environment (including a clean CI runner).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def trained_model() -> Path:
    """Guarantee a model artifact exists for the test session.

    Trains a small model once per session if ``models/model.joblib`` is absent
    (e.g. on a fresh CI checkout). Returns the model path.
    """
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        # Importing here keeps import-time light and avoids heavy deps unless needed.
        from mlops.train import train

        # Use the project defaults so metrics are deterministic and comfortably
        # above the 0.90 quality-gate threshold.
        train()
    assert model_path.exists(), "Model artifact should exist after training."
    return model_path
