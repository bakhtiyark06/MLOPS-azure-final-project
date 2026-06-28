"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Tests for the training stage - data loading, splitting, metric
         computation, confusion-matrix rendering and the full train() flow.
"""

from __future__ import annotations

from pathlib import Path

from mlops import train as train_module
from mlops.ingest_data import FEATURE_COLUMNS, TARGET_COLUMN


def test_load_training_data_has_expected_columns() -> None:
    """Training data loads with all feature columns and the target column."""
    frame = train_module.load_training_data()
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert column in frame.columns


def test_split_data_is_stratified_and_sized() -> None:
    """The split produces correctly sized, non-overlapping partitions."""
    frame = train_module.load_training_data()
    x_train, x_test, y_train, y_test = train_module.split_data(frame, test_size=0.2)
    assert len(x_train) + len(x_test) == len(frame)
    assert len(y_test) == len(x_test)
    assert list(x_train.columns) == FEATURE_COLUMNS


def test_compute_metrics_perfect_prediction() -> None:
    """Identical predictions yield perfect metrics."""
    import pandas as pd

    y = pd.Series([0, 1, 2, 0, 1, 2])
    metrics = train_module.compute_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_save_confusion_matrix(tmp_path: Path) -> None:
    """A confusion-matrix PNG is written to disk."""
    import pandas as pd

    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = pd.Series([0, 1, 2, 0, 2, 1])
    out = tmp_path / "cm.png"
    train_module.save_confusion_matrix(y_true, y_pred, path=out)
    assert out.exists() and out.stat().st_size > 0


def test_full_train_produces_artifacts() -> None:
    """End-to-end training writes the model and metrics with passing accuracy."""
    metrics = train_module.train()
    assert metrics["accuracy"] >= 0.90
    assert train_module.MODEL_PATH.exists()
    assert train_module.METRICS_PATH.exists()
    assert train_module.CONFUSION_MATRIX_PATH.exists()
