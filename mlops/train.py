"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Phase 2 - model training. Trains a scikit-learn classifier on the Iris
         dataset with a train/test split, tracks the run in MLflow (parameters,
         metrics and artifacts) and saves the model plus evaluation reports
         (metrics JSON and a confusion-matrix PNG).
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Allow direct execution (e.g. ``python mlops/train.py``) by ensuring the repo
# root is on sys.path so the top-level ``mlops`` package can be imported.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib

# Use a non-interactive backend so the script works headless in CI/containers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from mlops.ingest_data import (  # noqa: E402
    FEATURE_COLUMNS,
    RAW_CSV,
    TARGET_COLUMN,
    ingest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops.train")

# Default artifact locations.
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("reports/metrics.json")
CONFUSION_MATRIX_PATH = Path("reports/confusion_matrix.png")

# Default hyper-parameters (overridable via CLI flags).
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 3
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def load_training_data(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the training dataframe, running ingestion first if it is missing.

    Args:
        csv_path: Path to the raw dataset CSV.

    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    if not csv_path.exists():
        logger.info("Raw dataset not found at %s; running ingestion.", csv_path)
        ingest(upload=False)
    return pd.read_csv(csv_path)


def split_data(
    frame: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataframe into stratified train/test feature/label sets.

    Args:
        frame: Source dataframe containing features and the target column.
        test_size: Fraction of data held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        tuple: ``(X_train, X_test, y_train, y_test)``.
    """
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def compute_metrics(y_true: pd.Series, y_pred) -> Dict[str, float]:
    """Compute the standard multi-class classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        dict[str, float]: accuracy, precision, recall and f1 (macro-averaged).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred, path: Path = CONFUSION_MATRIX_PATH) -> Path:
    """Render and save a confusion-matrix figure.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        path: Destination PNG path.

    Returns:
        Path: The path the figure was written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved confusion matrix  -> %s", path)
    return path


def _read_dataset_hash() -> str:
    """Best-effort read of the dataset hash produced during ingestion."""
    metadata_path = Path("data/raw/metadata.json")
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8")).get("dataset_hash", "")
        except Exception:  # noqa: BLE001
            return ""
    return ""


def train(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, float]:
    """Train the model, log to MLflow and persist all artifacts.

    Args:
        n_estimators: Number of trees in the random forest.
        max_depth: Maximum tree depth.
        test_size: Fraction held out for testing.
        random_state: Reproducibility seed.

    Returns:
        dict[str, float]: The evaluation metrics on the held-out test set.
    """
    logger.info("=== Phase 2: Training ===")
    frame = load_training_data()
    x_train, x_test, y_train, y_test = split_data(frame, test_size, random_state)

    params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_size": test_size,
        "random_state": random_state,
    }
    logger.info("Training with params: %s", params)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    # Fit on plain NumPy arrays (not DataFrames) so the model does not store
    # feature names. The serving layer sends NumPy arrays, and this keeps
    # predictions warning-free without shipping pandas in the API image.
    model.fit(x_train.to_numpy(), y_train)
    predictions = model.predict(x_test.to_numpy())
    metrics = compute_metrics(y_test, predictions)
    logger.info("Test metrics: %s", metrics)

    # --- Persist artifacts locally ----------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Saved model             -> %s", MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {**metrics, "dataset_hash": _read_dataset_hash(), **params}
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Saved metrics           -> %s", METRICS_PATH)

    save_confusion_matrix(y_test, predictions)

    # --- MLflow tracking ---------------------------------------------------
    _log_to_mlflow(model, params, metrics)

    logger.info("Training complete. accuracy=%.4f", metrics["accuracy"])
    return metrics


def _log_to_mlflow(model, params: Dict, metrics: Dict[str, float]) -> None:
    """Log parameters, metrics and artifacts to MLflow.

    Wrapped defensively so a tracking-server hiccup never fails the build; the
    model and reports are already saved on disk before this runs.
    """
    try:
        # Recent MLflow raises unless this is set when using a file-based store;
        # opt in so the simple ``file:./mlruns`` backend keeps working in CI.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
        import mlflow.sklearn

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-mlops")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if CONFUSION_MATRIX_PATH.exists():
                mlflow.log_artifact(str(CONFUSION_MATRIX_PATH))
            if METRICS_PATH.exists():
                mlflow.log_artifact(str(METRICS_PATH))
            mlflow.sklearn.log_model(model, name="model")
        logger.info("Logged run to MLflow at %s (experiment=%s).", tracking_uri, experiment)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging skipped: %s", exc)


def main() -> None:
    """CLI entry point for ``python mlops/train.py``."""
    parser = argparse.ArgumentParser(description="Train the Iris classifier.")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()
    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
