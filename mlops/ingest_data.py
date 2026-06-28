"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Phase 1 - data ingestion. Loads the Iris dataset, persists a raw copy
         and a reference copy (used later for drift detection), computes a
         deterministic dataset hash, writes a metadata JSON sidecar and, when
         Azure Blob Storage is configured, uploads the artifacts. Azure upload
         is skipped gracefully when credentials are absent.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Allow direct execution (e.g. ``python mlops/ingest_data.py``) by ensuring the
# repo root is on sys.path so the top-level ``mlops`` package can be imported.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.datasets import load_iris

from mlops.azure_clients import get_blob_service_client, get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops.ingest")

# Standardised, snake_case feature column names used across the whole pipeline.
FEATURE_COLUMNS: List[str] = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "target"
SPECIES_COLUMN = "species"

# Default output locations (relative to the repo root).
RAW_DIR = Path("data/raw")
REFERENCE_DIR = Path("data/reference")
RAW_CSV = RAW_DIR / "iris.csv"
REFERENCE_CSV = REFERENCE_DIR / "reference.csv"
METADATA_JSON = RAW_DIR / "metadata.json"


def load_iris_dataframe() -> pd.DataFrame:
    """Load the Iris dataset into a tidy, snake_case :class:`pandas.DataFrame`.

    Returns:
        pandas.DataFrame: Columns are the four features plus ``target`` (int)
        and ``species`` (str).
    """
    bunch = load_iris(as_frame=True)
    frame = bunch.frame.copy()

    # Rename sklearn's "sepal length (cm)" style columns to clean snake_case.
    rename_map = dict(zip(bunch.feature_names, FEATURE_COLUMNS))
    frame = frame.rename(columns=rename_map)
    frame = frame.rename(columns={"target": TARGET_COLUMN})

    # Add a readable species column for downstream reporting / drift analysis.
    target_names = list(bunch.target_names)
    frame[SPECIES_COLUMN] = frame[TARGET_COLUMN].map(lambda i: target_names[int(i)])
    return frame


def compute_dataframe_hash(frame: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash of a dataframe's contents.

    The hash is computed over a canonical CSV serialisation so it is stable
    across runs and machines (used as the ``dataset_hash`` model tag).

    Args:
        frame: The dataframe to hash.

    Returns:
        str: Hex-encoded SHA-256 digest.
    """
    canonical = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_datasets(
    frame: pd.DataFrame,
    raw_csv: Path = RAW_CSV,
    reference_csv: Path = REFERENCE_CSV,
) -> Tuple[Path, Path]:
    """Persist the raw dataset and a reference copy to disk.

    Args:
        frame: The dataset to save.
        raw_csv: Destination for the raw dataset.
        reference_csv: Destination for the reference dataset.

    Returns:
        tuple[Path, Path]: Paths to the written raw and reference files.
    """
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    reference_csv.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(raw_csv, index=False)
    # The reference dataset is a frozen snapshot representing the training
    # distribution; drift detection later compares production data against it.
    frame.to_csv(reference_csv, index=False)

    logger.info("Saved raw dataset       -> %s (%d rows)", raw_csv, len(frame))
    logger.info("Saved reference dataset -> %s (%d rows)", reference_csv, len(frame))
    return raw_csv, reference_csv


def write_metadata(
    frame: pd.DataFrame,
    dataset_hash: str,
    metadata_path: Path = METADATA_JSON,
) -> Dict[str, object]:
    """Write a metadata sidecar describing the ingested dataset.

    Args:
        frame: The ingested dataframe.
        dataset_hash: SHA-256 hash from :func:`compute_dataframe_hash`.
        metadata_path: Destination for the metadata JSON.

    Returns:
        dict: The metadata dictionary that was written.
    """
    metadata: Dict[str, object] = {
        "dataset_name": "iris",
        "source": "sklearn.datasets.load_iris",
        "n_rows": int(len(frame)),
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "classes": sorted(frame[SPECIES_COLUMN].unique().tolist()),
        "dataset_hash": dataset_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote dataset metadata  -> %s", metadata_path)
    return metadata


def upload_to_blob(paths: List[Path]) -> bool:
    """Upload the given files to the configured Azure Blob container.

    Skips gracefully (returning ``False``) when storage is not configured or the
    SDK/credentials are unavailable.

    Args:
        paths: Local files to upload.

    Returns:
        bool: ``True`` if all uploads succeeded, ``False`` if skipped/failed.
    """
    config = get_config()
    if not config.storage_account or not config.storage_container:
        logger.info("Azure Blob Storage not configured; skipping upload.")
        return False

    client = get_blob_service_client(config)
    if client is None:
        logger.info("Blob client unavailable; skipping upload.")
        return False

    try:
        container = client.get_container_client(config.storage_container)
        # Create the container on first use if it does not already exist.
        try:
            container.create_container()
        except Exception:  # noqa: BLE001 - already exists / no permission to create
            pass

        for path in paths:
            blob_name = f"raw/{path.name}"
            with path.open("rb") as handle:
                container.upload_blob(name=blob_name, data=handle, overwrite=True)
            logger.info("Uploaded %s -> blob '%s'", path, blob_name)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Blob upload failed (continuing): %s", exc)
        return False


def ingest(upload: bool = True) -> Dict[str, object]:
    """Run the full ingestion stage end-to-end.

    Args:
        upload: Whether to attempt an Azure Blob upload after saving locally.

    Returns:
        dict: The dataset metadata (including the hash).
    """
    logger.info("=== Phase 1: Data ingestion ===")
    frame = load_iris_dataframe()
    dataset_hash = compute_dataframe_hash(frame)
    raw_csv, reference_csv = save_datasets(frame)
    metadata = write_metadata(frame, dataset_hash)

    if upload:
        upload_to_blob([raw_csv, reference_csv, METADATA_JSON])

    logger.info("Ingestion complete. dataset_hash=%s", dataset_hash)
    return metadata


def main() -> None:
    """CLI entry point for ``python mlops/ingest_data.py``."""
    parser = argparse.ArgumentParser(description="Iris data ingestion stage.")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Do not attempt to upload artifacts to Azure Blob Storage.",
    )
    args = parser.parse_args()
    ingest(upload=not args.no_upload)


if __name__ == "__main__":
    main()
