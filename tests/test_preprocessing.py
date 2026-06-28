"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Tests for the data ingestion / preprocessing stage - dataframe shape,
         column naming, deterministic hashing and metadata generation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlops.ingest_data import (
    FEATURE_COLUMNS,
    SPECIES_COLUMN,
    TARGET_COLUMN,
    compute_dataframe_hash,
    load_iris_dataframe,
    save_datasets,
    write_metadata,
)


def test_load_iris_dataframe_shape_and_columns() -> None:
    """The loaded Iris frame has 150 rows and the expected snake_case columns."""
    frame = load_iris_dataframe()
    assert len(frame) == 150
    for column in FEATURE_COLUMNS + [TARGET_COLUMN, SPECIES_COLUMN]:
        assert column in frame.columns


def test_species_mapping_is_consistent() -> None:
    """Each integer target maps to exactly one species name."""
    frame = load_iris_dataframe()
    grouped = frame.groupby(TARGET_COLUMN)[SPECIES_COLUMN].nunique()
    assert (grouped == 1).all()
    assert set(frame[SPECIES_COLUMN].unique()) == {"setosa", "versicolor", "virginica"}


def test_hash_is_deterministic_and_sensitive() -> None:
    """Hashing the same data is stable; a change alters the hash."""
    frame = load_iris_dataframe()
    h1 = compute_dataframe_hash(frame)
    h2 = compute_dataframe_hash(frame.copy())
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest length

    mutated = frame.copy()
    mutated.loc[0, FEATURE_COLUMNS[0]] = mutated.loc[0, FEATURE_COLUMNS[0]] + 1.0
    assert compute_dataframe_hash(mutated) != h1


def test_save_and_metadata(tmp_path: Path) -> None:
    """Datasets and metadata are written with correct content."""
    frame = load_iris_dataframe()
    raw = tmp_path / "raw.csv"
    ref = tmp_path / "ref.csv"
    save_datasets(frame, raw_csv=raw, reference_csv=ref)
    assert raw.exists() and ref.exists()
    assert len(pd.read_csv(raw)) == 150

    meta_path = tmp_path / "metadata.json"
    metadata = write_metadata(frame, compute_dataframe_hash(frame), metadata_path=meta_path)
    assert meta_path.exists()
    assert metadata["n_rows"] == 150
    assert metadata["n_features"] == len(FEATURE_COLUMNS)
    assert len(metadata["dataset_hash"]) == 64
