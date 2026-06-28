"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Tests for drift detection - PSI computation, drift injection, the
         compute/summary logic and the end-to-end report generation (normal and
         simulated-drift modes).
"""

from __future__ import annotations

import numpy as np

from mlops import drift_detection as drift
from mlops.ingest_data import FEATURE_COLUMNS


def test_psi_identical_is_zero() -> None:
    """Identical distributions have ~zero PSI."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=1000)
    assert drift.population_stability_index(data, data) < 1e-6


def test_psi_shifted_is_positive() -> None:
    """A shifted distribution produces a clearly positive PSI."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(3, 1, 1000)
    assert drift.population_stability_index(ref, cur) > drift.PSI_DRIFT_THRESHOLD


def test_inject_drift_shifts_means() -> None:
    """Injected drift increases feature means relative to the original."""
    reference, current = drift.load_datasets()
    drifted = drift.inject_drift(current)
    for column in FEATURE_COLUMNS:
        assert drifted[column].mean() > current[column].mean()


def test_compute_drift_no_drift() -> None:
    """Comparing a dataset to itself reports no drift."""
    reference, current = drift.load_datasets()
    summary = drift.compute_drift(reference, reference)
    assert summary["drift_detected"] is False
    assert summary["n_drifted_features"] == 0


def test_compute_drift_with_injected_drift() -> None:
    """Comparing against drifted data reports drift on every feature."""
    reference, current = drift.load_datasets()
    drifted = drift.inject_drift(current)
    summary = drift.compute_drift(reference, drifted)
    assert summary["drift_detected"] is True
    assert summary["n_drifted_features"] == summary["n_features"]


def test_render_html_contains_verdict() -> None:
    """The built-in HTML renderer embeds the verdict text."""
    reference, _ = drift.load_datasets()
    summary = drift.compute_drift(reference, reference)
    html = drift._render_html(summary)
    assert "Data Drift Report" in html
    assert "No significant drift" in html


def test_detect_drift_end_to_end() -> None:
    """detect_drift writes both the HTML report and JSON summary."""
    summary = drift.detect_drift(simulate=False)
    assert drift.DRIFT_REPORT_HTML.exists()
    assert drift.DRIFT_SUMMARY_JSON.exists()
    assert "features" in summary

    simulated = drift.detect_drift(simulate=True)
    assert simulated["drift_detected"] is True
