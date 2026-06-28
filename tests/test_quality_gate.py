"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Tests for the quality gate - pass/fail logic against the threshold and
         the demo-failure override.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlops.evaluate import (
    evaluate_gate,
    load_metrics,
    run_quality_gate,
    write_result,
)


def test_gate_passes_above_threshold() -> None:
    """Accuracy above the threshold passes the gate."""
    result = evaluate_gate({"accuracy": 0.97}, threshold=0.90)
    assert result["passed"] is True
    assert "1" not in str(result["reason"]) or "0.97" in str(result["reason"])


def test_gate_fails_below_threshold() -> None:
    """Accuracy below the threshold fails the gate."""
    result = evaluate_gate({"accuracy": 0.80}, threshold=0.90)
    assert result["passed"] is False


def test_gate_demo_fail_forces_failure() -> None:
    """Demo mode forces a failure even with perfect accuracy."""
    result = evaluate_gate({"accuracy": 1.0}, threshold=0.90, demo_fail=True)
    assert result["passed"] is False
    assert "demo" in str(result["reason"]).lower()


def test_gate_handles_missing_accuracy() -> None:
    """Missing accuracy is treated as 0.0 and fails the gate."""
    result = evaluate_gate({}, threshold=0.90)
    assert result["passed"] is False


def test_write_and_load_result(tmp_path: Path) -> None:
    """Gate results round-trip through JSON correctly."""
    out = tmp_path / "gate.json"
    result = evaluate_gate({"accuracy": 0.95}, threshold=0.90)
    write_result(result, path=out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["passed"] is True


def test_run_quality_gate_returns_zero_on_real_metrics() -> None:
    """End-to-end: with a trained model the real metrics pass the gate (exit 0).

    Relies on the session-scoped ``trained_model`` fixture having produced
    ``reports/metrics.json``.
    """
    metrics = load_metrics()
    assert "accuracy" in metrics
    assert run_quality_gate(threshold=0.90, demo_fail=False) == 0


def test_run_quality_gate_demo_fail_returns_nonzero() -> None:
    """Demo-fail mode returns a non-zero exit code (blocks deployment)."""
    assert run_quality_gate(threshold=0.90, demo_fail=True) == 1


def test_run_quality_gate_threshold_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An impossible threshold from the environment fails the gate."""
    monkeypatch.setenv("ACCURACY_THRESHOLD", "1.01")
    monkeypatch.delenv("DEMO_FAIL", raising=False)
    assert run_quality_gate() == 1


def test_load_metrics_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """load_metrics raises FileNotFoundError when metrics are absent."""
    with pytest.raises(FileNotFoundError):
        load_metrics(path=tmp_path / "missing.json")
