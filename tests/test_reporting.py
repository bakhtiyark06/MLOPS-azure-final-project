"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Tests for the reporting/registration stages - the OpenRouter report's
         deterministic fallback and the Azure ML registration's graceful skip
         behaviour when no credentials are configured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlops import openrouter_report as orr
from mlops import register_model as reg


def test_load_json_missing_returns_empty(tmp_path: Path) -> None:
    """Loading a non-existent JSON file returns an empty dict."""
    assert orr._load_json(tmp_path / "nope.json") == {}


def test_gather_context_has_sections() -> None:
    """The gathered context exposes all expected sections."""
    context = orr.gather_context()
    for key in ["metrics", "quality_gate", "drift", "registration", "generated_at"]:
        assert key in context


def test_fallback_report_ship_recommendation() -> None:
    """A passing gate with no drift yields a SHIP recommendation."""
    context = {
        "metrics": {"accuracy": 0.97},
        "quality_gate": {"passed": True, "threshold": 0.9},
        "drift": {"drift_detected": False},
        "registration": {},
        "generated_at": "now",
    }
    report = orr.build_fallback_report(context)
    assert "SHIP" in report
    assert "0.97" in report


def test_fallback_report_block_recommendation() -> None:
    """A failed gate yields a do-not-ship recommendation."""
    context = {
        "metrics": {"accuracy": 0.5},
        "quality_gate": {"passed": False},
        "drift": {"drift_detected": False},
        "registration": {},
        "generated_at": "now",
    }
    report = orr.build_fallback_report(context)
    assert "DO NOT SHIP" in report


def test_generate_report_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no API key, a markdown fallback report is written."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path = orr.generate_report()
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() != ""


def test_call_openrouter_returns_none_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """call_openrouter short-circuits to None when the key is absent."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert orr.call_openrouter({"metrics": {}}) is None


def test_build_tags_contains_required_keys() -> None:
    """Registration tags include all required governance keys."""
    tags = reg.build_tags({"accuracy": 0.97, "dataset_hash": "abc"})
    for key in ["accuracy", "dataset_hash", "git_sha", "created_by", "project_name"]:
        assert key in tags
    assert tags["accuracy"] == "0.97"


def test_register_model_skips_without_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration returns None and writes a record when Azure is unconfigured."""
    for var in ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME"]:
        monkeypatch.delenv(var, raising=False)
    version = reg.register_model()
    assert version is None
    assert reg.REGISTRATION_RECORD.exists()
    record = json.loads(reg.REGISTRATION_RECORD.read_text(encoding="utf-8"))
    assert record["registered"] is False


def test_register_model_missing_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Registration returns None when the model artifact is missing."""
    monkeypatch.setattr(reg, "MODEL_PATH", tmp_path / "missing.joblib")
    assert reg.register_model() is None


def test_register_model_client_none_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When configured but the MLClient cannot be built, registration is skipped."""
    monkeypatch.setattr(reg, "azure_is_configured", lambda *a, **k: True)
    monkeypatch.setattr(reg, "get_ml_client", lambda *a, **k: None)
    assert reg.register_model() is None


def test_git_sha_returns_value() -> None:
    """_git_sha returns a non-empty string (real SHA or 'unknown')."""
    assert isinstance(reg._git_sha(), str)
    assert reg._git_sha() != ""


class _FakeResponse:
    """Minimal stand-in for a ``requests`` response object."""

    def raise_for_status(self) -> None:  # noqa: D401 - test helper
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "# AI summary\nLooks good."}}]}


def test_call_openrouter_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mocked successful API call returns the model's content."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(orr.requests, "post", lambda *a, **k: _FakeResponse())
    content = orr.call_openrouter({"metrics": {"accuracy": 0.97}})
    assert content is not None
    assert "AI summary" in content


def test_call_openrouter_handles_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing API call is swallowed and returns None (fallback is used)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def _boom(*args, **kwargs):
        raise orr.requests.RequestException("network down")

    monkeypatch.setattr(orr.requests, "post", _boom)
    assert orr.call_openrouter({"metrics": {}}) is None
