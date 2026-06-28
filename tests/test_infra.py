"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Tests for the infrastructure-facing modules - Azure client factories,
         container image naming, deployment environment validation, manifest
         rendering and the HTTP smoke-test helper. These run fully offline.
"""

from __future__ import annotations

import pytest

from mlops import azure_clients as az
from mlops import build_container as build
from mlops import deploy_aci as aci
from mlops import deploy_aks as aks


# --- azure_clients ---------------------------------------------------------
def test_config_from_env_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env reads variables and missing() reports absent fields."""
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-123")
    monkeypatch.delenv("AZURE_RESOURCE_GROUP", raising=False)
    config = az.AzureConfig.from_env()
    assert config.subscription_id == "sub-123"
    assert "resource_group" in config.missing(["subscription_id", "resource_group"])


def test_azure_is_configured_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """azure_is_configured is False when core variables are missing."""
    for var in ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME"]:
        monkeypatch.delenv(var, raising=False)
    assert az.azure_is_configured() is False


def test_ml_client_none_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLClient construction is skipped (None) without core config."""
    for var in ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME"]:
        monkeypatch.delenv(var, raising=False)
    assert az.get_ml_client() is None


def test_blob_client_none_without_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blob client construction is skipped (None) without a storage account."""
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT", raising=False)
    assert az.get_blob_service_client() is None


# --- build_container -------------------------------------------------------
def test_resolve_image_names_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ACR, image tags are local (repo:tag) names."""
    monkeypatch.delenv("AZURE_ACR_NAME", raising=False)
    primary, tags = build.resolve_image_names()
    assert primary.startswith("mlops-api:")
    assert any(t.endswith(":latest") for t in tags)


def test_resolve_image_names_with_acr(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ACR set, tags are fully-qualified registry references."""
    monkeypatch.setenv("AZURE_ACR_NAME", "myreg")
    primary, tags = build.resolve_image_names()
    assert primary.startswith("myreg.azurecr.io/mlops-api:")


def test_push_skipped_without_acr(monkeypatch: pytest.MonkeyPatch) -> None:
    """push_image returns False when no ACR is configured."""
    monkeypatch.delenv("AZURE_ACR_NAME", raising=False)
    assert build.push_image(["mlops-api:latest"]) is False


def test_run_command_success() -> None:
    """run_command returns a zero exit code for a trivial command."""
    import sys

    code, _ = build.run_command([sys.executable, "--version"])
    assert code == 0


# --- deploy_aci ------------------------------------------------------------
def test_aci_validate_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACI validation lists problems when config is missing."""
    for var in ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ACR_NAME"]:
        monkeypatch.delenv(var, raising=False)
    problems = aci.validate_environment(az.AzureConfig.from_env())
    assert len(problems) > 0


def test_smoke_test_fails_fast_on_unreachable() -> None:
    """The smoke test returns False quickly for an unreachable host."""
    assert aci.smoke_test("http://127.0.0.1:1", retries=1, delay=0) is False


# --- deploy_aks ------------------------------------------------------------
def test_aks_validate_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """AKS validation lists problems when config is missing."""
    for var in ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ACR_NAME", "AZURE_AKS_CLUSTER"]:
        monkeypatch.delenv(var, raising=False)
    problems = aks.validate_environment(az.AzureConfig.from_env())
    assert len(problems) > 0


def test_render_manifest_contains_resources() -> None:
    """The rendered manifest references the image and a LoadBalancer service."""
    manifest = aks.render_manifest("myreg.azurecr.io/mlops-api:abc123", replicas=3)
    assert "myreg.azurecr.io/mlops-api:abc123" in manifest
    assert "LoadBalancer" in manifest
    assert "replicas: 3" in manifest
