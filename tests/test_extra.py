"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Additional coverage for graceful-degradation branches - ingestion's
         blob-skip path, deployment early-returns, credential factories, the
         build "no Docker" path and the FastAPI application lifespan.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlops import azure_clients as az
from mlops import build_container as build
from mlops import deploy_aci as aci
from mlops import deploy_aks as aks
from mlops import ingest_data as ingest


# --- ingestion -------------------------------------------------------------
def test_ingest_without_upload_returns_metadata() -> None:
    """ingest(upload=False) returns metadata including a 64-char hash."""
    metadata = ingest.ingest(upload=False)
    assert len(metadata["dataset_hash"]) == 64
    assert metadata["n_rows"] == 150


def test_upload_to_blob_skips_without_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blob upload returns False when storage is not configured."""
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONTAINER", raising=False)
    assert ingest.upload_to_blob([]) is False


# --- deployment early returns ---------------------------------------------
def test_aci_deploy_returns_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACI deploy returns None (and does not raise) when unconfigured."""
    for var in ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ACR_NAME"]:
        monkeypatch.delenv(var, raising=False)
    assert aci.deploy() is None


def test_aks_deploy_returns_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """AKS deploy returns None (and does not raise) when unconfigured."""
    for var in ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ACR_NAME", "AZURE_AKS_CLUSTER"]:
        monkeypatch.delenv(var, raising=False)
    assert aks.deploy() is None


def test_acr_credentials_returns_none_on_failure() -> None:
    """Reading ACR credentials for a bogus registry returns None gracefully."""
    assert aci._acr_credentials("nonexistent-registry-xyz") is None


def test_aci_run_command_success() -> None:
    """The ACI run_command helper succeeds for a trivial command."""
    import sys

    result = aci.run_command([sys.executable, "--version"])
    assert result.returncode == 0


# --- credential factories --------------------------------------------------
def test_get_credential_with_service_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured service principal yields a non-None credential object."""
    monkeypatch.setenv("AZURE_CLIENT_ID", "id")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant")
    assert az.get_credential() is not None


def test_get_blob_client_with_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured storage account + SP yields a Blob client object."""
    monkeypatch.setenv("AZURE_CLIENT_ID", "id")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT", "mystorage")
    assert az.get_blob_service_client() is not None


# --- build: no Docker path -------------------------------------------------
def test_build_image_without_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_image returns False when the Docker CLI is unavailable."""
    monkeypatch.setattr(build, "_command_exists", lambda name: False)
    assert build.build_image(["mlops-api:test"]) is False
    assert build.build_and_push(push=False) is None


def test_push_image_without_az(monkeypatch: pytest.MonkeyPatch) -> None:
    """push_image returns False when ACR is set but the az CLI is missing."""
    monkeypatch.setenv("AZURE_ACR_NAME", "myreg")
    monkeypatch.setattr(build, "_command_exists", lambda name: False)
    assert build.push_image(["myreg.azurecr.io/mlops-api:latest"]) is False


# --- FastAPI lifespan ------------------------------------------------------
def test_app_lifespan_runs_startup() -> None:
    """Entering the TestClient context triggers startup (model load) cleanly."""
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_configure_app_insights_no_connstring(monkeypatch: pytest.MonkeyPatch) -> None:
    """App Insights configuration is a safe no-op without a connection string."""
    from app.main import _configure_app_insights

    monkeypatch.delenv("APPINSIGHTS_CONNECTION_STRING", raising=False)
    _configure_app_insights()  # must not raise
