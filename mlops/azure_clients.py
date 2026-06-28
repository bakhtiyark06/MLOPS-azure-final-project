"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Centralised Azure client factories and configuration helpers. All
         Azure SDK imports are lazy and every factory degrades gracefully when
         credentials or optional packages are missing, so the rest of the
         pipeline can run unchanged on a developer laptop without Azure access.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

# Load variables from a local .env if python-dotenv is available. This is a
# convenience only - in CI the values come from the real environment.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

logger = logging.getLogger("mlops.azure")


# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------
@dataclass
class AzureConfig:
    """Strongly-typed view over the Azure-related environment variables."""

    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    subscription_id: Optional[str] = None
    resource_group: Optional[str] = None
    workspace_name: Optional[str] = None
    storage_account: Optional[str] = None
    storage_container: Optional[str] = None
    acr_name: Optional[str] = None
    aks_cluster: Optional[str] = None
    region: str = "eastus"

    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Build an :class:`AzureConfig` from the current environment.

        Returns:
            AzureConfig: Populated configuration (fields may be ``None``).
        """
        return cls(
            client_id=os.getenv("AZURE_CLIENT_ID"),
            client_secret=os.getenv("AZURE_CLIENT_SECRET"),
            tenant_id=os.getenv("AZURE_TENANT_ID"),
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZURE_WORKSPACE_NAME"),
            storage_account=os.getenv("AZURE_STORAGE_ACCOUNT"),
            storage_container=os.getenv("AZURE_STORAGE_CONTAINER"),
            acr_name=os.getenv("AZURE_ACR_NAME"),
            aks_cluster=os.getenv("AZURE_AKS_CLUSTER"),
            region=os.getenv("AZURE_REGION", "eastus"),
        )

    def missing(self, required: List[str]) -> List[str]:
        """Return the subset of ``required`` field names that are empty.

        Args:
            required: Attribute names that must be set for an operation.

        Returns:
            list[str]: Names of the missing fields (empty if all present).
        """
        return [name for name in required if not getattr(self, name, None)]


def get_config() -> AzureConfig:
    """Convenience wrapper returning :meth:`AzureConfig.from_env`."""
    return AzureConfig.from_env()


# ---------------------------------------------------------------------------
# Credential + client factories (all lazy / graceful)
# ---------------------------------------------------------------------------
def get_credential(config: Optional[AzureConfig] = None) -> Optional[Any]:
    """Create an Azure credential object.

    Prefers an explicit service principal (``ClientSecretCredential``) when the
    three SP variables are present, otherwise falls back to
    ``DefaultAzureCredential`` (az login / managed identity). Returns ``None``
    if the ``azure-identity`` package is not installed.

    Args:
        config: Optional pre-built config; loaded from env when omitted.

    Returns:
        An Azure credential instance, or ``None`` when unavailable.
    """
    config = config or get_config()
    try:
        from azure.identity import ClientSecretCredential, DefaultAzureCredential
    except ImportError:
        logger.warning("azure-identity not installed; cannot build credential.")
        return None

    if config.client_id and config.client_secret and config.tenant_id:
        logger.info("Using ClientSecretCredential (service principal).")
        return ClientSecretCredential(
            tenant_id=config.tenant_id,
            client_id=config.client_id,
            client_secret=config.client_secret,
        )

    logger.info("Service principal not fully configured; trying DefaultAzureCredential.")
    try:
        return DefaultAzureCredential()
    except Exception as exc:  # noqa: BLE001
        logger.warning("DefaultAzureCredential unavailable: %s", exc)
        return None


def get_ml_client(config: Optional[AzureConfig] = None) -> Optional[Any]:
    """Create an Azure Machine Learning ``MLClient`` if possible.

    Args:
        config: Optional pre-built config; loaded from env when omitted.

    Returns:
        ``azure.ai.ml.MLClient`` or ``None`` if prerequisites are missing.
    """
    config = config or get_config()
    required = ["subscription_id", "resource_group", "workspace_name"]
    missing = config.missing(required)
    if missing:
        logger.warning("Cannot build MLClient; missing config: %s", ", ".join(missing))
        return None

    credential = get_credential(config)
    if credential is None:
        return None

    try:
        from azure.ai.ml import MLClient
    except ImportError:
        logger.warning("azure-ai-ml not installed; cannot build MLClient.")
        return None

    try:
        return MLClient(
            credential=credential,
            subscription_id=config.subscription_id,
            resource_group_name=config.resource_group,
            workspace_name=config.workspace_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to create MLClient: %s", exc)
        return None


def get_blob_service_client(config: Optional[AzureConfig] = None) -> Optional[Any]:
    """Create a ``BlobServiceClient`` for the configured storage account.

    Args:
        config: Optional pre-built config; loaded from env when omitted.

    Returns:
        ``azure.storage.blob.BlobServiceClient`` or ``None`` if unavailable.
    """
    config = config or get_config()
    if not config.storage_account:
        logger.warning("AZURE_STORAGE_ACCOUNT not set; skipping Blob client.")
        return None

    credential = get_credential(config)
    if credential is None:
        return None

    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.warning("azure-storage-blob not installed; cannot build Blob client.")
        return None

    account_url = f"https://{config.storage_account}.blob.core.windows.net"
    try:
        return BlobServiceClient(account_url=account_url, credential=credential)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to create BlobServiceClient: %s", exc)
        return None


def azure_is_configured(config: Optional[AzureConfig] = None) -> bool:
    """Return ``True`` when the core Azure ML variables are all present.

    Args:
        config: Optional pre-built config; loaded from env when omitted.

    Returns:
        bool: Whether Azure ML operations can be attempted.
    """
    config = config or get_config()
    return not config.missing(["subscription_id", "resource_group", "workspace_name"])
