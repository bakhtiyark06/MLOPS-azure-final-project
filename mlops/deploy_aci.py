"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Phase 9 - staging deployment to Azure Container Instances (ACI).
         Validates the required environment, deploys the container image from
         ACR to ACI and runs an HTTP smoke test against the live endpoint.
         Provides clear, actionable errors when prerequisites are missing.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Allow direct execution (e.g. ``python mlops/deploy_aci.py``) by ensuring the
# repo root is on sys.path so the top-level ``mlops`` package can be imported.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import shutil
import subprocess
import time
from typing import List, Optional

import requests

from mlops.azure_clients import AzureConfig, get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops.deploy.aci")

IMAGE_REPO = "mlops-api"
CONTAINER_PORT = 8000


def _command_exists(name: str) -> bool:
    """Return ``True`` if an executable is discoverable on PATH."""
    return shutil.which(name) is not None


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and log it. Raises on failure when ``check`` is True."""
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(result.stderr.strip())
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
    return result


def validate_environment(config: AzureConfig) -> List[str]:
    """Validate that everything required for an ACI deploy is present.

    Args:
        config: Azure configuration loaded from the environment.

    Returns:
        list[str]: Human-readable list of problems (empty when ready).
    """
    problems: List[str] = []
    if not _command_exists("az"):
        problems.append("Azure CLI ('az') is not installed or not on PATH.")
    for field in ["subscription_id", "resource_group", "acr_name"]:
        if not getattr(config, field):
            problems.append(f"Missing environment variable for '{field}'.")
    return problems


def smoke_test(base_url: str, retries: int = 10, delay: float = 6.0) -> bool:
    """Run an HTTP smoke test against a deployed API.

    Polls ``/health`` until ready then exercises ``/predict`` with a known
    sample, verifying the response shape.

    Args:
        base_url: Base URL of the deployed service (e.g. ``http://host:8000``).
        retries: Number of health-check attempts before giving up.
        delay: Seconds to wait between attempts.

    Returns:
        bool: ``True`` if both health and prediction checks pass.
    """
    health_url = f"{base_url}/health"
    predict_url = f"{base_url}/predict"
    logger.info("Smoke testing %s", base_url)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(health_url, timeout=10)
            if resp.status_code == 200:
                logger.info("Health check OK on attempt %d: %s", attempt, resp.json())
                break
        except requests.RequestException as exc:
            logger.info("Health attempt %d/%d not ready: %s", attempt, retries, exc)
        time.sleep(delay)
    else:
        logger.error("Service did not become healthy in time.")
        return False

    sample = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    try:
        resp = requests.post(predict_url, json=sample, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        assert "prediction" in body and "confidence" in body, "Unexpected response body."
        logger.info("Prediction smoke test OK: %s", body)
        return True
    except (requests.RequestException, AssertionError) as exc:
        logger.error("Prediction smoke test failed: %s", exc)
        return False


def _acr_credentials(acr_name: str) -> Optional[dict]:
    """Fetch ACR admin credentials via the Azure CLI (returns None on failure)."""
    try:
        result = run_command(["az", "acr", "credential", "show", "--name", acr_name])
        return json.loads(result.stdout)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not read ACR credentials (is admin user enabled?): %s", exc)
        return None


def deploy(image_tag: str = "latest", container_name: Optional[str] = None) -> Optional[str]:
    """Deploy the image to ACI and smoke-test it.

    Args:
        image_tag: Image tag to deploy (defaults to ``latest``).
        container_name: Optional override for the ACI container group name.

    Returns:
        str | None: The deployed service base URL on success, else ``None``.
    """
    logger.info("=== Phase 9: Deploy to ACI (staging) ===")
    config = get_config()
    problems = validate_environment(config)
    if problems:
        logger.error("ACI deployment cannot proceed:")
        for problem in problems:
            logger.error("  - %s", problem)
        return None

    import os

    container_name = container_name or os.getenv("AZURE_ACI_NAME", "mlops-api-staging")
    registry = f"{config.acr_name}.azurecr.io"
    image = f"{registry}/{IMAGE_REPO}:{image_tag}"
    dns_label = f"{container_name}-{config.subscription_id[:8]}".lower()

    creds = _acr_credentials(config.acr_name)
    if creds is None:
        return None

    cmd = [
        "az", "container", "create",
        "--resource-group", config.resource_group,
        "--name", container_name,
        "--image", image,
        "--registry-login-server", registry,
        "--registry-username", creds["username"],
        "--registry-password", creds["passwords"][0]["value"],
        "--dns-name-label", dns_label,
        "--ports", str(CONTAINER_PORT),
        "--os-type", "Linux",
        "--cpu", "1",
        "--memory", "1.5",
        "--location", config.region,
        "--restart-policy", "OnFailure",
    ]
    try:
        run_command(cmd)
    except subprocess.CalledProcessError as exc:
        logger.error("ACI deployment failed: %s", exc)
        return None

    fqdn = f"{dns_label}.{config.region}.azurecontainer.io"
    base_url = f"http://{fqdn}:{CONTAINER_PORT}"
    logger.info("ACI deployed at %s", base_url)

    if not smoke_test(base_url):
        logger.error("Staging smoke test failed; investigate before promoting to AKS.")
        return None

    logger.info("ACI staging deployment succeeded and passed smoke tests.")
    return base_url


def main() -> None:
    """CLI entry point for ``python mlops/deploy_aci.py``."""
    parser = argparse.ArgumentParser(description="Deploy the API to Azure Container Instances.")
    parser.add_argument("--tag", default="latest", help="Image tag to deploy.")
    parser.add_argument("--name", default=None, help="Container group name override.")
    args = parser.parse_args()
    url = deploy(image_tag=args.tag, container_name=args.name)
    if url is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
