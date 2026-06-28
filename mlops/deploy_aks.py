"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Phase 9 - production deployment to Azure Kubernetes Service (AKS).
         Fetches AKS credentials, applies a Deployment + LoadBalancer Service
         manifest, waits for the external IP and runs an HTTP smoke test.
         Reuses the smoke-test helper from the ACI module and degrades
         gracefully when prerequisites are missing.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Allow direct execution (e.g. ``python mlops/deploy_aks.py``) by ensuring the
# repo root is on sys.path so the top-level ``mlops`` package can be imported.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from mlops.azure_clients import AzureConfig, get_config
from mlops.deploy_aci import run_command, smoke_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops.deploy.aks")

IMAGE_REPO = "mlops-api"
APP_NAME = "mlops-api"
CONTAINER_PORT = 8000
SERVICE_PORT = 80


def _command_exists(name: str) -> bool:
    """Return ``True`` if an executable is discoverable on PATH."""
    return shutil.which(name) is not None


def validate_environment(config: AzureConfig) -> List[str]:
    """Validate prerequisites for an AKS deployment.

    Args:
        config: Azure configuration loaded from the environment.

    Returns:
        list[str]: Human-readable problems (empty when ready).
    """
    problems: List[str] = []
    if not _command_exists("az"):
        problems.append("Azure CLI ('az') is not installed or not on PATH.")
    if not _command_exists("kubectl"):
        problems.append("'kubectl' is not installed or not on PATH.")
    for field in ["subscription_id", "resource_group", "acr_name", "aks_cluster"]:
        if not getattr(config, field):
            problems.append(f"Missing environment variable for '{field}'.")
    return problems


def render_manifest(image: str, replicas: int = 2) -> str:
    """Render a Kubernetes Deployment + LoadBalancer Service manifest.

    Args:
        image: Fully-qualified container image reference.
        replicas: Number of pod replicas for high availability.

    Returns:
        str: The YAML manifest as a string.
    """
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {APP_NAME}
  labels:
    app: {APP_NAME}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {APP_NAME}
  template:
    metadata:
      labels:
        app: {APP_NAME}
    spec:
      containers:
        - name: {APP_NAME}
          image: {image}
          ports:
            - containerPort: {CONTAINER_PORT}
          readinessProbe:
            httpGet:
              path: /health
              port: {CONTAINER_PORT}
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: {CONTAINER_PORT}
            initialDelaySeconds: 20
            periodSeconds: 20
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: {APP_NAME}
spec:
  type: LoadBalancer
  selector:
    app: {APP_NAME}
  ports:
    - port: {SERVICE_PORT}
      targetPort: {CONTAINER_PORT}
"""


def _wait_for_external_ip(retries: int = 30, delay: float = 10.0) -> Optional[str]:
    """Poll the Service until its external LoadBalancer IP is assigned."""
    for attempt in range(1, retries + 1):
        result = run_command(
            [
                "kubectl", "get", "service", APP_NAME,
                "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}",
            ],
            check=False,
        )
        ip = result.stdout.strip()
        if ip:
            logger.info("External IP assigned: %s", ip)
            return ip
        logger.info("Waiting for external IP (%d/%d)...", attempt, retries)
        time.sleep(delay)
    logger.error("External IP was not assigned in time.")
    return None


def deploy(image_tag: str = "latest", replicas: int = 2) -> Optional[str]:
    """Deploy the image to AKS and smoke-test the production endpoint.

    Args:
        image_tag: Image tag to deploy.
        replicas: Number of replicas.

    Returns:
        str | None: The production base URL on success, else ``None``.
    """
    logger.info("=== Phase 9: Deploy to AKS (production) ===")
    config = get_config()
    problems = validate_environment(config)
    if problems:
        logger.error("AKS deployment cannot proceed:")
        for problem in problems:
            logger.error("  - %s", problem)
        return None

    registry = f"{config.acr_name}.azurecr.io"
    image = f"{registry}/{IMAGE_REPO}:{image_tag}"

    try:
        # Fetch kubeconfig for the target cluster (overwrite to stay idempotent).
        run_command(
            [
                "az", "aks", "get-credentials",
                "--resource-group", config.resource_group,
                "--name", config.aks_cluster,
                "--overwrite-existing",
            ]
        )
        # Ensure AKS can pull from ACR (no-op if already attached).
        run_command(
            ["az", "aks", "update", "--resource-group", config.resource_group,
             "--name", config.aks_cluster, "--attach-acr", config.acr_name],
            check=False,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to configure AKS access: %s", exc)
        return None

    manifest = render_manifest(image, replicas=replicas)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(manifest)
        manifest_path = handle.name
    logger.info("Rendered manifest -> %s", manifest_path)

    try:
        run_command(["kubectl", "apply", "-f", manifest_path])
        run_command(
            ["kubectl", "rollout", "status", f"deployment/{APP_NAME}", "--timeout=180s"],
            check=False,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("kubectl apply failed: %s", exc)
        return None
    finally:
        Path(manifest_path).unlink(missing_ok=True)

    external_ip = _wait_for_external_ip()
    if not external_ip:
        return None

    base_url = f"http://{external_ip}:{SERVICE_PORT}"
    logger.info("AKS service reachable at %s", base_url)

    if not smoke_test(base_url):
        logger.error("Production smoke test failed.")
        return None

    logger.info("AKS production deployment succeeded and passed smoke tests.")
    return base_url


def main() -> None:
    """CLI entry point for ``python mlops/deploy_aks.py``."""
    parser = argparse.ArgumentParser(description="Deploy the API to Azure Kubernetes Service.")
    parser.add_argument("--tag", default="latest", help="Image tag to deploy.")
    parser.add_argument("--replicas", type=int, default=2, help="Number of pod replicas.")
    args = parser.parse_args()
    url = deploy(image_tag=args.tag, replicas=args.replicas)
    if url is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
