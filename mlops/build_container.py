"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Phase 6 - container build & publish. Builds the FastAPI Docker image,
         tags it with the git SHA and ``latest`` and pushes it to Azure
         Container Registry (ACR). Each step degrades gracefully: if Docker or
         the Azure CLI is unavailable the script reports exactly what is missing
         instead of crashing the pipeline.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Allow direct execution (e.g. ``python mlops/build_container.py``) by ensuring the
# repo root is on sys.path so the top-level ``mlops`` package can be imported.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

from mlops.azure_clients import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops.build")

IMAGE_REPO = "mlops-api"


def _command_exists(name: str) -> bool:
    """Return ``True`` if an executable is discoverable on PATH."""
    return shutil.which(name) is not None


def run_command(cmd: List[str], check: bool = True) -> Tuple[int, str]:
    """Run a shell command, streaming a friendly log line first.

    Args:
        cmd: Command and arguments.
        check: When ``True`` a non-zero exit raises ``CalledProcessError``.

    Returns:
        tuple[int, str]: Return code and combined stdout/stderr text.
    """
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    if output.strip():
        logger.info(output.strip())
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output)
    return result.returncode, output


def _git_sha_short() -> str:
    """Return a short git SHA for image tagging (or ``local``)."""
    sha = os.getenv("GITHUB_SHA") or os.getenv("GIT_SHA")
    if sha:
        return sha[:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "local"


def resolve_image_names() -> Tuple[str, List[str]]:
    """Compute the local image name and the full set of tags to apply.

    Returns:
        tuple[str, list[str]]: The primary local tag and all tags. When ACR is
        configured the tags are fully-qualified ``<registry>/<repo>:<tag>``.
    """
    config = get_config()
    sha = _git_sha_short()
    if config.acr_name:
        registry = f"{config.acr_name}.azurecr.io"
        tags = [f"{registry}/{IMAGE_REPO}:{sha}", f"{registry}/{IMAGE_REPO}:latest"]
        return tags[0], tags
    # No ACR configured: build local-only tags.
    tags = [f"{IMAGE_REPO}:{sha}", f"{IMAGE_REPO}:latest"]
    return tags[0], tags


def build_image(tags: List[str], context: str = ".") -> bool:
    """Build the Docker image and apply every requested tag.

    Args:
        tags: All tags to apply (first is used for ``docker build -t``).
        context: Build context directory.

    Returns:
        bool: ``True`` on success, ``False`` if Docker is unavailable/failed.
    """
    if not _command_exists("docker"):
        logger.error("Docker CLI not found. Install Docker Desktop and ensure it is running.")
        return False
    try:
        run_command(["docker", "build", "-t", tags[0], context])
        for extra in tags[1:]:
            run_command(["docker", "tag", tags[0], extra])
        logger.info("Built image with tags: %s", ", ".join(tags))
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Docker build failed: %s", exc)
        return False


def push_image(tags: List[str]) -> bool:
    """Authenticate to ACR and push all tags.

    Args:
        tags: Fully-qualified ACR tags to push.

    Returns:
        bool: ``True`` if pushed, ``False`` if skipped or failed.
    """
    config = get_config()
    if not config.acr_name:
        logger.info("AZURE_ACR_NAME not set; skipping push (built locally only).")
        return False
    if not _command_exists("az"):
        logger.warning("Azure CLI ('az') not found; cannot log in to ACR. Skipping push.")
        return False
    try:
        run_command(["az", "acr", "login", "--name", config.acr_name])
        for tag in tags:
            run_command(["docker", "push", tag])
        logger.info("Pushed image tags to ACR '%s'.", config.acr_name)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("ACR push failed: %s", exc)
        return False


def build_and_push(push: bool = True) -> Optional[str]:
    """Build the image and optionally push it to ACR.

    Args:
        push: Whether to attempt pushing to ACR after a successful build.

    Returns:
        str | None: The primary image tag on success, else ``None``.
    """
    logger.info("=== Phase 6: Container build ===")
    primary, tags = resolve_image_names()
    if not build_image(tags):
        return None
    if push:
        push_image(tags)
    return primary


def main() -> None:
    """CLI entry point for ``python mlops/build_container.py``."""
    parser = argparse.ArgumentParser(description="Build and push the API container image.")
    parser.add_argument("--no-push", action="store_true", help="Build locally without pushing to ACR.")
    args = parser.parse_args()
    image = build_and_push(push=not args.no_push)
    if image is None:
        raise SystemExit(1)
    logger.info("Primary image: %s", image)


if __name__ == "__main__":
    main()
