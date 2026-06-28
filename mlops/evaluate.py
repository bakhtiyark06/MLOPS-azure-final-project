"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Phase 3 - quality gate. Reads the metrics produced by training and
         compares accuracy against a configurable threshold. Exits 0 when the
         gate passes and a non-zero code when it fails, so GitHub Actions halts
         deployment automatically. Supports a demo mode that forces a failure.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Allow direct execution (e.g. ``python mlops/evaluate.py``) by ensuring the repo
# root is on sys.path so the top-level ``mlops`` package can be imported.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops.evaluate")

METRICS_PATH = Path("reports/metrics.json")
GATE_RESULT_PATH = Path("reports/quality_gate.json")
DEFAULT_THRESHOLD = 0.90


def load_metrics(path: Path = METRICS_PATH) -> Dict[str, float]:
    """Load the metrics JSON produced by the training stage.

    Args:
        path: Path to ``metrics.json``.

    Returns:
        dict[str, float]: Parsed metrics.

    Raises:
        FileNotFoundError: If metrics have not been generated yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at '{path}'. Run 'python mlops/train.py' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _env_flag(name: str) -> bool:
    """Return ``True`` when an environment flag is set to a truthy value."""
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def evaluate_gate(
    metrics: Dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
    demo_fail: bool = False,
) -> Dict[str, object]:
    """Evaluate the quality gate against the accuracy threshold.

    Args:
        metrics: Metrics dict containing at least ``accuracy``.
        threshold: Minimum accuracy required to pass.
        demo_fail: When ``True``, force the gate to fail regardless of metrics.

    Returns:
        dict: Structured gate result with ``passed`` flag and details.
    """
    accuracy = float(metrics.get("accuracy", 0.0))
    passed = (accuracy >= threshold) and not demo_fail

    result: Dict[str, object] = {
        "passed": passed,
        "accuracy": accuracy,
        "threshold": threshold,
        "demo_fail": demo_fail,
        "reason": (
            "demo failure forced via DEMO_FAIL"
            if demo_fail
            else (
                f"accuracy {accuracy:.4f} >= threshold {threshold:.2f}"
                if passed
                else f"accuracy {accuracy:.4f} < threshold {threshold:.2f}"
            )
        ),
    }
    return result


def write_result(result: Dict[str, object], path: Path = GATE_RESULT_PATH) -> None:
    """Persist the quality-gate result to ``reports/quality_gate.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote quality gate result -> %s", path)


def run_quality_gate(threshold: float | None = None, demo_fail: bool | None = None) -> int:
    """Run the full quality gate and return a process exit code.

    Args:
        threshold: Optional explicit threshold (defaults to env / 0.90).
        demo_fail: Optional explicit demo flag (defaults to env DEMO_FAIL).

    Returns:
        int: ``0`` if the gate passed, ``1`` otherwise.
    """
    logger.info("=== Phase 3: Quality gate ===")
    if threshold is None:
        threshold = float(os.getenv("ACCURACY_THRESHOLD", str(DEFAULT_THRESHOLD)))
    if demo_fail is None:
        demo_fail = _env_flag("DEMO_FAIL")

    metrics = load_metrics()
    result = evaluate_gate(metrics, threshold=threshold, demo_fail=demo_fail)
    write_result(result)

    if result["passed"]:
        logger.info("QUALITY GATE PASSED: %s", result["reason"])
        return 0
    logger.error("QUALITY GATE FAILED: %s", result["reason"])
    return 1


def main() -> None:
    """CLI entry point. Exits with the gate's return code."""
    parser = argparse.ArgumentParser(description="Model quality gate.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum accuracy required (default: env ACCURACY_THRESHOLD or 0.90).",
    )
    parser.add_argument(
        "--demo-fail",
        action="store_true",
        help="Force the gate to fail (demonstration mode).",
    )
    args = parser.parse_args()
    demo_fail = True if args.demo_fail else None
    sys.exit(run_quality_gate(threshold=args.threshold, demo_fail=demo_fail))


if __name__ == "__main__":
    main()
