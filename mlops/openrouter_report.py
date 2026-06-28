"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Phase 11 - AI-generated pipeline report. Collates the run's metrics,
         quality-gate result, drift summary and deployment record and asks an
         OpenRouter-hosted LLM to produce a concise human-readable summary. If
         no API key is configured (or the call fails) it writes a deterministic
         local fallback report instead of failing the pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops.openrouter")

REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.json"
GATE_PATH = REPORTS_DIR / "quality_gate.json"
DRIFT_PATH = REPORTS_DIR / "drift_summary.json"
REGISTRATION_PATH = REPORTS_DIR / "model_registration.json"
OUTPUT_PATH = REPORTS_DIR / "ai_report.md"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, returning ``{}`` if it is absent or invalid."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def gather_context() -> Dict[str, Any]:
    """Collect all available report artifacts into a single context dict.

    Returns:
        dict: Combined metrics, quality gate, drift and registration data.
    """
    return {
        "metrics": _load_json(METRICS_PATH),
        "quality_gate": _load_json(GATE_PATH),
        "drift": _load_json(DRIFT_PATH),
        "registration": _load_json(REGISTRATION_PATH),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_prompt(context: Dict[str, Any]) -> str:
    """Build the user prompt sent to the LLM from the gathered context."""
    return (
        "You are an MLOps assistant. Summarise the following CI/CD pipeline run "
        "for an engineering audience. Cover: model metrics, the quality gate "
        "outcome, data drift status, and deployment/registration. Be concise, "
        "use markdown headings and bullet points, and end with a clear "
        "recommendation (ship / investigate / rollback).\n\n"
        f"Pipeline data (JSON):\n```json\n{json.dumps(context, indent=2)}\n```"
    )


def call_openrouter(context: Dict[str, Any]) -> Optional[str]:
    """Call the OpenRouter chat completions API.

    Args:
        context: The gathered pipeline context.

    Returns:
        str | None: The model's markdown summary, or ``None`` on any failure /
        missing key (caller then uses the local fallback).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.info("OPENROUTER_API_KEY not set; using local fallback report.")
        return None

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "MLOps Azure Final Project",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise MLOps reporting assistant."},
            {"role": "user", "content": _build_prompt(context)},
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        logger.info("Received AI summary from OpenRouter (model=%s).", model)
        return content
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenRouter call failed (%s); using local fallback.", exc)
        return None


def build_fallback_report(context: Dict[str, Any]) -> str:
    """Build a deterministic markdown report without calling any external API.

    Args:
        context: The gathered pipeline context.

    Returns:
        str: A markdown report summarising the run.
    """
    metrics = context.get("metrics", {})
    gate = context.get("quality_gate", {})
    drift = context.get("drift", {})
    reg = context.get("registration", {})

    accuracy = metrics.get("accuracy", "n/a")
    gate_passed = gate.get("passed")
    drift_detected = drift.get("drift_detected")

    if gate_passed and not drift_detected:
        recommendation = "**SHIP** - quality gate passed and no significant drift detected."
    elif gate_passed and drift_detected:
        recommendation = "**INVESTIGATE** - model passed the gate but data drift was detected."
    else:
        recommendation = "**DO NOT SHIP** - quality gate failed; block deployment."

    return f"""# MLOps Pipeline Report (local fallback)

_Generated at {context.get('generated_at')}_

## Model metrics
- Accuracy: `{accuracy}`
- Precision: `{metrics.get('precision', 'n/a')}`
- Recall: `{metrics.get('recall', 'n/a')}`
- F1: `{metrics.get('f1', 'n/a')}`

## Quality gate
- Passed: `{gate_passed}`
- Threshold: `{gate.get('threshold', 'n/a')}`
- Reason: {gate.get('reason', 'n/a')}

## Data drift
- Drift detected: `{drift_detected}`
- Drifted features: `{drift.get('n_drifted_features', 'n/a')}` / `{drift.get('n_features', 'n/a')}`

## Model registration
- Registered: `{reg.get('registered', False)}`
- Version: `{reg.get('version', 'n/a')}`

## Recommendation
{recommendation}
"""


def generate_report() -> Path:
    """Generate the AI (or fallback) report and write it to ``reports/ai_report.md``.

    Returns:
        Path: The path of the written report.
    """
    logger.info("=== Phase 11: OpenRouter AI report ===")
    context = gather_context()
    summary = call_openrouter(context)
    if summary is None:
        summary = build_fallback_report(context)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(summary, encoding="utf-8")
    logger.info("AI report written -> %s", OUTPUT_PATH)
    return OUTPUT_PATH


def main() -> None:
    """CLI entry point for ``python mlops/openrouter_report.py``."""
    parser = argparse.ArgumentParser(description="Generate an AI pipeline summary report.")
    parser.parse_args()
    generate_report()


if __name__ == "__main__":
    main()
