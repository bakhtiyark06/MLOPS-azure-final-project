"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Phase 10 - data drift detection. Compares a "current" dataset against
         the frozen reference dataset and produces an HTML drift report plus a
         JSON summary. Uses Evidently AI when available for a rich report and
         always falls back to a self-contained PSI-based report so the stage
         never fails. Supports a simulated-drift demo mode.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Allow direct execution (e.g. ``python mlops/drift_detection.py``) by ensuring the
# repo root is on sys.path so the top-level ``mlops`` package can be imported.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from mlops.ingest_data import FEATURE_COLUMNS, RAW_CSV, REFERENCE_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops.drift")

DRIFT_REPORT_HTML = Path("reports/drift_report.html")
DRIFT_SUMMARY_JSON = Path("reports/drift_summary.json")

# Population Stability Index thresholds (industry convention):
#   PSI < 0.1  -> no significant change
#   0.1-0.25   -> moderate shift
#   > 0.25     -> significant drift
PSI_DRIFT_THRESHOLD = 0.25


def _env_flag(name: str) -> bool:
    """Return ``True`` when an env flag is set to a truthy value."""
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def load_datasets(
    reference_path: Path = REFERENCE_CSV,
    current_path: Path = RAW_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the reference and current datasets.

    Args:
        reference_path: Path to the frozen reference dataset.
        current_path: Path to the current/production dataset.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: ``(reference, current)``.

    Raises:
        FileNotFoundError: If either dataset is missing.
    """
    if not reference_path.exists() or not current_path.exists():
        raise FileNotFoundError(
            "Reference or current dataset missing. Run 'python mlops/ingest_data.py' first."
        )
    return pd.read_csv(reference_path), pd.read_csv(current_path)


def inject_drift(frame: pd.DataFrame, magnitude: float = 1.5) -> pd.DataFrame:
    """Inject synthetic drift into a copy of the dataset (demo mode).

    Shifts and scales the feature columns so the distribution clearly diverges
    from the reference, which makes the drift report demonstrably "drifted".

    Args:
        frame: Source dataframe.
        magnitude: Strength of the injected shift (in std-devs).

    Returns:
        pandas.DataFrame: A drifted copy of the dataset.
    """
    drifted = frame.copy()
    rng = np.random.default_rng(42)
    for column in FEATURE_COLUMNS:
        std = float(drifted[column].std()) or 1.0
        drifted[column] = drifted[column] + magnitude * std + rng.normal(0, std * 0.3, len(drifted))
    logger.info("Injected synthetic drift (magnitude=%.2f std).", magnitude)
    return drifted


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Compute the Population Stability Index between two numeric arrays.

    Args:
        reference: Reference distribution values.
        current: Current distribution values.
        bins: Number of quantile bins.

    Returns:
        float: The PSI value (0 = identical distributions).
    """
    # Build bin edges from the reference quantiles so bins are well populated.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:  # degenerate (constant) feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Convert to proportions, flooring zeros to avoid div-by-zero / log(0).
    ref_prop = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_prop = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> Dict[str, object]:
    """Compute per-feature drift statistics and an overall verdict.

    Args:
        reference: Reference dataframe.
        current: Current dataframe.

    Returns:
        dict: Summary with per-feature PSI/means and an aggregate ``drift_detected``.
    """
    features: List[Dict[str, object]] = []
    drifted_count = 0
    for column in FEATURE_COLUMNS:
        psi = population_stability_index(
            reference[column].to_numpy(), current[column].to_numpy()
        )
        is_drift = psi > PSI_DRIFT_THRESHOLD
        drifted_count += int(is_drift)
        features.append(
            {
                "feature": column,
                "psi": round(psi, 4),
                "reference_mean": round(float(reference[column].mean()), 4),
                "current_mean": round(float(current[column].mean()), 4),
                "drift": is_drift,
            }
        )

    summary: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_features": len(FEATURE_COLUMNS),
        "n_drifted_features": drifted_count,
        "share_drifted": round(drifted_count / len(FEATURE_COLUMNS), 4),
        "psi_threshold": PSI_DRIFT_THRESHOLD,
        "drift_detected": drifted_count > 0,
        "features": features,
    }
    return summary


def _render_html(summary: Dict[str, object]) -> str:
    """Render a self-contained HTML drift report from the summary dict."""
    rows = "".join(
        f"<tr class='{'drift' if f['drift'] else 'ok'}'>"
        f"<td>{f['feature']}</td><td>{f['psi']}</td>"
        f"<td>{f['reference_mean']}</td><td>{f['current_mean']}</td>"
        f"<td>{'DRIFT' if f['drift'] else 'stable'}</td></tr>"
        for f in summary["features"]  # type: ignore[index]
    )
    verdict = "DRIFT DETECTED" if summary["drift_detected"] else "No significant drift"
    verdict_color = "#c0392b" if summary["drift_detected"] else "#27ae60"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Data Drift Report</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #222; }}
    h1 {{ margin-bottom: 0.2rem; }}
    .verdict {{ font-size: 1.3rem; font-weight: 700; color: {verdict_color}; }}
    table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; max-width: 760px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #f4f6f8; }}
    tr.drift td {{ background: #fdecea; }}
    tr.ok td {{ background: #eafaf1; }}
    .meta {{ color: #666; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Data Drift Report</h1>
  <p class="meta">Generated at {summary['generated_at']} &middot; PSI threshold {summary['psi_threshold']}</p>
  <p class="verdict">{verdict}</p>
  <p>{summary['n_drifted_features']} of {summary['n_features']} features drifted
     ({float(summary['share_drifted']) * 100:.0f}%).</p>
  <table>
    <thead>
      <tr><th>Feature</th><th>PSI</th><th>Reference mean</th><th>Current mean</th><th>Status</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""


def _try_evidently(reference: pd.DataFrame, current: pd.DataFrame) -> bool:
    """Attempt to generate an Evidently report; return ``True`` on success.

    Evidently's public API has shifted across releases (the modern 0.7+ API in
    ``evidently`` / ``evidently.presets`` differs from the legacy
    ``evidently.report`` API), so both are attempted. This is treated as a
    best-effort enhancement over the always-produced built-in report.
    """
    cols = FEATURE_COLUMNS
    ref, cur = reference[cols], current[cols]

    # --- Modern API (Evidently >= 0.7) ------------------------------------
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        result = report.run(reference_data=ref, current_data=cur)
        result.save_html(str(DRIFT_REPORT_HTML))
        logger.info("Evidently (0.7+) drift report written -> %s", DRIFT_REPORT_HTML)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.info("Modern Evidently API unavailable (%s); trying legacy API.", exc)

    # --- Legacy API (Evidently 0.4.x) -------------------------------------
    try:
        from evidently.metric_preset import DataDriftPreset as LegacyPreset
        from evidently.report import Report as LegacyReport

        report = LegacyReport(metrics=[LegacyPreset()])
        report.run(reference_data=ref, current_data=cur)
        report.save_html(str(DRIFT_REPORT_HTML))
        logger.info("Evidently (legacy) drift report written -> %s", DRIFT_REPORT_HTML)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.info("Evidently unavailable or failed (%s); using built-in report.", exc)
        return False


def detect_drift(simulate: bool | None = None) -> Dict[str, object]:
    """Run drift detection end-to-end and write the HTML + JSON outputs.

    Args:
        simulate: Force simulated drift on/off; defaults to env ``SIMULATE_DRIFT``.

    Returns:
        dict: The drift summary.
    """
    logger.info("=== Phase 10: Drift detection ===")
    if simulate is None:
        simulate = _env_flag("SIMULATE_DRIFT")

    reference, current = load_datasets()
    if simulate:
        current = inject_drift(current)

    summary = compute_drift(reference, current)

    DRIFT_REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    # Prefer Evidently's report; fall back to the built-in renderer.
    if not _try_evidently(reference, current):
        DRIFT_REPORT_HTML.write_text(_render_html(summary), encoding="utf-8")
        logger.info("Built-in drift report written -> %s", DRIFT_REPORT_HTML)

    DRIFT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Drift summary written -> %s", DRIFT_SUMMARY_JSON)
    logger.info(
        "Drift detection complete: drift_detected=%s (%d/%d features).",
        summary["drift_detected"],
        summary["n_drifted_features"],
        summary["n_features"],
    )
    return summary


def main() -> None:
    """CLI entry point for ``python mlops/drift_detection.py``."""
    parser = argparse.ArgumentParser(description="Data drift detection.")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Inject synthetic drift to demonstrate a drifted report.",
    )
    args = parser.parse_args()
    simulate = True if args.simulate else None
    detect_drift(simulate=simulate)


if __name__ == "__main__":
    main()
