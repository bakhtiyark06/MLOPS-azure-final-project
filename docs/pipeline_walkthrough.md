# Pipeline Walkthrough

> Author: Bakhtiyar Khan · Date: 2026-06-27

A stage-by-stage tour of the pipeline, mapping each phase to its script and
outputs.

## Phase 1 — Data ingestion (`mlops/ingest_data.py`)
- Loads the Iris dataset via scikit-learn into a clean snake_case dataframe.
- Saves `data/raw/iris.csv` and a frozen `data/reference/reference.csv`.
- Computes a deterministic SHA-256 `dataset_hash`.
- Writes `data/raw/metadata.json` (rows, features, classes, hash, timestamp).
- Uploads artifacts to Azure Blob **only if** storage is configured; otherwise
  skips with a log message.

```bash
python mlops/ingest_data.py            # with optional Blob upload
python mlops/ingest_data.py --no-upload
```

## Phase 2 — Training (`mlops/train.py`)
- Stratified train/test split.
- Trains a `RandomForestClassifier`.
- Logs params, metrics and artifacts to MLflow (`file:./mlruns` by default).
- Saves `models/model.joblib`, `reports/metrics.json`,
  `reports/confusion_matrix.png`.

```bash
python mlops/train.py --n-estimators 200 --max-depth 5
```

## Phase 3 — Quality gate (`mlops/evaluate.py`)
- Reads `reports/metrics.json`, compares accuracy to `ACCURACY_THRESHOLD` (0.90).
- Exit `0` = pass, non-zero = fail → CI/CD stops.
- Demo failure: `python mlops/evaluate.py --demo-fail` or `DEMO_FAIL=true`.

## Phase 4 — Model registration (`mlops/register_model.py`)
- Registers the model in Azure ML with tags: `accuracy`, `dataset_hash`,
  `git_sha`, `created_by`, `project_name`, `version`.
- Writes `reports/model_registration.json`. Skips gracefully without Azure.
- **Rollback**: re-deploy a previous version (see [`demo_script.md`](demo_script.md)).

## Phase 5 — Serving (`app/`)
- `GET /health`, `POST /predict` returning `{prediction, class, confidence}`.

## Phase 6 — Container build (`mlops/build_container.py`)
- Builds the image, tags with git SHA + `latest`, pushes to ACR (if configured).

```bash
python mlops/build_container.py --no-push   # local build only
```

## Phase 7 — Testing (`tests/`)
- pytest suite for API, inference, preprocessing and quality gate, ≥70% coverage.

## Phase 8 — CI/CD (`.github/workflows/`)
- `ci.yml`: deps → tests/coverage → ingest → train → gate → drift → AI report.
- `cd.yml`: Azure login → train → gate → register → build/push → ACI → smoke →
  AKS → smoke.

## Phase 9 — Deployment (`mlops/deploy_aci.py`, `mlops/deploy_aks.py`)
- ACI (staging) and AKS (production), each with environment validation and an
  HTTP smoke test of `/health` and `/predict`.

## Phase 10 — Monitoring & drift (`mlops/drift_detection.py`)
- Generates `reports/drift_report.html`; supports `--simulate`.

## Phase 11 — AI report (`mlops/openrouter_report.py`)
- Summarises metrics, gate, drift and deployment via OpenRouter; falls back to a
  deterministic local report if no API key.

## End-to-end (local, no Azure)

```bash
python mlops/ingest_data.py --no-upload
python mlops/train.py
python mlops/evaluate.py
python mlops/drift_detection.py
python mlops/openrouter_report.py
```
