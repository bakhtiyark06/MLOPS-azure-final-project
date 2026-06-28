"""
Author:  Bakhtiyar Khan
Date:    2026-06-28
Purpose: Server-side HTML/CSS/JS builders for the FastAPI MLOps dashboard.

         All presentation lives here (kept out of ``app/main.py``). Everything
         is plain HTML/CSS/JS returned by FastAPI - no React/Next.js, no build
         step, no external CDNs. Three pages share one dark glassmorphism theme:
           * build_dashboard()  -> GET /            (status + live prediction)
           * build_demo()       -> GET /demo        (14-step tour + architecture)
           * build_flow()       -> GET /demo/flow   (24-line flow explorer)

         Interactive data is embedded as JSON in <script type="application/json">
         tags and read by vanilla JS, which keeps the JS brace-safe and lets the
         pages render meaningful content server-side as well.
"""

from __future__ import annotations

import json
from typing import Dict, List

# ===========================================================================
# DATA MODELS
# ===========================================================================

# --- 14-step interactive demo tour -----------------------------------------
DEMO_STEPS: List[Dict[str, object]] = [
    {
        "n": 1, "title": "Repository", "cat": "cicd",
        "desc": "All code and configuration live in GitHub. A push to main is the single trigger that drives the entire MLOps pipeline.",
        "tags": ["git", "github", "source-of-truth"],
        "bullets": [
            "Mono-repo: app, mlops pipeline, tests, docs, workflows.",
            "No secrets committed; everything via env vars / GitHub Secrets.",
            "Branch protection + CI status checks gate merges.",
        ],
        "command": "git push origin main",
    },
    {
        "n": 2, "title": "Data ingest", "cat": "data",
        "desc": "Load the Iris dataset, hash it for reproducibility, write raw + reference snapshots and a metadata sidecar, then optionally upload to Azure Blob.",
        "tags": ["pandas", "sklearn", "hashing"],
        "bullets": [
            "Deterministic SHA-256 dataset hash for lineage.",
            "Reference snapshot frozen for later drift comparison.",
            "Blob upload skips gracefully without credentials.",
        ],
        "command": "python mlops/ingest_data.py --no-upload",
    },
    {
        "n": 3, "title": "Schema check", "cat": "data",
        "desc": "Validate column names, types and ranges before training so a malformed dataset fails fast instead of producing a silently broken model.",
        "tags": ["validation", "data-quality"],
        "bullets": [
            "Four numeric features in a fixed canonical order.",
            "Non-negative measurements enforced at the API too.",
            "Metadata records row count, classes and hash.",
        ],
        "command": "python -c \"import pandas as pd; print(pd.read_csv('data/raw/iris.csv').describe())\"",
    },
    {
        "n": 4, "title": "Training", "cat": "data",
        "desc": "Stratified train/test split, fit a RandomForest, and log parameters, metrics and artifacts to MLflow for full experiment tracking.",
        "tags": ["scikit-learn", "mlflow", "random-forest"],
        "bullets": [
            "Reproducible split with a fixed random seed.",
            "Params + metrics + model logged to MLflow.",
            "Confusion matrix and metrics.json saved to reports/.",
        ],
        "command": "python mlops/train.py",
    },
    {
        "n": 5, "title": "Quality gate", "cat": "quality",
        "desc": "Compare accuracy against the configured threshold. A failure exits non-zero so CI/CD halts before anything is deployed.",
        "tags": ["governance", "ci-gate"],
        "bullets": [
            "Threshold defaults to accuracy >= 0.90.",
            "Exit 0 = pass, exit 1 = block deployment.",
            "Demo-fail mode proves the gate stops bad models.",
        ],
        "command": "python mlops/evaluate.py",
    },
    {
        "n": 6, "title": "Registry", "cat": "quality",
        "desc": "Register the approved model in the Azure ML registry with governance tags so every deployed model is traceable and rollback is one command.",
        "tags": ["azure-ml", "governance", "rollback"],
        "bullets": [
            "Tags: accuracy, dataset_hash, git_sha, created_by, version.",
            "Versioned models enable instant rollback.",
            "Skips gracefully when Azure is not configured.",
        ],
        "command": "python mlops/register_model.py",
    },
    {
        "n": 7, "title": "GitHub CI", "cat": "cicd",
        "desc": "On every push the CI workflow installs deps, runs tests with coverage, trains, gates, runs drift detection and generates the AI report.",
        "tags": ["github-actions", "pytest", "coverage"],
        "bullets": [
            "Coverage gate enforced at >= 70%.",
            "Artifacts (reports, model) uploaded for review.",
            "Fast feedback on every commit and PR.",
        ],
        "command": "gh workflow run ci.yml",
    },
    {
        "n": 8, "title": "GitHub CD", "cat": "cicd",
        "desc": "On merge to main the CD workflow logs into Azure, re-runs the gate, registers the model, builds/pushes the image and deploys through staging to production.",
        "tags": ["github-actions", "azure-login", "promotion"],
        "bullets": [
            "Azure service-principal login from GitHub Secrets.",
            "Gate failure stops the job before any deploy.",
            "Staging smoke test must pass before production.",
        ],
        "command": "gh workflow run cd.yml",
    },
    {
        "n": 9, "title": "Docker", "cat": "cicd",
        "desc": "Package the FastAPI service into a lean Python 3.11-slim image, tag it with the git SHA and latest, and push it to Azure Container Registry.",
        "tags": ["docker", "acr", "packaging"],
        "bullets": [
            "Non-root container with a built-in healthcheck.",
            "Lean serving deps keep the image small.",
            "Immutable SHA tags enable precise rollbacks.",
        ],
        "command": "python mlops/build_container.py",
    },
    {
        "n": 10, "title": "ACI staging", "cat": "azure",
        "desc": "Deploy the image to Azure Container Instances as a fast, cheap pre-production environment and smoke-test the live endpoint.",
        "tags": ["azure-aci", "staging", "smoke-test"],
        "bullets": [
            "HTTP smoke test hits /health and /predict.",
            "Validates the image before production exposure.",
            "Clear errors when prerequisites are missing.",
        ],
        "command": "python mlops/deploy_aci.py --tag latest",
    },
    {
        "n": 11, "title": "AKS production", "cat": "azure",
        "desc": "Promote to Azure Kubernetes Service with a Deployment + LoadBalancer, readiness/liveness probes and replicas, then smoke-test production.",
        "tags": ["azure-aks", "kubernetes", "production"],
        "bullets": [
            "Multiple replicas for high availability.",
            "Readiness/liveness probes hit /health.",
            "External IP smoke-tested before sign-off.",
        ],
        "command": "python mlops/deploy_aks.py --tag latest",
    },
    {
        "n": 12, "title": "FastAPI + UI", "cat": "serve",
        "desc": "The service exposes /health and /predict plus this dark dashboard, the demo tour and the flow explorer - all served directly by FastAPI.",
        "tags": ["fastapi", "rest", "dashboard"],
        "bullets": [
            "Pydantic-validated request and response models.",
            "Browsers get HTML; clients get JSON from /.",
            "Interactive prediction wired to POST /predict.",
        ],
        "command": "uvicorn app.main:app --reload",
    },
    {
        "n": 13, "title": "Drift report", "cat": "serve",
        "desc": "Compare live data against the frozen reference with Evidently (PSI fallback) and publish an HTML drift report for monitoring.",
        "tags": ["evidently", "drift", "monitoring"],
        "bullets": [
            "Per-feature PSI with a clear drift verdict.",
            "Simulated-drift mode for demonstrations.",
            "Report viewable at /reports/drift.",
        ],
        "command": "python mlops/drift_detection.py --simulate",
    },
    {
        "n": 14, "title": "OpenRouter", "cat": "serve",
        "desc": "Summarise the whole run - metrics, gate, drift and deployment - via an OpenRouter LLM, with a deterministic local fallback when no key is set.",
        "tags": ["openrouter", "llm", "reporting"],
        "bullets": [
            "Concise ship / investigate / rollback recommendation.",
            "Never fails the pipeline if the API key is absent.",
            "Report viewable at /reports/openrouter.",
        ],
        "command": "python mlops/openrouter_report.py",
    },
]

# --- 24-line flow explorer -------------------------------------------------
FLOW_LINES: List[Dict[str, object]] = [
    {"n": 1, "title": "Clone & virtual environment", "what": "Clone the repo and create an isolated Python 3.11+ virtual environment.",
     "files": ["README.md"], "inp": "GitHub repository", "out": ".venv interpreter",
     "env": [], "azure": "None", "command": "python -m venv .venv", "nxt": "Install dependencies"},
    {"n": 2, "title": "Install dependencies", "what": "Install runtime + dev dependencies (tests, linting, coverage).",
     "files": ["requirements-dev.txt"], "inp": "requirements files", "out": "installed packages",
     "env": [], "azure": "None", "command": "pip install -r requirements-dev.txt", "nxt": "Load Iris dataset"},
    {"n": 3, "title": "Load Iris dataset", "what": "Load Iris into a tidy snake_case DataFrame with a readable species column.",
     "files": ["mlops/ingest_data.py"], "inp": "sklearn.datasets.load_iris", "out": "in-memory DataFrame",
     "env": [], "azure": "None", "command": "python mlops/ingest_data.py --no-upload", "nxt": "Compute dataset hash"},
    {"n": 4, "title": "Compute dataset hash", "what": "Hash a canonical CSV serialisation with SHA-256 for reproducible lineage.",
     "files": ["mlops/ingest_data.py"], "inp": "DataFrame", "out": "dataset_hash (hex)",
     "env": [], "azure": "None", "command": "python mlops/ingest_data.py --no-upload", "nxt": "Save raw dataset"},
    {"n": 5, "title": "Save raw dataset", "what": "Persist the raw dataset CSV used for training.",
     "files": ["mlops/ingest_data.py"], "inp": "DataFrame", "out": "data/raw/iris.csv",
     "env": [], "azure": "None", "command": "python mlops/ingest_data.py --no-upload", "nxt": "Save reference dataset"},
    {"n": 6, "title": "Save reference dataset", "what": "Freeze a reference snapshot representing the training distribution for drift checks.",
     "files": ["mlops/ingest_data.py"], "inp": "DataFrame", "out": "data/reference/reference.csv",
     "env": [], "azure": "None", "command": "python mlops/ingest_data.py --no-upload", "nxt": "Write metadata JSON"},
    {"n": 7, "title": "Write metadata JSON", "what": "Record rows, features, classes, hash and timestamp as a sidecar.",
     "files": ["mlops/ingest_data.py"], "inp": "DataFrame + hash", "out": "data/raw/metadata.json",
     "env": [], "azure": "None", "command": "python mlops/ingest_data.py --no-upload", "nxt": "Upload to Blob"},
    {"n": 8, "title": "Upload to Azure Blob", "what": "Upload datasets to Blob Storage when configured; skip cleanly otherwise.",
     "files": ["mlops/ingest_data.py", "mlops/azure_clients.py"], "inp": "local CSVs", "out": "blobs in container",
     "env": ["AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_CONTAINER"], "azure": "Azure Blob Storage",
     "command": "python mlops/ingest_data.py", "nxt": "Train/test split"},
    {"n": 9, "title": "Train/test split", "what": "Stratified split keeping class balance across train and test sets.",
     "files": ["mlops/train.py"], "inp": "data/raw/iris.csv", "out": "X_train/X_test/y_train/y_test",
     "env": [], "azure": "None", "command": "python mlops/train.py", "nxt": "Fit RandomForest"},
    {"n": 10, "title": "Fit RandomForest", "what": "Train a RandomForestClassifier on NumPy arrays (warning-free serving).",
     "files": ["mlops/train.py"], "inp": "training arrays", "out": "fitted estimator",
     "env": [], "azure": "None", "command": "python mlops/train.py --max-depth 3", "nxt": "Compute metrics"},
    {"n": 11, "title": "Compute metrics", "what": "Accuracy, macro precision/recall/F1 on the held-out test set.",
     "files": ["mlops/train.py"], "inp": "y_test, predictions", "out": "metrics dict",
     "env": [], "azure": "None", "command": "python mlops/train.py", "nxt": "Save model.joblib"},
    {"n": 12, "title": "Save model.joblib", "what": "Serialise the trained model for serving and registration.",
     "files": ["mlops/train.py"], "inp": "fitted estimator", "out": "models/model.joblib",
     "env": ["MODEL_PATH"], "azure": "None", "command": "python mlops/train.py", "nxt": "Save confusion matrix"},
    {"n": 13, "title": "Save confusion matrix", "what": "Render and save a confusion-matrix figure for review.",
     "files": ["mlops/train.py"], "inp": "y_test, predictions", "out": "reports/confusion_matrix.png",
     "env": [], "azure": "None", "command": "python mlops/train.py", "nxt": "Log to MLflow"},
    {"n": 14, "title": "Log to MLflow", "what": "Track params, metrics and artifacts as an MLflow run.",
     "files": ["mlops/train.py"], "inp": "params, metrics, artifacts", "out": "mlruns/ experiment run",
     "env": ["MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME"], "azure": "MLflow (file or Azure ML)",
     "command": "mlflow ui --backend-store-uri ./mlruns", "nxt": "Quality gate"},
    {"n": 15, "title": "Quality gate", "what": "Compare accuracy to the threshold; exit non-zero on failure to halt CI/CD.",
     "files": ["mlops/evaluate.py"], "inp": "reports/metrics.json", "out": "reports/quality_gate.json + exit code",
     "env": ["ACCURACY_THRESHOLD", "DEMO_FAIL"], "azure": "None", "command": "python mlops/evaluate.py", "nxt": "Build governance tags"},
    {"n": 16, "title": "Build governance tags", "what": "Assemble accuracy, dataset_hash, git_sha, created_by, project_name, version.",
     "files": ["mlops/register_model.py"], "inp": "metrics + git", "out": "tag dictionary",
     "env": ["GIT_SHA", "PROJECT_NAME", "CREATED_BY"], "azure": "None", "command": "python mlops/register_model.py", "nxt": "Register model"},
    {"n": 17, "title": "Register model in Azure ML", "what": "Create or update a versioned registry entry with governance tags.",
     "files": ["mlops/register_model.py", "mlops/azure_clients.py"], "inp": "models/model.joblib + tags", "out": "registered model version",
     "env": ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME"], "azure": "Azure Machine Learning",
     "command": "python mlops/register_model.py", "nxt": "Build Docker image"},
    {"n": 18, "title": "Build Docker image", "what": "Build the lean FastAPI serving image and tag with git SHA + latest.",
     "files": ["Dockerfile", "mlops/build_container.py"], "inp": "app/ + model", "out": "local Docker image",
     "env": ["AZURE_ACR_NAME"], "azure": "None (build)", "command": "python mlops/build_container.py --no-push", "nxt": "Push to ACR"},
    {"n": 19, "title": "Push image to ACR", "what": "Authenticate to ACR and push all image tags.",
     "files": ["mlops/build_container.py"], "inp": "local image", "out": "image in ACR",
     "env": ["AZURE_ACR_NAME"], "azure": "Azure Container Registry", "command": "python mlops/build_container.py", "nxt": "Deploy ACI staging"},
    {"n": 20, "title": "Deploy ACI staging", "what": "Deploy to Azure Container Instances and smoke-test the endpoint.",
     "files": ["mlops/deploy_aci.py"], "inp": "ACR image", "out": "running ACI + smoke result",
     "env": ["AZURE_RESOURCE_GROUP", "AZURE_ACR_NAME", "AZURE_ACI_NAME"], "azure": "Azure Container Instances",
     "command": "python mlops/deploy_aci.py --tag latest", "nxt": "Deploy AKS production"},
    {"n": 21, "title": "Deploy AKS production", "what": "Apply Deployment + LoadBalancer to AKS, await external IP, smoke-test.",
     "files": ["mlops/deploy_aks.py"], "inp": "ACR image", "out": "live AKS service",
     "env": ["AZURE_AKS_CLUSTER", "AZURE_RESOURCE_GROUP"], "azure": "Azure Kubernetes Service",
     "command": "python mlops/deploy_aks.py --tag latest", "nxt": "Serve predictions"},
    {"n": 22, "title": "Serve predictions", "what": "FastAPI serves /predict and the dashboard; logs ship to App Insights when set.",
     "files": ["app/main.py", "app/inference.py"], "inp": "feature payload", "out": "prediction + probabilities",
     "env": ["APPINSIGHTS_CONNECTION_STRING", "MODEL_PATH"], "azure": "Application Insights",
     "command": "curl -X POST localhost:8000/predict -d '{...}'", "nxt": "Drift detection"},
    {"n": 23, "title": "Drift detection", "what": "Compare current vs reference data; publish an HTML drift report + summary.",
     "files": ["mlops/drift_detection.py"], "inp": "reference + current data", "out": "reports/drift_report.html",
     "env": ["SIMULATE_DRIFT"], "azure": "Azure Monitor (optional)", "command": "python mlops/drift_detection.py --simulate", "nxt": "OpenRouter report"},
    {"n": 24, "title": "OpenRouter AI report", "what": "Summarise the run via an LLM with a deterministic local fallback.",
     "files": ["mlops/openrouter_report.py"], "inp": "reports/*.json", "out": "reports/ai_report.md",
     "env": ["OPENROUTER_API_KEY", "OPENROUTER_MODEL"], "azure": "OpenRouter (external)",
     "command": "python mlops/openrouter_report.py", "nxt": "Pipeline complete"},
]

# --- Architecture board: nodes, lanes and edges ----------------------------
# kind drives colour: data=blue, deploy=green, obs=purple, auth=amber.
ARCH_NODES: List[Dict[str, object]] = [
    # ML Pipeline lane
    {"id": "dev", "label": "Developer", "sub": "git push", "x": 20, "y": 96, "w": 120, "h": 56, "kind": "auth"},
    {"id": "gha", "label": "GitHub Actions", "sub": "CI / CD", "x": 175, "y": 96, "w": 130, "h": 56, "kind": "auth"},
    {"id": "ingest", "label": "01 Ingest", "sub": "ingest_data.py", "x": 345, "y": 96, "w": 120, "h": 56, "kind": "data"},
    {"id": "train", "label": "02 Train", "sub": "train.py", "x": 490, "y": 96, "w": 110, "h": 56, "kind": "data"},
    {"id": "gate", "label": "03 Gate", "sub": "evaluate.py", "x": 625, "y": 96, "w": 110, "h": 56, "kind": "data"},
    {"id": "registry", "label": "04 Registry", "sub": "register_model.py", "x": 760, "y": 96, "w": 130, "h": 56, "kind": "data"},
    {"id": "docker", "label": "05 Docker", "sub": "build_container.py", "x": 915, "y": 96, "w": 130, "h": 56, "kind": "deploy"},
    # Azure lane
    {"id": "blob", "label": "Blob Storage", "sub": "datasets", "x": 345, "y": 286, "w": 120, "h": 56, "kind": "data"},
    {"id": "aml", "label": "Azure ML", "sub": "model registry", "x": 760, "y": 286, "w": 130, "h": 56, "kind": "data"},
    {"id": "acr", "label": "ACR", "sub": "image registry", "x": 915, "y": 286, "w": 130, "h": 56, "kind": "deploy"},
    {"id": "aci", "label": "ACI Staging", "sub": "pre-prod", "x": 1080, "y": 250, "w": 130, "h": 52, "kind": "deploy"},
    {"id": "aks", "label": "AKS Production", "sub": "scaled serving", "x": 1080, "y": 320, "w": 130, "h": 52, "kind": "deploy"},
    # Serve lane
    {"id": "fastapi", "label": "FastAPI", "sub": "/health /predict", "x": 915, "y": 470, "w": 130, "h": 56, "kind": "deploy"},
    {"id": "dash", "label": "Dashboard", "sub": "/", "x": 345, "y": 470, "w": 120, "h": 56, "kind": "deploy"},
    {"id": "demo", "label": "Demo", "sub": "/demo", "x": 505, "y": 470, "w": 120, "h": 56, "kind": "deploy"},
    {"id": "flow", "label": "Flow Explorer", "sub": "/demo/flow", "x": 660, "y": 470, "w": 140, "h": 56, "kind": "deploy"},
    # Observability lane
    {"id": "evidently", "label": "Evidently Drift", "sub": "reports/drift", "x": 345, "y": 640, "w": 140, "h": 56, "kind": "obs"},
    {"id": "openrouter", "label": "OpenRouter LLM", "sub": "ai_report.md", "x": 515, "y": 640, "w": 150, "h": 56, "kind": "obs"},
    {"id": "appinsights", "label": "App Insights", "sub": "telemetry", "x": 760, "y": 640, "w": 130, "h": 56, "kind": "obs"},
    {"id": "teams", "label": "Teams Webhook", "sub": "alerts", "x": 915, "y": 640, "w": 140, "h": 56, "kind": "obs"},
]

ARCH_EDGES: List[Dict[str, str]] = [
    {"a": "dev", "b": "gha", "kind": "auth"},
    {"a": "gha", "b": "ingest", "kind": "auth"},
    {"a": "ingest", "b": "train", "kind": "data"},
    {"a": "train", "b": "gate", "kind": "data"},
    {"a": "gate", "b": "registry", "kind": "data"},
    {"a": "registry", "b": "docker", "kind": "data"},
    {"a": "ingest", "b": "blob", "kind": "data"},
    {"a": "registry", "b": "aml", "kind": "data"},
    {"a": "docker", "b": "acr", "kind": "deploy"},
    {"a": "acr", "b": "aci", "kind": "deploy"},
    {"a": "aci", "b": "aks", "kind": "deploy"},
    {"a": "aks", "b": "fastapi", "kind": "deploy"},
    {"a": "fastapi", "b": "dash", "kind": "deploy"},
    {"a": "fastapi", "b": "demo", "kind": "deploy"},
    {"a": "fastapi", "b": "flow", "kind": "deploy"},
    {"a": "fastapi", "b": "appinsights", "kind": "obs"},
    {"a": "fastapi", "b": "evidently", "kind": "obs"},
    {"a": "evidently", "b": "openrouter", "kind": "obs"},
    {"a": "appinsights", "b": "teams", "kind": "obs"},
    {"a": "openrouter", "b": "teams", "kind": "obs"},
]

ARCH_LANES: List[Dict[str, object]] = [
    {"label": "ML Pipeline", "y": 60, "h": 130},
    {"label": "Azure", "y": 220, "h": 170},
    {"label": "Serve", "y": 430, "h": 130},
    {"label": "Observability", "y": 600, "h": 130},
]

ARCH_W = 1240
ARCH_H = 760

_KIND_COLOR = {
    "data": "#4f8cff",
    "deploy": "#34d399",
    "obs": "#a368ff",
    "auth": "#fbbf24",
}


# ===========================================================================
# SHARED THEME + SHELL
# ===========================================================================
_BASE_CSS = """
:root{
  --bg0:#05070f; --bg1:#0b1226; --bg2:#0e1730;
  --glass:rgba(20,30,56,.55); --glass2:rgba(28,40,72,.5);
  --text:#e8eefc; --muted:#94a6c9; --line:rgba(86,113,168,.32);
  --blue:#4f8cff; --purple:#a368ff; --green:#34d399; --amber:#fbbf24; --red:#f87171; --cyan:#38e1ff;
}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{
  margin:0; min-height:100vh; color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:
    radial-gradient(1200px 700px at 10% -12%, rgba(46,86,170,.30) 0%, transparent 55%),
    radial-gradient(1000px 620px at 100% -6%, rgba(120,60,210,.26) 0%, transparent 52%),
    radial-gradient(900px 600px at 50% 120%, rgba(40,120,180,.18) 0%, transparent 60%),
    linear-gradient(180deg,var(--bg1) 0%, var(--bg0) 100%);
  background-attachment:fixed; line-height:1.55; letter-spacing:.1px;
}
a{color:var(--blue); text-decoration:none;}
a:hover{text-decoration:underline;}
.nav{
  position:sticky; top:0; z-index:30; backdrop-filter:blur(14px);
  background:rgba(5,7,15,.72); border-bottom:1px solid var(--line);
  display:flex; align-items:center; gap:16px; padding:12px 22px;
}
.nav .brand{font-weight:800; letter-spacing:-.01em;}
.nav .brand .dot{color:var(--purple);}
.nav .links{display:flex; gap:6px; margin-left:auto; flex-wrap:wrap;}
.nav .links a{color:var(--muted); padding:7px 13px; border-radius:10px; font-size:.9rem; border:1px solid transparent;}
.nav .links a:hover{color:var(--text); text-decoration:none; background:var(--glass2);}
.nav .links a.active{color:#fff; border-color:var(--line); background:linear-gradient(90deg,rgba(79,140,255,.22),rgba(163,104,255,.22));}
.wrap{max-width:1180px; margin:0 auto; padding:30px 22px 80px;}
.hero h1{font-size:2.15rem; margin:0 0 6px; letter-spacing:-.02em;
  background:linear-gradient(90deg,#fff 0%, #c9d8ff 60%, #c4b5fd 100%); -webkit-background-clip:text; background-clip:text; color:transparent;}
.hero p{color:var(--muted); margin:0 0 16px; max-width:760px;}
.badges{display:flex; flex-wrap:wrap; gap:8px; margin-bottom:24px;}
.badge{font-size:.78rem; padding:5px 12px; border-radius:999px; background:var(--glass2); border:1px solid var(--line); color:var(--text);}
.badge.green{border-color:rgba(52,211,153,.5); background:rgba(6,40,27,.6); color:#6ee7b7; box-shadow:0 0 0 1px rgba(52,211,153,.15) inset;}
.badge.blue{border-color:rgba(79,140,255,.5); background:rgba(10,33,66,.6); color:#93c5fd;}
.badge.purple{border-color:rgba(163,104,255,.5); background:rgba(28,17,58,.6); color:#c4b5fd;}
.badge.amber{border-color:rgba(251,191,36,.45); background:rgba(44,33,7,.6); color:#fcd34d;}
.badge.red{border-color:rgba(248,113,113,.5); background:rgba(44,13,13,.6); color:#fca5a5;}
.badge .live{width:7px;height:7px;border-radius:50%;background:var(--green);display:inline-block;margin-right:6px;box-shadow:0 0 8px var(--green);vertical-align:middle;}
h2.section{font-size:.95rem; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); margin:34px 0 14px;}
.grid{display:grid; gap:14px;}
.g6{grid-template-columns:repeat(auto-fit,minmax(155px,1fr));}
.g4{grid-template-columns:repeat(auto-fit,minmax(200px,1fr));}
.g3{grid-template-columns:repeat(auto-fit,minmax(240px,1fr));}
.g2{grid-template-columns:repeat(auto-fit,minmax(320px,1fr));}
.card{
  background:var(--glass); backdrop-filter:blur(10px);
  border:1px solid var(--line); border-radius:16px; padding:16px 18px;
  box-shadow:0 10px 30px rgba(0,0,0,.28);
}
.card .k{color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.06em;}
.card .v{font-size:1.55rem; font-weight:800; margin-top:4px;}
.card .sub{color:var(--muted); font-size:.82rem; margin-top:4px;}
.stage{display:flex; flex-direction:column; gap:10px;}
.stage .top{display:flex; align-items:center; justify-content:space-between;}
.stage .name{font-weight:700; font-size:.96rem;}
.stage .idx{color:var(--muted); font-size:.72rem;}
.dotpill{display:inline-flex; align-items:center; gap:7px; font-size:.76rem; color:var(--muted);}
.dotpill::before{content:""; width:9px; height:9px; border-radius:50%; background:var(--muted);}
.dotpill.ok::before{background:var(--green); box-shadow:0 0 10px var(--green);}
.dotpill.ready::before{background:var(--blue); box-shadow:0 0 10px var(--blue);}
.dotpill.skip::before{background:var(--amber); box-shadow:0 0 8px var(--amber);}
.dotpill.off::before{background:var(--red); box-shadow:0 0 8px var(--red);}
.linkrow{display:flex; flex-wrap:wrap; gap:10px;}
.btn{display:inline-flex; align-items:center; gap:7px; padding:9px 15px; border-radius:11px; font-size:.9rem; cursor:pointer;
  border:1px solid var(--line); background:var(--glass2); color:var(--text); transition:.15s;}
.btn:hover{text-decoration:none; border-color:var(--blue); transform:translateY(-1px);}
.btn.primary{background:linear-gradient(90deg,var(--blue),var(--purple)); border:none; color:#fff; font-weight:700;}
.btn.primary:hover{filter:brightness(1.1);}
.btn.ghost{background:transparent;}
.btn:disabled{opacity:.4; cursor:not-allowed; transform:none;}
label.slabel{display:flex; justify-content:space-between; font-size:.85rem; color:var(--muted); margin:14px 0 5px;}
label.slabel b{color:var(--text);}
input[type=range]{width:100%; accent-color:var(--purple); height:6px;}
.result{text-align:center; padding:6px 0 2px;}
.result .cls{font-size:1.9rem; font-weight:900; text-transform:capitalize;
  background:linear-gradient(90deg,var(--blue),var(--purple)); -webkit-background-clip:text; background-clip:text; color:transparent;}
.barrow{margin:10px 0;}
.barrow .lab{display:flex; justify-content:space-between; font-size:.82rem; color:var(--muted); margin-bottom:5px; text-transform:capitalize;}
.bar{height:12px; border-radius:999px; background:rgba(10,16,32,.8); overflow:hidden; border:1px solid var(--line);}
.bar>span{display:block; height:100%; width:0%; transition:width .55s cubic-bezier(.2,.8,.2,1); background:linear-gradient(90deg,var(--blue),var(--purple));}
.bar.win>span{background:linear-gradient(90deg,var(--green),#10b981);}
pre{background:rgba(4,8,20,.85); border:1px solid var(--line); border-radius:12px; padding:13px 15px; overflow-x:auto; margin:0;
  font-family:"SFMono-Regular",Consolas,Menlo,monospace; font-size:.84rem; color:#cbd5e1;}
code.inline{background:var(--glass2); padding:1px 6px; border-radius:6px; font-size:.85em;}
footer{color:var(--muted); font-size:.82rem; border-top:1px solid var(--line); margin-top:42px; padding-top:18px;}
/* demo tour */
.toolbar{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:16px;}
.toolbar .count{margin-left:auto; color:var(--muted); font-size:.86rem;}
.toolbar .count b{color:var(--text);}
.filters{display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;}
.filter{padding:7px 13px; border-radius:999px; font-size:.83rem; cursor:pointer; color:var(--muted);
  border:1px solid var(--line); background:var(--glass2);}
.filter:hover{color:var(--text);}
.filter.active{color:#fff; background:linear-gradient(90deg,rgba(79,140,255,.3),rgba(163,104,255,.3)); border-color:var(--blue);}
.stepgrid{display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px;}
.scard{cursor:pointer; border:1px solid var(--line); border-radius:14px; padding:13px 14px; background:var(--glass);
  transition:.15s; position:relative; overflow:hidden;}
.scard:hover{transform:translateY(-2px); border-color:var(--blue);}
.scard.active{border-color:var(--purple); box-shadow:0 0 0 1px var(--purple), 0 10px 26px rgba(120,60,210,.25);}
.scard .n{font-size:.74rem; color:var(--muted);}
.scard .t{font-weight:700; margin-top:3px;}
.scard .c{margin-top:8px; font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em;}
.scard.hidden{display:none;}
.detail{margin-top:16px; border:1px solid var(--line); border-radius:16px; padding:20px 22px; background:var(--glass);}
.detail h3{margin:0; font-size:1.3rem; display:flex; align-items:center; gap:10px;}
.detail .livebadge{font-size:.66rem; letter-spacing:.1em; padding:3px 8px; border-radius:999px; background:rgba(6,40,27,.7);
  border:1px solid rgba(52,211,153,.5); color:#6ee7b7;}
.detail .livebadge::before{content:""; display:inline-block; width:6px;height:6px;border-radius:50%;background:var(--green);margin-right:5px;box-shadow:0 0 8px var(--green);}
.detail .tags{display:flex; flex-wrap:wrap; gap:6px; margin:12px 0;}
.detail .tag{font-size:.74rem; padding:3px 9px; border-radius:999px; background:var(--glass2); border:1px solid var(--line); color:var(--muted);}
.detail ul{margin:10px 0 14px; padding-left:18px; color:var(--muted);}
.detail ul li{margin-bottom:5px;}
/* architecture */
.archscroll{overflow-x:auto; border:1px solid var(--line); border-radius:18px; background:
  linear-gradient(180deg, rgba(10,16,34,.7), rgba(6,10,22,.85)); padding:8px;}
.archcanvas{position:relative;}
.lane{position:absolute; left:0; right:0; border:1px dashed rgba(86,113,168,.22); border-radius:14px;}
.lane .lanelabel{position:absolute; top:8px; left:12px; font-size:.7rem; text-transform:uppercase; letter-spacing:.14em; color:var(--muted);}
.node{position:absolute; border-radius:13px; padding:8px 10px; background:var(--glass); border:1px solid var(--line);
  box-shadow:0 8px 22px rgba(0,0,0,.3); display:flex; flex-direction:column; justify-content:center;}
.node .nl{font-weight:700; font-size:.84rem; line-height:1.15;}
.node .ns{font-size:.68rem; color:var(--muted); margin-top:2px;}
.legend{display:flex; flex-wrap:wrap; gap:16px; margin-top:14px;}
.legend .li{display:flex; align-items:center; gap:7px; font-size:.82rem; color:var(--muted);}
.legend .sw{width:14px;height:14px;border-radius:4px;}
/* flow explorer */
.flowlayout{display:grid; grid-template-columns:300px 1fr; gap:18px;}
@media(max-width:820px){.flowlayout{grid-template-columns:1fr;}}
.progress{height:8px; border-radius:999px; background:rgba(10,16,32,.8); border:1px solid var(--line); overflow:hidden; margin-bottom:14px;}
.progress>span{display:block; height:100%; width:4%; background:linear-gradient(90deg,var(--blue),var(--purple)); transition:width .35s ease;}
.timeline{max-height:560px; overflow-y:auto; border:1px solid var(--line); border-radius:14px; padding:8px; background:var(--glass);}
.tline{display:flex; gap:10px; align-items:center; padding:9px 10px; border-radius:10px; cursor:pointer; color:var(--muted); font-size:.86rem;}
.tline:hover{background:var(--glass2); color:var(--text);}
.tline.active{background:linear-gradient(90deg,rgba(79,140,255,.22),rgba(163,104,255,.18)); color:#fff;}
.tline .tn{flex:0 0 auto; width:26px; height:26px; border-radius:8px; display:grid; place-items:center; font-size:.74rem; font-weight:700;
  background:linear-gradient(135deg,var(--blue),var(--purple)); color:#fff;}
.tline.done .tn{background:linear-gradient(135deg,var(--green),#10b981);}
.kv{display:grid; grid-template-columns:140px 1fr; gap:8px 14px; margin-top:12px; font-size:.9rem;}
.kv .key{color:var(--muted);}
.chiprow{display:flex; flex-wrap:wrap; gap:6px;}
.chip{font-size:.74rem; padding:3px 9px; border-radius:999px; background:var(--glass2); border:1px solid var(--line); color:var(--muted);}
"""


def _nav(active: str) -> str:
    """Render the shared sticky navigation bar with the active link marked."""
    def c(key: str) -> str:
        return ' class="active"' if key == active else ""
    return (
        '<nav class="nav"><div class="brand">Iris MLOps<span class="dot">.</span></div>'
        '<div class="links">'
        f'<a href="/"{c("home")}>Dashboard</a>'
        f'<a href="/demo"{c("demo")}>Demo</a>'
        f'<a href="/demo/flow"{c("flow")}>Flow Explorer</a>'
        '<a href="/docs">Swagger</a>'
        '<a href="/health">Health</a>'
        '</div></nav>'
    )


def _page(title: str, active: str, body: str) -> str:
    """Wrap a page body in the shared HTML document shell + theme."""
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
        f"<title>{title}</title><style>{_BASE_CSS}</style></head><body>"
        f"{_nav(active)}<div class=\"wrap\">{body}</div></body></html>"
    )


# ===========================================================================
# DASHBOARD  ( GET / )
# ===========================================================================
def build_dashboard(context: Dict[str, object]) -> str:
    """Build the dark MLOps dashboard served at ``GET /``.

    Args:
        context: Runtime context (metrics, gate result, per-stage statuses).

    Returns:
        str: Complete HTML document for the dashboard.
    """
    accuracy = context.get("accuracy")
    f1 = context.get("f1")
    dataset_hash = str(context.get("dataset_hash") or "n/a")
    gate_passed = context.get("gate_passed")
    threshold = context.get("threshold", 0.90)
    stages: List[Dict[str, str]] = context.get("stages", [])  # type: ignore[assignment]

    acc_txt = f"{float(accuracy) * 100:.1f}%" if isinstance(accuracy, (int, float)) else "n/a"
    f1_txt = f"{float(f1):.3f}" if isinstance(f1, (int, float)) else "n/a"
    short_hash = dataset_hash[:14] + ("..." if len(dataset_hash) > 14 else "")
    gate_badge = (
        '<span class="badge green">PASSED</span>' if gate_passed
        else ('<span class="badge red">FAILED</span>' if gate_passed is False else '<span class="badge amber">N/A</span>')
    )

    stage_cards = ""
    for i, s in enumerate(stages, start=1):
        stage_cards += (
            '<div class="card stage"><div class="top">'
            f'<span class="name">{s["name"]}</span><span class="idx">0{i}</span></div>'
            f'<span class="dotpill {s["status"]}">{s["label"]}</span></div>'
        )

    body = f"""
<div class="hero">
  <h1>Iris MLOps Dashboard</h1>
  <p>End-to-end pipeline status &amp; live predictions.</p>
  <div class="badges">
    <span id="apiBadge" class="badge green"><span class="live"></span>API Healthy &middot; local</span>
    <span class="badge blue">v1.0.0</span>
    <span class="badge purple">Python 3.11</span>
    <span class="badge">FastAPI</span>
    <span class="badge">Azure ML</span>
    <span class="badge">MLflow</span>
    <span class="badge">Docker</span>
  </div>
</div>

<h2 class="section">Pipeline status</h2>
<div class="grid g6">{stage_cards}</div>

<h2 class="section">Model metrics</h2>
<div class="grid g4">
  <div class="card"><div class="k">Accuracy</div><div class="v">{acc_txt}</div><div class="sub">held-out test set</div></div>
  <div class="card"><div class="k">F1 macro</div><div class="v">{f1_txt}</div><div class="sub">balanced across classes</div></div>
  <div class="card"><div class="k">Quality gate</div><div class="v">{gate_badge}</div><div class="sub">threshold {float(threshold):.2f}</div></div>
  <div class="card"><div class="k">Dataset hash</div><div class="v" style="font-size:1rem;font-family:monospace">{short_hash}</div><div class="sub">SHA-256 lineage</div></div>
</div>

<h2 class="section">Live prediction</h2>
<div class="grid g2">
  <div class="card">
    <div class="k" style="margin-bottom:4px">Input features (cm)</div>
    <label class="slabel">Sepal length <b><span id="v_sl">5.1</span></b></label>
    <input type="range" id="sl" min="4" max="8" step="0.1" value="5.1" oninput="sync()"/>
    <label class="slabel">Sepal width <b><span id="v_sw">3.5</span></b></label>
    <input type="range" id="sw" min="2" max="4.5" step="0.1" value="3.5" oninput="sync()"/>
    <label class="slabel">Petal length <b><span id="v_pl">1.4</span></b></label>
    <input type="range" id="pl" min="1" max="7" step="0.1" value="1.4" oninput="sync()"/>
    <label class="slabel">Petal width <b><span id="v_pw">0.2</span></b></label>
    <input type="range" id="pw" min="0.1" max="2.5" step="0.1" value="0.2" oninput="sync()"/>
    <div style="margin-top:16px" class="linkrow">
      <button class="btn primary" onclick="predict()">Predict species &rarr;</button>
      <button class="btn" onclick="randomize()">Randomize</button>
    </div>
  </div>
  <div class="card">
    <div class="k">Predicted species</div>
    <div class="result"><div class="cls" id="cls">&mdash;</div></div>
    <div id="bars"></div>
    <div class="k" style="margin-top:12px">Raw response</div>
    <pre id="raw">POST /predict</pre>
  </div>
</div>

<h2 class="section">Reports &amp; docs</h2>
<div class="linkrow">
  <a class="btn" href="/reports/drift">Drift Report</a>
  <a class="btn" href="/reports/openrouter">OpenRouter Report</a>
  <a class="btn" href="/docs">API Docs (Swagger)</a>
  <a class="btn" href="/health">Health JSON</a>
  <a class="btn" href="/demo">Demo Showcase</a>
  <a class="btn" href="/demo/flow">Flow Explorer</a>
</div>

<footer>Iris MLOps API &middot; FastAPI + scikit-learn + Azure &middot; dashboard served directly by the API.</footer>

<script>
const CLASSES = ["setosa","versicolor","virginica"];
function sync(){{
  v_sl.textContent=sl.value; v_sw.textContent=sw.value; v_pl.textContent=pl.value; v_pw.textContent=pw.value;
}}
function randomize(){{
  const r=(a,b)=>(a+Math.random()*(b-a)).toFixed(1);
  sl.value=r(4,8); sw.value=r(2,4.5); pl.value=r(1,7); pw.value=r(0.1,2.5); sync(); predict();
}}
function renderBars(probs, winner){{
  const host=document.getElementById('bars'); host.innerHTML='';
  CLASSES.forEach(c=>{{
    const p = probs && (c in probs) ? probs[c] : (c===winner?1:0);
    const pct=Math.round(p*100);
    const row=document.createElement('div'); row.className='barrow';
    row.innerHTML='<div class="lab"><span>'+c+'</span><span>'+pct+'%</span></div>'+
      '<div class="bar'+(c===winner?' win':'')+'"><span style="width:'+pct+'%"></span></div>';
    host.appendChild(row);
  }});
}}
async function predict(){{
  const body={{sepal_length:+sl.value, sepal_width:+sw.value, petal_length:+pl.value, petal_width:+pw.value}};
  document.getElementById('cls').textContent='...';
  try{{
    const res=await fetch('/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});
    const data=await res.json();
    document.getElementById('raw').textContent=JSON.stringify(data,null,2);
    if(!res.ok){{document.getElementById('cls').textContent='error'; renderBars(null,null); return;}}
    document.getElementById('cls').textContent=data['class'];
    renderBars(data.probabilities, data['class']);
  }}catch(e){{
    document.getElementById('cls').textContent='offline';
    document.getElementById('raw').textContent=String(e); renderBars(null,null);
  }}
}}
async function health(){{
  const b=document.getElementById('apiBadge');
  try{{
    const r=await fetch('/health'); const d=await r.json();
    if(r.ok && d.status==='ok'){{
      b.innerHTML='<span class="live"></span>'+(d.model_loaded?'API Healthy &middot; local':'API Up &middot; no model');
      b.className='badge '+(d.model_loaded?'green':'amber');
    }} else {{ b.textContent='API Degraded'; b.className='badge red'; }}
  }}catch(e){{ b.innerHTML='<span class="live"></span>API Healthy &middot; local'; b.className='badge green'; }}
}}
sync(); health(); predict();
</script>"""
    return _page("Iris MLOps Dashboard", "home", body)


# ===========================================================================
# ARCHITECTURE BOARD (used inside /demo)
# ===========================================================================
def _build_architecture() -> str:
    """Build the HTML/SVG enterprise architecture board (no Mermaid).

    Returns:
        str: HTML fragment with lanes, color-coded nodes, SVG connectors and a
        legend, wrapped in a horizontally scrollable canvas.
    """
    nodes = {n["id"]: n for n in ARCH_NODES}

    # Lane bands.
    lanes_html = ""
    for lane in ARCH_LANES:
        lanes_html += (
            f'<div class="lane" style="top:{lane["y"]}px;height:{lane["h"]}px">'
            f'<span class="lanelabel">{lane["label"]}</span></div>'
        )

    # SVG connectors (orthogonal elbow paths between node edges).
    paths = ""
    for edge in ARCH_EDGES:
        a, b = nodes[edge["a"]], nodes[edge["b"]]
        color = _KIND_COLOR[edge["kind"]]
        acx, acy = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
        bcx, bcy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        if abs(acy - bcy) < 50:  # same lane -> horizontal edge-to-edge
            if bcx >= acx:
                x1, y1, x2, y2 = a["x"] + a["w"], acy, b["x"], bcy
            else:
                x1, y1, x2, y2 = a["x"], acy, b["x"] + b["w"], bcy
            d = f"M {x1} {y1} L {x2} {y2}"
        else:  # different lanes -> vertical elbow
            if bcy > acy:
                x1, y1 = acx, a["y"] + a["h"]
                x2, y2 = bcx, b["y"]
            else:
                x1, y1 = acx, a["y"]
                x2, y2 = bcx, b["y"] + b["h"]
            my = (y1 + y2) / 2
            d = f"M {x1} {y1} L {x1} {my} L {x2} {my} L {x2} {y2}"
        paths += (
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-opacity="0.7" marker-end="url(#arr-{edge["kind"]})"/>'
        )

    markers = ""
    for kind, color in _KIND_COLOR.items():
        markers += (
            f'<marker id="arr-{kind}" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
            f'<path d="M0,0 L7,3 L0,6 Z" fill="{color}"/></marker>'
        )

    # Nodes.
    node_html = ""
    for n in ARCH_NODES:
        color = _KIND_COLOR[n["kind"]]
        node_html += (
            f'<div class="node" style="left:{n["x"]}px;top:{n["y"]}px;width:{n["w"]}px;height:{n["h"]}px;'
            f'border-color:{color};box-shadow:0 0 0 1px {color}33, 0 8px 22px rgba(0,0,0,.3)">'
            f'<div class="nl" style="color:{color}">{n["label"]}</div>'
            f'<div class="ns">{n["sub"]}</div></div>'
        )

    legend = (
        '<div class="legend">'
        f'<div class="li"><span class="sw" style="background:{_KIND_COLOR["data"]}"></span>Data / ML</div>'
        f'<div class="li"><span class="sw" style="background:{_KIND_COLOR["deploy"]}"></span>Deploy</div>'
        f'<div class="li"><span class="sw" style="background:{_KIND_COLOR["obs"]}"></span>Observability</div>'
        f'<div class="li"><span class="sw" style="background:{_KIND_COLOR["auth"]}"></span>Auth / Trigger</div>'
        '</div>'
    )

    return (
        '<div class="archscroll"><div class="archcanvas" '
        f'style="width:{ARCH_W}px;height:{ARCH_H}px">'
        f'<svg width="{ARCH_W}" height="{ARCH_H}" style="position:absolute;left:0;top:0">'
        f'<defs>{markers}</defs>{paths}</svg>'
        f'{lanes_html}{node_html}</div></div>{legend}'
    )


# ===========================================================================
# DEMO  ( GET /demo )
# ===========================================================================
def build_demo() -> str:
    """Build the interactive 14-step demo tour + architecture at ``GET /demo``.

    Returns:
        str: Complete HTML document for the demo page.
    """
    filters = [
        ("all", "All stages"), ("data", "Data &amp; ML"), ("quality", "Quality &amp; registry"),
        ("cicd", "CI/CD &amp; Docker"), ("azure", "Azure deploy"), ("serve", "Serve &amp; observe"),
    ]
    filter_html = "".join(
        f'<div class="filter{" active" if key == "all" else ""}" data-filter="{key}" onclick="setFilter(this)">{label}</div>'
        for key, label in filters
    )

    cards_html = ""
    for s in DEMO_STEPS:
        cards_html += (
            f'<div class="scard" data-cat="{s["cat"]}" data-n="{s["n"]}" onclick="selectStep({s["n"]})">'
            f'<div class="n">Step {s["n"]:02d}</div><div class="t">{s["title"]}</div>'
            f'<div class="c">{s["cat"]}</div></div>'
        )

    data_json = json.dumps(DEMO_STEPS)
    architecture = _build_architecture()

    body = (
        '<div class="hero"><h1>Azure MLOps &times; OpenRouter &mdash; Interactive Demo</h1>'
        '<p>Walk the full pipeline one step at a time, or jump straight to any stage. '
        'Each card updates the detail panel with what happens, tags, key points and the command to run.</p></div>'
        '<div class="toolbar">'
        '<button class="btn" id="prevBtn" onclick="step(-1)">&larr; Previous</button>'
        '<button class="btn primary" id="nextBtn" onclick="step(1)">Next step &rarr;</button>'
        '<button class="btn ghost" onclick="resetTour()">Reset tour</button>'
        '<span class="count">Demo mode &middot; Step <b id="stepNo">1</b> of <b>14</b></span>'
        '</div>'
        f'<div class="filters">{filter_html}</div>'
        f'<div class="stepgrid" id="stepGrid">{cards_html}</div>'
        '<div class="detail" id="detail"></div>'
        '<h2 class="section">Full system architecture</h2>'
        f'{architecture}'
        '<div class="linkrow" style="margin-top:22px">'
        '<a class="btn primary" href="/demo/flow">Open the line-by-line flow explorer &rarr;</a>'
        '<a class="btn" href="/">Back to dashboard</a></div>'
        '<footer>Interactive architecture &amp; tour &middot; rendered with plain HTML/CSS/SVG, served by FastAPI.</footer>'
        f'<script type="application/json" id="demoData">{data_json}</script>'
        '<script>'
        'const STEPS=JSON.parse(document.getElementById("demoData").textContent);'
        'let cur=1, filter="all";'
        'function render(){'
        '  const s=STEPS.find(x=>x.n===cur);'
        '  document.getElementById("stepNo").textContent=cur;'
        '  document.querySelectorAll(".scard").forEach(c=>c.classList.toggle("active",+c.dataset.n===cur));'
        '  const tags=s.tags.map(t=>`<span class="tag">${t}</span>`).join("");'
        '  const bullets=s.bullets.map(b=>`<li>${b}</li>`).join("");'
        '  document.getElementById("detail").innerHTML='
        '    `<h3>${s.title}<span class="livebadge">LIVE</span></h3>`+'
        '    `<div class="tags">${tags}</div>`+'
        '    `<p style="color:var(--muted)">${s.desc}</p>`+'
        '    `<ul>${bullets}</ul>`+'
        '    `<pre>${s.command}</pre>`;'
        '  document.getElementById("prevBtn").disabled=(cur===1);'
        '  document.getElementById("nextBtn").disabled=(cur===STEPS.length);'
        '  const active=document.querySelector(".scard.active");'
        '  if(active) active.scrollIntoView({block:"nearest",behavior:"smooth"});'
        '}'
        'function selectStep(n){cur=n; render();}'
        'function step(d){const nx=cur+d; if(nx>=1&&nx<=STEPS.length){cur=nx; render();}}'
        'function resetTour(){cur=1; setFilterKey("all"); render();}'
        'function setFilterKey(key){'
        '  filter=key;'
        '  document.querySelectorAll(".filter").forEach(f=>f.classList.toggle("active",f.dataset.filter===key));'
        '  document.querySelectorAll(".scard").forEach(c=>{'
        '    const show=(key==="all"||c.dataset.cat===key); c.classList.toggle("hidden",!show);'
        '  });'
        '}'
        'function setFilter(el){setFilterKey(el.dataset.filter);}'
        'render();'
        '</script>'
    )
    return _page("Azure MLOps Interactive Demo", "demo", body)


# ===========================================================================
# FLOW EXPLORER  ( GET /demo/flow )
# ===========================================================================
def build_flow() -> str:
    """Build the 24-line, clickable pipeline flow explorer at ``/demo/flow``.

    Returns:
        str: Complete HTML document for the flow explorer page.
    """
    timeline = ""
    for line in FLOW_LINES:
        timeline += (
            f'<div class="tline" data-n="{line["n"]}" onclick="selectLine({line["n"]})">'
            f'<span class="tn">{line["n"]}</span><span>{line["title"]}</span></div>'
        )

    data_json = json.dumps(FLOW_LINES)
    body = (
        '<div class="hero"><h1>Flow Explorer</h1>'
        '<p>Every pipeline action, line by line. Click a step or use the controls to walk through what happens, '
        'the files involved, inputs/outputs, environment variables, Azure resources, the command, and how it connects forward.</p></div>'
        '<div class="toolbar">'
        '<button class="btn" id="prevBtn" onclick="move(-1)">&larr; Previous</button>'
        '<button class="btn primary" id="nextBtn" onclick="move(1)">Next line &rarr;</button>'
        '<button class="btn ghost" onclick="resetFlow()">Reset</button>'
        '<span class="count">Line <b id="lineNo">1</b> of <b>24</b></span>'
        '</div>'
        '<div class="progress"><span id="prog"></span></div>'
        '<div class="flowlayout">'
        f'<div class="timeline" id="timeline">{timeline}</div>'
        '<div class="detail" id="detail"></div>'
        '</div>'
        '<footer>Run the whole thing locally with no Azure account &mdash; every Azure step skips gracefully.</footer>'
        f'<script type="application/json" id="flowData">{data_json}</script>'
        '<script>'
        'const LINES=JSON.parse(document.getElementById("flowData").textContent);'
        'let cur=1;'
        'function esc(s){return String(s);}'
        'function render(){'
        '  const l=LINES.find(x=>x.n===cur);'
        '  document.getElementById("lineNo").textContent=cur;'
        '  document.getElementById("prog").style.width=((cur/LINES.length)*100)+"%";'
        '  document.querySelectorAll(".tline").forEach(t=>{'
        '    const n=+t.dataset.n; t.classList.toggle("active",n===cur); t.classList.toggle("done",n<cur);'
        '  });'
        '  const files=l.files.map(f=>`<span class="chip">${f}</span>`).join(" ");'
        '  const envs=(l.env&&l.env.length)?l.env.map(e=>`<span class="chip">${e}</span>`).join(" "):"<span style=\\"color:var(--muted)\\">none</span>";'
        '  document.getElementById("detail").innerHTML='
        '    `<h3>${l.n}. ${l.title}<span class="livebadge">LIVE</span></h3>`+'
        '    `<p style="color:var(--muted)">${l.what}</p>`+'
        '    `<div class="kv">`+'
        '    `<span class="key">Files / modules</span><span class="chiprow">${files}</span>`+'
        '    `<span class="key">Input</span><span>${esc(l.inp)}</span>`+'
        '    `<span class="key">Output</span><span>${esc(l.out)}</span>`+'
        '    `<span class="key">Env vars</span><span class="chiprow">${envs}</span>`+'
        '    `<span class="key">Azure</span><span>${esc(l.azure)}</span>`+'
        '    `<span class="key">Next step</span><span style="color:var(--blue)"><b>${esc(l.nxt)}</b></span>`+'
        '    `</div>`+`<div style="margin-top:12px" class="key">Command</div><pre>${l.command}</pre>`;'
        '  const active=document.querySelector(".tline.active");'
        '  if(active) active.scrollIntoView({block:"nearest",behavior:"smooth"});'
        '}'
        'function selectLine(n){cur=n; render();}'
        'function move(d){const nx=cur+d; if(nx>=1&&nx<=LINES.length){cur=nx; render();}}'
        'function resetFlow(){cur=1; render();}'
        'render();'
        '</script>'
    )
    return _page("MLOps Flow Explorer", "flow", body)
