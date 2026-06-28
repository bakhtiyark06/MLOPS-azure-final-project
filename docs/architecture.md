# Architecture

> Author: Bakhtiyar Khan · Date: 2026-06-27

This document describes the end-to-end architecture of the Iris MLOps pipeline.

## Overview

The system implements a complete, production-grade MLOps CI/CD pipeline on
Microsoft Azure. Code pushed to GitHub triggers Continuous Integration (tests,
training, quality gate, drift, AI report). Merges to `main` additionally trigger
Continuous Deployment (model registration, container build/push, staged
deployment to ACI, then production deployment to AKS), each protected by smoke
tests.

## Architecture diagram

```mermaid
flowchart TD
    Dev[Developer] -->|git push| GH[GitHub Repository]

    subgraph CI["GitHub Actions - CI (ci.yml)"]
        I1[Install deps] --> I2[pytest + coverage >= 70%]
        I2 --> I3[Ingest data]
        I3 --> I4[Train model + MLflow]
        I4 --> I5[Quality gate]
        I5 --> I6[Drift detection]
        I6 --> I7[OpenRouter AI report]
    end

    subgraph CD["GitHub Actions - CD (cd.yml)"]
        D0[Azure login] --> D1[Train]
        D1 --> D2[Quality gate]
        D2 -->|pass| D3[Register model in Azure ML]
        D2 -->|fail| STOP[Stop - block deploy]
        D3 --> D4[Build image]
        D4 --> D5[Push to ACR]
        D5 --> D6[Deploy ACI staging]
        D6 --> D7[Smoke test]
        D7 --> D8[Deploy AKS production]
        D8 --> D9[Smoke test]
    end

    GH --> CI
    GH --> CD

    subgraph Azure["Microsoft Azure"]
        Blob[(Azure Blob Storage)]
        AML[Azure ML Workspace + Model Registry]
        ACR[(Azure Container Registry)]
        ACI[Azure Container Instances - Staging]
        AKS[Azure Kubernetes Service - Production]
        MON[Azure Monitor + Application Insights]
    end

    I3 -. upload raw/reference .-> Blob
    D3 --> AML
    D5 --> ACR
    ACR --> ACI
    ACR --> AKS
    ACI -. telemetry .-> MON
    AKS -. telemetry .-> MON

    Users[API Clients] -->|POST /predict| AKS
```

## Components

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| Data ingestion | scikit-learn, pandas, Azure Blob | Load Iris, hash, version, upload |
| Training | scikit-learn, MLflow | Train model, log params/metrics/artifacts |
| Quality gate | Python | Enforce minimum accuracy, block bad models |
| Model registry | Azure ML | Versioned, tagged model governance |
| Serving | FastAPI, uvicorn | REST inference (`/health`, `/predict`) |
| Packaging | Docker, ACR | Reproducible container image |
| Staging | Azure Container Instances | Fast, cheap pre-prod validation |
| Production | Azure Kubernetes Service | Scalable, resilient serving |
| Monitoring | Azure Monitor, App Insights | Logs, metrics, telemetry |
| Drift | Evidently AI (+ PSI fallback) | Detect data distribution shift |
| Reporting | OpenRouter LLM | Human-readable run summary |

## Design principles

1. **Graceful degradation** – every Azure/optional integration is lazy and
   skips cleanly when credentials or packages are missing, so the core flow runs
   on any laptop.
2. **Fail fast, fail safe** – the quality gate exits non-zero on bad models,
   halting deployment before anything reaches Azure.
3. **Twelve-factor config** – everything is configured via environment variables
   / GitHub Secrets; no secrets are committed.
4. **Promotion path** – staging (ACI) is always validated with smoke tests
   before production (AKS).

See [`pipeline_walkthrough.md`](pipeline_walkthrough.md) for a stage-by-stage
narrative and [`azure_setup.md`](azure_setup.md) for resource provisioning.
