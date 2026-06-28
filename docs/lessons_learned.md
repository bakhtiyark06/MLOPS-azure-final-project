# Lessons Learned

> Author: Bakhtiyar Khan · Date: 2026-06-27

## What went well
- **Graceful degradation** made local development painless: every Azure call is
  lazy and skips cleanly, so the full train → evaluate → drift → report loop runs
  offline with zero credentials.
- **Quality gate as a hard stop** is a simple but powerful guardrail — a single
  non-zero exit code prevents a bad model from ever reaching Azure.
- **Separating the lean serving image** (`requirements-api.txt`) from the full
  pipeline deps kept the production container small and fast to build.

## Challenges & resolutions
- *Evidently API churn*: Evidently's public API changes between releases, so
  drift detection ships a self-contained PSI-based report as the guaranteed path
  and treats Evidently as a best-effort enhancement.
- *Python version drift*: the local interpreter was newer than the 3.11 target;
  pinning the Docker base to `python:3.11-slim` keeps the production runtime
  deterministic regardless of the developer's machine.
- *Secret hygiene*: all credentials flow through `.env` (git-ignored) locally and
  GitHub Secrets in CI; `.env.example` documents every variable without leaking
  values.

## What I'd do next
- Add canary / blue-green deployments on AKS for zero-downtime releases.
- Schedule drift detection against live production data and auto-open an issue
  when PSI breaches the threshold.
- Add model explainability (SHAP) to the AI report.
- Introduce a feature store and data validation (Great Expectations) upstream of
  training.

## Key takeaways
1. Reproducibility (hashing, pinned base image, MLflow) is the backbone of trust.
2. Automation is only as safe as its gates — invest in the quality gate and
   smoke tests.
3. Observability must be built in from day one, not bolted on later.
