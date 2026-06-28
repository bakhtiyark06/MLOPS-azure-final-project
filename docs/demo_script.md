# Demo Script

> Author: Bakhtiyar Khan · Date: 2026-06-27

A ~10 minute live demonstration script for graders.

## 0. Setup (once)

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements-dev.txt
```

## 1. Run the test suite with coverage (Phase 7)

```bash
pytest --cov
```
Show the coverage summary ≥ 70% and all tests passing.

## 2. Train and pass the quality gate (Phases 1–3)

```bash
python mlops/ingest_data.py --no-upload
python mlops/train.py
python mlops/evaluate.py
echo "exit code: $?"   # 0 = passed
```
Open `reports/metrics.json` and `reports/confusion_matrix.png`.

## 3. Demonstrate the quality gate BLOCKING a bad model (Phase 3)

```bash
python mlops/evaluate.py --demo-fail
echo "exit code: $?"   # non-zero = blocked
```
Explain: in CI/CD this non-zero exit stops the deployment job.

## 4. Show MLflow tracking (Phase 2)

```bash
mlflow ui --backend-store-uri ./mlruns
# open http://127.0.0.1:5000
```

## 5. Run the API locally (Phase 5)

```bash
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## 6. Build the Docker image (Phase 6)

```bash
docker build -t mlops-api:demo .
docker run -p 8000:8000 mlops-api:demo
```

## 7. Drift detection — normal vs simulated (Phase 10)

```bash
python mlops/drift_detection.py            # no drift
python mlops/drift_detection.py --simulate # drift detected
```
Open `reports/drift_report.html` for both runs.

## 8. AI report (Phase 11)

```bash
python mlops/openrouter_report.py
```
Open `reports/ai_report.md`.

## 9. Show CI/CD on GitHub (Phase 8)

- Open the Actions tab; show a green CI run and the CD pipeline stages.
- Show the registered model + tags in the Azure ML studio (if provisioned).

## 10. Rollback demonstration (Phase 4)

```bash
# Re-deploy a known-good earlier image tag to AKS:
python mlops/deploy_aks.py --tag <previous-git-sha>
```
Explain that model versions in Azure ML and image tags in ACR make rollback a
one-command operation.
