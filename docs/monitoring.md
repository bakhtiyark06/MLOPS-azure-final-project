# Monitoring & Observability

> Author: Bakhtiyar Khan · Date: 2026-06-27

## Layers of observability

| Layer | Tool | What it captures |
|-------|------|------------------|
| Application logs | Python `logging` → stdout | Request lifecycle, errors, latency |
| Telemetry export | Application Insights (OpenCensus) | Traces, custom dimensions, exceptions |
| Platform metrics | Azure Monitor | CPU/memory, restarts, availability |
| Data quality | Evidently AI / PSI | Feature drift over time |

## Application Insights

The API (`app/main.py`) attaches an `AzureLogHandler` when
`APPINSIGHTS_CONNECTION_STRING` is set. Every served prediction emits a
structured log line with `custom_dimensions`:

```python
logger.info("prediction served", extra={"custom_dimensions": {
    "prediction": 0, "class": "setosa", "confidence": 0.99, "latency_ms": 1.2,
}})
```

These appear in Application Insights under **Logs → traces** and
**customDimensions**. Example Kusto query:

```kusto
traces
| where message == "prediction served"
| extend cls = tostring(customDimensions["class"])
| summarize count() by cls, bin(timestamp, 5m)
```

If the connection string is absent, telemetry export is skipped silently and
logs still go to stdout (captured by ACI/AKS).

## Azure Monitor

- **ACI**: `az container logs --resource-group <RG> --name mlops-api-staging`
- **AKS**: `kubectl logs deployment/mlops-api` and Azure Monitor for Containers
  (Container Insights) for cluster-level metrics, alerts and dashboards.

Recommended alerts:
- HTTP 5xx rate > 1% over 5 min
- p95 latency > 500 ms
- Pod restart count > 0 in 10 min

## Drift monitoring

See [`pipeline_walkthrough.md`](pipeline_walkthrough.md) and run:

```bash
python mlops/drift_detection.py            # normal
python mlops/drift_detection.py --simulate # demo: inject drift
```

Outputs `reports/drift_report.html` (Evidently when available, otherwise a
self-contained PSI report) and `reports/drift_summary.json`. PSI thresholds:
`<0.1` stable, `0.1–0.25` moderate, `>0.25` drift.

## Health checks

- API: `GET /health` returns `{status, model_loaded, model_path}`.
- Docker `HEALTHCHECK` hits `/health` every 30s.
- Kubernetes readiness/liveness probes hit `/health`.
