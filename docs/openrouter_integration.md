# OpenRouter Integration

> Author: Bakhtiyar Khan · Date: 2026-06-27

The pipeline uses [OpenRouter](https://openrouter.ai) to generate a concise,
human-readable summary of each run (metrics, quality gate, drift, deployment).

## How it works

`mlops/openrouter_report.py`:
1. Gathers `reports/metrics.json`, `reports/quality_gate.json`,
   `reports/drift_summary.json` and `reports/model_registration.json`.
2. Builds a prompt and calls the OpenRouter chat-completions API.
3. Writes the result to `reports/ai_report.md`.
4. **If `OPENROUTER_API_KEY` is missing or the call fails, it writes a
   deterministic local fallback report instead of failing the pipeline.**

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | _(none)_ | API key; when unset the fallback is used |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Model slug to use |

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_MODEL="openai/gpt-4o-mini"
python mlops/openrouter_report.py
```

## API request shape

```http
POST https://openrouter.ai/api/v1/chat/completions
Authorization: Bearer $OPENROUTER_API_KEY
Content-Type: application/json

{
  "model": "openai/gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are a precise MLOps reporting assistant."},
    {"role": "user", "content": "...pipeline JSON + instructions..."}
  ],
  "temperature": 0.2
}
```

## Safety / cost

- Temperature is low (0.2) for stable, factual summaries.
- A 60s timeout and broad exception handling guarantee the pipeline never hangs
  or fails because of the LLM call.
- The key is provided only through GitHub Secrets / environment variables and is
  never committed.
