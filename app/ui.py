"""
Author:  Bakhtiyar Khan
Date:    2026-06-27
Purpose: Server-side HTML/CSS/JS builders for the FastAPI dashboard. Keeps all
         presentation logic out of ``app/main.py``. Everything here is plain
         HTML returned by FastAPI - no React/Next.js or external build step. The
         three pages (dashboard, demo, flow) share one dark theme and nav bar.
"""

from __future__ import annotations

import json
from typing import Dict, List

# ---------------------------------------------------------------------------
# Pipeline step model. Drives both the demo architecture page and the
# line-by-line flow explorer so the two never drift out of sync.
# ---------------------------------------------------------------------------
PIPELINE_STEPS: List[Dict[str, object]] = [
    {
        "n": 1,
        "title": "Data Ingestion",
        "what": "Load the Iris dataset, hash it, write raw + reference copies and a metadata sidecar, then optionally upload to Azure Blob Storage.",
        "files": ["mlops/ingest_data.py"],
        "command": "python mlops/ingest_data.py --no-upload",
        "inputs": "scikit-learn Iris bunch",
        "outputs": "data/raw/iris.csv, data/reference/reference.csv, data/raw/metadata.json",
        "azure": "Azure Blob Storage (optional)",
        "next": "Model Training",
    },
    {
        "n": 2,
        "title": "Model Training",
        "what": "Stratified train/test split, fit a RandomForest, log params/metrics/artifacts to MLflow and save the model + reports.",
        "files": ["mlops/train.py"],
        "command": "python mlops/train.py",
        "inputs": "data/raw/iris.csv",
        "outputs": "models/model.joblib, reports/metrics.json, reports/confusion_matrix.png",
        "azure": "MLflow tracking (file or Azure ML)",
        "next": "Quality Gate",
    },
    {
        "n": 3,
        "title": "Quality Gate",
        "what": "Compare accuracy to the threshold. Exit non-zero on failure so CI/CD halts before any deployment. Supports a demo-fail mode.",
        "files": ["mlops/evaluate.py"],
        "command": "python mlops/evaluate.py",
        "inputs": "reports/metrics.json",
        "outputs": "reports/quality_gate.json (exit 0 pass / 1 fail)",
        "azure": "None",
        "next": "Model Registry",
    },
    {
        "n": 4,
        "title": "Model Registry",
        "what": "Register the approved model in Azure ML with governance tags: accuracy, dataset_hash, git_sha, created_by, project_name, version.",
        "files": ["mlops/register_model.py", "mlops/azure_clients.py"],
        "command": "python mlops/register_model.py",
        "inputs": "models/model.joblib, reports/metrics.json",
        "outputs": "reports/model_registration.json",
        "azure": "Azure Machine Learning workspace",
        "next": "Container Build",
    },
    {
        "n": 5,
        "title": "Container Build",
        "what": "Build the FastAPI Docker image, tag with git SHA + latest and push to Azure Container Registry.",
        "files": ["Dockerfile", "mlops/build_container.py"],
        "command": "python mlops/build_container.py",
        "inputs": "app/, models/model.joblib, requirements-api.txt",
        "outputs": "Docker image in ACR",
        "azure": "Azure Container Registry",
        "next": "Staging Deploy",
    },
    {
        "n": 6,
        "title": "Staging Deploy (ACI)",
        "what": "Deploy the image to Azure Container Instances and run an HTTP smoke test against /health and /predict.",
        "files": ["mlops/deploy_aci.py"],
        "command": "python mlops/deploy_aci.py --tag latest",
        "inputs": "ACR image",
        "outputs": "Running ACI container + smoke-test result",
        "azure": "Azure Container Instances",
        "next": "Production Deploy",
    },
    {
        "n": 7,
        "title": "Production Deploy (AKS)",
        "what": "Apply a Deployment + LoadBalancer to Azure Kubernetes Service, wait for the external IP and smoke-test production.",
        "files": ["mlops/deploy_aks.py"],
        "command": "python mlops/deploy_aks.py --tag latest",
        "inputs": "ACR image",
        "outputs": "Live AKS service + smoke-test result",
        "azure": "Azure Kubernetes Service",
        "next": "Monitoring & Drift",
    },
    {
        "n": 8,
        "title": "Monitoring & Drift",
        "what": "Ship telemetry to Application Insights / Azure Monitor and run Evidently drift detection to produce an HTML report.",
        "files": ["app/main.py", "mlops/drift_detection.py"],
        "command": "python mlops/drift_detection.py --simulate",
        "inputs": "data/reference/reference.csv, current data",
        "outputs": "reports/drift_report.html, reports/drift_summary.json",
        "azure": "Application Insights, Azure Monitor",
        "next": "AI Report",
    },
    {
        "n": 9,
        "title": "OpenRouter AI Report",
        "what": "Summarise metrics, quality gate, drift and deployment via an OpenRouter LLM. Falls back to a deterministic local report without a key.",
        "files": ["mlops/openrouter_report.py"],
        "command": "python mlops/openrouter_report.py",
        "inputs": "reports/*.json",
        "outputs": "reports/ai_report.md",
        "azure": "OpenRouter API (external)",
        "next": "Done",
    },
]

# Components rendered on the /demo architecture page.
DEMO_COMPONENTS: List[Dict[str, str]] = [
    {"name": "GitHub Actions", "role": "CI/CD orchestration", "group": "Pipeline"},
    {"name": "MLflow", "role": "Experiment tracking", "group": "Pipeline"},
    {"name": "Quality Gate", "role": "Accuracy threshold guard", "group": "Pipeline"},
    {"name": "Azure ML", "role": "Model registry + governance", "group": "Azure"},
    {"name": "Azure Blob Storage", "role": "Dataset versioning", "group": "Azure"},
    {"name": "Azure Container Registry", "role": "Image storage (ACR)", "group": "Azure"},
    {"name": "Docker", "role": "Reproducible packaging", "group": "Build"},
    {"name": "ACI Staging", "role": "Pre-prod validation", "group": "Deploy"},
    {"name": "AKS Production", "role": "Scalable serving", "group": "Deploy"},
    {"name": "FastAPI", "role": "REST inference API", "group": "Serve"},
    {"name": "Application Insights", "role": "Telemetry + logs", "group": "Monitor"},
    {"name": "Evidently Drift", "role": "Data drift detection", "group": "Monitor"},
    {"name": "OpenRouter Report", "role": "AI run summary", "group": "Monitor"},
]


# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------
_BASE_CSS = """
:root {
  --bg0:#070b16; --bg1:#0f1830; --panel:#121c34; --panel2:#1a2848;
  --text:#e6edf7; --muted:#93a4c4; --line:#23314f;
  --blue:#4f8cff; --purple:#a368ff; --green:#34d399; --amber:#fbbf24; --red:#f87171;
}
* { box-sizing:border-box; }
body {
  margin:0; min-height:100vh; color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:
    radial-gradient(1100px 600px at 12% -8%, #1b2c54 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #2a1a4d 0%, transparent 50%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%);
  background-attachment:fixed; line-height:1.55;
}
a { color:var(--blue); text-decoration:none; }
a:hover { text-decoration:underline; }
.nav {
  position:sticky; top:0; z-index:10; backdrop-filter:blur(10px);
  background:rgba(7,11,22,.7); border-bottom:1px solid var(--line);
  display:flex; align-items:center; gap:18px; padding:12px 24px;
}
.nav .brand { font-weight:700; letter-spacing:-.01em; }
.nav .brand .dot { color:var(--purple); }
.nav .links { display:flex; gap:6px; margin-left:auto; flex-wrap:wrap; }
.nav .links a {
  color:var(--muted); padding:6px 12px; border-radius:8px; font-size:.9rem;
  border:1px solid transparent;
}
.nav .links a:hover { color:var(--text); text-decoration:none; background:var(--panel2); }
.nav .links a.active { color:var(--text); border-color:var(--line); background:var(--panel); }
.wrap { max-width:1080px; margin:0 auto; padding:32px 24px 80px; }
.hero h1 { font-size:2.05rem; margin:0 0 6px; letter-spacing:-.02em; }
.hero p { color:var(--muted); margin:0 0 16px; max-width:720px; }
.badges { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:26px; }
.badge {
  font-size:.78rem; padding:4px 11px; border-radius:999px;
  background:var(--panel2); border:1px solid var(--line); color:var(--text);
}
.badge.green { border-color:#14532d; background:#06281b; color:#6ee7b7; }
.badge.blue  { border-color:#13335f; background:#0a2142; color:#93c5fd; }
.badge.purple{ border-color:#3b1f63; background:#1c113a; color:#c4b5fd; }
.badge.amber { border-color:#5a4410; background:#2c2107; color:#fcd34d; }
.badge.red   { border-color:#5b1d1d; background:#2c0d0d; color:#fca5a5; }
h2.section { font-size:1.05rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:34px 0 14px; }
.grid { display:grid; gap:14px; }
.g6 { grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }
.g4 { grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); }
.g3 { grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); }
.g2 { grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
.card {
  background:linear-gradient(180deg,var(--panel) 0%, #0e1730 100%);
  border:1px solid var(--line); border-radius:14px; padding:16px 18px;
}
.card .k { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }
.card .v { font-size:1.5rem; font-weight:700; margin-top:4px; }
.card .sub { color:var(--muted); font-size:.82rem; margin-top:4px; }
.stage { display:flex; flex-direction:column; gap:8px; }
.stage .top { display:flex; align-items:center; justify-content:space-between; }
.stage .name { font-weight:600; }
.dotpill { display:inline-flex; align-items:center; gap:6px; font-size:.76rem; color:var(--muted); }
.dotpill::before { content:""; width:9px; height:9px; border-radius:50%; background:var(--muted); }
.dotpill.ok::before { background:var(--green); box-shadow:0 0 10px var(--green); }
.dotpill.ready::before { background:var(--blue); box-shadow:0 0 10px var(--blue); }
.dotpill.skip::before { background:var(--amber); }
.dotpill.off::before { background:var(--red); }
.linkrow { display:flex; flex-wrap:wrap; gap:10px; }
.btn {
  display:inline-block; padding:9px 14px; border-radius:10px; font-size:.9rem; cursor:pointer;
  border:1px solid var(--line); background:var(--panel2); color:var(--text);
}
.btn:hover { text-decoration:none; border-color:var(--blue); }
.btn.primary { background:linear-gradient(90deg,var(--blue),var(--purple)); border:none; color:#fff; font-weight:600; }
.btn.primary:hover { filter:brightness(1.08); }
label.slabel { display:flex; justify-content:space-between; font-size:.85rem; color:var(--muted); margin:12px 0 4px; }
label.slabel b { color:var(--text); }
input[type=range] { width:100%; accent-color:var(--purple); }
.result { text-align:center; padding:8px 0; }
.result .cls { font-size:1.8rem; font-weight:800; background:linear-gradient(90deg,var(--blue),var(--purple)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.bar { height:12px; border-radius:999px; background:var(--panel2); overflow:hidden; border:1px solid var(--line); }
.bar > span { display:block; height:100%; background:linear-gradient(90deg,var(--green),#10b981); width:0%; transition:width .5s ease; }
.barrow { margin:10px 0; }
.barrow .lab { display:flex; justify-content:space-between; font-size:.8rem; color:var(--muted); margin-bottom:4px; }
pre { background:#070d1c; border:1px solid var(--line); border-radius:12px; padding:14px 16px; overflow-x:auto; margin:0;
  font-family:"SFMono-Regular",Consolas,Menlo,monospace; font-size:.84rem; color:#cbd5e1; }
.step { border:1px solid var(--line); border-radius:14px; overflow:hidden; margin-bottom:12px; background:var(--panel); }
.step > button {
  width:100%; text-align:left; cursor:pointer; border:none; background:transparent; color:var(--text);
  display:flex; align-items:center; gap:14px; padding:16px 18px; font-size:1rem;
}
.step > button:hover { background:var(--panel2); }
.step .num {
  flex:0 0 auto; width:34px; height:34px; border-radius:10px; display:grid; place-items:center;
  font-weight:700; background:linear-gradient(135deg,var(--blue),var(--purple)); color:#fff;
}
.step .arrow { margin-left:auto; color:var(--muted); transition:transform .2s ease; }
.step.open .arrow { transform:rotate(90deg); }
.step .body { display:none; padding:4px 18px 18px 66px; border-top:1px solid var(--line); }
.step.open .body { display:block; }
.kv { display:grid; grid-template-columns:130px 1fr; gap:6px 14px; margin-top:10px; font-size:.9rem; }
.kv .key { color:var(--muted); }
.flowline { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:16px 0 6px; }
.chip { padding:7px 12px; border-radius:10px; background:var(--panel); border:1px solid var(--line); font-size:.85rem; }
.chip .sub { color:var(--muted); font-size:.74rem; display:block; }
.sep { color:var(--purple); font-weight:700; }
footer { color:var(--muted); font-size:.82rem; border-top:1px solid var(--line); margin-top:40px; padding-top:18px; }
code.inline { background:var(--panel2); padding:1px 6px; border-radius:6px; font-size:.85em; }
"""


def _nav(active: str) -> str:
    """Render the shared top navigation bar.

    Args:
        active: Key of the active page ("home", "demo", "flow").

    Returns:
        str: HTML for the nav bar.
    """
    def cls(key: str) -> str:
        return ' class="active"' if key == active else ""

    return f"""<nav class="nav">
  <div class="brand">Iris MLOps<span class="dot">.</span></div>
  <div class="links">
    <a href="/"{cls('home')}>Dashboard</a>
    <a href="/demo"{cls('demo')}>Architecture</a>
    <a href="/demo/flow"{cls('flow')}>Flow Explorer</a>
    <a href="/docs">Swagger</a>
    <a href="/health">Health</a>
  </div>
</nav>"""


def _page(title: str, active: str, body: str) -> str:
    """Wrap page body in the shared HTML document shell.

    Args:
        title: Document title.
        active: Active nav key.
        body: Inner HTML for the page.

    Returns:
        str: A complete HTML document.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>{_BASE_CSS}</style>
</head>
<body>
{_nav(active)}
<div class="wrap">
{body}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Dashboard ( GET / )
# ---------------------------------------------------------------------------
def build_dashboard(context: Dict[str, object]) -> str:
    """Build the dark MLOps dashboard served at ``GET /``.

    Args:
        context: Runtime context (metrics, gate result, stage statuses).

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
    short_hash = dataset_hash[:12] + ("..." if len(dataset_hash) > 12 else "")
    gate_badge = (
        '<span class="badge green">PASSED</span>' if gate_passed
        else ('<span class="badge red">FAILED</span>' if gate_passed is False else '<span class="badge amber">N/A</span>')
    )

    stage_cards = "".join(
        f"""<div class="card stage">
      <div class="top"><span class="name">{s['name']}</span></div>
      <span class="dotpill {s['status']}">{s['label']}</span>
    </div>"""
        for s in stages
    )

    sample = json.dumps({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})

    body = f"""
<div class="hero">
  <h1>Iris MLOps Dashboard</h1>
  <p>Production-grade Iris classification service for the Azure MLOps CI/CD pipeline.
     Train, gate, register, containerise, deploy and monitor &mdash; all from one repo.</p>
  <div class="badges">
    <span id="apiBadge" class="badge amber">API: checking...</span>
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
  <div class="card"><div class="k">F1 (macro)</div><div class="v">{f1_txt}</div><div class="sub">balanced across classes</div></div>
  <div class="card"><div class="k">Quality gate</div><div class="v">{gate_badge}</div><div class="sub">threshold {float(threshold):.2f}</div></div>
  <div class="card"><div class="k">Dataset hash</div><div class="v" style="font-size:1rem;font-family:monospace">{short_hash}</div><div class="sub">SHA-256 of dataset</div></div>
</div>

<h2 class="section">Live prediction</h2>
<div class="grid g2">
  <div class="card">
    <div class="k" style="margin-bottom:6px">Input features (cm)</div>
    <label class="slabel">Sepal length <b><span id="v_sl">5.1</span></b></label>
    <input type="range" id="sl" min="4" max="8" step="0.1" value="5.1" oninput="sync()"/>
    <label class="slabel">Sepal width <b><span id="v_sw">3.5</span></b></label>
    <input type="range" id="sw" min="2" max="4.5" step="0.1" value="3.5" oninput="sync()"/>
    <label class="slabel">Petal length <b><span id="v_pl">1.4</span></b></label>
    <input type="range" id="pl" min="1" max="7" step="0.1" value="1.4" oninput="sync()"/>
    <label class="slabel">Petal width <b><span id="v_pw">0.2</span></b></label>
    <input type="range" id="pw" min="0.1" max="2.5" step="0.1" value="0.2" oninput="sync()"/>
    <div style="margin-top:16px" class="linkrow">
      <button class="btn primary" onclick="predict()">Run prediction &rarr;</button>
      <button class="btn" onclick="randomize()">Randomize</button>
    </div>
  </div>
  <div class="card">
    <div class="k">Prediction result</div>
    <div class="result"><div class="cls" id="cls">&mdash;</div></div>
    <div class="barrow">
      <div class="lab"><span>Confidence</span><span id="confTxt">&mdash;</span></div>
      <div class="bar"><span id="confBar"></span></div>
    </div>
    <div class="k" style="margin-top:14px">Raw response</div>
    <pre id="raw">POST /predict {sample}</pre>
  </div>
</div>

<h2 class="section">Reports &amp; docs</h2>
<div class="linkrow">
  <a class="btn" href="/docs">Swagger UI</a>
  <a class="btn" href="/redoc">ReDoc</a>
  <a class="btn" href="/health">Health JSON</a>
  <a class="btn" href="/reports/drift">Drift report</a>
  <a class="btn" href="/reports/openrouter">OpenRouter report</a>
  <a class="btn" href="/demo">Architecture</a>
  <a class="btn" href="/demo/flow">Flow explorer</a>
</div>

<footer>Iris MLOps API &middot; FastAPI + scikit-learn + Azure &middot; dashboard served directly by the API.</footer>

<script>
function sync() {{
  v_sl.textContent = sl.value; v_sw.textContent = sw.value;
  v_pl.textContent = pl.value; v_pw.textContent = pw.value;
}}
function randomize() {{
  const r=(a,b)=>(a+Math.random()*(b-a)).toFixed(1);
  sl.value=r(4,8); sw.value=r(2,4.5); pl.value=r(1,7); pw.value=r(0.1,2.5); sync();
}}
async function predict() {{
  const body = {{
    sepal_length:parseFloat(sl.value), sepal_width:parseFloat(sw.value),
    petal_length:parseFloat(pl.value), petal_width:parseFloat(pw.value)
  }};
  document.getElementById('cls').textContent='...';
  try {{
    const res = await fetch('/predict', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body)}});
    const data = await res.json();
    document.getElementById('raw').textContent = JSON.stringify(data, null, 2);
    if (!res.ok) {{ document.getElementById('cls').textContent='error'; return; }}
    document.getElementById('cls').textContent = data['class'];
    const pct = Math.round((data.confidence||0)*100);
    document.getElementById('confBar').style.width = pct+'%';
    document.getElementById('confTxt').textContent = pct+'%';
  }} catch (e) {{
    document.getElementById('cls').textContent='offline';
    document.getElementById('raw').textContent = String(e);
  }}
}}
async function health() {{
  const b = document.getElementById('apiBadge');
  try {{
    const r = await fetch('/health'); const d = await r.json();
    if (r.ok && d.status==='ok') {{ b.textContent = d.model_loaded ? 'API: Healthy' : 'API: Up (no model)';
      b.className = 'badge ' + (d.model_loaded ? 'green':'amber'); }}
    else {{ b.textContent='API: Degraded'; b.className='badge red'; }}
  }} catch (e) {{ b.textContent='API: Local'; b.className='badge blue'; }}
}}
sync(); health(); predict();
</script>"""
    return _page("Iris MLOps Dashboard", "home", body)


# ---------------------------------------------------------------------------
# Architecture demo ( GET /demo )
# ---------------------------------------------------------------------------
def build_demo() -> str:
    """Build the full-system architecture walkthrough at ``GET /demo``.

    Returns:
        str: Complete HTML document for the architecture page.
    """
    groups: Dict[str, List[Dict[str, str]]] = {}
    for comp in DEMO_COMPONENTS:
        groups.setdefault(comp["group"], []).append(comp)

    group_blocks = ""
    for group, comps in groups.items():
        cards = "".join(
            f"""<div class="card"><div class="name" style="font-weight:600">{c['name']}</div>
            <div class="sub" style="color:var(--muted);font-size:.84rem;margin-top:4px">{c['role']}</div></div>"""
            for c in comps
        )
        group_blocks += f'<h2 class="section">{group}</h2><div class="grid g3">{cards}</div>'

    flow_chips = ""
    chips = [
        ("Developer", "git push"), ("GitHub Actions", "CI + CD"), ("Tests", "pytest 70%+"),
        ("Train", "MLflow"), ("Quality Gate", "accuracy >= 0.90"), ("Azure ML", "register"),
        ("Docker", "build"), ("ACR", "push image"), ("ACI", "staging + smoke"),
        ("AKS", "production + smoke"), ("FastAPI", "serve /predict"),
        ("App Insights", "telemetry"), ("Evidently", "drift"), ("OpenRouter", "AI report"),
    ]
    for i, (name, sub) in enumerate(chips):
        flow_chips += f'<div class="chip">{name}<span class="sub">{sub}</span></div>'
        if i < len(chips) - 1:
            flow_chips += '<span class="sep">&rarr;</span>'

    body = f"""
<div class="hero">
  <h1>System Architecture</h1>
  <p>End-to-end MLOps pipeline on Microsoft Azure. A push to GitHub runs CI; a merge to main runs CD,
     promoting a gated model through staging and into production with monitoring at every layer.</p>
  <div class="badges">
    <span class="badge green">CI/CD automated</span>
    <span class="badge blue">Azure native</span>
    <span class="badge purple">Graceful offline mode</span>
  </div>
</div>

<h2 class="section">End-to-end flow</h2>
<div class="card"><div class="flowline">{flow_chips}</div></div>

{group_blocks}

<h2 class="section">How it fits together</h2>
<div class="grid g2">
  <div class="card">
    <div style="font-weight:600;margin-bottom:6px">Continuous Integration</div>
    <div style="color:var(--muted);font-size:.9rem">Install deps &rarr; run tests &amp; coverage &rarr; ingest data &rarr;
    train with MLflow &rarr; quality gate &rarr; drift detection &rarr; OpenRouter report. Artifacts uploaded for review.</div>
  </div>
  <div class="card">
    <div style="font-weight:600;margin-bottom:6px">Continuous Deployment</div>
    <div style="color:var(--muted);font-size:.9rem">Azure login &rarr; train &rarr; quality gate &rarr; register in Azure ML &rarr;
    build &amp; push to ACR &rarr; ACI staging + smoke test &rarr; AKS production + smoke test.</div>
  </div>
</div>

<div class="linkrow" style="margin-top:22px">
  <a class="btn primary" href="/demo/flow">Open the line-by-line flow explorer &rarr;</a>
  <a class="btn" href="/">Back to dashboard</a>
</div>

<footer>Architecture overview &middot; see docs/architecture.md for the full write-up.</footer>"""
    return _page("MLOps Architecture", "demo", body)


# ---------------------------------------------------------------------------
# Flow explorer ( GET /demo/flow )
# ---------------------------------------------------------------------------
def build_flow() -> str:
    """Build the clickable, line-by-line pipeline flow explorer at ``/demo/flow``.

    Returns:
        str: Complete HTML document for the flow explorer page.
    """
    steps_html = ""
    for step in PIPELINE_STEPS:
        files = ", ".join(f"<code class='inline'>{f}</code>" for f in step["files"])  # type: ignore[index]
        steps_html += f"""
<div class="step" id="step{step['n']}">
  <button onclick="toggle({step['n']})">
    <span class="num">{step['n']}</span>
    <span>{step['title']}</span>
    <span class="arrow">&rsaquo;</span>
  </button>
  <div class="body">
    <p style="color:var(--muted);margin:6px 0 0">{step['what']}</p>
    <div class="kv">
      <span class="key">Files / modules</span><span>{files}</span>
      <span class="key">Command</span><span><code class="inline">{step['command']}</code></span>
      <span class="key">Inputs</span><span>{step['inputs']}</span>
      <span class="key">Outputs</span><span>{step['outputs']}</span>
      <span class="key">Azure</span><span>{step['azure']}</span>
      <span class="key">Next step</span><span><b style="color:var(--blue)">{step['next']}</b></span>
    </div>
  </div>
</div>"""

    body = f"""
<div class="hero">
  <h1>Flow Explorer</h1>
  <p>Every pipeline stage, line by line. Click a step to see what happens, the files involved,
     the command to run, its inputs and outputs, the Azure resources used, and what runs next.</p>
  <div class="linkrow" style="margin-bottom:6px">
    <button class="btn" onclick="expandAll()">Expand all</button>
    <button class="btn" onclick="collapseAll()">Collapse all</button>
    <a class="btn" href="/demo">Architecture</a>
    <a class="btn" href="/">Dashboard</a>
  </div>
</div>

{steps_html}

<footer>Run the whole thing locally with no Azure account &mdash; every Azure step skips gracefully.</footer>

<script>
function toggle(n) {{ document.getElementById('step'+n).classList.toggle('open'); }}
function expandAll() {{ document.querySelectorAll('.step').forEach(s=>s.classList.add('open')); }}
function collapseAll() {{ document.querySelectorAll('.step').forEach(s=>s.classList.remove('open')); }}
document.getElementById('step1').classList.add('open');
</script>"""
    return _page("MLOps Flow Explorer", "flow", body)
