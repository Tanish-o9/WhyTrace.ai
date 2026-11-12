"""Self-contained interactive HTML report generator for whyTrace.ai evaluation runs."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .trace import Trace, load_traces


def generate_html_report(run_dir: str | Path, output_path: str | Path | None = None) -> str:
    """Read traces.jsonl and graphs.json from `run_dir` and generate a self-contained interactive HTML report."""
    run_path = Path(run_dir)
    traces_file = run_path / "traces.jsonl"
    graphs_file = run_path / "graphs.json"

    if not traces_file.exists():
        raise FileNotFoundError(f"No traces.jsonl found in {run_path}")

    traces = load_traces(traces_file)
    graphs: dict[str, dict[str, Any]] = {}
    if graphs_file.exists():
        try:
            graphs = {g["task_id"]: g for g in json.loads(graphs_file.read_text(encoding="utf-8"))}
        except Exception:
            graphs = {}

    total_runs = len(traces)
    successes = sum(bool(t.success) for t in traces)
    failures = total_runs - successes
    pass_rate = (successes / total_runs * 100) if total_runs else 0.0
    root_caused = sum(g.get("root_step") is not None for g in graphs.values())
    attacked_count = sum("attack_success" in t.meta for t in traces)
    attack_successes = sum(bool(t.meta.get("attack_success")) for t in traces)

    tasks_data = []
    for t in traces:
        g = graphs.get(t.task_id, {})
        tasks_data.append({
            "task_id": t.task_id,
            "task": t.task,
            "gold": t.gold,
            "final_answer": t.final_answer,
            "success": bool(t.success),
            "root_step": g.get("root_step"),
            "interventions": g.get("interventions", []),
            "steps": [
                {
                    "idx": s.idx,
                    "agent": s.agent,
                    "kind": s.kind,
                    "content": s.content,
                    "is_root": (g.get("root_step") == s.idx),
                    "meta": s.meta,
                }
                for s in t.steps
            ],
            "meta": t.meta,
        })

    tasks_json = json.dumps(tasks_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>whyTrace.ai — Diagnostic Evaluation Report</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --text-muted: #8b949e;
      --primary: #58a6ff;
      --success: #238636;
      --danger: #da3633;
      --warning: #d29922;
      --code-bg: #0b0e14;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }}
    .header h1 {{ font-size: 24px; color: var(--primary); display: flex; align-items: center; gap: 8px; }}
    .header .tag {{ background: #1f6feb22; color: var(--primary); padding: 4px 10px; border-radius: 12px; font-size: 13px; border: 1px solid #1f6feb44; }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metric-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric-card .title {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }}
    .metric-card .val {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    .metric-card .sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}
    .val.success {{ color: #3fb950; }}
    .val.danger {{ color: #f85149; }}
    .val.primary {{ color: var(--primary); }}

    .main-layout {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
    }}
    @media(max-width: 900px) {{
      .main-layout {{ grid-template-columns: 1fr; }}
    }}
    .task-list-panel {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      max-height: 80vh;
      overflow-y: auto;
    }}
    .task-filter {{
      margin-bottom: 12px;
      display: flex;
      gap: 8px;
    }}
    .task-filter input {{
      width: 100%;
      background: var(--code-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 13px;
    }}
    .task-item {{
      padding: 10px 12px;
      border-radius: 6px;
      border: 1px solid transparent;
      margin-bottom: 6px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: background 0.15s;
    }}
    .task-item:hover {{ background: #21262d; }}
    .task-item.active {{ background: #1f6feb33; border-color: var(--primary); }}
    .badge {{
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 10px;
      text-transform: uppercase;
    }}
    .badge.pass {{ background: #23863633; color: #3fb950; border: 1px solid #23863666; }}
    .badge.fail {{ background: #da363333; color: #f85149; border: 1px solid #da363366; }}
    .badge.root {{ background: #d2992233; color: #e3b341; border: 1px solid #d2992266; }}

    .detail-panel {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
    }}
    .detail-header {{
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 20px;
    }}
    .detail-header h2 {{ font-size: 18px; margin-bottom: 8px; }}
    .kv-row {{ display: flex; gap: 24px; font-size: 13px; margin-top: 6px; color: var(--text-muted); }}
    .kv-row span b {{ color: var(--text); }}

    .step-card {{
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      margin-bottom: 12px;
      overflow: hidden;
    }}
    .step-card.is-root {{
      border-color: #f85149;
      box-shadow: 0 0 10px #f8514933;
    }}
    .step-head {{
      padding: 8px 12px;
      background: #161b22;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 600;
    }}
    .step-card.is-root .step-head {{
      background: #da363322;
      color: #ff7b72;
    }}
    .step-content {{
      padding: 12px;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-all;
    }}

    .interventions-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 13px;
    }}
    .interventions-table th, .interventions-table td {{
      padding: 8px 12px;
      border: 1px solid var(--border);
      text-align: left;
    }}
    .interventions-table th {{ background: var(--code-bg); color: var(--text-muted); }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>whyTrace.ai <span class="tag">Diagnostic Report</span></h1>
      <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">Run Path: <code>{html.escape(str(run_path))}</code></p>
    </div>
  </div>

  <div class="metrics-grid">
    <div class="metric-card">
      <div class="title">Total Evaluated Tasks</div>
      <div class="val primary">{total_runs}</div>
      <div class="sub">Machine-checkable executions</div>
    </div>
    <div class="metric-card">
      <div class="title">Task Pass Rate</div>
      <div class="val {'success' if pass_rate >= 80 else ('danger' if pass_rate < 50 else 'primary')}">{pass_rate:.1f}%</div>
      <div class="sub">{successes} succeeded / {failures} failed</div>
    </div>
    <div class="metric-card">
      <div class="title">Root-Caused by Replay</div>
      <div class="val primary">{root_caused}</div>
      <div class="sub">{round(root_caused / max(failures, 1) * 100)}% of failures localized</div>
    </div>
    <div class="metric-card">
      <div class="title">Attack Success Rate (ASR)</div>
      <div class="val {'danger' if attack_successes > 0 else 'success'}">{((attack_successes / max(attacked_count, 1)) * 100) if attacked_count else 0:.1f}%</div>
      <div class="sub">{attack_successes} successful / {attacked_count} tested</div>
    </div>
  </div>

  <div class="main-layout">
    <div class="task-list-panel">
      <div class="task-filter">
        <input type="text" id="taskSearch" placeholder="Filter tasks..." oninput="renderTaskList()">
      </div>
      <div id="taskList"></div>
    </div>

    <div class="detail-panel" id="detailPanel">
      <p style="color: var(--text-muted);">Select a task on the left to inspect its causal trajectory.</p>
    </div>
  </div>

  <script>
    const tasks = {tasks_json};
    let selectedTaskId = tasks.length > 0 ? tasks[0].task_id : null;

    function renderTaskList() {{
      const filter = document.getElementById("taskSearch").value.toLowerCase();
      const listEl = document.getElementById("taskList");
      listEl.innerHTML = "";

      tasks.filter(t => t.task_id.toLowerCase().includes(filter) || t.task.toLowerCase().includes(filter)).forEach(t => {{
        const item = document.createElement("div");
        item.className = "task-item" + (t.task_id === selectedTaskId ? " active" : "");
        item.onclick = () => selectTask(t.task_id);

        const label = document.createElement("span");
        label.style.fontSize = "13px";
        label.style.fontWeight = "500";
        label.innerText = t.task_id;

        const badge = document.createElement("span");
        badge.className = "badge " + (t.success ? "pass" : "fail");
        badge.innerText = t.success ? "PASS" : "FAIL";

        item.appendChild(label);
        item.appendChild(badge);
        listEl.appendChild(item);
      }});
    }}

    function selectTask(taskId) {{
      selectedTaskId = taskId;
      renderTaskList();
      const t = tasks.find(x => x.task_id === taskId);
      if (!t) return;

      const panel = document.getElementById("detailPanel");
      let stepsHtml = "";
      t.steps.forEach(s => {{
        const rootBadge = s.is_root ? '<span class="badge root">🚨 ROOT CAUSE</span>' : '';
        stepsHtml += `
          <div class="step-card ${{s.is_root ? 'is-root' : ''}}">
            <div class="step-head">
              <span>Step ${{s.idx}} — ${{s.agent}} / <code>${{s.kind}}</code></span>
              ${{rootBadge}}
            </div>
            <div class="step-content">${{escapeHtml(s.content)}}</div>
          </div>
        `;
      }});

      let interventionsHtml = "";
      if (t.interventions && t.interventions.length > 0) {{
        let rows = "";
        t.interventions.forEach(i => {{
          rows += `
            <tr>
              <td>Step ${{i.step}}</td>
              <td style="color: ${{i.fixed ? '#3fb950' : '#f85149'}}">${{i.fixed ? '✅ Resolved' : '❌ Still Failed'}}</td>
              <td><code>${{escapeHtml(i.new_answer || '(none)')}}</code></td>
              <td><code>${{escapeHtml(i.patched_with || '')}}</code></td>
            </tr>
          `;
        }});
        interventionsHtml = `
          <h3 style="font-size: 15px; margin-top: 24px; margin-bottom: 8px;">Counterfactual Replay Interventions</h3>
          <table class="interventions-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Failure Resolved?</th>
                <th>New Answer</th>
                <th>Patched Value</th>
              </tr>
            </thead>
            <tbody>${{rows}}</tbody>
          </table>
        `;
      }}

      panel.innerHTML = `
        <div class="detail-header">
          <h2>${{t.task_id}} <span class="badge ${{t.success ? 'pass' : 'fail'}}">${{t.success ? 'PASS' : 'FAIL'}}</span></h2>
          <p style="font-size: 14px; margin-top: 4px;">${{escapeHtml(t.task)}}</p>
          <div class="kv-row">
            <span>Expected Gold: <b>${{escapeHtml(t.gold || '(none)')}}</b></span>
            <span>Agent Answer: <b>${{escapeHtml(t.final_answer || '(none)')}}</b></span>
            ${{t.root_step !== null && t.root_step !== undefined ? `<span>Diagnosed Root Cause: <b style="color: #f85149;">Step ${{t.root_step}}</b></span>` : ''}}
          </div>
        </div>

        <h3 style="font-size: 15px; margin-bottom: 12px;">Execution Trace</h3>
        ${{stepsHtml}}

        ${{interventionsHtml}}
      `;
    }}

    function escapeHtml(str) {{
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }}

    renderTaskList();
    if (selectedTaskId) selectTask(selectedTaskId);
  </script>
</body>
</html>
"""

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_content, encoding="utf-8")

    return html_content

