"""GitHub PR Commenter and CI/CD Summary Generator for whyTrace.ai."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .trace import load_traces


def generate_pr_comment(run_dir: str | Path, regressions_file: str | Path | None = None) -> str:
    """Generate a GitHub-flavored Markdown comment for pull request status updates."""
    p = Path(run_dir)
    traces_file = p / "traces.jsonl"
    graphs_file = p / "graphs.json"

    if not traces_file.exists():
        return "⚠️ **whyTrace.ai**: No execution traces found."

    traces = load_traces(traces_file)
    graphs: dict[str, dict[str, Any]] = {}
    if graphs_file.exists():
        try:
            graphs = {g["task_id"]: g for g in json.loads(graphs_file.read_text(encoding="utf-8"))}
        except Exception:
            graphs = {}

    total = len(traces)
    ok = sum(bool(t.success) for t in traces)
    failed = total - ok
    rate = (ok / total * 100) if total else 0.0

    reg_passed = 0
    reg_total = 0
    reg_file = Path(regressions_file) if regressions_file else p / "regressions.jsonl"
    if reg_file.exists():
        try:
            cases = [json.loads(line) for line in reg_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            reg_total = len(cases)
            reg_passed = sum(bool(c.get("passed", True)) for c in cases)
        except Exception:
            pass

    status_emoji = "✅" if failed == 0 else "🚨"

    lines = [
        f"## {status_emoji} whyTrace.ai Diagnostic Audit Summary",
        "",
        "| Metric | Result | Status |",
        "|---|---|---|",
        f"| **Task Success Rate** | `{ok}/{total}` ({rate:.1f}%) | {'🟢 Pass' if rate >= 80 else ('🔴 Fail' if rate < 50 else '🟡 Warning')} |",
        f"| **Failures Detected** | `{failed}` | {'🟢 None' if failed == 0 else '🔴 Failures Found'} |",
        f"| **Root-Caused by Replay** | `{sum(g.get('root_step') is not None for g in graphs.values())}` | 🔍 Localized |",
    ]

    if reg_total > 0:
        lines.append(f"| **Regression Gate** | `{reg_passed}/{reg_total}` pass | {'🟢 Clean' if reg_passed == reg_total else '🔴 Regressed'} |")

    lines.append("")

    if failed > 0:
        lines.append("### 🔍 Diagnosed Failure Breakdown")
        lines.append("")
        lines.append("| Task ID | Question | Agent Answer | Gold | Root Cause Step |")
        lines.append("|---|---|---|---|---|")
        for t in traces:
            if not t.success:
                g = graphs.get(t.task_id, {})
                root = f"Step {g.get('root_step')}" if g.get("root_step") is not None else "(Attribution Pending)"
                clean_q = t.task.replace("|", "\\|")[:50]
                lines.append(f"| `{t.task_id}` | {clean_q} | `{t.final_answer or 'none'}` | `{t.gold or 'none'}` | **{root}** |")
        lines.append("")

    lines.append("> *Report generated automatically by [whyTrace.ai](https://github.com/Tanish-o9/WhyTrace.ai)*")
    return "\n".join(lines)


def write_to_step_summary_if_in_ci(markdown: str) -> None:
    """If running in GitHub Actions, append markdown to GITHUB_STEP_SUMMARY."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path and Path(summary_path).exists():
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + markdown + "\n")
