"""Trace viewer + Causal Failure Graph. Run: streamlit run dashboard/app.py -- runs/demo"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

from whytrace.trace import load_traces

st.set_page_config(page_title="whyTrace.ai", page_icon="🔍", layout="wide")
st.title("whyTrace.ai — Agent Trace & Causal Failure Viewer")

# Sidebar for run selection
st.sidebar.header("Evaluation Runs")
default_run = sys.argv[-1] if len(sys.argv) > 1 and not sys.argv[-1].endswith(".py") else "runs/demo"

available_runs = [str(p) for p in Path("runs").glob("*") if p.is_dir()] if Path("runs").exists() else []
if default_run not in available_runs:
    available_runs.insert(0, default_run)

selected_run_str = st.sidebar.selectbox("Select Run Directory", available_runs, index=0)
run = Path(selected_run_str)

traces_path = run / "traces.jsonl"
graphs_path = run / "graphs.json"

if not traces_path.exists():
    st.warning(f"No traces found at `{traces_path}`. Run an evaluation or demo first (e.g. `whytrace demo --out {selected_run_str}`).")
    st.stop()

traces = load_traces(traces_path)
graphs = {g["task_id"]: g for g in json.loads(graphs_path.read_text())} if graphs_path.exists() else {}

ok = sum(bool(t.success) for t in traces)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Runs", len(traces))
c2.metric("Task Success Rate", f"{ok}/{len(traces)} ({ok / max(len(traces), 1):.1%})")
c3.metric("Failures", len(traces) - ok)
c4.metric("Root-Caused by Replay", sum(g.get("root_step") is not None for g in graphs.values()))

only_fail = st.checkbox("Show only failed runs", True)
pool = [t for t in traces if not (only_fail and t.success)]

if not pool:
    st.success("No failures in this run selection! All tasks succeeded.")
    st.stop()

t = st.selectbox("Select Task Run", pool, format_func=lambda tr: f"{'PASS' if tr.success else 'FAIL'}  |  {tr.task_id}")

left, right = st.columns([3, 2])
g = graphs.get(t.task_id)

with left:
    st.subheader("Step-by-Step Execution Trace")
    st.caption(f"**Task:** {t.task}")
    st.caption(f"**Expected Gold:** `{t.gold}`  |  **Agent Answer:** `{t.final_answer}`")

    for s in t.steps:
        root = g and s.idx == g.get("root_step")
        with st.container(border=True):
            badge = "🚨 **ROOT CAUSE OF FAILURE**" if root else f"**Step {s.idx}**"
            st.markdown(f"{badge} — `{s.agent}/{s.kind}`")
            lang = "sql" if s.kind == "tool_call" or "sql" in s.content.lower() else None
            st.code(s.content, language=lang)

with right:
    if g:
        st.subheader("Causal Failure Graph")
        try:
            st.graphviz_chart(g["dot"])
        except Exception as e:
            st.info("Graphviz visualizer unavailable on system. Displaying graph nodes:")
            st.json({"root_step": g.get("root_step"), "nodes": g.get("nodes"), "edges": g.get("edges")})

        st.subheader("Counterfactual Interventions")
        if g.get("interventions"):
            st.table([
                {
                    "Step": i["step"],
                    "Resolved Failure?": "✅ Yes" if i["fixed"] else "❌ No",
                    "New Answer": i.get("new_answer") or "(none)",
                    "Patched With": i.get("patched_with", "")[:60]
                }
                for i in g["interventions"]
            ])
        else:
            st.caption("No counterfactual interventions recorded for this task.")
    else:
        st.info("No causal graph generated for this task.")
