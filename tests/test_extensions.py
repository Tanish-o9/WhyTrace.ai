import json
from pathlib import Path

from whytrace.benchmarks.coding import CodingTask, evaluate_coding_agent
from whytrace.ci_bot import generate_pr_comment
from whytrace.integrations import TraceContext, WhyTraceCallbackHandler, trace_agent
from whytrace.profiler import profile_trace
from whytrace.report import generate_html_report
from whytrace.trace import Trace


def test_trace_context_and_decorator():
    # Context manager test
    with TraceContext("t1", "what is 2+2?", gold="4") as ctx:
        ctx.step("thought", "calculating...")
        ctx.step("tool_call", "add(2, 2)")
        ctx.step("tool_result", "4")
        tr = ctx.finish("4")

    assert tr.success is True
    assert len(tr.steps) == 5  # input, thought, tool_call, tool_result, final

    # Decorator test
    captured = []

    @trace_agent(output_traces=captured)
    def simple_agent(question: str) -> str:
        return "solved"

    ans = simple_agent("my question")
    assert ans == "solved"
    assert len(captured) == 1
    assert captured[0].final_answer == "solved"


def test_langchain_callback_handler():
    handler = WhyTraceCallbackHandler(task_id="lc_1", task="calculate", gold="42")
    handler.on_tool_start({"name": "calculator"}, "21 * 2")
    handler.on_tool_end("42")
    handler.on_chain_end({"output": "42"})

    tr = handler.get_trace()
    assert tr.task_id == "lc_1"
    assert tr.final_answer == "42"
    assert tr.success is True
    assert any(s.kind == "tool_call" for s in tr.steps)


def test_profiler_detects_redundancies_and_estimates_cost():
    t = Trace(task_id="p1", task="search")
    t.add("input", "search query")
    t.add("tool_call", "search('apple')")
    t.add("tool_result", "result 1")
    t.add("tool_call", "search('apple')")  # Redundant duplicate!
    t.add("tool_result", "result 1")
    t.add("final", "apple is fruit")

    prof = profile_trace(t)
    assert len(prof.redundant_calls) == 1
    assert prof.redundant_calls[0].duplicate_of == 1
    assert prof.estimated_cost_usd > 0


def test_report_and_ci_bot_generation(tmp_path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()

    t = Trace(task_id="task_x", task="test task", gold="10", final_answer="10", success=True)
    t.add("input", "test task")
    t.add("final", "10")

    (run_dir / "traces.jsonl").write_text(t.model_dump_json() + "\n", encoding="utf-8")
    (run_dir / "graphs.json").write_text(json.dumps([{"task_id": "task_x", "root_step": None, "interventions": []}]), encoding="utf-8")

    html_file = run_dir / "report.html"
    generate_html_report(run_dir, html_file)
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")
    assert "whyTrace.ai" in content
    assert "task_x" in content

    comment = generate_pr_comment(run_dir)
    assert "whyTrace.ai Diagnostic Audit Summary" in comment
    assert "task_x" not in comment or "100.0%" in comment


def test_coding_benchmark():
    def dummy_coder(task: CodingTask):
        if task.id == "code_002":
            return "def is_palindrome(s):\n    filt = [c.lower() for c in s if c.isalnum()]\n    return filt == filt[::-1]", []
        return "def two_sum(n, t): return [0, 1]", []

    traces = evaluate_coding_agent(dummy_coder)
    assert len(traces) == 2
    assert any(t.success is True for t in traces)

