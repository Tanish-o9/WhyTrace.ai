"""Quickstart: Instrument, evaluate, profile, and report a custom AI agent with whyTrace.ai."""
from pathlib import Path

from whytrace.integrations import trace_agent
from whytrace.profiler import profile_run
from whytrace.report import generate_html_report
from whytrace.trace import Trace, save_traces

captured_traces: list[Trace] = []


# 1. Instrument your agent with a single decorator
@trace_agent(output_traces=captured_traces)
def my_clinical_or_data_agent(question: str) -> str:
    """A simulated multi-step agent performing reasoning and tool execution."""
    print(f"Agent received task: {question}")
    if "creatinine" in question.lower():
        # Agent correctly solves this query
        return "1.25"
    # Agent simulates a calculation/retrieval failure
    return "unknown"


def main():
    run_dir = Path("runs/quickstart_demo")
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {"question": "What is the creatinine for patient Ava Chen?", "gold": "1.25"},
        {"question": "What is the mean sodium across measurements?", "gold": "142.50"},
    ]

    print("\n--- Running Custom Agent ---")
    for t in tasks:
        answer = my_clinical_or_data_agent(t["question"])
        # Compare against gold answer
        captured_traces[-1].gold = t["gold"]
        captured_traces[-1].success = (answer.strip() == t["gold"].strip())

    # 2. Persist execution traces
    traces_file = run_dir / "traces.jsonl"
    save_traces(captured_traces, traces_file)
    print(f"Traces saved to: {traces_file}")

    # 3. Profile Token Consumption & Efficiency
    print("\n--- Efficiency & Cost Profile ---")
    profile = profile_run(captured_traces, model_tier="gpt-4o")
    summary = profile.summary()
    print(f"Total Traces: {summary['total_traces']}")
    print(f"Estimated Tokens: {summary['total_estimated_tokens']}")
    print(f"Estimated Cost: ${summary['total_estimated_cost_usd']:.5f}")

    # 4. Generate Interactive Standalone HTML Report
    report_file = run_dir / "report.html"
    generate_html_report(run_dir, report_file)
    print(f"\nInteractive HTML Diagnostic Report generated at: {report_file}")
    print("Open this file in your browser to inspect the visual post-mortem!\n")


if __name__ == "__main__":
    main()

