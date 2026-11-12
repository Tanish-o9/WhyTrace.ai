"""Example demonstrating LangChain / LangGraph callback auditing with whyTrace.ai."""
from pathlib import Path

from whytrace.integrations import WhyTraceCallbackHandler
from whytrace.report import generate_html_report
from whytrace.trace import save_traces


def simulate_langchain_agent_run(question: str, handler: WhyTraceCallbackHandler) -> str:
    """Simulates how LangChain chains invoke callbacks during execution."""
    # 1. Chain starts
    handler.on_chain_start({"name": "AgentExecutor"}, {"input": question})

    # 2. Agent thinks
    handler.on_llm_start({"name": "ChatOpenAI"}, [f"Thought: Need to search database for {question}"])

    # 3. Agent calls a tool
    handler.on_tool_start({"name": "database_search"}, f"SELECT info FROM records WHERE query='{question}'")

    # 4. Tool returns result
    tool_output = "Record 42: Verified temperature 98.6 F"
    handler.on_tool_end(tool_output)

    # 5. Agent produces final output
    final_answer = "98.6 F"
    handler.on_chain_end({"output": final_answer})
    return final_answer


def main():
    run_dir = Path("runs/langchain_demo")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate whyTrace callback handler
    handler = WhyTraceCallbackHandler(
        task_id="lc_task_001",
        task="What is the patient temperature?",
        gold="98.6 F",
    )

    print("Running simulated LangChain agent with WhyTraceCallbackHandler...")
    answer = simulate_langchain_agent_run("What is the patient temperature?", handler)
    print(f"Agent finished with answer: {answer}")

    trace = handler.get_trace()
    save_traces([trace], run_dir / "traces.jsonl")

    # Generate interactive report
    report_file = run_dir / "report.html"
    generate_html_report(run_dir, report_file)
    print(f"Diagnostic report written to: {report_file}")


if __name__ == "__main__":
    main()

