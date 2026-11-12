# whyTrace.ai

Interpretability-first evaluation harness for agentic systems. It diagnoses *why* an agent failed instead of
only scoring it, stress-tests agents with adversarial prompt-injection, detects cost/token bloat, and gates regressions in CI.

| Piece | Module | What it does |
|---|---|---|
| Traces | `trace.py` | Structured per-step JSON traces (JSONL, OpenTelemetry format) |
| Taxonomy | `taxonomy.py` | 7 report categories + SQL-logic, medical-code lookup, injection-compliance |
| Judges | `judge.py` | Binary verdicts, concurrent multi-model jury majority vote, order-swapped pairwise |
| Attribution | `attribution.py` | "First wrong step" localization (all-at-once and step-by-step) |
| Counterfactual replay | `replay.py` | Patch a suspected step, re-run with transactional isolation, flip failures to build Causal Failure Graphs |
| Red-teaming | `attacks.py` | Indirect prompt injection with robust compliance vs refusal discrimination |
| Profiler | `profiler.py` | Detects redundant tool calls, cyclic loops, bloated steps, and token/USD costs |
| Reports | `report.py` | Generates self-contained interactive standalone HTML diagnostic reports |
| CI Bot | `ci_bot.py` | Formats GitHub PR comment markdown and reports directly to `$GITHUB_STEP_SUMMARY` |
| Integrations | `integrations/` | `@trace_agent` decorator, `TraceContext`, and `WhyTraceCallbackHandler` for LangChain |
| Benchmarks | `whowhen.py`, `benchmarks/` | Who&When ICML 2025 benchmark loader and coding agent testbed |
| Collector API | `server.py` | Lightweight REST server (`whytrace serve`) for remote trace ingestion |
| Regression | `regression.py` | Diagnosed failures become permanent test cases; `whytrace regress` exits 1 on regression |
| Dashboard | `dashboard/app.py` | Streamlit trace viewer with dynamic run selector and causal graph visualizer |

## Quick Start

```bash
pip install -e ".[dev]"
pytest -q                          # 18 tests, fully offline

# Run offline end-to-end demo
whytrace demo --out runs/demo

# Generate standalone interactive HTML report (no server needed)
whytrace report --run runs/demo --out runs/demo/report.html

# Analyze efficiency, duplicate calls, and token cost
whytrace profile --run runs/demo --tier default

# Check regression gate (fixed agent passes; buggy exits 1)
whytrace regress --path runs/demo/regressions.jsonl
whytrace regress --bug latest mean --path runs/demo/regressions.jsonl

# Generate GitHub PR comment summary markdown
whytrace comment --run runs/demo

# Adversarial prompt-injection evaluation
whytrace attack                    # template attacks vs baseline
whytrace attack --hardened         # hardened agent defense

# Optional Streamlit UI dashboard
pip install -e ".[dashboard]"
streamlit run dashboard/app.py -- runs/demo
```

## Tracing Your Own Agents

### 1. Decorator (`@trace_agent`)
```python
from whytrace.integrations import trace_agent

@trace_agent()
def my_autonomous_agent(question: str) -> str:
    # Any agent implementation (ReAct, function calling, custom loop)
    return "final answer"
```

### 2. Context Manager (`TraceContext`)
```python
from whytrace.integrations import TraceContext

with TraceContext(task_id="task_001", task="What is the stock price?") as ctx:
    ctx.step("thought", "Querying market data API...")
    ctx.step("tool_call", "get_quote('AAPL')", agent="finance_tool")
    ctx.step("tool_result", "$220.50", agent="finance_tool")
    trace = ctx.finish("$220.50", success=True)
```

### 3. LangChain / LangGraph Callback Handler
```python
from whytrace.integrations import WhyTraceCallbackHandler

handler = WhyTraceCallbackHandler(task_id="task_100", task="Explain gravity")
agent_executor.invoke({"input": "Explain gravity"}, config={"callbacks": [handler]})
trace = handler.get_trace()
```

## Running Real Model Jury
Evaluate real LLM agents or judges with Who&When or custom endpoints:
```bash
whytrace whowhen --subset algo --method step --judges anthropic:claude-sonnet-5-5,openai:gpt-4o,ollama:llama3.1:8b
```
Supports Anthropic, OpenAI, Ollama (`ollama:<model>`), vLLM, and any OpenAI-compatible base URL.
