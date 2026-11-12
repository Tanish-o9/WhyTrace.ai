"""LangChain callback handler to stream agent execution directly into whyTrace Trace objects."""
from __future__ import annotations

import json
from typing import Any

from ..trace import Trace

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    class BaseCallbackHandler:  # type: ignore
        """Fallback base class when langchain is not installed."""
        pass


class WhyTraceCallbackHandler(BaseCallbackHandler):
    """Callback handler for LangChain and LangGraph workflows to generate structured whyTrace traces.

    Usage:
        handler = WhyTraceCallbackHandler(task_id="langchain_001", task="Find weather in Paris")
        agent_executor.invoke({"input": "Find weather in Paris"}, config={"callbacks": [handler]})
        trace = handler.get_trace()
    """

    def __init__(self, task_id: str = "lc_task", task: str = "", gold: str | None = None):
        super().__init__()
        self.trace = Trace(task_id=task_id, task=task, gold=gold)
        if task:
            self.trace.add("input", task, agent="user")

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs) -> None:
        if not self.trace.steps:
            inp_text = str(inputs.get("input", inputs))
            if not self.trace.task:
                self.trace.task = inp_text
            self.trace.add("input", inp_text, agent="user")

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "tool") if serialized else "tool"
        self.trace.add("tool_call", str(input_str), agent=tool_name, tool=tool_name)

    def on_tool_end(self, output: str, **kwargs) -> None:
        self.trace.add("tool_result", str(output), agent="tool_result")

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs) -> None:
        if prompts:
            self.trace.add("thought", prompts[0][:400], agent="agent")

    def on_llm_end(self, response: Any, **kwargs) -> None:
        pass

    def on_chain_error(self, error: BaseException, **kwargs) -> None:
        self.trace.add("tool_result", f"ERROR: {error}", agent="system")
        self.trace.success = False

    def on_chain_end(self, outputs: dict[str, Any], **kwargs) -> None:
        ans = outputs.get("output", str(outputs)) if isinstance(outputs, dict) else str(outputs)
        self.trace.add("final", str(ans), agent="agent")
        self.trace.final_answer = str(ans)
        if self.trace.gold is not None:
            self.trace.success = str(ans).strip().lower() == str(self.trace.gold).strip().lower()

    def get_trace(self) -> Trace:
        return self.trace
