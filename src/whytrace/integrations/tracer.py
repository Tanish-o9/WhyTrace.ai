"""Lightweight agent decorator and context manager to trace any Python agent callable."""
from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable

from ..trace import StepKind, Trace


class TraceContext:
    """Context manager for instrumenting arbitrary agent workflows into a whyTrace Trace object."""

    def __init__(self, task_id: str, task: str, gold: str | None = None):
        self.trace = Trace(task_id=task_id, task=task, gold=gold)
        self.trace.add("input", task, agent="user")

    def __enter__(self) -> TraceContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.step("tool_result", f"ERROR ({exc_type.__name__}): {exc_val}", agent="system")
            self.finish("", success=False, error=str(exc_val))

    def step(self, kind: StepKind, content: str, agent: str = "agent", **meta) -> None:
        """Record an intermediate execution step (thought, tool_call, tool_result, message)."""
        self.trace.add(kind, content, agent=agent, **meta)

    def finish(self, final_answer: str, success: bool | None = None, **meta) -> Trace:
        """Mark the final answer and finalize the trace."""
        self.trace.add("final", final_answer, agent="agent", **meta)
        self.trace.final_answer = final_answer
        if success is not None:
            self.trace.success = success
        elif self.trace.gold is not None:
            self.trace.success = str(final_answer).strip() == str(self.trace.gold).strip()
        return self.trace


def trace_agent(
    task_id_key: str = "task_id",
    task_key: str = "question",
    gold_key: str = "gold",
    output_traces: list[Trace] | None = None,
) -> Callable:
    """Decorator to wrap any function `run_agent(task, ...)` and automatically produce a whyTrace Trace.

    Usage:
        @trace_agent()
        def my_agent(question: str) -> str:
            # your agent logic here
            return "final answer"
    """
    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments

            # Extract task details or use sensible fallbacks
            task_id = str(params.get(task_id_key) or f"task_{int(time.time() * 1000)}")
            task = str(params.get(task_key) or (args[0] if args else "custom_task"))
            gold = str(params.get(gold_key)) if gold_key in params else None

            with TraceContext(task_id, task, gold=gold) as ctx:
                result = fn(*args, **kwargs)
                tr = ctx.finish(str(result))
                if output_traces is not None:
                    output_traces.append(tr)
                return result

        return wrapper
    return decorator

