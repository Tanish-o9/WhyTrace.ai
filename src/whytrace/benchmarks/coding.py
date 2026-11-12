"""Coding agent failure attribution and evaluation benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..trace import Trace


@dataclass
class CodingTask:
    id: str
    prompt: str
    entry_point: str
    test_code: str
    canonical_solution: str


SAMPLE_TASKS = [
    CodingTask(
        id="code_001",
        prompt="Write a function `two_sum(nums: list[int], target: int) -> list[int]` returning indices of the two numbers that add up to target.",
        entry_point="two_sum",
        test_code="assert two_sum([2, 7, 11, 15], 9) == [0, 1]\nassert two_sum([3, 2, 4], 6) == [1, 2]",
        canonical_solution="def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target - n], i]\n        seen[n] = i\n    return []",
    ),
    CodingTask(
        id="code_002",
        prompt="Write a function `is_palindrome(s: str) -> bool` ignoring non-alphanumeric characters and case.",
        entry_point="is_palindrome",
        test_code="assert is_palindrome('A man, a plan, a canal: Panama') is True\nassert is_palindrome('race a car') is False",
        canonical_solution="def is_palindrome(s):\n    filt = [c.lower() for c in s if c.isalnum()]\n    return filt == filt[::-1]",
    ),
]


def execute_test(code: str, test_code: str) -> tuple[bool, str]:
    """Execute code alongside test_code in an isolated namespace."""
    ns: dict = {}
    try:
        exec(code, ns)
        exec(test_code, ns)
        return True, "Passed"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def evaluate_coding_agent(
    agent_fn: Callable[[CodingTask], tuple[str, list[dict]]],
    tasks: list[CodingTask] | None = None,
) -> list[Trace]:
    """Evaluate a coding agent on code generation and attribution tasks.

    `agent_fn(task) -> (code_string, list_of_intermediate_steps)`
    """
    tasks = tasks or SAMPLE_TASKS
    traces: list[Trace] = []

    for task in tasks:
        trace = Trace(task_id=task.id, task=task.prompt, gold=task.canonical_solution)
        trace.add("input", task.prompt, agent="user")

        try:
            code_out, steps = agent_fn(task)
            for s in steps:
                trace.add(s.get("kind", "thought"), s.get("content", ""), agent=s.get("agent", "coder"))

            passed, msg = execute_test(code_out, task.test_code)
            trace.add("final", code_out, agent="coder", test_result=msg)
            trace.final_answer = code_out
            trace.success = passed
        except Exception as e:
            trace.add("final", f"ERROR: {e}", agent="coder")
            trace.final_answer = ""
            trace.success = False

        traces.append(trace)

    return traces

