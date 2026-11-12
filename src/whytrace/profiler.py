"""Token profiling, cost estimation, and efficiency diagnostics for agent execution traces."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .trace import Trace, load_traces

# Approximate pricing per 1M tokens ($)
PRICING_ESTIMATES = {
    "claude-sonnet": {"input": 3.0, "output": 15.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "default": {"input": 2.0, "output": 8.0},
}


def _estimate_tokens(text: str) -> int:
    """Fast character-to-token heuristic (~4 chars per token)."""
    return max(1, len(text.strip()) // 4)


@dataclass
class RedundantCall:
    step_idx: int
    tool: str
    content: str
    duplicate_of: int


@dataclass
class TraceProfile:
    task_id: str
    total_steps: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    redundant_calls: list[RedundantCall] = field(default_factory=list)
    has_loop: bool = False
    bloated_steps: list[int] = field(default_factory=list)


@dataclass
class EfficiencyReport:
    total_traces: int
    total_estimated_tokens: int
    total_estimated_cost_usd: float
    total_redundant_calls: int
    loop_detected_count: int
    per_trace: list[TraceProfile] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "total_traces": self.total_traces,
            "total_estimated_tokens": self.total_estimated_tokens,
            "total_estimated_cost_usd": round(self.total_estimated_cost_usd, 5),
            "total_redundant_calls": self.total_redundant_calls,
            "loop_detected_count": self.loop_detected_count,
        }


def profile_trace(trace: Trace, model_tier: str = "default") -> TraceProfile:
    """Analyze a single trace for token consumption, cost, and efficiency anomalies."""
    pricing = PRICING_ESTIMATES.get(model_tier, PRICING_ESTIMATES["default"])
    input_tokens = 0
    output_tokens = 0
    seen_calls: dict[str, int] = {}
    redundancies: list[RedundantCall] = []
    bloated_steps: list[int] = []

    # Detect loops or cyclic thoughts
    thoughts: list[str] = []

    for s in trace.steps:
        toks = _estimate_tokens(s.content)
        if s.kind in ("input", "tool_result"):
            input_tokens += toks
        else:
            output_tokens += toks

        # Flag unusually bloated single steps (> 1200 tokens)
        if toks > 1200:
            bloated_steps.append(s.idx)

        # Redundant tool call detection
        if s.kind == "tool_call":
            call_sig = s.content.strip().lower()
            if call_sig in seen_calls:
                redundancies.append(RedundantCall(
                    step_idx=s.idx,
                    tool=s.agent,
                    content=s.content[:80],
                    duplicate_of=seen_calls[call_sig],
                ))
            else:
                seen_calls[call_sig] = s.idx

        if s.kind == "thought":
            thoughts.append(s.content.strip()[:60])

    # Check for simple oscillation / loop
    has_loop = False
    if len(thoughts) >= 3:
        counts = Counter(thoughts)
        if any(cnt >= 2 for cnt in counts.values()):
            has_loop = True

    cost = (input_tokens / 1_000_000 * pricing["input"]) + (output_tokens / 1_000_000 * pricing["output"])

    return TraceProfile(
        task_id=trace.task_id,
        total_steps=len(trace.steps),
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=cost,
        redundant_calls=redundancies,
        has_loop=has_loop,
        bloated_steps=bloated_steps,
    )


def profile_run(run_dir_or_traces: str | Path | list[Trace], model_tier: str = "default") -> EfficiencyReport:
    """Analyze all traces in a directory or list of traces for cost and efficiency."""
    if isinstance(run_dir_or_traces, (str, Path)):
        p = Path(run_dir_or_traces)
        traces_file = p if p.is_file() else p / "traces.jsonl"
        traces = load_traces(traces_file)
    else:
        traces = run_dir_or_traces

    profiles = [profile_trace(t, model_tier=model_tier) for t in traces]
    total_tokens = sum(p.estimated_input_tokens + p.estimated_output_tokens for p in profiles)
    total_cost = sum(p.estimated_cost_usd for p in profiles)
    total_redundant = sum(len(p.redundant_calls) for p in profiles)
    total_loops = sum(bool(p.has_loop) for p in profiles)

    return EfficiencyReport(
        total_traces=len(profiles),
        total_estimated_tokens=total_tokens,
        total_estimated_cost_usd=total_cost,
        total_redundant_calls=total_redundant,
        loop_detected_count=total_loops,
        per_trace=profiles,
    )

