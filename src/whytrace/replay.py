"""Counterfactual replay: find the root cause of a failure by intervention, not by asking a judge.

For each candidate step (earliest first) we patch that step's output with a reference or LLM-generated counterfactual value,
re-run the agent from there with transactional isolation, and check whether the failure disappears.
The resulting Causal Failure Graph records every intervention as causal evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .agent import ReActAgent
from .ehr import EHR, Task
from .llm import LLM
from .trace import Trace

Patcher = Callable[[Trace, int], str | None]


@dataclass
class Intervention:
    step: int
    patched_with: str
    fixed: bool
    new_answer: str | None


@dataclass
class CausalFailureGraph:
    task_id: str
    nodes: list[dict]
    edges: list[tuple[int, int, str]]
    root_step: int | None
    interventions: list[Intervention] = field(default_factory=list)

    def to_dot(self) -> str:
        lines = ["digraph G {", "  rankdir=LR; node [shape=box, fontsize=10];"]
        for n in self.nodes:
            color = "tomato" if n["idx"] == self.root_step else "white"
            label = f'{n["idx"]} {n["agent"]}/{n["kind"]}\\n{n["preview"]}'.replace('"', "'")
            lines.append(f'  s{n["idx"]} [label="{label}", style=filled, fillcolor={color}];')
        for a, b, kind in self.edges:
            style = "bold" if kind == "data" else "dashed"
            lines.append(f"  s{a} -> s{b} [style={style}, label={kind}];")
        return "\n".join(lines + ["}"])

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "root_step": self.root_step,
            "interventions": [i.__dict__ for i in self.interventions],
        }


def reference_patcher(ehr: EHR, task: Task) -> Patcher:
    """Reference repair per step kind, derived from the task's gold reference SQL."""
    def patch(trace: Trace, idx: int) -> str | None:
        kind = trace.steps[idx].kind
        if kind == "tool_call":
            return task.reference_sql
        if kind == "tool_result":
            return ehr.query(task.reference_sql)
        if kind == "final":
            return task.gold
        return None
    return patch


def llm_counterfactual_patcher(llm: LLM, task: Task) -> Patcher:
    """LLM-guided counterfactual generator for open-ended agents when no pre-defined reference action exists."""
    def patch(trace: Trace, idx: int) -> str | None:
        target_step = trace.steps[idx]
        history = trace.render(upto=idx - 1)
        system = (
            "You are an expert debugger generating a minimal counterfactual repair for an AI agent's step.\n"
            f"Task: {task.question}\n"
            f"Expected gold answer: {task.gold}\n"
            "Given the history prior to this step, output ONLY the corrected content that replaces this step's output."
        )
        user = (
            f"History:\n{history}\n\n"
            f"Step {idx} ({target_step.kind}) produced:\n{target_step.content}\n\n"
            f"Provide the corrected {target_step.kind} content only (JSON or text), no explanations."
        )
        try:
            return llm.complete(system, [{"role": "user", "content": user}], 0.0, 300).strip()
        except Exception:
            return None
    return patch


def build_graph(trace: Trace, root: int | None, interventions: list[Intervention]) -> CausalFailureGraph:
    nodes = [
        {"idx": s.idx, "agent": s.agent, "kind": s.kind, "preview": s.content[:60].replace("\n", " ")}
        for s in trace.steps
    ]
    edges: list[tuple[int, int, str]] = [(i, i + 1, "seq") for i in range(len(trace.steps) - 1)]
    for s in trace.steps:  # data-flow: a tool_result feeds every later agent step that consumes it
        if s.kind == "tool_result":
            edges += [(s.idx, o.idx, "data") for o in trace.steps[s.idx + 1:] if o.kind in ("tool_call", "final")]
    return CausalFailureGraph(trace.task_id, nodes, edges, root, interventions)


def find_root_cause(
    agent: ReActAgent,
    task: Task,
    trace: Trace,
    patcher: Patcher | None = None,
) -> CausalFailureGraph:
    """Replay each candidate step with a counterfactual patch under transactional isolation."""
    patcher = patcher or reference_patcher(agent.ehr, task)
    interventions: list[Intervention] = []
    root = None

    if trace.success:
        return build_graph(trace, None, [])

    for s in trace.steps:
        if s.kind not in ("tool_call", "tool_result", "final"):
            continue
        patch = patcher(trace, s.idx)
        if patch is None:
            continue

        # Transactional isolation: create savepoint before replay if environment supports it
        sp_name = f"replay_{s.idx}"
        if getattr(agent, "ehr", None) and hasattr(agent.ehr, "savepoint"):
            agent.ehr.savepoint(sp_name)

        try:
            new = agent.replay(trace, task, s.idx, patch)
        finally:
            if getattr(agent, "ehr", None) and hasattr(agent.ehr, "rollback_to"):
                agent.ehr.rollback_to(sp_name)

        interventions.append(Intervention(s.idx, patch[:200], bool(new.success), new.final_answer))
        if new.success:
            root = s.idx
            break

    return build_graph(trace, root, interventions)
