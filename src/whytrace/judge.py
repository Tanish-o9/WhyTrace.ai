"""LLM-as-judge with bias mitigations: binary verdicts, concurrent jury majority, order swapping."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .llm import LLM
from .taxonomy import FailureMode, parse_mode, taxonomy_prompt
from .trace import Trace

logger = logging.getLogger("whytrace.judge")


def extract_json(text: str) -> dict:
    """Pull the first valid JSON object out of an LLM reply (tolerates fences, prose, and internal braces). Returns {} on failure."""
    if not text:
        return {}
    # First, try direct json.loads after stripping markdown code fences
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        res = json.loads(cleaned)
        if isinstance(res, dict):
            return res
    except json.JSONDecodeError:
        pass

    # Find candidate JSON objects with string-aware brace matching
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            escape = False
            start = i
            for j in range(start, n):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\" and in_str:
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[start : j + 1]
                            try:
                                res = json.loads(candidate)
                                if isinstance(res, dict):
                                    return res
                            except json.JSONDecodeError:
                                pass
                            break
        i += 1
    return {}


class Jury:
    """N judge models (or N samples of one model); verdicts are aggregated by majority vote concurrently."""

    def __init__(self, judges: list[LLM], temperature: float = 0.0):
        if not judges:
            raise ValueError("a jury needs at least one judge")
        self.judges, self.temperature = judges, temperature

    def ask(self, system: str, user: str, max_tokens: int = 700) -> list[dict]:
        def _ask_judge(j: LLM) -> dict:
            try:
                msgs = [{"role": "user", "content": user}]
                reply = j.complete(system, msgs, self.temperature, max_tokens)
                vote = extract_json(reply)
                if not vote:  # retry once with format reminder if prose returned
                    msgs += [{"role": "assistant", "content": reply[:500]},
                             {"role": "user", "content": "Reply with ONLY the JSON object described, nothing else."}]
                    vote = extract_json(j.complete(system, msgs, self.temperature, max_tokens))
                return vote
            except Exception as e:
                logger.warning(f"[jury] {j.name} failed: {type(e).__name__}: {e}")
                return {"_error": str(e)}

        if len(self.judges) <= 1:
            return [_ask_judge(j) for j in self.judges]
        with ThreadPoolExecutor(max_workers=min(len(self.judges), 8)) as ex:
            return list(ex.map(_ask_judge, self.judges))


def majority(values: list) -> tuple[object | None, float]:
    """Most common non-None value and its vote share among the non-None votes."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, 0.0
    v, n = Counter(vals).most_common(1)[0]
    return v, n / len(vals)


@dataclass
class StepVerdict:
    step: int
    is_error: bool
    mode: FailureMode = FailureMode.OTHER
    rationale: str = ""
    agreement: float = 1.0
    votes: list[dict] = field(default_factory=list)


STEP_SYSTEM = (
    "You audit an AI agent's execution trace. A step is WRONG if its output is incorrect, unsupported, "
    "or unhelpful GIVEN only its input and the steps before it. Judge only the step you are asked about.\n"
    "Failure taxonomy:\n" + taxonomy_prompt() + "\n"
    'Reply with JSON only: {"error": true|false, "mode": "<taxonomy id or null>", "reason": "<one sentence>"}'
)


def judge_step(jury: Jury, trace: Trace, idx: int, context: str = "") -> StepVerdict:
    user = (f"Task: {trace.task}\n{('Reference: ' + context) if context else ''}\n\n"
            f"Trace so far:\n{trace.render(idx)}\n\nIs step {idx} wrong? Answer with the JSON object.")
    votes = jury.ask(STEP_SYSTEM, user)
    flags = [bool(v["error"]) if "error" in v else None for v in votes]
    is_err, share = majority(flags)
    mode, _ = majority([parse_mode(v.get("mode")) for v in votes if v.get("error")] or [None])
    reason = next((v.get("reason", "") for v in votes if bool(v.get("error")) == is_err and v.get("reason")), "")
    return StepVerdict(idx, bool(is_err), mode or FailureMode.OTHER, reason, share, votes)


PAIR_SYSTEM = ('You compare two candidate answers to a task. Reply with JSON only: {"winner": "A"|"B"|"tie"}')


def pairwise(jury: Jury, task: str, a: str, b: str) -> str:
    """Judge A-vs-B in both presentation orders; return 'a', 'b' or 'tie' (disagreement across orders = tie)."""
    def run(first: str, second: str) -> list[str | None]:
        u = f"Task: {task}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}"
        return [v.get("winner") for v in jury.ask(PAIR_SYSTEM, u)]

    fwd, _ = majority(run(a, b))
    rev, _ = majority(run(b, a))
    rev = {"A": "B", "B": "A"}.get(rev, rev)  # map back to the original labelling
    if fwd == rev and fwd in ("A", "B"):
        return fwd.lower()
    return "tie"
