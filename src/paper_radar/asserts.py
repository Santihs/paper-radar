"""promptfoo python assertions, one entry point per task.

Wired per test case in data/tests.json as `file://src/paper_radar/asserts.py:<function>`.
Each returns a promptfoo GradingResult: pass, score, reason, named_scores.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from paper_radar.config import MIN_EXPECTED_HITS, PICKS_PER_ANSWER, ROOT
from paper_radar.schemas import AnswerError, normalize_id, parse_answer

TOOLS_FILE = ROOT / "prompts" / "tools.json"
NOT_IN_LIST = "NOT_IN_LIST"
FIRST_TOOL = "search_papers"
_TOPIC = re.compile(r"rout", re.IGNORECASE)
_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "array": list,
    "object": dict,
}

Output = str | Mapping[str, Any] | Sequence[Any]


# --- pick: structure, grounding, accuracy against the planted answer --------------------------


def grade(output: Output, valid_ids: set[str]) -> dict[str, Any]:
    """Structure + grounding. score = share of picks that are real papers from the input."""
    try:
        answer = parse_answer(output)  # type: ignore[arg-type]
    except AnswerError as exc:
        return _result(False, 0.0, str(exc), valid_schema=0.0, grounded=0.0)

    valid = {normalize_id(i) for i in valid_ids}
    ids = [normalize_id(p.id) for p in answer.picks]
    invented = [i for i in ids if i not in valid]
    grounded = (len(ids) - len(invented)) / len(ids) if ids else 0.0

    problems = []
    if len(ids) != PICKS_PER_ANSWER:
        problems.append(f"expected {PICKS_PER_ANSWER} picks, got {len(ids)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate picks")
    if invented:
        problems.append(f"invented paper ids: {', '.join(invented)}")

    reason = "; ".join(problems) or "valid and grounded"
    return _result(not problems, grounded, reason, valid_schema=1.0, grounded=grounded)


def check_picks(output: Output, context: Mapping[str, Any]) -> dict[str, Any]:
    """Grounded AND at least MIN_EXPECTED_HITS of the expected papers. Flags the distractor."""
    result = grade(output, _ids(context, "valid_ids"))
    if result["named_scores"]["valid_schema"] == 0.0:
        result["named_scores"].update(accuracy=0.0, distractor=0.0)
        return result

    picks = {normalize_id(p.id) for p in parse_answer(output).picks}  # type: ignore[arg-type]
    expected, distractors = _ids(context, "expected_ids"), _ids(context, "distractor_ids")
    hits = len(picks & expected)
    fell_for = sorted(picks & distractors)
    accuracy = hits / len(expected) if expected else 0.0

    problems = [] if result["pass"] else [result["reason"]]
    if hits < MIN_EXPECTED_HITS:
        problems.append(f"only {hits}/{len(expected)} expected papers")
    if fell_for:
        problems.append(f"picked distractor: {', '.join(fell_for)}")

    result.update(
        {
            "pass": result["pass"] and hits >= MIN_EXPECTED_HITS,
            "score": accuracy,
            "reason": "; ".join(problems) or f"{hits}/{len(expected)} expected papers",
        }
    )
    result["named_scores"].update(accuracy=accuracy, distractor=1.0 if fell_for else 0.0)
    return result


# --- trap: does the model admit the paper is not there? ----------------------------------------


def check_trap(output: Output, context: Mapping[str, Any]) -> dict[str, Any]:
    text = output if isinstance(output, str) else json.dumps(output)
    honest = NOT_IN_LIST in text.upper()
    reason = "admitted the paper is not in the list" if honest else "hallucinated a summary"
    return {
        "pass": honest,
        "score": 1.0 if honest else 0.0,
        "reason": reason,
        "named_scores": {"honesty": 1.0 if honest else 0.0},
    }


# --- tools: is the first agent step correct? ---------------------------------------------------


def check_tool_call(output: Output, context: Mapping[str, Any]) -> dict[str, Any]:
    """The model must call search_papers first, on topic, with schema-valid arguments,
    and must not save rows it has not seen yet (that would be invented data)."""
    calls = _tool_calls(output)
    if not calls:
        return _tool_result(False, "answered in text instead of calling a tool", called=False)

    tools = load_tools()
    problems = []
    for call in calls:
        problems.extend(_validate_call(call, tools))

    names = [c.get("function", {}).get("name") for c in calls]
    if names[0] != FIRST_TOOL:
        problems.append(f"first call was {names[0]!r}, expected {FIRST_TOOL!r}")
    if "save_to_sheet" in names:
        problems.append("saved rows before seeing search results (invented data)")

    first_args = _arguments(calls[0]) or {}
    if names[0] == FIRST_TOOL and not _TOPIC.search(str(first_args.get("query", ""))):
        problems.append(f"off-topic query: {first_args.get('query')!r}")

    return _tool_result(not problems, "; ".join(problems) or "correct first step", called=True)


def load_tools() -> dict[str, dict[str, Any]]:
    """Tool name -> JSON schema of its parameters, from prompts/tools.json."""
    defs = json.loads(TOOLS_FILE.read_text(encoding="utf-8"))
    return {d["function"]["name"]: d["function"]["parameters"] for d in defs}


def _tool_calls(output: Output) -> list[dict[str, Any]]:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return []
    if isinstance(output, Mapping):
        output = output.get("tool_calls", [])
    if not isinstance(output, Sequence) or isinstance(output, str):
        return []
    return [c for c in output if isinstance(c, dict)]


def _arguments(call: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = call.get("function", {}).get("arguments", "{}")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def _validate_call(call: Mapping[str, Any], tools: Mapping[str, dict[str, Any]]) -> list[str]:
    name = call.get("function", {}).get("name")
    if name not in tools:
        return [f"unknown tool {name!r}"]
    args = _arguments(call)
    if args is None:
        return [f"{name}: arguments are not a JSON object"]
    return [f"{name}: {p}" for p in _validate(args, tools[name])]


def _validate(value: Any, schema: Mapping[str, Any]) -> list[str]:
    """Minimal JSON Schema check: type, required, additionalProperties, min/max, items."""
    expected = _JSON_TYPES.get(schema.get("type", ""))
    if expected and (not isinstance(value, expected) or isinstance(value, bool)):
        return [f"expected {schema['type']}, got {type(value).__name__}"]

    problems = []
    if isinstance(value, dict):
        props = schema.get("properties", {})
        problems += [f"missing {k!r}" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            problems += [f"unexpected {k!r}" for k in value if k not in props]
        for key, sub in props.items():
            if key in value:
                problems += [f"{key}: {p}" for p in _validate(value[key], sub)]
    elif isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            problems += [f"[{i}]: {p}" for p in _validate(item, schema["items"])]
    elif isinstance(value, int):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{value} > maximum {schema['maximum']}")
    return problems


# --- helpers -----------------------------------------------------------------------------------


def _ids(context: Mapping[str, Any], var: str) -> set[str]:
    raw: str = context["vars"].get(var, "")
    return {normalize_id(i) for i in raw.split(",") if i}


def _result(
    passed: bool, score: float, reason: str, *, valid_schema: float, grounded: float
) -> dict[str, Any]:
    return {
        "pass": passed,
        "score": score,
        "reason": reason,
        "named_scores": {"valid_schema": valid_schema, "grounded": grounded},
    }


def _tool_result(passed: bool, reason: str, *, called: bool) -> dict[str, Any]:
    return {
        "pass": passed,
        "score": 1.0 if passed else 0.0,
        "reason": reason,
        "named_scores": {
            "tool_called": 1.0 if called else 0.0,
            "correct_first_step": float(passed),
        },
    }
