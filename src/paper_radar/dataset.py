"""Build the evaluation dataset: recent papers + planted classics -> promptfoo test cases."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from paper_radar.config import PLANTED, TRAP_PAPER_ID, TRAP_PAPER_TITLE, Planted, Role, Task
from paper_radar.schemas import Paper, normalize_id

_PAPERS = TypeAdapter(list[Paper])
DEFAULT_ASSERTS = "file://src/paper_radar/asserts.py"


def assemble(
    recent: Sequence[Paper], classics: Sequence[Paper], planted: Sequence[Planted] = PLANTED
) -> list[Paper]:
    """Insert each planted classic at its fixed position among the recent papers.

    Fixed positions make position bias part of the experiment (e.g. a paper in the middle).
    """
    by_id = {normalize_id(p.id): p for p in classics}
    missing = [p.arxiv_id for p in planted if p.arxiv_id not in by_id]
    if missing:
        raise ValueError(f"planted papers not fetched: {', '.join(missing)}")

    papers = [p for p in recent if normalize_id(p.id) not in by_id]
    for item in sorted(planted, key=lambda p: p.position):
        papers.insert(min(item.position, len(papers)), by_id[item.arxiv_id])
    return papers


def format_papers(papers: Sequence[Paper]) -> str:
    return "\n\n".join(f"[{p.id}] {p.title}\n{p.summary}" for p in papers)


def build_tests(
    papers: Sequence[Paper],
    planted: Sequence[Planted] = PLANTED,
    asserts: str = DEFAULT_ASSERTS,
) -> list[dict[str, Any]]:
    """One promptfoo test case per task, each bound to its own prompt and assertion.

    Id lists are comma-joined strings, not lists: promptfoo expands list vars
    into one test case per element.
    """
    ids = {normalize_id(p.id) for p in papers}
    if normalize_id(TRAP_PAPER_ID) in ids:
        raise ValueError("trap paper must not be in the list")

    def by_role(role: Role) -> str:
        return ",".join(p.arxiv_id for p in planted if p.role is role and p.arxiv_id in ids)

    shared = {"papers": format_papers(papers), "valid_ids": ",".join(p.id for p in papers)}
    return [
        _case(
            Task.PICK,
            "Judgment: papers that help choose a model by cost and quality",
            {
                **shared,
                "expected_ids": by_role(Role.EXPECTED),
                "distractor_ids": by_role(Role.DISTRACTOR),
            },
            f"{asserts}:check_picks",
        ),
        _case(
            Task.TRAP,
            "Honesty: summarize a paper that is not in the list",
            {**shared, "trap_id": TRAP_PAPER_ID, "trap_title": TRAP_PAPER_TITLE},
            f"{asserts}:check_trap",
        ),
        _case(
            Task.TOOLS,
            "Agent readiness: first tool call to build the Tech Radar",
            {},
            f"{asserts}:check_tool_call",
        ),
    ]


def _case(task: Task, description: str, vars_: dict[str, str], assert_ref: str) -> dict[str, Any]:
    return {
        "description": description,
        "prompts": [task.value],
        "vars": {"task": task.value, **vars_},
        "assert": [{"type": "python", "value": assert_ref, "metric": task.value}],
    }


def save_dataset(
    papers: Sequence[Paper], tests: Sequence[dict[str, Any]], papers_file: Path, tests_file: Path
) -> None:
    papers_file.parent.mkdir(parents=True, exist_ok=True)
    papers_file.write_bytes(_PAPERS.dump_json(list(papers), indent=2))
    tests_file.write_text(json.dumps(list(tests), indent=2), encoding="utf-8")


def load_papers(papers_file: Path) -> list[Paper]:
    return _PAPERS.validate_json(papers_file.read_bytes())
