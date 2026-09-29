"""Cross-model views promptfoo does not give: pass rate per task and agreement on picks."""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from paper_radar.config import Task
from paper_radar.schemas import AnswerError, normalize_id, parse_answer


@dataclass(frozen=True)
class ModelRun:
    """One promptfoo result row: one model, one task, one repeat."""

    model: str
    task: str
    passed: bool
    picks: tuple[str, ...]
    cost_usd: float | None
    latency_ms: int | None
    note: str


@dataclass(frozen=True)
class TaskScore:
    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def load_runs(eval_output: Mapping[str, Any]) -> list[ModelRun]:
    """Read the JSON written by `promptfoo eval -o`."""
    runs = []
    for row in eval_output["results"]["results"]:
        provider = row["provider"]
        grading = row.get("gradingResult") or {}
        task = (row.get("vars") or {}).get("task", "")
        output = (row.get("response") or {}).get("output")

        picks: tuple[str, ...] = ()
        note = row.get("error") or grading.get("reason") or ""
        if task == Task.PICK and output is not None:
            try:
                picks = tuple(normalize_id(p.id) for p in parse_answer(output).picks)
            except AnswerError as exc:
                note = str(exc)

        runs.append(
            ModelRun(
                model=provider.get("label") or provider["id"],
                task=task,
                passed=bool(row.get("success")),
                picks=picks,
                cost_usd=row.get("cost"),
                latency_ms=row.get("latencyMs"),
                note=note,
            )
        )
    return runs


def task_scores(runs: Sequence[ModelRun]) -> dict[str, dict[str, TaskScore]]:
    """model -> task -> passed/total across repeats."""
    counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for run in runs:
        cell = counts[run.model][run.task]
        cell[0] += run.passed
        cell[1] += 1
    return {m: {t: TaskScore(*c) for t, c in tasks.items()} for m, tasks in counts.items()}


def count_votes(runs: Sequence[ModelRun], valid_ids: set[str]) -> Counter[str]:
    """Votes per real paper across pick runs; invented ids are ignored so they cannot win."""
    valid = {normalize_id(i) for i in valid_ids}
    return Counter(
        pid for run in runs if run.task == Task.PICK for pid in set(run.picks) if pid in valid
    )
