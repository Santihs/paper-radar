from typing import Any

from paper_radar.config import Task
from paper_radar.consensus import ModelRun, TaskScore, count_votes, load_runs, task_scores


def test_load_runs_reads_task_result_and_picks(eval_output: dict[str, Any]) -> None:
    runs = {(r.model, r.task): r for r in load_runs(eval_output)}

    honest_pick = runs["fake-honest", "pick"]
    assert honest_pick.passed is True
    assert set(honest_pick.picks) == {"2306.05685", "2406.18665", "2305.05176"}

    bad_pick = runs["fake-hallucinating", "pick"]
    assert bad_pick.passed is False
    assert "picked distractor" in bad_pick.note

    assert runs["fake-hallucinating", "trap"].note == "hallucinated a summary"
    assert runs["fake-hallucinating", "tools"].picks == ()  # picks only parsed for the pick task

    broken = runs["broken", "pick"]
    assert broken.picks == ()
    assert "No endpoints found" in broken.note


def test_task_scores_per_model(eval_output: dict[str, Any]) -> None:
    scores = task_scores(load_runs(eval_output))
    assert scores["fake-honest"] == {t.value: TaskScore(1, 1) for t in Task}
    assert all(s.passed == 0 for s in scores["fake-hallucinating"].values())


def run(*picks: str, task: str = Task.PICK, passed: bool = True) -> ModelRun:
    return ModelRun("m", task, passed, picks, cost_usd=None, latency_ms=None, note="")


def test_task_scores_aggregate_repeats() -> None:
    runs = [run(passed=True), run(passed=False), run(passed=True)]
    score = task_scores(runs)["m"]["pick"]
    assert (score.passed, score.total) == (2, 3)
    assert score.rate == 2 / 3


def test_count_votes_ignores_invented_ids_and_versions() -> None:
    votes = count_votes([run("a", "b", "c"), run("a", "b", "fake")], valid_ids={"av1", "bv2", "c"})
    assert votes == {"a": 2, "b": 2, "c": 1}


def test_count_votes_only_counts_pick_task_once_per_run() -> None:
    runs = [run("a", "a"), run("a", task=Task.TOOLS)]
    assert count_votes(runs, valid_ids={"a"}) == {"a": 1}
