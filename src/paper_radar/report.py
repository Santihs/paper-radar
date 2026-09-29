"""Terminal rendering of the cross-model views."""

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from paper_radar.config import PLANTED, Role, Task
from paper_radar.consensus import ModelRun, count_votes, task_scores
from paper_radar.schemas import Paper, normalize_id

_MARKS = {Role.EXPECTED: "[green]expected[/]", Role.DISTRACTOR: "[red]distractor[/]"}


def render(runs: Sequence[ModelRun], papers: Sequence[Paper], console: Console) -> None:
    console.print(_scoreboard(runs))
    console.print(_agreement(runs, papers))


def _scoreboard(runs: Sequence[ModelRun]) -> Table:
    table = Table(title="Which model fits which task? (passed / runs)", show_lines=True)
    table.add_column("Model")
    for task in Task:
        table.add_column(task.value, justify="center")
    table.add_column("Cost USD", justify="right")
    table.add_column("Avg latency", justify="right")

    for model, tasks in task_scores(runs).items():
        mine = [r for r in runs if r.model == model]
        cost = sum(r.cost_usd or 0.0 for r in mine)
        latencies = [r.latency_ms for r in mine if r.latency_ms is not None]
        avg = f"{sum(latencies) / len(latencies) / 1000:.1f}s" if latencies else "-"
        cells = []
        for task in Task:
            score = tasks.get(task.value)
            if score is None:
                cells.append("-")
                continue
            color = "green" if score.rate == 1 else "yellow" if score.rate > 0 else "red"
            cells.append(f"[{color}]{score.passed}/{score.total}[/]")
        table.add_row(model, *cells, f"${cost:.4f}", avg)
    return table


def _agreement(runs: Sequence[ModelRun], papers: Sequence[Paper]) -> Table:
    titles = {normalize_id(p.id): p.title for p in papers}
    roles = {p.arxiv_id: p.role for p in PLANTED}
    pick_runs = [r for r in runs if r.task == Task.PICK and r.picks]

    table = Table(title=f"Pick task: which papers did the models choose? ({len(pick_runs)} runs)")
    table.add_column("Votes", justify="center")
    table.add_column("Paper")
    table.add_column("Planted as")
    for pid, n in count_votes(pick_runs, set(titles)).most_common():
        role = roles.get(pid)
        mark = _MARKS.get(role, role.value) if role else ""
        table.add_row(f"{n}/{len(pick_runs)}", titles[pid][:80], mark)
    return table
