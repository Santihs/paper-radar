"""One folder per eval run (data/runs/YYYYMMDD-HHMMSS), so results are never overwritten."""

from datetime import datetime
from pathlib import Path

_STAMP = "%Y%m%d-%H%M%S"


def new_run_dir(runs_dir: Path, now: datetime | None = None) -> Path:
    """Create a fresh, uniquely named run folder."""
    base = (now or datetime.now()).strftime(_STAMP)
    run, n = runs_dir / base, 1
    while run.exists():  # two runs in the same second
        n += 1
        run = runs_dir / f"{base}-{n}"
    run.mkdir(parents=True)
    return run


def latest_run(runs_dir: Path, results_file: str) -> Path | None:
    """Newest run folder that actually has results (a failed eval leaves none)."""
    if not runs_dir.is_dir():
        return None
    done = [d for d in runs_dir.iterdir() if (d / results_file).is_file()]
    return max(done, key=lambda d: d.name, default=None)
