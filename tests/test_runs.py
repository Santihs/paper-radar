from datetime import datetime
from pathlib import Path

from paper_radar.runs import latest_run, new_run_dir

NOW = datetime(2026, 9, 29, 0, 15, 12)


def test_new_run_dir_is_timestamped(tmp_path: Path) -> None:
    run = new_run_dir(tmp_path / "runs", NOW)
    assert run.name == "20260929-001512"
    assert run.is_dir()


def test_new_run_dir_never_reuses_a_folder(tmp_path: Path) -> None:
    first = new_run_dir(tmp_path, NOW)
    second = new_run_dir(tmp_path, NOW)
    assert second != first
    assert second.name == "20260929-001512-2"


def test_latest_run_skips_runs_without_results(tmp_path: Path) -> None:
    runs = {"20260928-230000": True, "20260929-000000": True, "20260929-010000": False}
    for name, done in runs.items():
        (tmp_path / name).mkdir()
        if done:
            (tmp_path / name / "eval.json").write_text("{}")
    latest = latest_run(tmp_path, "eval.json")
    assert latest is not None
    assert latest.name == "20260929-000000"


def test_latest_run_without_runs(tmp_path: Path) -> None:
    assert latest_run(tmp_path / "missing", "eval.json") is None
    assert latest_run(tmp_path, "eval.json") is None
