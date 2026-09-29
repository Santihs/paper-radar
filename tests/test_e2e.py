"""Full promptfoo run against fake providers: no API key, no cost."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from paper_radar.config import PROMPTFOO_VERSION, Task
from paper_radar.consensus import load_runs, task_scores
from paper_radar.dataset import build_tests
from paper_radar.schemas import Paper

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.e2e
def test_promptfoo_grades_every_task_with_our_asserts(papers: list[Paper], tmp_path: Path) -> None:
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        pytest.skip("pnpm not installed")

    # asserts path is relative to the fake config's directory
    tests = build_tests(papers, asserts="file://../../src/paper_radar/asserts.py")
    (FIXTURES / "tests.fake.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")

    out = tmp_path / "eval.json"
    env = {
        **os.environ,
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_SHARING": "1",
        "PROMPTFOO_DISABLE_UPDATE": "true",
        "PROMPTFOO_PYTHON": sys.executable,
    }
    config = FIXTURES / "promptfooconfig.fake.yaml"
    # promptfoo exits non-zero when any assert fails, which is the point here
    subprocess.run(
        [
            pnpm,
            "dlx",
            f"promptfoo@{PROMPTFOO_VERSION}",
            "eval",
            "-c",
            str(config),
            "-o",
            str(out),
            "--no-cache",
            "--no-share",
        ],
        env=env,
        check=False,
        capture_output=True,
        timeout=300,
    )

    runs = load_runs(json.loads(out.read_text("utf-8")))
    assert len(runs) == 2 * len(Task)  # each test ran only against its own prompt
    scores = task_scores(runs)
    assert all(scores["fake-honest"][t].passed == 1 for t in Task)
    assert all(scores["fake-hallucinating"][t].passed == 0 for t in Task)
