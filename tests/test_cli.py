import json
import urllib.error
from collections.abc import Sequence
from email.message import Message
from pathlib import Path

import pytest

from paper_radar import cli, config


@pytest.fixture
def promptfoo_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_promptfoo(args: Sequence[str]) -> int:
        calls.append(list(args))
        return 0

    monkeypatch.setattr(cli, "_promptfoo", fake_promptfoo)
    return calls


def test_eval_passes_extra_flags_and_never_shares(
    promptfoo_calls: list[list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_file = tmp_path / "tests.json"
    tests_file.write_text("[]")
    monkeypatch.setattr(config, "TESTS_FILE", tests_file)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")

    assert cli.main(["eval", "--no-cache"]) == 0
    [args] = promptfoo_calls
    assert args[0] == "eval"
    assert "--no-share" in args
    assert args[-1] == "--no-cache"
    [run] = (tmp_path / "runs").iterdir()
    assert str(run / config.EVAL_CSV) in args
    assert (run / "promptfooconfig.yaml").is_file()  # config snapshot next to the results


def test_eval_blocked_by_governance(
    promptfoo_calls: list[list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "promptfooconfig.yaml"
    bad.write_text("providers:\n  - openrouter:deepseek/deepseek-v4\n")
    monkeypatch.setattr(config, "PROMPTFOO_CONFIG", bad)

    assert cli.main(["eval"]) == 1
    assert promptfoo_calls == []


def test_eval_requires_dataset(
    promptfoo_calls: list[list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "TESTS_FILE", tmp_path / "missing.json")
    assert cli.main(["eval"]) == 1
    assert promptfoo_calls == []


def test_consensus_without_eval_is_friendly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    assert cli.main(["consensus"]) == 1


def test_unknown_flags_rejected_outside_passthrough_commands() -> None:
    with pytest.raises(SystemExit):
        cli.main(["consensus", "--bogus"])


def test_fetch_rate_limited_keeps_existing_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_file = tmp_path / "tests.json"
    tests_file.write_text("[]")
    monkeypatch.setattr(config, "TESTS_FILE", tests_file)

    def rate_limited(*_: object, **__: object) -> None:
        raise urllib.error.HTTPError("https://export.arxiv.org", 429, "Rate", Message(), None)

    monkeypatch.setattr(cli, "fetch_papers", rate_limited)
    assert cli.main(["fetch"]) == 1
    assert tests_file.read_text() == "[]"


def test_read_env_file_parses_keys_without_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\n\nOPENROUTER_API_KEY="sk-or-test"\nexport OTHER=1\nbroken\n')
    assert cli.read_env_file(env_file) == {"OPENROUTER_API_KEY": "sk-or-test", "OTHER": "1"}


def test_read_env_file_missing_is_empty(tmp_path: Path) -> None:
    assert cli.read_env_file(tmp_path / "missing.env") == {}


def test_consensus_reads_latest_run_or_the_one_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_output: dict[str, object]
) -> None:
    runs = tmp_path / "runs"
    for name in ("20260928-230000", "20260929-010000"):
        (runs / name).mkdir(parents=True)
        (runs / name / config.EVAL_JSON).write_text(json.dumps(eval_output))
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    shown: list[str] = []
    monkeypatch.setattr(cli, "render", lambda *_: None)
    monkeypatch.setattr(cli.console, "print", lambda msg, *_: shown.append(str(msg)))

    assert cli.main(["consensus"]) == 0
    assert "20260929-010000" in shown[0]
    assert cli.main(["consensus", "--run", "20260928-230000"]) == 0
    assert "20260928-230000" in shown[1]
    assert cli.main(["consensus", "--run", "missing"]) == 1
