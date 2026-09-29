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

    assert cli.main(["eval", "--no-cache"]) == 0
    [args] = promptfoo_calls
    assert args[0] == "eval"
    assert "--no-share" in args
    assert args[-1] == "--no-cache"
    assert str(config.EVAL_CSV_FILE) in args


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
    monkeypatch.setattr(config, "EVAL_OUTPUT_FILE", tmp_path / "missing.json")


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
