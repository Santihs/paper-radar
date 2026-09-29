import json
from pathlib import Path
from typing import Any

import pytest

from paper_radar.config import PLANTED, TRAP_PAPER_ID, Planted, Role, Task
from paper_radar.dataset import assemble, build_tests, load_papers, save_dataset
from paper_radar.schemas import Paper, normalize_id


def paper(pid: str) -> Paper:
    return Paper(id=pid, title=f"title {pid}", summary="s")


def test_assemble_places_classics_at_fixed_positions() -> None:
    recent = [paper(f"2609.0000{i}v1") for i in range(5)]
    classics = [paper("1111.11111v2"), paper("2222.22222v1")]
    planted = [
        Planted("1111.11111", Role.DISTRACTOR, 0, ""),
        Planted("2222.22222", Role.EXPECTED, 3, ""),
    ]
    ids = [p.id for p in assemble(recent, classics, planted)]
    assert ids[0] == "1111.11111v2"
    assert ids[3] == "2222.22222v1"
    assert len(ids) == 7


def test_assemble_real_design_puts_lost_in_the_middle_mid_list(papers: list[Paper]) -> None:
    ids = [normalize_id(p.id) for p in papers]
    assert ids[0] == "1706.03762"  # Attention, the famous distractor, is seen first
    middle = ids.index("2307.03172")
    assert 0 < middle < len(ids) - 1
    assert len(ids) == len(set(ids))


def test_assemble_drops_recent_duplicates_of_classics() -> None:
    classic = paper("1111.11111v2")
    planted = [Planted("1111.11111", Role.EXPECTED, 0, "")]
    assert assemble([paper("1111.11111v1")], [classic], planted) == [classic]


def test_assemble_fails_loudly_when_a_classic_is_missing() -> None:
    with pytest.raises(ValueError, match=r"1706.03762"):
        assemble([paper("2609.00001v1")], [], PLANTED)


def test_one_case_per_task_bound_to_its_prompt(cases: dict[str, dict[str, Any]]) -> None:
    assert set(cases) == {t.value for t in Task}
    for task, case in cases.items():
        assert case["prompts"] == [task]
        [check] = case["assert"]
        assert check["type"] == "python"
        assert check["value"].startswith("file://src/paper_radar/asserts.py:check_")


def test_pick_case_carries_the_answer_key(cases: dict[str, dict[str, Any]]) -> None:
    v = cases[Task.PICK]["vars"]
    assert set(v["expected_ids"].split(",")) == {"2306.05685", "2406.18665", "2305.05176"}
    assert set(v["distractor_ids"].split(",")) == {"1706.03762", "2210.03629"}


def test_id_lists_are_strings_not_lists(cases: dict[str, dict[str, Any]]) -> None:
    # promptfoo would expand a list var into one test case per element
    for case in cases.values():
        assert all(isinstance(value, str) for value in case["vars"].values())


def test_trap_paper_is_not_in_the_list(cases: dict[str, dict[str, Any]]) -> None:
    v = cases[Task.TRAP]["vars"]
    assert v["trap_id"] == TRAP_PAPER_ID
    assert TRAP_PAPER_ID not in v["valid_ids"]


def test_build_tests_rejects_a_trap_that_exists() -> None:
    with pytest.raises(ValueError, match="trap"):
        build_tests([paper(TRAP_PAPER_ID)], planted=[])


def test_save_and_load_roundtrip(papers: list[Paper], tmp_path: Path) -> None:
    papers_file = tmp_path / "data" / "papers.json"
    tests_file = tmp_path / "data" / "tests.json"
    tests = build_tests(papers)
    save_dataset(papers, tests, papers_file, tests_file)
    assert load_papers(papers_file) == papers
    assert json.loads(tests_file.read_text("utf-8")) == tests
