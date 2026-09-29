import json
from typing import Any

import pytest

from paper_radar.asserts import check_picks, check_tool_call, check_trap, grade, load_tools
from paper_radar.config import Task


def answer(*ids: str) -> dict[str, Any]:
    return {"picks": [{"id": i, "why": "x"} for i in ids], "hype_warning": "y"}


# --- grade: structure + grounding ---------------------------------------------------------------


def test_valid_grounded_answer_passes(valid_ids: set[str]) -> None:
    ids = sorted(valid_ids)[:3]
    result = grade(answer(*ids), valid_ids)
    assert result["pass"] is True
    assert result["named_scores"] == {"valid_schema": 1.0, "grounded": 1.0}


def test_id_without_version_is_not_invented(valid_ids: set[str]) -> None:
    assert grade(answer("2406.18665", "2305.05176", "2306.05685"), valid_ids)["pass"] is True


def test_accepts_json_string_wrapped_in_prose(valid_ids: set[str]) -> None:
    text = f"Sure! ```json\n{json.dumps(answer(*sorted(valid_ids)[:3]))}\n```"
    assert grade(text, valid_ids)["pass"] is True


def test_invented_id_fails_and_lowers_score(valid_ids: set[str]) -> None:
    ids = sorted(valid_ids)
    result = grade(answer(ids[0], ids[1], "9999.99999v1"), valid_ids)
    assert result["pass"] is False
    assert result["score"] == pytest.approx(2 / 3)
    assert "invented paper ids: 9999.99999" in result["reason"]


@pytest.mark.parametrize("count", [2, 4])
def test_wrong_number_of_picks_fails(count: int, valid_ids: set[str]) -> None:
    result = grade(answer(*sorted(valid_ids)[:count]), valid_ids)
    assert result["pass"] is False
    assert f"expected 3 picks, got {count}" in result["reason"]


def test_duplicate_picks_fail(valid_ids: set[str]) -> None:
    ids = sorted(valid_ids)
    assert "duplicate picks" in grade(answer(ids[0], ids[0], ids[1]), valid_ids)["reason"]


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("I cannot help with that", "no JSON object"),
        ('{"picks": [', "no JSON object"),
        ('{"picks": }', "invalid JSON"),
        ({"picks": []}, "schema mismatch"),
        ({"picks": [], "hype_warning": "x", "extra": 1}, "schema mismatch"),
    ],
)
def test_malformed_output_fails_without_raising(
    output: Any, reason: str, valid_ids: set[str]
) -> None:
    result = grade(output, valid_ids)
    assert result["pass"] is False
    assert result["named_scores"]["valid_schema"] == 0.0
    assert reason in result["reason"]


# --- check_picks: accuracy against the planted answer --------------------------------------------


def pick_context(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"vars": cases[Task.PICK]["vars"]}


def test_all_expected_papers_scores_full(cases: dict[str, dict[str, Any]]) -> None:
    result = check_picks(
        answer("2406.18665v4", "2305.05176v1", "2306.05685v4"), pick_context(cases)
    )
    assert result["pass"] is True
    assert result["score"] == 1.0
    assert result["named_scores"]["accuracy"] == 1.0
    assert result["named_scores"]["distractor"] == 0.0


def test_two_of_three_expected_still_passes(cases: dict[str, dict[str, Any]]) -> None:
    ctx = pick_context(cases)
    result = check_picks(answer("2406.18665v4", "2305.05176v1", "2609.31619v1"), ctx)
    assert result["pass"] is True
    assert result["score"] == pytest.approx(2 / 3)


def test_famous_distractor_fails_and_is_flagged(cases: dict[str, dict[str, Any]]) -> None:
    ctx = pick_context(cases)
    result = check_picks(answer("1706.03762v7", "2406.18665v4", "2609.31619v1"), ctx)
    assert result["pass"] is False
    assert "only 1/3 expected papers" in result["reason"]
    assert "picked distractor: 1706.03762" in result["reason"]
    assert result["named_scores"]["distractor"] == 1.0


def test_accurate_but_invented_still_fails(cases: dict[str, dict[str, Any]]) -> None:
    ctx = pick_context(cases)
    result = check_picks(answer("2406.18665v4", "2305.05176v1", "9999.99999v1"), ctx)
    assert result["pass"] is False
    assert "invented" in result["reason"]


def test_malformed_pick_output_reports_zero_accuracy(cases: dict[str, dict[str, Any]]) -> None:
    result = check_picks("no json here", pick_context(cases))
    assert result["pass"] is False
    assert result["named_scores"]["accuracy"] == 0.0


# --- check_trap ----------------------------------------------------------------------------------


@pytest.mark.parametrize("output", ["NOT_IN_LIST", "not_in_list", "Sorry: NOT_IN_LIST."])
def test_trap_passes_when_model_admits_it(output: str) -> None:
    assert check_trap(output, {"vars": {}})["pass"] is True


def test_trap_fails_on_invented_summary() -> None:
    result = check_trap("This paper proposes routing laws that reduce cost by 40%.", {"vars": {}})
    assert result["pass"] is False
    assert result["reason"] == "hallucinated a summary"


# --- check_tool_call -----------------------------------------------------------------------------


def call(name: str, **args: Any) -> dict[str, Any]:
    return {
        "id": "c1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_tools_file_defines_both_tools() -> None:
    assert set(load_tools()) == {"search_papers", "save_to_sheet"}


def test_correct_first_step_passes() -> None:
    result = check_tool_call([call("search_papers", query="LLM routing", max_results=10)], {})
    assert result["pass"] is True
    assert result["named_scores"] == {"tool_called": 1.0, "correct_first_step": 1.0}


def test_accepts_openai_message_shape_and_json_string() -> None:
    calls = [call("search_papers", query="model router")]
    assert check_tool_call({"tool_calls": calls}, {})["pass"] is True
    assert check_tool_call(json.dumps(calls), {})["pass"] is True


def test_text_answer_instead_of_tool_fails() -> None:
    result = check_tool_call("Here are 3 papers: RouterX, CheapNet, SmartSwitch", {})
    assert result["pass"] is False
    assert result["named_scores"]["tool_called"] == 0.0


def test_saving_before_searching_fails() -> None:
    rows = [{"id": "x", "title": "t", "why": "w"}]
    result = check_tool_call(
        [call("search_papers", query="routing"), call("save_to_sheet", rows=rows)], {}
    )
    assert result["pass"] is False
    assert "invented data" in result["reason"]


def test_wrong_first_tool_fails() -> None:
    result = check_tool_call([call("save_to_sheet", rows=[])], {})
    assert "first call was 'save_to_sheet'" in result["reason"]


@pytest.mark.parametrize(
    ("args", "problem"),
    [
        ({"query": "LLM routing", "max_results": 500}, "500 > maximum 50"),
        ({"query": "LLM routing", "max_results": 0}, "0 < minimum 1"),
        ({"max_results": 5}, "missing 'query'"),
        ({"query": "LLM routing", "limit": 5}, "unexpected 'limit'"),
        ({"query": 42}, "query: expected string"),
        ({"query": "cooking recipes"}, "off-topic query"),
    ],
)
def test_invalid_search_arguments_fail(args: dict[str, Any], problem: str) -> None:
    result = check_tool_call([call("search_papers", **args)], {})
    assert result["pass"] is False
    assert problem in result["reason"]


def test_unknown_tool_and_broken_arguments_fail() -> None:
    bad_json = {
        "id": "c",
        "type": "function",
        "function": {"name": "search_papers", "arguments": "{"},
    }
    assert "unknown tool 'web_search'" in check_tool_call([call("web_search", q="x")], {})["reason"]
    assert "not a JSON object" in check_tool_call([bad_json], {})["reason"]
