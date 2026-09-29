"""Governance as a test: the real promptfooconfig.yaml must follow policy."""

from typing import Any

import pytest

from paper_radar.config import PROMPTFOO_CONFIG
from paper_radar.governance import find_violations, load_config


def test_repo_config_complies_with_policy() -> None:
    config = load_config(PROMPTFOO_CONFIG)
    assert config["providers"]
    assert find_violations(config) == []


def test_repo_config_starts_with_the_model_maker() -> None:
    first_host = {
        p["id"]: p["config"]["provider"]["order"][0]
        for p in load_config(PROMPTFOO_CONFIG)["providers"]
    }
    assert first_host == {
        "openrouter:anthropic/claude-sonnet-5.5": "anthropic",
        "openrouter:openai/gpt-6-sol": "openai",
        "openrouter:google/gemini-3.8-flash": "google-ai-studio",
        "openrouter:x-ai/grok-4.7": "xai",
    }


def provider(pid: str, **routing: Any) -> dict[str, Any]:
    base = {"data_collection": "deny", "order": ["openai"], "allow_fallbacks": False}
    merged = {k: v for k, v in {**base, **routing}.items() if v is not None}
    return {"id": pid, "config": {"provider": merged}}


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (provider("openrouter:deepseek/deepseek-v4"), "vendor 'deepseek' is not allowed"),
        (provider("openrouter:qwen/qwen-4"), "vendor 'qwen' is not allowed"),
        (provider("openai:gpt-6-sol"), "must be called through OpenRouter"),
        (provider("openrouter:openai/gpt-6-sol", data_collection=None), "data_collection: deny"),
        (provider("openrouter:openai/gpt-6-sol", data_collection="allow"), "data_collection: deny"),
        (provider("openrouter:openai/gpt-6-sol", order=None), "missing provider.order"),
        (
            provider("openrouter:openai/gpt-6-sol", order=["openai", "deepinfra"]),
            "host 'deepinfra'",
        ),
        (
            provider("openrouter:openai/gpt-6-sol", allow_fallbacks=True),
            "allow_fallbacks must be false",
        ),
        (
            provider("openrouter:openai/gpt-6-sol", allow_fallbacks=None),
            "allow_fallbacks must be false",
        ),
    ],
)
def test_single_violation_detected(entry: dict[str, Any], reason: str) -> None:
    [violation] = find_violations({"providers": [entry]})
    assert reason in violation.reason


def test_bare_string_provider_reports_every_missing_rule() -> None:
    reasons = [v.reason for v in find_violations({"providers": ["openrouter:openai/gpt-6-sol"]})]
    assert any("data_collection" in r for r in reasons)
    assert any("provider.order" in r for r in reasons)


def test_compliant_provider_passes() -> None:
    compliant = provider("openrouter:anthropic/claude-sonnet-5.5", order=["anthropic", "azure"])
    assert find_violations({"providers": [compliant]}) == []


def test_prompts_with_config_are_not_txt() -> None:
    # promptfoo 0.123.1 silently drops prompt-level config (tools, response_format) for .txt
    prompts = load_config(PROMPTFOO_CONFIG)["prompts"]
    with_config = [p["id"] for p in prompts if isinstance(p, dict) and p.get("config")]
    assert with_config
    assert not [pid for pid in with_config if pid.endswith(".txt")]


def test_gpt_never_sends_temperature() -> None:
    # GPT-6 Sol endpoints reject temperature; with require_parameters that is a 404
    [gpt] = [
        p["config"]
        for p in load_config(PROMPTFOO_CONFIG)["providers"]
        if p["id"] == "openrouter:openai/gpt-6-sol"
    ]
    assert "temperature" not in gpt
    assert gpt["omitDefaults"] is True


def test_every_model_is_attributed_to_paper_radar() -> None:
    providers = load_config(PROMPTFOO_CONFIG)["providers"]
    assert all(p["config"]["headers"]["X-Title"] == "paper-radar" for p in providers)
