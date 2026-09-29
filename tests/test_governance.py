"""Governance as a test: every promptfoo config in the repo must follow policy."""

from pathlib import Path
from typing import Any

import pytest

from paper_radar.config import PROMPTFOO_CONFIG, ROOT
from paper_radar.governance import find_violations, load_config

CONFIGS = sorted(ROOT.glob("promptfooconfig*.yaml"))
MAKER_HOST = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google-ai-studio",
    "x-ai": "xai",
}


def test_both_configs_are_checked() -> None:
    assert {c.name for c in CONFIGS} >= {"promptfooconfig.yaml", "promptfooconfig.cheap.yaml"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_repo_config_complies_with_policy(path: Path) -> None:
    config = load_config(path)
    assert config["providers"]
    assert find_violations(config) == []


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_repo_config_starts_with_the_model_maker(path: Path) -> None:
    for p in load_config(path)["providers"]:
        vendor = p["id"].removeprefix("openrouter:").split("/")[0]
        assert p["config"]["provider"]["order"][0] == MAKER_HOST[vendor], p["id"]


def test_demo_config_models() -> None:
    ids = [p["id"] for p in load_config(PROMPTFOO_CONFIG)["providers"]]
    assert ids == [
        "openrouter:anthropic/claude-sonnet-5.5",
        "openrouter:openai/gpt-6-sol",
        "openrouter:google/gemini-3.8-flash",
        "openrouter:x-ai/grok-4.7",
    ]


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


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_prompts_with_config_are_not_txt(path: Path) -> None:
    # promptfoo 0.123.1 silently drops prompt-level config (tools, response_format) for .txt
    prompts = load_config(path)["prompts"]
    with_config = [p["id"] for p in prompts if isinstance(p, dict) and p.get("config")]
    assert with_config
    assert not [pid for pid in with_config if pid.endswith(".txt")]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_gpt_never_sends_temperature(path: Path) -> None:
    # GPT-6 endpoints reject temperature; with require_parameters that is a 404
    gpts = [p["config"] for p in load_config(path)["providers"] if "/gpt-6" in p["id"]]
    assert gpts
    for gpt in gpts:
        assert "temperature" not in gpt
        assert gpt["omitDefaults"] is True


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_model_is_attributed_to_paper_radar(path: Path) -> None:
    providers = load_config(path)["providers"]
    assert all(p["config"]["headers"]["X-Title"] == "paper-radar" for p in providers)
