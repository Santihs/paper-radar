"""Governance as a test: the real promptfooconfig.yaml must follow policy."""

import pytest

from paper_radar.config import PROMPTFOO_CONFIG
from paper_radar.governance import find_violations, load_config


def test_repo_config_complies_with_policy() -> None:
    config = load_config(PROMPTFOO_CONFIG)
    assert config["providers"]
    assert find_violations(config) == []


def provider(pid: str, data_collection: str | None = "deny") -> dict[str, object]:
    routing = {"data_collection": data_collection} if data_collection else {}
    return {"id": pid, "config": {"provider": routing}}


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (provider("openrouter:deepseek/deepseek-v4"), "vendor 'deepseek' is not allowed"),
        (provider("openrouter:qwen/qwen-4"), "vendor 'qwen' is not allowed"),
        (provider("openai:gpt-6-sol"), "must be called through OpenRouter"),
        (provider("openrouter:openai/gpt-6-sol", None), "data_collection: deny"),
        (provider("openrouter:openai/gpt-6-sol", "allow"), "data_collection: deny"),
        ("openrouter:openai/gpt-6-sol", "data_collection: deny"),
    ],
)
def test_violations_detected(entry: object, reason: str) -> None:
    [violation] = find_violations({"providers": [entry]})
    assert reason in violation.reason


def test_compliant_provider_passes() -> None:
    compliant = provider("openrouter:anthropic/claude-sonnet-5.5")
    assert find_violations({"providers": [compliant]}) == []
