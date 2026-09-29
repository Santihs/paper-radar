"""Policy checks on promptfooconfig.yaml, enforced before any eval runs."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from paper_radar.config import ALLOWED_VENDORS

_GATEWAY = "openrouter:"


@dataclass(frozen=True)
class Violation:
    provider: str
    reason: str


def load_config(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config


def find_violations(
    config: Mapping[str, Any], allowed_vendors: frozenset[str] = ALLOWED_VENDORS
) -> list[Violation]:
    """Every provider must go through OpenRouter, be an allowed vendor,
    and only route to endpoints that do not retain or train on prompts."""
    violations = []
    for entry in config.get("providers", []):
        provider_id, options = (entry, {}) if isinstance(entry, str) else (entry["id"], entry)

        if not provider_id.startswith(_GATEWAY):
            violations.append(Violation(provider_id, "must be called through OpenRouter"))
            continue

        vendor = provider_id.removeprefix(_GATEWAY).split("/", 1)[0]
        if vendor not in allowed_vendors:
            violations.append(Violation(provider_id, f"vendor '{vendor}' is not allowed"))

        routing = options.get("config", {}).get("provider", {})
        if routing.get("data_collection") != "deny":
            violations.append(Violation(provider_id, "missing provider.data_collection: deny"))

    return violations
