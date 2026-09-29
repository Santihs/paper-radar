"""Domain models and parsing of model answers."""

import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    summary: str


class Pick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    why: str


class Answer(BaseModel):
    """What every model must return (mirrors prompts/answer_schema.json)."""

    model_config = ConfigDict(extra="forbid")

    picks: list[Pick]
    hype_warning: str


def normalize_id(arxiv_id: str) -> str:
    """'2406.18665v4' -> '2406.18665', so a pick without a version still matches."""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


class AnswerError(ValueError):
    """The model output is not a valid Answer."""


def parse_answer(output: str | Mapping[str, Any]) -> Answer:
    """Parse a model output into an Answer.

    promptfoo hands over a dict when structured output parsed cleanly, otherwise raw text,
    which may wrap the JSON in prose or a markdown fence.
    """
    if isinstance(output, str):
        start, end = output.find("{"), output.rfind("}")
        if start == -1 or end < start:
            raise AnswerError("no JSON object in output")
        try:
            output = json.loads(output[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AnswerError(f"invalid JSON: {exc.msg}") from exc
    try:
        return Answer.model_validate(output)
    except ValidationError as exc:
        raise AnswerError(f"schema mismatch: {exc.error_count()} error(s)") from exc
