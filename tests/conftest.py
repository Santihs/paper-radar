import json
from pathlib import Path
from typing import Any

import pytest

from paper_radar.arxiv import parse_feed
from paper_radar.dataset import assemble, build_tests
from paper_radar.schemas import Paper

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def recent() -> list[Paper]:
    return parse_feed((FIXTURES / "arxiv_sample.xml").read_bytes())


@pytest.fixture
def classics() -> list[Paper]:
    return parse_feed((FIXTURES / "arxiv_classics.xml").read_bytes())


@pytest.fixture
def papers(recent: list[Paper], classics: list[Paper]) -> list[Paper]:
    return assemble(recent, classics)


@pytest.fixture
def valid_ids(papers: list[Paper]) -> set[str]:
    return {p.id for p in papers}


@pytest.fixture
def cases(papers: list[Paper]) -> dict[str, dict[str, Any]]:
    """Test cases keyed by task, as promptfoo would receive them."""
    return {c["vars"]["task"]: c for c in build_tests(papers)}


@pytest.fixture
def eval_output() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / "promptfoo_output.json").read_text("utf-8"))
    return data
