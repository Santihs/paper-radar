import io
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from paper_radar.arxiv import (
    USER_AGENT,
    _fetch,
    build_ids_url,
    build_url,
    fetch_by_ids,
    fetch_papers,
    parse_feed,
)
from paper_radar.config import PLANTED
from paper_radar.schemas import Paper, normalize_id

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_url_queries_categories_newest_first() -> None:
    query = parse_qs(urlparse(build_url(["cs.AI", "cs.CL"], 5)).query)
    assert query["search_query"] == ["cat:cs.AI OR cat:cs.CL"]
    assert query["sortBy"] == ["submittedDate"]
    assert query["sortOrder"] == ["descending"]
    assert query["max_results"] == ["5"]


def test_build_ids_url_requests_exact_papers() -> None:
    query = parse_qs(urlparse(build_ids_url(["1706.03762", "2406.18665"])).query)
    assert query["id_list"] == ["1706.03762,2406.18665"]
    assert query["max_results"] == ["2"]


def test_parse_feed_extracts_ids_and_normalises_text(recent: list[Paper]) -> None:
    assert len(recent) == 4
    assert recent[0].id == "2609.31619v1"
    for p in recent:
        assert "\n" not in p.title
        assert "  " not in p.title
        assert p.summary
        assert len(p.summary) <= 600


def test_parse_feed_truncates_summary() -> None:
    xml = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/1234.5678v2</id>
      <title>A
        title</title>
      <summary>abcdefghij</summary></entry></feed>"""
    [paper] = parse_feed(xml, summary_chars=4)
    assert paper == Paper(id="1234.5678v2", title="A title", summary="abcd")


def test_parse_feed_empty() -> None:
    assert parse_feed(b'<feed xmlns="http://www.w3.org/2005/Atom"/>') == []


def test_classics_fixture_has_every_planted_paper(classics: list[Paper]) -> None:

    assert {normalize_id(p.id) for p in classics} == {p.arxiv_id for p in PLANTED}


@pytest.mark.live
def test_fetch_papers_live() -> None:
    assert len(fetch_papers(["cs.AI"], max_results=2)) == 2


@pytest.mark.live
def test_fetch_by_ids_live() -> None:
    [paper] = fetch_by_ids(["1706.03762"])
    assert paper.title == "Attention Is All You Need"


class _Resp(io.BytesIO):
    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _rate_limited(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 429, "Too Many Requests", Message(), None)


def test_fetch_retries_on_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    feed = FIXTURES.joinpath("arxiv_sample.xml").read_bytes()
    calls: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _Resp:
        calls.append(req)
        if len(calls) == 1:
            raise _rate_limited(req.full_url)
        return _Resp(feed)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    papers = _fetch("https://example.test", delays=(0,))
    assert len(papers) == 4
    assert len(calls) == 2
    assert calls[0].get_header("User-agent") == USER_AGENT


def test_fetch_gives_up_after_last_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _Resp:
        raise _rate_limited(req.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        _fetch("https://example.test", delays=(0, 0))
