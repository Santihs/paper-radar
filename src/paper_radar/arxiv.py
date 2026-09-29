"""Fetch the latest papers from the public arXiv API."""

import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from urllib.parse import urlencode

from paper_radar.schemas import Paper

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
# arXiv API etiquette: identify the client and wait >= 3 s between calls.
USER_AGENT = "paper-radar/0.1 (public-research demo)"
_RETRY_STATUS = frozenset({429, 503})
_RETRY_DELAYS = (10.0, 20.0, 40.0)


def build_url(categories: Sequence[str], max_results: int) -> str:
    query = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    return f"{ARXIV_API}?{urlencode(params)}"


def build_ids_url(ids: Sequence[str]) -> str:
    return f"{ARXIV_API}?{urlencode({'id_list': ','.join(ids), 'max_results': len(ids)})}"


def parse_feed(xml: bytes, summary_chars: int = 600) -> list[Paper]:
    """Turn an arXiv Atom feed into Papers, normalising whitespace."""
    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall("a:entry", _ATOM):
        raw_id = entry.findtext("a:id", default="", namespaces=_ATOM)
        title = entry.findtext("a:title", default="", namespaces=_ATOM)
        summary = entry.findtext("a:summary", default="", namespaces=_ATOM)
        papers.append(
            Paper(
                id=raw_id.rsplit("/", 1)[-1],
                title=" ".join(title.split()),
                summary=" ".join(summary.split())[:summary_chars],
            )
        )
    return papers


def fetch_papers(categories: Sequence[str], max_results: int = 20) -> list[Paper]:
    return _fetch(build_url(categories, max_results))


def fetch_by_ids(ids: Sequence[str]) -> list[Paper]:
    return _fetch(build_ids_url(ids))


def _fetch(url: str, delays: Sequence[float] = _RETRY_DELAYS) -> list[Paper]:
    """GET the feed, backing off on arXiv rate limits (429/503)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(len(delays) + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                return parse_feed(resp.read())
        except urllib.error.HTTPError as err:
            if err.code not in _RETRY_STATUS or attempt == len(delays):
                raise
            retry_after = err.headers.get("Retry-After", "") if err.headers else ""
            time.sleep(float(retry_after) if retry_after.isdigit() else delays[attempt])
    raise AssertionError("unreachable")
