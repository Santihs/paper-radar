"""Fetch the latest papers from the public arXiv API."""

import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from urllib.parse import urlencode

from paper_radar.schemas import Paper

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}


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


def _fetch(url: str) -> list[Paper]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return parse_feed(resp.read())
