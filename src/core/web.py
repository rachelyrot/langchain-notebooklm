"""Firecrawl access: search the web, scrape a page, crawl a site.

A thin wrapper that hides the SDK's response shapes behind two small dataclasses, so the
research tools in ``agents/research.py`` deal with plain values and never with
``firecrawl.v2.types``. The client is built lazily — a missing key must surface as a
handled error on the request that needs the web, not as a crash at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

SEARCH_LIMIT = 5
CRAWL_LIMIT = 5
SCRAPE_TIMEOUT_MS = 60_000

_client = None


class WebUnavailable(Exception):
    """Raised when web research cannot run (no API key, or Firecrawl failed)."""


@dataclass
class WebResult:
    """One hit from a search — a candidate the agent may decide to scrape."""

    url: str
    title: str
    description: str


@dataclass
class WebPage:
    """A fetched page, ready to become a source."""

    url: str
    title: str
    markdown: str


def get_client():
    """The Firecrawl client, built on first use."""
    global _client
    if _client is None:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise WebUnavailable(
                "Web research needs FIRECRAWL_API_KEY in .env — get a key at firecrawl.dev."
            )
        from firecrawl import Firecrawl

        _client = Firecrawl(api_key=api_key)
    return _client


def _title_of(item, fallback: str) -> str:
    """Titles live directly on search hits but under ``metadata`` on scraped documents."""
    title = getattr(item, "title", None)
    if not title:
        metadata = getattr(item, "metadata", None)
        title = getattr(metadata, "title", None) if metadata else None
    return (title or fallback).strip()


def _url_of(item) -> str | None:
    url = getattr(item, "url", None)
    if not url:
        metadata = getattr(item, "metadata", None)
        url = getattr(metadata, "url", None) or getattr(metadata, "source_url", None)
    return url


def search(query: str, limit: int = SEARCH_LIMIT) -> list[WebResult]:
    """Web search results for one query phrasing (no page content is fetched)."""
    try:
        data = get_client().search(query=query, limit=limit)
    except WebUnavailable:
        raise
    except Exception as exc:  # SDK raises its own error types; the caller only needs the message
        raise WebUnavailable(f"Search failed: {exc}") from exc

    results = []
    for item in data.web or []:
        url = _url_of(item)
        if not url:
            continue
        description = getattr(item, "description", None) or getattr(item, "summary", None)
        results.append(
            WebResult(url=url, title=_title_of(item, url), description=(description or "").strip())
        )
    return results


def scrape(url: str) -> WebPage:
    """Fetch one page as markdown."""
    try:
        document = get_client().scrape(
            url, formats=["markdown"], only_main_content=True, timeout=SCRAPE_TIMEOUT_MS
        )
    except WebUnavailable:
        raise
    except Exception as exc:
        raise WebUnavailable(f"Scrape of {url} failed: {exc}") from exc

    markdown = (document.markdown or "").strip()
    if not markdown:
        raise WebUnavailable(f"{url} returned no readable text.")
    return WebPage(url=_url_of(document) or url, title=_title_of(document, url), markdown=markdown)


def crawl(url: str, limit: int = CRAWL_LIMIT) -> list[WebPage]:
    """Crawl a site and return its pages — for when one site is the authority on a topic."""
    try:
        job = get_client().crawl(url, limit=limit, formats=["markdown"], only_main_content=True)
    except WebUnavailable:
        raise
    except Exception as exc:
        raise WebUnavailable(f"Crawl of {url} failed: {exc}") from exc

    pages = []
    for document in job.data or []:
        markdown = (document.markdown or "").strip()
        page_url = _url_of(document)
        if markdown and page_url:
            pages.append(
                WebPage(url=page_url, title=_title_of(document, page_url), markdown=markdown)
            )
    return pages
