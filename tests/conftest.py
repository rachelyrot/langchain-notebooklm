"""Shared fixtures.

Every test runs against fakes: no API keys, no network, no cost. The embedding model is
a deterministic bag-of-words vector — crude, but similar text really does score higher,
which is all the retrieval tests need.
"""

from __future__ import annotations

import hashlib

import pytest

import agents.research as research_module
import core.store as store_module
from core.store import SourceStore
from core.web import WebPage, WebResult


class FakeEmbeddings:
    """Deterministic embeddings: one dimension per hashed word."""

    DIM = 64

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIM
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def store(monkeypatch):
    """A fresh, empty notebook wired into every module that holds the singleton."""
    fresh = SourceStore()
    monkeypatch.setattr(store_module, "_embeddings", FakeEmbeddings())
    monkeypatch.setattr(store_module, "store", fresh)
    monkeypatch.setattr(research_module, "store", fresh)
    monkeypatch.setattr(store_module, "_spent", store_module.deque())
    return fresh


@pytest.fixture
def pages():
    """The web, as far as these tests are concerned."""
    return {
        "https://a.example/x": WebPage(
            "https://a.example/x", "Page A", "Solar panel efficiency rose to 24 percent. " * 30
        ),
        "https://b.example/y": WebPage(
            "https://b.example/y", "Page B", "Battery storage costs fell sharply. " * 30
        ),
        "https://c.example/z": WebPage(
            "https://c.example/z", "Page C", "Critics question the field data. " * 30
        ),
    }


@pytest.fixture
def fake_web(monkeypatch, pages):
    """Replace Firecrawl with the ``pages`` fixture."""
    from core.web import WebUnavailable

    def fake_scrape(url):
        if url not in pages:
            raise WebUnavailable(f"Scrape of {url} failed: 404")
        return pages[url]

    monkeypatch.setattr(
        research_module,
        "search",
        lambda query, limit=5: [WebResult(u, p.title, "a description") for u, p in pages.items()],
    )
    monkeypatch.setattr(research_module, "scrape", fake_scrape)
    monkeypatch.setattr(research_module, "crawl", lambda url, limit=5: list(pages.values()))
    return pages
