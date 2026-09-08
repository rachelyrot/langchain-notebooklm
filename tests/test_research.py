"""The research tools: what they index, what they refuse, and how they fail."""

from __future__ import annotations

import pytest

import agents.research as research
from core.web import WebPage


def _call(tool, **args):
    """Invoke a tool the way the agent does, so we get the ToolMessage back."""
    return tool.invoke({"name": tool.name, "args": args, "id": "1", "type": "tool_call"})


def test_web_search_lists_candidates_without_fetching(store, fake_web):
    out = research.web_search.invoke({"query": "solar efficiency", "limit": 5})

    assert "https://a.example/x" in out
    assert store.list() == [], "searching must not add anything to the notebook"


def test_scrape_page_indexes_the_page(store, fake_web):
    message = _call(research.scrape_page, url="https://a.example/x", reason="angle 1")

    source = store.find_by_url("https://a.example/x")
    assert source is not None
    assert source.name == "Page A"
    assert [s.id for s in message.artifact] == [source.id]


def test_the_same_url_is_never_indexed_twice(store, fake_web):
    _call(research.scrape_page, url="https://a.example/x", reason="angle 1")
    again = _call(research.scrape_page, url="https://a.example/x", reason="angle 1")

    assert again.artifact == []
    assert "Already in the notebook" in again.content
    assert len(store.list()) == 1


def test_a_failed_fetch_is_reported_not_raised(store, fake_web):
    message = _call(research.scrape_page, url="https://missing.example", reason="angle")

    assert message.artifact == []
    assert "Could not add" in message.content
    assert store.list() == []


def test_an_indexing_failure_does_not_kill_the_run(store, monkeypatch):
    """A provider error must come back as a tool result, or the whole graph dies."""

    def boom(**kwargs):
        raise RuntimeError("429 trial token rate limit exceeded")

    monkeypatch.setattr(store, "add", boom)

    content, added = research._index(WebPage("https://e/boom", "Boom", "some content"))

    assert added == []
    assert "Could not index" in content and "RuntimeError" in content
    assert store.find_by_url("https://e/boom") is None


def test_crawl_site_indexes_only_what_is_new(store, fake_web):
    _call(research.scrape_page, url="https://a.example/x", reason="angle 1")

    message = _call(research.crawl_site, url="https://a.example", reason="authority", limit=5)

    assert len(message.artifact) == 2, "the page already in the notebook should be skipped"
    assert len(store.list()) == 3


def test_added_sources_dedupes_across_the_transcript(store, fake_web):
    first = _call(research.scrape_page, url="https://a.example/x", reason="angle 1")
    second = _call(research.scrape_page, url="https://b.example/y", reason="angle 2")

    added = research._added_sources([first, second, first])

    assert [s.name for s in added] == ["Page A", "Page B"]


def test_indexed_web_content_is_retrievable(store, fake_web):
    _call(research.scrape_page, url="https://b.example/y", reason="angle 2")

    hits = store.search("battery storage costs", k=1)
    assert hits[0].metadata["url"] == "https://b.example/y"


def test_a_decision_for_an_unknown_run_is_refused():
    with pytest.raises(research.UnknownRun):
        research.decide("nosuchrun", approved=[0])
