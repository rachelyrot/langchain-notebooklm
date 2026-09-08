"""Citations: an answer is credited with what this turn actually retrieved."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.chat import _cited_sources
from core.sources import format_docs


def _tool_message(docs, call_id):
    return ToolMessage(format_docs(docs), tool_call_id=call_id, artifact=docs)


def test_only_the_current_turn_is_cited(store):
    """With a checkpointer the whole thread comes back — earlier turns must not leak in."""
    store.add(name="pricing.md", content="Our pricing tiers are Basic, Pro and Enterprise.")
    store.add(name="hiring.md", content="We hired twelve engineers this quarter.")

    messages = [
        HumanMessage("turn 1"),
        _tool_message(store.search("hiring engineers", k=1), "1"),
        AIMessage("answer 1"),
        HumanMessage("turn 2"),
        _tool_message(store.search("pricing tiers", k=1), "2"),
        AIMessage("answer 2"),
    ]

    assert _cited_sources(messages) == ["pricing.md"]


def test_sources_are_deduped_but_keep_their_order(store):
    store.add(name="a.md", content="alpha beta gamma delta")
    store.add(name="b.md", content="epsilon zeta eta theta")

    docs = store.search("alpha epsilon")
    messages = [HumanMessage("q"), _tool_message(docs, "1"), _tool_message(docs, "2")]

    cited = _cited_sources(messages)
    assert cited == list(dict.fromkeys(d.metadata["source_name"] for d in docs))


def test_a_turn_without_retrieval_cites_nothing(store):
    messages = [HumanMessage("hello"), AIMessage("hi")]
    assert _cited_sources(messages) == []

    empty = ToolMessage("No relevant documents found", tool_call_id="1", artifact=[])
    assert _cited_sources([HumanMessage("q"), empty]) == []
    assert _cited_sources([]) == []


def test_format_docs_labels_every_chunk_with_its_source(store):
    store.add(name="pricing.md", content="Our pricing tiers are Basic and Pro.")

    rendered = format_docs(store.search("pricing"))
    assert "[pricing.md]" in rendered
