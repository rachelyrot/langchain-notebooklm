"""Streaming a chat turn: tool events, tokens, and the correction after the fact."""

from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage

import agents.chat as chat
from agents.guardrails import REFUSAL, guardrails
from agents.retrieval import make_retrieval_tools

SEARCH_CALL = [
    {"name": "search_sources", "args": {"query": "pricing tiers"}, "id": "t1", "type": "tool_call"}
]


class StreamingModel(GenericFakeChatModel):
    """Emits its message word by word, the way a real model does."""

    def bind_tools(self, tools, **kwargs):
        return self


class WholeMessageModel(FakeMessagesListChatModel):
    """Returns each turn in one piece - the only way to script a tool call.

    `GenericFakeChatModel` streams a message's *text*, so a turn that is nothing but a
    tool call has nothing to stream and raises.
    """

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def agent_factory(monkeypatch, store):
    def build(messages, middleware=(), token_by_token=True):
        model = (
            StreamingModel(messages=iter(messages))
            if token_by_token
            else WholeMessageModel(responses=messages)
        )
        agent = create_agent(
            model=model,
            system_prompt="answer from the notebook",
            tools=make_retrieval_tools(store),
            middleware=list(middleware),
            checkpointer=chat._agent.checkpointer,
        )
        monkeypatch.setattr(chat, "_agent", agent)
        return agent

    return build


def test_the_answer_arrives_as_tokens(agent_factory, store, unique_thread):
    store.add(name="pricing.md", content="Our pricing tiers are Basic, Pro and Enterprise.")
    agent_factory([AIMessage("Basic, Pro and Enterprise.")])

    events = list(chat.stream("what are the tiers?", thread_id=unique_thread))

    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert len(tokens) > 1, "the answer arrived in one lump, not as a stream"
    assert "".join(tokens).strip() == "Basic, Pro and Enterprise."
    assert events[-1]["type"] == "done"


def test_a_tool_call_is_announced_before_the_answer(agent_factory, store, unique_thread):
    store.add(name="pricing.md", content="Our pricing tiers are Basic, Pro and Enterprise.")
    agent_factory(
        [AIMessage(content="", tool_calls=SEARCH_CALL), AIMessage("Basic, Pro and Enterprise.")],
        token_by_token=False,
    )

    events = list(chat.stream("what are the tiers?", thread_id=unique_thread))
    kinds = [e["type"] for e in events]

    assert "tool" in kinds
    assert kinds.index("tool") < kinds.index("token"), "the tool ran after the answer?"
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["name"] == "search_sources"
    assert tool_event["args"]["query"] == "pricing tiers"


def test_the_citations_come_with_the_done_event(agent_factory, store, unique_thread):
    store.add(name="pricing.md", content="Our pricing tiers are Basic, Pro and Enterprise.")
    agent_factory(
        [AIMessage(content="", tool_calls=SEARCH_CALL), AIMessage("Basic, Pro and Enterprise.")],
        token_by_token=False,
    )

    events = list(chat.stream("tiers?", thread_id=unique_thread))

    assert events[-1] == {"type": "done", "sources": ["pricing.md"]}


def test_a_guardrail_correction_reaches_the_client(agent_factory, store, unique_thread):
    """The refusal happens after the model, so the ungrounded text is already streamed."""
    store.add(name="pricing.md", content="Our pricing tiers are Basic, Pro and Enterprise.")
    agent_factory([AIMessage("Made up from general knowledge.")], middleware=guardrails())

    events = list(chat.stream("something unrelated", thread_id=unique_thread))

    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    replacement = next(e for e in events if e["type"] == "replace")

    assert "general knowledge" in streamed, "nothing was streamed, so nothing to correct"
    assert replacement["text"] == REFUSAL


def test_citations_survive_a_round_trip_through_the_checkpointer(store):
    """Restored artifacts are plain dicts, not Documents - the reader must cope."""
    restored = ToolMessage(
        "[pricing.md] ...",
        tool_call_id="1",
        artifact=[
            {
                "id": "abc-0",
                "metadata": {"source_id": "abc", "source_name": "pricing.md", "chunk": 0},
                "page_content": "Our pricing tiers are Basic and Pro.",
                "type": "Document",
            }
        ],
    )

    assert chat._cited_sources([restored]) == ["pricing.md"]
