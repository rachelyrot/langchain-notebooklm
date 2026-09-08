"""The human-in-the-loop step: the agent proposes, the person decides.

The model is scripted, so the whole interrupt/resume cycle runs offline.
"""

from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import agents.research as research


class ScriptedModel(FakeMessagesListChatModel):
    """Replays a fixed sequence of turns; create_agent's bind_tools must not disturb it."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def agent(monkeypatch, store, fake_web, pages):
    script = [
        AIMessage(
            content="Searching.",
            tool_calls=[
                {"name": "web_search", "args": {"query": "solar"}, "id": "s1", "type": "tool_call"}
            ],
        ),
        # every candidate proposed in one turn -> a single interrupt with three requests
        AIMessage(
            content="Here are the candidates.",
            tool_calls=[
                {
                    "name": "scrape_page",
                    "args": {"url": url, "reason": f"covers angle {i}"},
                    "id": f"p{i}",
                    "type": "tool_call",
                }
                for i, url in enumerate(pages, 1)
            ],
        ),
        AIMessage(content="Added what you picked."),
    ]

    scripted = create_agent(
        model=ScriptedModel(responses=script),
        system_prompt=research.SYSTEM_PROMPT,
        tools=[research.web_search, research.scrape_page, research.crawl_site],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"scrape_page": research._review, "crawl_site": research._review}
            )
        ],
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(research, "_agent", scripted)
    return scripted


def test_the_run_pauses_before_fetching_anything(agent, store):
    result = research.start("solar efficiency")

    assert result.status == "awaiting_selection"
    assert [c.url for c in result.candidates] == [
        "https://a.example/x",
        "https://b.example/y",
        "https://c.example/z",
    ]
    assert all(c.reason for c in result.candidates), "the person needs a reason to judge by"
    assert store.list() == [], "nothing may be fetched before the person decides"


def test_only_the_approved_pages_are_added(agent, store):
    paused = research.start("solar efficiency")

    done = research.decide(paused.run_id, approved=[0, 2])

    assert done.status == "done"
    assert [s.name for s in done.sources] == ["Page A", "Page C"]
    assert store.find_by_url("https://b.example/y") is None, "a rejected page was fetched"
    assert len(store.list()) == 2


def test_a_rejection_reaches_the_model_as_a_decision(agent):
    paused = research.start("solar efficiency")
    research.decide(paused.run_id, approved=[0, 2])

    state = agent.get_state({"configurable": {"thread_id": paused.run_id}})
    rejections = [m for m in state.values["messages"] if getattr(m, "status", None) == "error"]

    assert len(rejections) == 1
    assert "chose not to add it" in rejections[0].content


def test_skipping_everything_still_finishes_cleanly(agent, store):
    paused = research.start("solar efficiency")

    done = research.decide(paused.run_id, approved=[])

    assert done.status == "done"
    assert done.sources == []
    assert store.list() == []


def test_the_pending_batch_is_read_back_from_the_checkpointer(agent, store):
    """Nothing about a paused run lives in a module-level dict, so a restart is survivable."""
    paused = research.start("solar efficiency")

    recovered = research.pending(paused.run_id)

    assert [c.url for c in recovered] == [c.url for c in paused.candidates]
    assert [c.reason for c in recovered] == [c.reason for c in paused.candidates]


def test_a_finished_run_stops_accepting_decisions(agent):
    paused = research.start("solar efficiency")
    research.decide(paused.run_id, approved=[0])

    with pytest.raises(research.UnknownRun):
        research.decide(paused.run_id, approved=[1])
