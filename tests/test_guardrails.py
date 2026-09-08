"""Guardrails: an answer must be grounded, and personal data must not leak out."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.guardrails import REFUSAL, RequireGroundingMiddleware, guardrails


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _run(middleware, script, question="what is the record efficiency?"):
    agent = create_agent(
        model=ScriptedModel(responses=script),
        system_prompt="answer from the notebook",
        tools=[],
        middleware=middleware,
    )
    return agent.invoke({"messages": [{"role": "user", "content": question}]})


def test_an_answer_with_no_retrieval_is_replaced_by_a_refusal():
    result = _run([RequireGroundingMiddleware()], [AIMessage("Perovskite cells reach 34.85%.")])

    assert result["messages"][-1].content == REFUSAL


def test_an_answer_backed_by_retrieval_is_left_alone():
    """A thread that consulted the notebook keeps its answer, follow-ups included."""
    grounded = RequireGroundingMiddleware()
    state = {
        "messages": [
            HumanMessage("first question"),
            ToolMessage("[longi.md] 34.85 percent", tool_call_id="1"),
            AIMessage("It is 34.85%."),
        ]
    }

    assert grounded.after_model(state, None) is None


def test_a_turn_still_calling_tools_is_not_touched():
    grounded = RequireGroundingMiddleware()
    pending = AIMessage(
        content="",
        tool_calls=[{"name": "search_sources", "args": {"query": "x"}, "id": "1", "type": "tool_call"}],
    )

    assert grounded.after_model({"messages": [HumanMessage("q"), pending]}, None) is None


def test_the_refusal_replaces_rather_than_appends():
    answer = AIMessage("ungrounded", id="fixed-id")

    update = RequireGroundingMiddleware().after_model(
        {"messages": [HumanMessage("q"), answer]}, None
    )

    assert update["messages"][0].id == "fixed-id", "a new id would leave both answers in state"


def test_an_email_in_the_answer_is_redacted():
    result = _run(
        [PIIMiddleware("email", strategy="redact", apply_to_output=True)],
        [AIMessage("Contact the author at jane.doe@example.com for details.")],
    )

    content = result["messages"][-1].content
    assert "jane.doe@example.com" not in content
    assert "REDACTED" in content.upper()


def test_the_chat_agent_runs_with_all_of_them():
    kinds = {type(m).__name__ for m in guardrails()}

    assert "RequireGroundingMiddleware" in kinds
    assert "PIIMiddleware" in kinds
