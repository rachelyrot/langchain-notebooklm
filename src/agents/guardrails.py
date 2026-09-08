"""Guardrails for the chat agent.

Two risks are specific to this product, and each gets a middleware:

1. **An ungrounded answer.** The whole promise is that answers come from the notebook. A
   model that skips retrieval and answers from what it already knows is not a worse
   answer — it is a different product, and the user has no way to tell.
2. **Personal data in scraped pages.** Sources are pulled off the open web, so a page can
   carry someone's email or an address; those must not be echoed back out.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, PIIMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

REFUSAL = (
    "I can only answer from the sources in this notebook, and nothing in them was "
    "relevant to that question. Try asking about what your sources actually cover, or "
    "add a source that does."
)


class RequireGroundingMiddleware(AgentMiddleware):
    """Refuse a final answer that no source was ever consulted for.

    Scoped to the whole thread rather than the current turn on purpose: a follow-up
    ("and what about the second one?") legitimately answers from passages already
    retrieved earlier in the conversation.
    """

    def after_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None  # still working — tool calls are about to run

        if any(isinstance(m, ToolMessage) for m in messages):
            return None  # the thread has consulted the notebook at least once

        if not any(isinstance(m, HumanMessage) for m in messages):
            return None

        # Same id, so this replaces the answer rather than appending to it.
        return {"messages": [AIMessage(content=REFUSAL, id=last.id)]}


def guardrails() -> list[AgentMiddleware]:
    """The guardrails the chat agent runs with."""
    return [
        RequireGroundingMiddleware(),
        # Sources come from the open web: redact personal data on the way out, and in
        # the retrieved passages the model is shown.
        PIIMiddleware("email", strategy="redact", apply_to_output=True, apply_to_tool_results=True),
        PIIMiddleware(
            "credit_card", strategy="redact", apply_to_output=True, apply_to_tool_results=True
        ),
    ]
