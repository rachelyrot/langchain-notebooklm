from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from agents.guardrails import guardrails
from agents.retrieval import make_retrieval_tools
from core.memory import checkpointer
from core.store import store

@dataclass
class Answer:
    text: str
    sources: list[str]


MODEL = "anthropic:claude-sonnet-4-6"
SYSTEM_PROMPT = "You are the assistant for a notebook of source documents"


# Built once: the checkpointer is what gives the agent its short-term memory, so it has
# to outlive a single call (one thread of conversation per ``thread_id``).
_agent = create_agent(
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer(),
    tools=make_retrieval_tools(store),
    middleware=guardrails(),
)


def _source_name(doc: Any) -> str | None:
    """The source a retrieved chunk came from.

    A chunk is a `Document` while the turn is running and a plain dict once it has been
    through the checkpointer, so read the metadata out of either.
    """
    metadata = getattr(doc, "metadata", None)
    if metadata is None and isinstance(doc, dict):
        metadata = doc.get("metadata")
    return (metadata or {}).get("source_name")


def _cited_sources(messages: list[AnyMessage]) -> list[str]:
    """Source names the tools returned during the current turn, in order, deduped.

    ``messages`` is the whole thread, so start from the last user message — otherwise an
    answer would be credited with everything retrieved earlier in the conversation.
    """
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)

    names: list[str] = []
    for message in messages[start:]:
        if isinstance(message, ToolMessage) and message.artifact:
            for doc in message.artifact:
                name = _source_name(doc)
                if name and name not in names:
                    names.append(name)
    return names


def stream(question: str, thread_id: str) -> Iterator[dict[str, Any]]:
    """Answer as it is written, as a sequence of events.

    Events: ``tool`` when the notebook is consulted, ``token`` for each fragment of the
    answer, ``replace`` when the finished answer differs from what was streamed, and
    ``done`` with the citations.

    ``replace`` is not a nicety. Guardrails run *after* the model, so an ungrounded
    answer is refused only once its tokens have already left for the browser; the client
    is told to drop what it showed.
    """
    config = {"configurable": {"thread_id": thread_id}}
    written: list[str] = []

    for mode, payload in _agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            chunk, _meta = payload
            text = getattr(chunk, "text", "") or ""
            # AIMessageChunk while a model streams; a whole AIMessage if one does not.
            # ToolMessages travel this channel too and are not part of the answer.
            if text and isinstance(chunk, AIMessage):
                written.append(text)
                yield {"type": "token", "text": text}
        elif mode == "updates":
            for call in _tool_calls(payload):
                yield {"type": "tool", "name": call["name"], "args": call.get("args", {})}

    messages = _agent.get_state(config).values["messages"]
    final = messages[-1].text if messages else ""
    if final != "".join(written):
        yield {"type": "replace", "text": final}

    yield {"type": "done", "sources": _cited_sources(messages)}


def _tool_calls(update: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool calls a node just decided on, if any."""
    calls = []
    for node in update.values():
        for message in (node or {}).get("messages", []) if isinstance(node, dict) else []:
            calls.extend(getattr(message, "tool_calls", None) or [])
    return calls


def answer(question: str, thread_id: str) -> Answer:
    config = {"configurable": {"thread_id": thread_id}}
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config=config
    )

    messages = result["messages"]
    return Answer(text=messages[-1].text, sources=_cited_sources(messages))
