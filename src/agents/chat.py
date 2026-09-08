from dataclasses import dataclass

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage

from agents.retrieval import make_retrieval_tools
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
    checkpointer=InMemorySaver(),
    tools=make_retrieval_tools(store),
)


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
                name = doc.metadata.get("source_name")
                if name and name not in names:
                    names.append(name)
    return names


def answer(question: str, thread_id: str) -> Answer:
    config = {"configurable": {"thread_id": thread_id}}
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config=config
    )

    messages = result["messages"]
    return Answer(text=messages[-1].text, sources=_cited_sources(messages))
