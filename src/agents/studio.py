"""Studio artifacts: standalone agents that return a *shape*, not prose.

Each artifact is a Pydantic schema handed to `create_agent` as `response_format`, so the
model fills in fields instead of writing a document. The rendering to markdown then
happens here, in code — which is the point of structured output: the same artifact could
be rendered as a slide deck or a web page without asking the model again.

These agents are stateless and are invoked directly by the Studio panel; they are not
part of the chat.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from agents.retrieval import make_retrieval_tools
from core.store import store

MODEL = "anthropic:claude-sonnet-4-6"
RECURSION_LIMIT = 30

GROUNDING = """You build an artifact out of a notebook of source documents.

Start with `list_sources` to see what the notebook holds, then call `search_sources` \
several times with different, specific queries — enough to cover the material rather than \
one broad sweep. Write only what the sources support: no outside knowledge, no filler. \
Every claim carries the name of the source it came from, exactly as `list_sources` spells \
it. If the sources do not cover something, leave it out or say so."""


# -- the shapes ----------------------------------------------------------------


class KeyPoint(BaseModel):
    """One substantive finding, tied to where it came from."""

    point: str = Field(description="The finding, in one or two sentences.")
    source: str = Field(description="Name of the source document it came from.")


class Summary(BaseModel):
    """A briefing on everything in the notebook."""

    title: str = Field(description="A specific title naming the actual subject matter.")
    overview: str = Field(description="One paragraph: what this material is about.")
    key_points: list[KeyPoint] = Field(description="The 5-8 findings that matter most.")
    disagreements: list[str] = Field(
        default_factory=list,
        description="Points where sources contradict each other, naming both sides. Empty if none.",
    )
    open_questions: list[str] = Field(
        default_factory=list, description="Questions the sources raise but do not answer."
    )


class FaqItem(BaseModel):
    question: str = Field(description="A question a reader of this material would actually ask.")
    answer: str = Field(description="The answer, drawn only from the sources.")
    source: str = Field(description="Name of the source document the answer came from.")


class Faq(BaseModel):
    """The questions this material answers."""

    title: str = Field(description="A specific title naming the actual subject matter.")
    items: list[FaqItem] = Field(description="6-10 questions, ordered from basic to specific.")


@dataclass
class Artifact:
    """A generated artifact, ready to be saved as a note."""

    title: str
    content: str  # markdown


# -- rendering -----------------------------------------------------------------


def _render_summary(summary: Summary) -> str:
    lines = [summary.overview, "", "## Key points"]
    lines += [f"- {p.point}  \n  *— {p.source}*" for p in summary.key_points]
    if summary.disagreements:
        lines += ["", "## Where the sources disagree"]
        lines += [f"- {d}" for d in summary.disagreements]
    if summary.open_questions:
        lines += ["", "## Open questions"]
        lines += [f"- {q}" for q in summary.open_questions]
    return "\n".join(lines)


def _render_faq(faq: Faq) -> str:
    blocks = [f"**{i.question}**\n\n{i.answer}\n\n*— {i.source}*" for i in faq.items]
    return "\n\n".join(blocks)


# -- the agents ----------------------------------------------------------------


def _build(schema: type[BaseModel], task: str):
    return create_agent(
        model=MODEL,
        system_prompt=f"{GROUNDING}\n\n{task}",
        tools=make_retrieval_tools(store),
        response_format=schema,
    )


_AGENTS = {
    "summary": (
        _build(Summary, "Your artifact is a summary of the whole notebook."),
        _render_summary,
    ),
    "faq": (
        _build(Faq, "Your artifact is a FAQ: the questions this material answers."),
        _render_faq,
    ),
}

KINDS = tuple(_AGENTS)


class EmptyNotebook(Exception):
    """Raised when there is nothing active to build an artifact from."""


def generate(kind: str) -> Artifact:
    """Run the artifact agent for ``kind`` over the active sources."""
    if kind not in _AGENTS:
        raise KeyError(kind)
    if not store.active_ids():
        raise EmptyNotebook("Enable at least one source before generating an artifact.")

    agent, render = _AGENTS[kind]
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Build the artifact."}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )

    structured = result["structured_response"]
    return Artifact(title=structured.title, content=render(structured))
