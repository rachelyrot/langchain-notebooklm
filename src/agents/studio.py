"""Studio artifacts: standalone agents that return a *shape*, not prose.

Each artifact is a Pydantic schema handed to `create_agent` as `response_format`, so the
model fills in fields instead of writing a document. The rendering to markdown then
happens here, in code — which is the point of structured output: the same artifact could
be rendered as a slide deck or a web page without asking the model again.

These agents are stateless and are invoked directly by the Studio panel; they are not
part of the chat.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from agents import artifacts
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


class Stat(BaseModel):
    """A single number worth putting in large type."""

    value: str = Field(description="The figure itself, short: '34.85%', '3 of 5', '2025'.")
    label: str = Field(description="What the figure measures, a few words.")
    source: str = Field(description="Name of the source document it came from.")


class Section(BaseModel):
    heading: str = Field(description="A short heading, a few words.")
    points: list[str] = Field(description="2-4 short lines. Not paragraphs.")


class Infographic(BaseModel):
    """A one-page visual summary: figures first, then a little structure."""

    title: str = Field(description="A specific title naming the actual subject matter.")
    subtitle: str = Field(description="One line of context under the title.")
    stats: list[Stat] = Field(description="3-5 concrete figures from the sources.")
    sections: list[Section] = Field(description="2-4 sections of short points.")
    takeaway: str = Field(description="The single sentence to remember.")


class Slide(BaseModel):
    title: str = Field(description="The slide's heading.")
    bullets: list[str] = Field(description="3-5 bullets, one line each. Not sentences to read aloud.")
    source: str = Field(description="Name of the source document this slide rests on.")


class Deck(BaseModel):
    """A short presentation over the notebook."""

    title: str = Field(description="A specific title naming the actual subject matter.")
    subtitle: str = Field(description="One line of context for the title slide.")
    slides: list[Slide] = Field(description="5-8 slides that build an argument in order.")


@dataclass
class Artifact:
    """A generated artifact: markdown for the notes panel, plus a file when it is one."""

    title: str
    content: str  # markdown
    file: object | None = None  # artifacts.StoredFile, when the artifact is a file


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


def _render_infographic(data: Infographic) -> str:
    """The note that accompanies the file: the same content, readable as text."""
    lines = [data.subtitle, ""]
    lines += [f"**{s.value}** — {s.label}  \n  *— {s.source}*" for s in data.stats]
    for section in data.sections:
        lines += ["", f"## {section.heading}"] + [f"- {p}" for p in section.points]
    lines += ["", f"**Takeaway:** {data.takeaway}"]
    return "\n".join(lines)


def _render_deck(deck: Deck) -> str:
    lines = [deck.subtitle]
    for i, slide in enumerate(deck.slides, 1):
        lines += ["", f"## {i}. {slide.title}"]
        lines += [f"- {b}" for b in slide.bullets]
        lines.append(f"*— {slide.source}*")
    return "\n".join(lines)


# -- the agents ----------------------------------------------------------------


def _build(schema: type[BaseModel], task: str):
    return create_agent(
        model=MODEL,
        system_prompt=f"{GROUNDING}\n\n{task}",
        tools=make_retrieval_tools(store),
        response_format=schema,
    )


@dataclass
class Kind:
    """How one artifact is produced: which agent, how it reads, and what file it makes."""

    agent: Any
    to_markdown: Callable[[Any], str]
    to_file: Callable[[Any], bytes] | None = None
    extension: str = ""


_AGENTS: dict[str, Kind] = {
    "summary": Kind(
        _build(Summary, "Your artifact is a summary of the whole notebook."),
        _render_summary,
    ),
    "faq": Kind(
        _build(Faq, "Your artifact is a FAQ: the questions this material answers."),
        _render_faq,
    ),
    "infographic": Kind(
        _build(
            Infographic,
            "Your artifact is a one-page infographic. Lead with concrete figures pulled "
            "from the sources — a section with no numbers in it is a section that belongs "
            "in the summary instead. Keep every line short enough to read at a glance.",
        ),
        _render_infographic,
        artifacts.render_infographic,
        ".html",
    ),
    "powerpoint": Kind(
        _build(
            Deck,
            "Your artifact is a short slide deck. The slides should build an argument in "
            "order rather than list facts. Bullets are headlines, not sentences to be "
            "read aloud.",
        ),
        _render_deck,
        artifacts.render_deck,
        ".pptx",
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

    spec = _AGENTS[kind]
    result = spec.agent.invoke(
        {"messages": [{"role": "user", "content": "Build the artifact."}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )

    structured = result["structured_response"]
    stored = None
    if spec.to_file is not None:
        stored = artifacts.store_file(
            spec.to_file(structured), structured.title, spec.extension
        )

    return Artifact(title=structured.title, content=spec.to_markdown(structured), file=stored)
