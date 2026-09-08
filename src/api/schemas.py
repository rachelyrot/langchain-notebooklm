"""The API contract shared between backend and client.

These Pydantic models are the *stable* boundary: the web client is written against them,
and each stage fills in the backend behind them. Keep them additive.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Impl = Literal["A", "B"]


# -- sources -------------------------------------------------------------------


class SourceInfo(BaseModel):
    """A document in the notebook (list view — no content)."""

    id: str
    name: str
    chars: int
    active: bool
    url: str | None = None  # set for sources found by web research


class SourceDetail(SourceInfo):
    """A source with its full text (for the viewer)."""

    content: str


class AddSourceRequest(BaseModel):
    """Add a source by pasting text."""

    name: str | None = None
    content: str


class SetActiveRequest(BaseModel):
    active: bool


class ResearchRequest(BaseModel):
    """Build sources for a topic from the open web."""

    topic: str
    max_candidates: int = 8


class ResearchCandidate(BaseModel):
    """A page the agent proposed. Nothing is fetched until the user accepts it."""

    index: int
    action: str  # "scrape_page" (one page) or "crawl_site" (a whole site)
    url: str
    reason: str


class ResearchDecision(BaseModel):
    """The user's picks: the indexes to accept. Everything else in the batch is dropped."""

    approved: list[int] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    """One leg of a research run.

    ``awaiting_selection`` means the run is paused on ``candidates`` and expects a
    decision at ``/api/sources/research/{run_id}/decide``; ``done`` means it finished.
    """

    run_id: str
    status: Literal["awaiting_selection", "done"]
    summary: str = ""
    candidates: list[ResearchCandidate] = Field(default_factory=list)
    sources: list[SourceInfo] = Field(default_factory=list)


# -- chat ----------------------------------------------------------------------


class Citation(BaseModel):
    """A source the answer was grounded on."""

    source: str
    snippet: str | None = None


class ChatRequest(BaseModel):
    """A grounded-chat turn."""

    message: str
    thread_id: str | None = None  # used from stage 4 (short-term memory)


class ChatResponse(BaseModel):
    """The model's grounded answer plus the sources it was based on."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    engine: str


# -- studio (artifacts) --------------------------------------------------------


class ArtifactKind(BaseModel):
    """A generatable artifact shown in the Studio panel."""

    key: str
    title: str
    icon: str
    status: Literal["ready", "planned"]


class GenerateArtifactRequest(BaseModel):
    kind: str
    impl: Impl = "A"


# -- notes ---------------------------------------------------------------------


class Note(BaseModel):
    id: str
    title: str
    content: str


class AddNoteRequest(BaseModel):
    title: str
    content: str

