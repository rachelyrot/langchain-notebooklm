"""Backend services: translate API requests into the LangChain code.

This is the thin layer between the stable API contract and the per-stage implementations.
Source management and notes are product glue (in-memory); chat is powered by the stage-2
agent; Studio artifacts raise ``ComingSoon`` until structured output lands in stage 3.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from api.schemas import (
    ArtifactKind,
    ChatRequest,
    ChatResponse,
    Citation,
    GeneratedArtifact,
    Note,
    ResearchCandidate,
    ResearchDecision,
    ResearchRequest,
    ResearchResponse,
    SourceDetail,
    SourceInfo,
)
from agents import chat, research, studio
from core import web
from core.store import store


class ComingSoon(Exception):
    """Raised for a product capability that is advertised but not wired up yet."""


# -- sources -------------------------------------------------------------------


def _to_info(source) -> SourceInfo:
    return SourceInfo(
        id=source.id,
        name=source.name,
        chars=len(source.content),
        active=source.active,
        url=source.url,
    )


def list_sources() -> list[SourceInfo]:
    return [_to_info(s) for s in store.list()]


def get_source(source_id: str) -> SourceDetail | None:
    source = store.get(source_id)
    if source is None:
        return None
    return SourceDetail(
        id=source.id,
        name=source.name,
        chars=len(source.content),
        active=source.active,
        content=source.content,
    )


def add_source(name: str | None, content: str) -> SourceInfo:
    name = (name or "").strip() or _auto_name(content)
    return _to_info(store.add(name=name, content=content))


def set_source_active(source_id: str, active: bool) -> SourceInfo | None:
    source = store.set_active(source_id, active)
    return _to_info(source) if source else None


def remove_source(source_id: str) -> bool:
    return store.remove(source_id)


def _to_research(result: research.Research) -> ResearchResponse:
    return ResearchResponse(
        run_id=result.run_id,
        status=result.status,
        summary=result.summary,
        candidates=[
            ResearchCandidate(index=c.index, action=c.action, url=c.url, reason=c.reason)
            for c in result.candidates
        ],
        sources=[_to_info(s) for s in result.sources],
    )


def start_research(req: ResearchRequest) -> ResearchResponse:
    """Search the web for a topic and come back with pages for the user to choose from."""
    # Fail fast on a missing key: inside the tools it would only look like a failed search.
    web.get_client()

    return _to_research(research.start(req.topic, max_candidates=req.max_candidates))


def decide_research(run_id: str, decision: ResearchDecision) -> ResearchResponse:
    """Apply the user's picks and let the run continue."""
    return _to_research(research.decide(run_id, decision.approved))


def _auto_name(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else "Pasted source"
    first_line = first_line.lstrip("# ").strip()
    return (first_line[:40] or "Pasted source") + ".txt"


# -- chat ----------------------------------------------------------------------


NO_SOURCES = "No active sources. Enable at least one source on the left to chat."


def run_chat(req: ChatRequest) -> ChatResponse:
    """Answer a chat turn with the chat agent, grounded in the active sources."""
    if not store.active_ids():
        return ChatResponse(answer=NO_SOURCES, engine="chat")

    result = chat.answer(req.message, thread_id=req.thread_id or "default")
    citations = [Citation(source=name) for name in result.sources]
    return ChatResponse(answer=result.text, citations=citations, engine="chat")


def stream_chat(req: ChatRequest) -> Iterator[dict]:
    """The same turn as ``run_chat``, event by event, for the streaming endpoint."""
    if not store.active_ids():
        yield {"type": "token", "text": NO_SOURCES}
        yield {"type": "done", "sources": []}
        return

    try:
        yield from chat.stream(req.message, thread_id=req.thread_id or "default")
    except Exception as exc:  # the stream has already started; the client cannot get a 500
        yield {"type": "error", "detail": f"{type(exc).__name__}: {exc}"}


# -- studio (artifacts) --------------------------------------------------------

def _status(key: str) -> str:
    return "ready" if key in studio.KINDS else "planned"


ARTIFACTS: list[ArtifactKind] = [
    ArtifactKind(key="infographic", title="Infographic", icon="📊", status=_status("infographic")),
    ArtifactKind(key="powerpoint", title="PowerPoint", icon="📑", status=_status("powerpoint")),
    ArtifactKind(key="summary", title="Summary", icon="📄", status=_status("summary")),
    ArtifactKind(key="faq", title="FAQ", icon="❓", status=_status("faq")),
]

_ARTIFACTS_BY_KEY = {a.key: a for a in ARTIFACTS}


def list_artifacts() -> list[ArtifactKind]:
    return ARTIFACTS


def generate_artifact(kind: str, impl: str) -> GeneratedArtifact:
    """Build an artifact from the active sources and keep it as a note."""
    if kind not in studio.KINDS:
        artifact = _ARTIFACTS_BY_KEY.get(kind)
        title = artifact.title if artifact else "This artifact"
        raise ComingSoon(f"{title} generation is coming soon.")

    result = studio.generate(kind)
    return GeneratedArtifact(
        kind=kind,
        note=add_note(title=result.title, content=result.content),
        download_url=result.file.url if result.file else None,
        download_name=result.file.download_name if result.file else None,
    )


# -- notes ---------------------------------------------------------------------

_NOTES: dict[str, Note] = {}


def list_notes() -> list[Note]:
    return list(_NOTES.values())


def add_note(title: str, content: str) -> Note:
    note = Note(id=uuid.uuid4().hex[:8], title=title.strip() or "Untitled note", content=content)
    _NOTES[note.id] = note
    return note


def remove_note(note_id: str) -> bool:
    return _NOTES.pop(note_id, None) is not None
