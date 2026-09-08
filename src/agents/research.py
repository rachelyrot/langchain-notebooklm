"""The research agent: builds sources for a topic out of the open web.

Standalone and stateless as a *product* feature — the Sources panel invokes it directly
with a topic, it is not part of the chat. Internally the run is not one shot: the agent
searches from several angles and then *proposes* pages, and
``HumanInTheLoopMiddleware`` pauses the graph before any page is fetched. The person
picks; the agent does not get to decide what enters the notebook.

A paused run lives in the checkpointer under its ``run_id``, so it survives between the
HTTP request that started it and the one that answers it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from core.memory import checkpointer
from core.sources import Source
from core.store import store
from core.web import CRAWL_LIMIT, WebUnavailable, crawl, scrape, search

MODEL = "anthropic:claude-sonnet-4-6"
RECURSION_LIMIT = 40

SYSTEM_PROMPT = """You research a topic on the open web and propose sources for a notebook.

You do not decide what the notebook keeps — a person does. Your job is to search well and \
to put a clear, varied set of candidates in front of them.

Work in this order:

1. Plan. Write 3-5 *differently angled* search queries for the topic — not rephrasings of \
each other. Cover it from several directions: the broad overview, a specific mechanism or \
detail, recent developments, criticism or the opposing view, and concrete numbers or cases.
2. Search. Call `web_search` once per query. Read the titles and descriptions.
3. Propose. Call `scrape_page` for every result that could plausibly be worth reading — \
**all of them in the same turn**, as parallel tool calls, so the person reviews one list \
instead of a trickle. Drop only the obvious junk: dead links, pure ad pages, and pages you \
have already proposed. Aim for a spread across different sites and different angles rather \
than five takes on the same story. In `reason`, say in one short line what this page \
contributes and which angle it covers — that line is what the person reads when choosing.
   If one site is clearly the authority and has several relevant pages, propose \
`crawl_site` on it instead of proposing its pages one by one.
4. Wait. Each proposal comes back either as an added source or as a rejection. A rejection \
is the person's decision, not an error: never re-propose a rejected page, and never \
propose a replacement for it unless they ask.
5. Report. Finish with a short plain-text summary: which angles you searched, what was \
added, and anything the searches did not turn up. Do not use markdown headings."""


@dataclass
class Candidate:
    """A page the agent proposed, waiting for the person to accept or drop it."""

    index: int  # position in the pending batch — this is what a decision refers to
    action: str  # "scrape_page" or "crawl_site"
    url: str
    reason: str


@dataclass
class Research:
    """The state of a research run after one leg of work."""

    run_id: str
    status: Literal["awaiting_selection", "done"]
    summary: str = ""
    sources: list[Source] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)


class UnknownRun(Exception):
    """Raised when a decision arrives for a run that is not waiting for one."""


# -- tools ---------------------------------------------------------------------


def _preview(text: str, limit: int = 400) -> str:
    """Enough for the agent to sanity-check a page without pulling it into the prompt."""
    snippet = " ".join(text.split())[:limit]
    return snippet + ("…" if len(snippet) == limit else "")


def _index(page) -> tuple[str, list[Source]]:
    """Add a fetched page to the store, unless it is already there.

    Indexing calls out to the embedding provider, which can fail (rate limits above all).
    That must stay a *tool result*, not an exception: an exception here kills the whole
    graph run mid-flight and the pages that already succeeded are lost with it.
    """
    existing = store.find_by_url(page.url)
    if existing is not None:
        return f"Already in the notebook: {existing.name} ({page.url}) — skipped.", []

    try:
        source = store.add(name=page.title, content=page.markdown, url=page.url)
    except Exception as exc:  # provider errors are the agent's problem to report, not to raise
        return f"Could not index {page.url}: {type(exc).__name__} — {exc}", []

    return (
        f"Added '{source.name}' ({page.url}), {len(page.markdown)} chars.\n"
        f"Preview: {_preview(page.markdown)}",
        [source],
    )


@tool
def web_search(query: str, limit: int = 5) -> str:
    """Search the web for one query. Returns candidate results — no page content yet."""
    try:
        results = search(query, limit=limit)
    except WebUnavailable as exc:
        return f"Search unavailable: {exc}"

    if not results:
        return f"No results for '{query}'. Try a different phrasing."
    return "\n\n".join(
        f"{i}. {r.title}\n   {r.url}\n   {r.description}" for i, r in enumerate(results, 1)
    )


@tool(response_format="content_and_artifact")
def scrape_page(url: str, reason: str) -> tuple[str, list[Source]]:
    """Propose one page as a source. Runs only if the person accepts it.

    Args:
        url: The page to fetch.
        reason: One short line — what this page contributes and which angle it covers.
    """
    try:
        return _index(scrape(url))
    except WebUnavailable as exc:
        return f"Could not add {url}: {exc}", []


@tool(response_format="content_and_artifact")
def crawl_site(url: str, reason: str, limit: int = CRAWL_LIMIT) -> tuple[str, list[Source]]:
    """Propose a whole site as sources. Runs only if the person accepts it.

    Args:
        url: The site to crawl.
        reason: One short line — why this site is the authority on the topic.
        limit: Maximum number of pages to take from it.
    """
    try:
        pages = crawl(url, limit=limit)
    except WebUnavailable as exc:
        return f"Could not crawl {url}: {exc}", []

    if not pages:
        return f"Crawling {url} returned no readable pages.", []

    lines, added = [], []
    for page in pages:
        message, sources = _index(page)
        lines.append(message.split("\n")[0])  # one line per page; skip the previews
        added.extend(sources)
    return "\n".join(lines), added


# -- the agent -----------------------------------------------------------------


def _describe(tool_call, state, runtime) -> str:
    """What the person sees for one proposal."""
    args = tool_call["args"]
    url, reason = args.get("url", ""), args.get("reason", "")
    if tool_call["name"] == "crawl_site":
        return f"Crawl {url} (up to {args.get('limit', CRAWL_LIMIT)} pages) — {reason}"
    return f"{url} — {reason}"


_review = InterruptOnConfig(allowed_decisions=["approve", "reject"], description=_describe)

_agent = create_agent(
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    tools=[web_search, scrape_page, crawl_site],
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"scrape_page": _review, "crawl_site": _review})
    ],
    checkpointer=checkpointer(),
)

def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}


def pending(run_id: str) -> list[Candidate] | None:
    """The batch a run is paused on, read back from the checkpointer.

    Deliberately not cached in a module-level dict: the interrupt is already durable
    state, so reading it from there is what lets a paused run outlive a restart.
    """
    snapshot = _agent.get_state(_config(run_id))
    if not snapshot.interrupts:
        return None
    return _candidates_of(snapshot.interrupts[0].value)

REJECTED = (
    "The person reviewed this page and chose not to add it. That is their decision — "
    "do not propose this page again and do not look for a replacement for it."
)


def _added_sources(messages: list[AnyMessage]) -> list[Source]:
    """The sources the tools actually indexed, in order, without duplicates.

    The artifacts carry ids rather than usable records once a run has been through the
    checkpointer, so the store — which is the durable copy anyway — is asked for each.
    """
    added: list[Source] = []
    seen: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage) and message.artifact:
            for item in message.artifact:
                source_id = getattr(item, "id", None)
                if source_id is None and isinstance(item, dict):
                    source_id = item.get("id")

                live = store.get(source_id) if source_id else None
                if live is not None and live.id not in seen:
                    seen.add(live.id)
                    added.append(live)
    return added


def _candidates_of(interrupt_value: dict[str, Any]) -> list[Candidate]:
    """Turn the middleware's HITL request into the list the client renders."""
    candidates = []
    for i, request in enumerate(interrupt_value.get("action_requests", [])):
        args = request.get("args", {})
        candidates.append(
            Candidate(
                index=i,
                action=request.get("name", "scrape_page"),
                url=args.get("url", ""),
                reason=args.get("reason", "") or request.get("description", ""),
            )
        )
    return candidates


def _run(run_id: str, payload: Any) -> Research:
    """Advance a run until it either needs the person or finishes."""
    result = _agent.invoke(payload, config=_config(run_id))
    messages = result.get("messages", [])

    interrupts = result.get("__interrupt__") or ()
    if interrupts:
        return Research(
            run_id=run_id,
            status="awaiting_selection",
            sources=_added_sources(messages),
            candidates=_candidates_of(interrupts[0].value),
        )

    return Research(
        run_id=run_id,
        status="done",
        summary=messages[-1].text if messages else "",
        sources=_added_sources(messages),
    )


def start(topic: str, max_candidates: int = 8) -> Research:
    """Search the web for ``topic`` and come back with pages to choose from."""
    instruction = (
        f"Research this topic and propose sources: {topic}\n"
        f"Propose at most {max_candidates} pages."
    )
    return _run(uuid.uuid4().hex[:8], {"messages": [{"role": "user", "content": instruction}]})


def decide(run_id: str, approved: list[int]) -> Research:
    """Apply the person's picks to the batch the run is paused on, then carry on.

    ``approved`` holds the indexes of the accepted candidates; every other candidate in
    the batch is rejected. The middleware wants exactly one decision per proposal, in the
    order the proposals were made.
    """
    candidates = pending(run_id)
    if candidates is None:
        raise UnknownRun(f"Research run {run_id} is not waiting for a selection.")

    picked = set(approved)
    decisions: list[dict[str, Any]] = [
        {"type": "approve"} if c.index in picked else {"type": "reject", "message": REJECTED}
        for c in candidates
    ]
    return _run(run_id, Command(resume={"decisions": decisions}))
