"""The tools every agent uses to read the notebook.

Shared so the chat agent and the Studio artifact agents ground themselves the same way —
and so a change to how retrieval is presented to a model happens in one place.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from core.sources import format_docs
from core.store import SourceStore


def make_retrieval_tools(store: SourceStore) -> list[BaseTool]:
    """Retrieval tools bound to one notebook."""

    @tool(response_format="content_and_artifact")
    def search_sources(query: str) -> tuple[str, list[Document]]:
        """Find passages in the active sources that are relevant to a query"""
        docs = store.search(query=query)
        if not docs:
            return "No relevant documents found in the active sources", []
        return format_docs(docs), docs

    @tool
    def list_sources() -> str:
        """List the active sources by name, so you know what the notebook contains"""
        sources = [s for s in store.list() if s.active]
        if not sources:
            return "The notebook has no active sources."
        return "\n".join(f"- {s.name} ({len(s.content)} chars)" for s in sources)

    return [search_sources, list_sources]
