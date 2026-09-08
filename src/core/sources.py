"""Source ingestion: the document model, chunking, and prompt formatting.

A *source* is one document in the notebook. Retrieval works on chunks, so every chunk
carries its parent's id and name in ``metadata`` — that is what lets an answer be cited.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@dataclass
class Source:
    """A document in the notebook.

    ``url`` is set for sources brought in from the web (see ``agents/research.py``);
    pasted and uploaded sources leave it ``None``.
    """

    name: str
    content: str
    active: bool = True
    url: str | None = None
    chunks: int = 0  # how many chunks it was split into — the store deletes them by id
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def split_source(source: Source) -> list[Document]:
    """Split a source into retrievable chunks, each tagged with its origin.

    ``url`` is left out entirely when there is none: the vector store rejects null
    metadata values, so an absent key is the only way to say "not from the web".
    """
    origin = {"source_id": source.id, "source_name": source.name}
    if source.url:
        origin["url"] = source.url

    return [
        Document(page_content=text, metadata={**origin, "chunk": i})
        for i, text in enumerate(_splitter.split_text(source.content))
    ]


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks for the model, each labelled with its source name."""
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source_name', 'unknown')}] {doc.page_content}"
        for doc in docs
    )
