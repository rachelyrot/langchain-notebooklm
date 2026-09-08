"""The live notebook: sources plus their embeddings, held in memory.

One process-wide ``store`` instance backs both the API (source CRUD) and the chat agent
(retrieval). Vectors live in numpy arrays — the corpus is small and short-lived, so there
is no vector database to run or persist.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.sources import Source, split_source

EMBEDDING_MODEL = os.getenv("NOTEBOOKLM_EMBEDDING_MODEL", "embed-multilingual-v3.0")

# Sources differ wildly in size — a scraped review can hold 200 chunks next to a press
# release's 5 — so the biggest document would win a pure top-k on sheer number of
# candidates. The per-source cap keeps an answer spread across the notebook instead of
# quoting one document eight times.
TOP_K = 8
MAX_PER_SOURCE = 3

# A scraped web page is large, and the research agent indexes several of them at once
# (approved tool calls run in parallel). Cohere's trial key allows 100k tokens/minute
# and its SDK fans its own sub-batches out concurrently, so the pacing has to happen
# here — on this side of the call — or the whole run dies on a 429.
EMBED_BATCH = 64  # texts per request
EMBED_TOKEN_BUDGET = 80_000  # per minute, kept under the trial cap
EMBED_WINDOW = 60.0
MAX_SOURCE_CHARS = 40_000  # bounds what one page can cost to embed

_embeddings: Embeddings | None = None
_embed_lock = threading.Lock()
_spent: deque[tuple[float, int]] = deque()  # (when, tokens) inside the trailing window


def get_embeddings() -> Embeddings:
    """The embedding model, built on first use.

    Deliberately lazy: the server should still start (and serve the client) when
    ``COHERE_API_KEY`` is missing — the failure then surfaces on the request that needs
    embeddings, not as a crash at import time.
    """
    global _embeddings
    if _embeddings is None:
        from langchain_cohere import CohereEmbeddings

        _embeddings = CohereEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def _reserve(tokens: int) -> None:
    """Block until ``tokens`` fit inside the trailing-minute budget."""
    while True:
        now = time.monotonic()
        while _spent and now - _spent[0][0] > EMBED_WINDOW:
            _spent.popleft()

        # A batch bigger than the whole budget can never fit; let it through alone.
        if not _spent or sum(t for _, t in _spent) + tokens <= EMBED_TOKEN_BUDGET:
            _spent.append((now, tokens))
            return

        time.sleep(EMBED_WINDOW - (now - _spent[0][0]) + 0.1)


def _embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed in paced batches, one caller at a time.

    The lock matters as much as the budget: several approved pages are indexed from
    parallel tool calls, and without it they would each spend the same allowance.
    """
    vectors: list[list[float]] = []
    with _embed_lock:
        embeddings = get_embeddings()
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            _reserve(sum(len(t) for t in batch) // 4 + 1)  # ~4 chars per token
            vectors.extend(embeddings.embed_documents(batch))
    return vectors


def _normalize(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class SourceStore:
    """Sources in the notebook, with their chunks and chunk embeddings."""

    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self._chunks: dict[str, list[Document]] = {}
        self._vectors: dict[str, np.ndarray] = {}

    # -- source management -----------------------------------------------------

    def list(self) -> list[Source]:
        return list(self._sources.values())

    def get(self, source_id: str) -> Source | None:
        return self._sources.get(source_id)

    def find_by_url(self, url: str) -> Source | None:
        """The source fetched from ``url``, if the notebook already has it."""
        return next((s for s in self._sources.values() if s.url == url), None)

    def add(self, name: str, content: str, url: str | None = None) -> Source:
        """Add a source and index it for retrieval."""
        if len(content) > MAX_SOURCE_CHARS:
            content = content[:MAX_SOURCE_CHARS] + "\n\n[… truncated]"

        source = Source(name=name, content=content, url=url)
        chunks = split_source(source)

        # Embed first: a source that failed to index must not be left in the notebook
        # as a silently unsearchable entry.
        vectors = _normalize(_embed_documents([c.page_content for c in chunks])) if chunks else None

        self._sources[source.id] = source
        self._chunks[source.id] = chunks
        if vectors is not None:
            self._vectors[source.id] = vectors
        return source

    def set_active(self, source_id: str, active: bool) -> Source | None:
        source = self._sources.get(source_id)
        if source is None:
            return None
        source.active = active
        return source

    def remove(self, source_id: str) -> bool:
        if source_id not in self._sources:
            return False
        del self._sources[source_id]
        self._chunks.pop(source_id, None)
        self._vectors.pop(source_id, None)
        return True

    def active_ids(self) -> list[str]:
        return [s.id for s in self._sources.values() if s.active]

    # -- retrieval -------------------------------------------------------------

    def search(
        self, query: str, k: int = TOP_K, max_per_source: int = MAX_PER_SOURCE
    ) -> list[Document]:
        """The ``k`` chunks most similar to ``query``, scoped to the active sources.

        Ranking is plain cosine similarity, but the picking is not: at most
        ``max_per_source`` chunks come from any one source, so a large document cannot
        crowd out the rest of the notebook. If the cap leaves room — few active sources,
        say — the remainder is filled with the next best chunks regardless of origin.
        """
        candidates = [
            (source_id, self._chunks[source_id], self._vectors[source_id])
            for source_id in self.active_ids()
            if source_id in self._vectors
        ]
        if not candidates:
            return []

        matrix = np.vstack([vectors for _, _, vectors in candidates])
        chunks = [chunk for _, source_chunks, _ in candidates for chunk in source_chunks]

        _reserve(len(query) // 4 + 1)  # a query is tiny, but it spends from the same budget
        query_vector = _normalize([get_embeddings().embed_query(query)])[0]
        scores = matrix @ query_vector  # cosine similarity — both sides are normalized

        ranked = [int(i) for i in np.argsort(scores)[::-1]]
        taken: dict[str, int] = {}
        picked: list[int] = []
        for i in ranked:
            source_id = chunks[i].metadata.get("source_id")
            if taken.get(source_id, 0) >= max_per_source:
                continue
            taken[source_id] = taken.get(source_id, 0) + 1
            picked.append(i)
            if len(picked) == k:
                break

        if len(picked) < k:  # the cap left room: top up with the best of what is left
            spare = [i for i in ranked if i not in set(picked)]
            picked.extend(spare[: k - len(picked)])
            picked.sort(key=ranked.index)

        return [chunks[i] for i in picked]


store = SourceStore()
