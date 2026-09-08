"""The notebook: sources on disk, their chunks in a Chroma vector store.

The store is the only thing above the model layer that knows retrieval exists, so it is
also the only place that had to change when the vectors moved out of process memory: the
methods below are the same ones `api/services.py` and the agents were already calling.

Two files make up a notebook, both under ``NOTEBOOKLM_DATA_DIR``:

* ``sources.json`` — the Source records (name, full text, active flag, url)
* ``chroma/``      — the chunk embeddings, queried with a metadata filter

Chroma is built lazily. Listing, toggling and removing sources need no embedding provider,
so the server still starts without a key; only indexing and searching do.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.sources import Source, split_source

EMBEDDING_MODEL = os.getenv("NOTEBOOKLM_EMBEDDING_MODEL", "embed-multilingual-v3.0")
DATA_DIR = Path(os.getenv("NOTEBOOKLM_DATA_DIR", "data"))
COLLECTION = "notebook"

# Sources differ wildly in size — a scraped review can hold 200 chunks next to a press
# release's 5 — so the biggest document would win a pure top-k on sheer number of
# candidates. The per-source cap keeps an answer spread across the notebook instead of
# quoting one document eight times.
TOP_K = 8
MAX_PER_SOURCE = 3

# A scraped web page is large, and the research agent indexes several of them at once
# (approved tool calls run in parallel). Cohere's trial key allows 100k tokens/minute
# and its SDK fans its own sub-batches out concurrently, so the pacing has to happen
# on this side of the call or the whole run dies on a 429.
EMBED_BATCH = 64  # texts per request
EMBED_TOKEN_BUDGET = 80_000  # per minute, kept under the trial cap
EMBED_WINDOW = 60.0
MAX_SOURCE_CHARS = 40_000  # bounds what one page can cost to embed

_embed_lock = threading.Lock()
_spent: deque[tuple[float, int]] = deque()  # (when, tokens) inside the trailing window


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


class ThrottledEmbeddings(Embeddings):
    """Paces another embedding model, and hands *that* to Chroma.

    The vector store calls the embedding model itself, so the pacing has to live inside
    the model rather than around the call sites — otherwise indexing through Chroma would
    quietly bypass it.
    """

    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with _embed_lock:  # parallel tool calls must not spend the same allowance twice
            for start in range(0, len(texts), EMBED_BATCH):
                batch = texts[start : start + EMBED_BATCH]
                _reserve(sum(len(t) for t in batch) // 4 + 1)  # ~4 chars per token
                vectors.extend(self._inner.embed_documents(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        _reserve(len(text) // 4 + 1)  # tiny, but it spends from the same budget
        return self._inner.embed_query(text)


def default_embeddings() -> Embeddings:
    """Cohere, paced. Built on first use so a missing key is not a startup crash."""
    from langchain_cohere import CohereEmbeddings

    return ThrottledEmbeddings(CohereEmbeddings(model=EMBEDDING_MODEL))


class SourceStore:
    """Sources in the notebook, with their chunks and chunk embeddings."""

    def __init__(self, data_dir: Path | str | None = None, embeddings: Embeddings | None = None):
        self._dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self._embeddings = embeddings
        self._vectorstore = None
        self._open_lock = threading.Lock()
        self._sources: dict[str, Source] = self._read()

    # -- persistence -----------------------------------------------------------

    @property
    def _record_file(self) -> Path:
        return self._dir / "sources.json"

    def _read(self) -> dict[str, Source]:
        if not self._record_file.is_file():
            return {}
        records = json.loads(self._record_file.read_text(encoding="utf-8"))
        return {r["id"]: Source(**r) for r in records}

    def _write(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "id": s.id,
                "name": s.name,
                "content": s.content,
                "active": s.active,
                "url": s.url,
                "chunks": s.chunks,
            }
            for s in self._sources.values()
        ]
        self._record_file.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    @property
    def _vectors(self):
        """The Chroma collection, opened on first use.

        Locked because the research agent indexes several approved pages from parallel
        tool calls: without it they race to create the client, and Chroma's own client
        registry is not safe against that.
        """
        with self._open_lock:
            if self._vectorstore is not None:
                return self._vectorstore

            from langchain_chroma import Chroma

            if self._embeddings is None:
                self._embeddings = default_embeddings()
            self._vectorstore = Chroma(
                collection_name=COLLECTION,
                embedding_function=self._embeddings,
                persist_directory=str(self._dir / "chroma"),
                # Chroma defaults to L2; text embeddings are compared by angle, not
                # distance, and the two only agree when every vector is normalized.
                collection_configuration={"hnsw": {"space": "cosine"}},
            )
            return self._vectorstore

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
        source.chunks = len(chunks)

        # Index first: a source that failed to embed must not be left in the notebook
        # as a silently unsearchable entry.
        if chunks:
            self._vectors.add_documents(chunks, ids=self._chunk_ids(source))

        self._sources[source.id] = source
        self._write()
        return source

    def set_active(self, source_id: str, active: bool) -> Source | None:
        source = self._sources.get(source_id)
        if source is None:
            return None
        source.active = active
        self._write()
        return source

    def remove(self, source_id: str) -> bool:
        source = self._sources.pop(source_id, None)
        if source is None:
            return False
        if source.chunks:
            self._vectors.delete(ids=self._chunk_ids(source))
        self._write()
        return True

    def active_ids(self) -> list[str]:
        return [s.id for s in self._sources.values() if s.active]

    @staticmethod
    def _chunk_ids(source: Source) -> list[str]:
        return [f"{source.id}-{i}" for i in range(source.chunks)]

    # -- retrieval -------------------------------------------------------------

    def search(
        self, query: str, k: int = TOP_K, max_per_source: int = MAX_PER_SOURCE
    ) -> list[Document]:
        """The ``k`` chunks most similar to ``query``, scoped to the active sources.

        Each active source is queried separately for its own best ``k`` chunks, and the
        results are merged and re-ranked. One global query would not do: a 200-chunk
        review fills the candidate list before the smaller sources are ever considered,
        so they would be crowded out one level above the cap. The query is embedded once
        and reused, so the extra queries cost nothing but local index lookups.

        At most ``max_per_source`` of the merged chunks come from any one source. If the
        cap leaves room — few active sources, say — the remainder is filled with the next
        best chunks regardless of origin.
        """
        active = self.active_ids()
        if not active:
            return []

        vectors = self._vectors  # also resolves the embedding model
        query_vector = self._embeddings.embed_query(query)

        scored: list[tuple[Document, float]] = []
        for source_id in active:
            scored += vectors.similarity_search_by_vector_with_relevance_scores(
                query_vector, k=k, filter={"source_id": source_id}
            )
        scored.sort(key=lambda pair: pair[1])  # Chroma returns a distance: lower is closer
        ranked = [doc for doc, _ in scored]

        taken: dict[str, int] = {}
        picked: list[Document] = []
        for doc in ranked:
            source_id = doc.metadata.get("source_id")
            if taken.get(source_id, 0) >= max_per_source:
                continue
            taken[source_id] = taken.get(source_id, 0) + 1
            picked.append(doc)
            if len(picked) == k:
                break

        if len(picked) < k:  # the cap left room: top up with the best of what is left
            picked += [d for d in ranked if d not in picked][: k - len(picked)]
            picked.sort(key=ranked.index)

        return picked


store = SourceStore()
