"""The notebook: ingestion, scoping to active sources, and retrieval spread."""

from __future__ import annotations

import threading
import time
from collections import Counter

import core.store as store_module


def test_add_indexes_and_tags_chunks_with_their_origin(store):
    source = store.add(name="pricing.md", content="Our pricing tiers are Basic and Pro.")

    chunks = store._chunks[source.id]
    assert chunks
    assert all(c.metadata["source_id"] == source.id for c in chunks)
    assert all(c.metadata["source_name"] == "pricing.md" for c in chunks)


def test_crud_round_trip(store):
    source = store.add(name="a.md", content="some content", url="https://a")

    assert store.get(source.id) is source
    assert store.find_by_url("https://a") is source
    assert store.active_ids() == [source.id]

    store.set_active(source.id, False)
    assert store.active_ids() == []

    assert store.remove(source.id) is True
    assert store.remove(source.id) is False
    assert store.list() == []


def test_search_is_scoped_to_active_sources(store):
    pricing = store.add(name="pricing.md", content="Our pricing tiers are Basic, Pro and Enterprise.")
    store.add(name="hiring.md", content="We hired twelve engineers this quarter.")

    hits = store.search("what are the pricing tiers?")
    assert hits[0].metadata["source_name"] == "pricing.md"

    store.set_active(pricing.id, False)
    hits = store.search("what are the pricing tiers?")
    assert all(h.metadata["source_name"] != "pricing.md" for h in hits)


def test_search_on_an_empty_notebook_returns_nothing(store):
    assert store.search("anything") == []


def test_a_large_source_cannot_crowd_out_the_others(store):
    """The reason MAX_PER_SOURCE exists: chunk count must not decide the answer."""
    store.add(name="big-review.md", content="perovskite tandem efficiency record data " * 800)
    for i, name in enumerate(["longi.md", "ossila.md", "doe.md"]):
        store.add(name=name, content=f"{name} reports a perovskite tandem efficiency record. " * 60)

    hits = store.search("perovskite tandem efficiency record")
    counts = Counter(h.metadata["source_name"] for h in hits)

    assert len(hits) == store_module.TOP_K
    assert max(counts.values()) <= store_module.MAX_PER_SOURCE
    assert len(counts) >= 3, f"answer would rest on too few sources: {counts}"

    # without the cap the biggest document takes every slot
    unc = store.search("perovskite tandem efficiency record", k=4, max_per_source=4)
    assert len({h.metadata["source_name"] for h in unc}) == 1


def test_the_cap_does_not_starve_a_single_source_notebook(store):
    store.add(name="only.md", content="perovskite tandem efficiency record data " * 800)

    hits = store.search("perovskite tandem efficiency record")
    assert len(hits) == store_module.TOP_K
    assert {h.metadata["source_name"] for h in hits} == {"only.md"}


def test_oversized_content_is_truncated(store):
    source = store.add(name="huge.md", content="x " * store_module.MAX_SOURCE_CHARS)

    assert len(source.content) <= store_module.MAX_SOURCE_CHARS + 20
    assert source.content.endswith("[… truncated]")


def test_embedding_calls_are_serialized_and_paced(store, monkeypatch):
    """Parallel indexing must not spend the same per-minute allowance twice."""
    calls: list[tuple[float, int]] = []
    lock = threading.Lock()

    class Counting:
        def embed_documents(self, texts):
            with lock:
                calls.append((time.monotonic(), sum(len(t) for t in texts) // 4))
            return [[1.0] * 8 for _ in texts]

        def embed_query(self, text):
            return [1.0] * 8

    monkeypatch.setattr(store_module, "_embeddings", Counting())
    monkeypatch.setattr(store_module, "EMBED_WINDOW", 1.0)
    monkeypatch.setattr(store_module, "EMBED_TOKEN_BUDGET", 4_000)

    # ~2.4k tokens per page: one fits inside the budget, two in the same window do not
    page = "efficiency data " * 500
    threads = [
        threading.Thread(target=store.add, kwargs={"name": f"p{i}", "content": page})
        for i in range(3)
    ]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - started

    assert len(store.list()) == 3
    assert elapsed > store_module.EMBED_WINDOW, "the throttle never waited"
    for at, _ in calls:
        spent = sum(tok for t, tok in calls if at <= t < at + store_module.EMBED_WINDOW)
        assert spent <= store_module.EMBED_TOKEN_BUDGET
