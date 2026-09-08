# langchain-notebooklm

A NotebookLM-style **grounded research assistant**, built as a project for learning
**LangChain v1**. The code is organized as a finished product, by feature — not by
development stage.

## The app

A NotebookLM-style 3-panel workspace, with a real client/server split:

```
┌ Sources ────────┬ Chat ───────────────┬ Studio ──────┐
│ paste / upload  │ grounded answers    │ artifacts    │
│ research the web│ with citations      │ + saved notes│
│ select / view   │                     │              │
└─────────────────┴─────────────────────┴──────────────┘
```

- **Sources** — add by pasting text, uploading `.md`/`.txt`, or **researching the web**:
  give a topic and the research agent searches it from several angles and proposes pages.
  The run then *pauses* (`HumanInTheLoopMiddleware`) and you choose which of them become
  sources — nothing is fetched before you decide. Toggle which sources are active
  (retrieval is scoped to them), view or remove a source.
- **Chat** — the main product: a single conversational agent for grounded Q&A with
  citations and short-term memory; save any answer to a note.
- **Studio** — **Summary** and **FAQ** are built: each is a standalone, stateless agent
  that returns a Pydantic object (`response_format`) rather than prose, which is then
  rendered to markdown in code and saved as a note. **Infographic** and **PowerPoint**
  still return `501` and are marked "Soon" in the UI.

## Quick start

```bash
uv sync
cp .env.example .env         # then fill in the keys
uv run notebooklm-serve      # then open http://127.0.0.1:4040
```

You need `ANTHROPIC_API_KEY` (chat) and `COHERE_API_KEY` (embeddings). `FIRECRAWL_API_KEY`
is only needed for web research — without it the server still runs and everything else
works; the Research button returns `503`.

The notebook is kept on disk under `data/` (source records in `sources.json`, chunk
embeddings in `chroma/`, agent memory in `memory.sqlite`), so it survives a restart.
Delete that directory to start over.

CLI (no server) — ground an answer in local files:

```bash
uv run notebooklm -s notes.md -s report.md "What changed between the two?"
```

Tests — no API keys needed, nothing hits the network:

```bash
uv run pytest
```

## Stack

- **Chat model:** Anthropic Claude — `anthropic:claude-sonnet-4-6`, set in
  [`agents/chat.py`](src/agents/chat.py) and [`agents/research.py`](src/agents/research.py)
- **Embeddings:** Cohere `embed-multilingual-v3.0`, in [`core/store.py`](src/core/store.py)
- **Retrieval:** Chroma, persisted to disk, cosine distance
- **Memory:** a SQLite checkpointer — chat threads and paused research runs survive a restart
- **Web:** Firecrawl (`search` / `scrape` / `crawl`)
- **Backend:** FastAPI; **client:** plain HTML/JS/CSS, no build step

### Configuration

Only these environment variables are read (`.env` is loaded automatically):

| Variable | Default | Used by |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | — | the chat and research agents |
| `COHERE_API_KEY` | — | embeddings |
| `FIRECRAWL_API_KEY` | — | web research only |
| `NOTEBOOKLM_EMBEDDING_MODEL` | `embed-multilingual-v3.0` | `core/store.py` |
| `NOTEBOOKLM_DATA_DIR` | `data` | where the notebook is kept on disk |
| `NOTEBOOKLM_HOST` | `127.0.0.1` | `api/serve.py` |
| `NOTEBOOKLM_PORT` | `4040` | `api/serve.py` |
| `NOTEBOOKLM_RELOAD` | `0` | `api/serve.py` (`1` enables uvicorn reload) |

The chat model and the embedding provider are **not** configurable by env var — change
them in the files above.

## Structure

```
client/                 web client — single-page HTML/JS/CSS, no build step
src/
  app.py                CLI: ask a question about local files
  agents/
    chat.py             the conversational chat agent (tools + short-term memory)
    research.py         web research + the human-in-the-loop selection step
    studio.py           artifact agents: a Pydantic shape per artifact
    retrieval.py        the retrieval tools every agent shares
  core/
    sources.py          the Source model, chunking, prompt formatting
    store.py            the SourceStore: Chroma, retrieval, rate limiting
    memory.py           the SQLite checkpointer both agents share
    web.py              Firecrawl access behind two small dataclasses
  api/
    schemas.py          the API contract shared with the client
    services.py         translates requests into the LangChain code
    app.py              FastAPI routes + static client
    serve.py            uvicorn entry point
tests/                  offline: fake embeddings, fake Firecrawl, scripted model
```

### API

```
GET    /api/health
GET    /api/sources                        POST /api/sources          (paste)
POST   /api/sources/upload                 GET  /api/sources/{id}     (full content)
PATCH  /api/sources/{id}   (toggle)        DELETE /api/sources/{id}
POST   /api/sources/research               → pages to choose from (or a finished run)
POST   /api/sources/research/{run_id}/decide  → accept the picks, drop the rest
POST   /api/chat
GET    /api/studio/artifacts               POST /api/studio/generate  (→ a note; 501 if unbuilt)
GET    /api/notes                          POST /api/notes            DELETE /api/notes/{id}
```

## Feature roadmap

| Feature | Status |
|---------|--------|
| Retrieval (grounded Q&A) | ✅ done — the `chat` agent's tools |
| Agent + tools | ✅ done — `agents/chat.py` |
| Short-term memory | ✅ done — checkpointer + `thread_id` on the chat agent |
| Citations | ✅ done — tools return their docs as a `ToolMessage` artifact |
| Web research | ✅ done — `agents/research.py` (needs `FIRECRAWL_API_KEY`) |
| Middlewares | ✅ done — `HumanInTheLoopMiddleware` on the research agent |
| Human in the loop | ✅ done — you pick which proposed pages become sources |
| Structured output | ✅ done — Summary + FAQ in `agents/studio.py` |
| Studio: Infographic + PowerPoint | ⏳ planned (they need file generation) |
| Event streaming | ⏳ planned |
| Guardrails | ✅ done — `agents/guardrails.py` |
| Persistence | ✅ done — Chroma + a SQLite checkpointer |
| MCP | ⏳ planned |

## Notes on the design

- **Citations come from the tool itself.** `search_sources` is a `content_and_artifact`
  tool, so the retrieved documents ride along on the `ToolMessage` and the answer is
  credited without parsing text back out of the prompt. Only the current turn counts —
  with a checkpointer the whole thread comes back, so `_cited_sources` reads from the
  last human message onward.
- **An artifact is a shape, not a document.** A Studio agent is given a Pydantic schema
  as `response_format`, so the model fills in fields — key points with their source,
  disagreements between sources, open questions — and the markdown is rendered from those
  fields in code. The same artifact could be rendered as a deck or a web page without
  asking the model again.
- **A paused research run survives between HTTP requests.** The interrupt lives in the
  agent's checkpointer under a `run_id`; the client sends the picks back to
  `/decide`, which resumes the graph with `Command(resume={"decisions": [...]})`.
- **Retrieval spreads across sources.** A scraped review can hold 200 chunks next to a
  press release's 5, so a plain top-k would always quote the biggest document. Search
  returns `TOP_K = 8` chunks with at most `MAX_PER_SOURCE = 3` from any one source, and
  tops up beyond the cap only when there is nothing else left to add. Each active source
  is queried for its own best chunks and the results are merged — one global query would
  let the big document fill the *candidate* list, crowding the others out one level above
  the cap. The query is embedded once and reused across those queries.
- **Nothing about a paused run lives in a variable.** The pending proposals are read back
  out of the checkpointer, so a research run waiting for your decision survives a restart.
- **Embedding is paced.** Several approved pages are indexed from parallel tool calls,
  and the Cohere SDK fans its own batches out concurrently, so `core/store.py` serializes
  the calls and holds them under a rolling budget of 80k tokens/minute. A single source is
  truncated at `MAX_SOURCE_CHARS = 40_000` to bound what one page can cost. Chunks are
  1000 characters with 150 of overlap.
- **Provider errors are tool results, not exceptions.** A failed scrape or a rate-limited
  embedding comes back to the model as a message, because raising would kill the whole
  graph run and lose the pages that already succeeded.
- **Guardrails protect the promise, not the model.** An answer the notebook was never
  consulted for is refused outright: it would look like every other answer while being a
  different product. Emails and card numbers are redacted from answers and from retrieved
  passages, because sources are scraped off the open web.
