# langchain-notebooklm

A NotebookLM-style **grounded research assistant**, built as a project for learning
**LangChain v1**. The code is organized as a finished product, by feature — not by
development stage.


## Stack
- **Chat model:** Anthropic Claude (`anthropic:claude-sonnet-4-6` by default)
- **Embeddings:** Cohere (`embed-multilingual-v3.0` by default)
- Everything is provider-agnostic via env vars — see [`.env.example`](.env.example).

## The app

A NotebookLM-style 3-panel workspace, with a real client/server split:

```
┌ Sources ────┬ Chat ───────────────┬ Studio ──────┐
│ add / upload│ grounded answers    │ artifacts    │
│ select      │ with citations      │ + saved notes│
│ view / del  │                     │              │
└─────────────┴─────────────────────┴──────────────┘
```

- **Sources** — add by pasting text, uploading `.md`/`.txt`, or **researching the web**:
  give a topic and the research agent searches it from several angles and proposes pages.
  The run then *pauses* (`HumanInTheLoopMiddleware`) and you choose which of them become
  sources — nothing is fetched before you decide. Toggle which sources are active
  (retrieval is scoped to them), view or remove a source.
- **Chat** — the main product: a single conversational agent for grounded Q&A with
  citations and short-term memory; save any answer to a note.
- **Studio** — generate artifacts: **Infographic · PowerPoint · Summary · FAQ**
  (each will be its own standalone agent; PowerPoint uses a `.pptx` generation skill).

## Structure

```
client/                 web client — single-page HTML/JS/CSS, no build step
src/
  app.py                CLI entry point (ask a question about local files)
  agents/               one agent per feature
    chat.py             the conversational chat agent (tools + short-term memory)
    research.py         web research + the human-in-the-loop selection step
  core/
    sources.py          the Source model, chunking, prompt formatting
    store.py            the live SourceStore: embeddings, retrieval, rate limiting
    web.py              Firecrawl access behind two small dataclasses
  api/                  FastAPI backend: schemas (contract), services, routes
tests/                  offline: fake embeddings, fake Firecrawl, scripted model
```

The chat is one simple agent with the retrieval tools and memory. The Studio artifact
generators will be standalone, stateless agents (invoked directly by the Studio, not part of
the chat). Capabilities not built yet return `501` and the UI shows a "coming soon" notice.

## Quick start

```bash
uv sync
cp .env.example .env         # ANTHROPIC_API_KEY + COHERE_API_KEY; FIRECRAWL_API_KEY for web research
uv run notebooklm-serve      # then open http://127.0.0.1:4040
```

The notebook starts empty and lives in memory: add sources by pasting, uploading, or
researching the web. Restarting the server clears it.

CLI (no server) — ground an answer in local files:

```bash
uv run notebooklm -s notes.md -s report.md "What changed between the two?"
```

Tests (no API keys needed, nothing hits the network):

```bash
uv run pytest
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
| Structured output (Studio artifacts) | ⏳ planned |
| Event streaming | ⏳ planned |
| Guardrails | ⏳ planned |
| MCP | ⏳ planned |

## Notes on the design

- **Citations** come from the tool itself: `search_sources` is a
  `content_and_artifact` tool, so the retrieved documents ride along on the `ToolMessage`
  and the answer is credited without parsing text back out of the prompt.
- **Retrieval spreads across sources.** A scraped review can hold 200 chunks next to a
  press release's 5, so a plain top-k would always quote the biggest document;
  `MAX_PER_SOURCE` caps how much any one source can contribute.
- **Embedding is rate limited** in `core/store.py` — several approved pages are indexed
  from parallel tool calls, and a trial embedding key will 429 without pacing.
- **Nothing persists.** Sources, chat memory and paused research runs all live in
  process memory; a restart is a clean slate.
