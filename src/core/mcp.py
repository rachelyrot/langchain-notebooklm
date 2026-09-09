"""Tools borrowed from MCP servers.

MCP lets the research agent use tools this project did not write — a filesystem server to
read local documents, a company's internal search, anything that speaks the protocol.
Servers are declared in one environment variable, so adding a capability means editing
`.env` rather than the code.

Two decisions worth stating:

* **They go to the research agent, not the chat agent.** The chat agent must answer only
  from the notebook — `RequireGroundingMiddleware` refuses anything else — so handing it
  outside tools would set the guardrail against the tools. Research is the part of the
  product whose job is to reach outward.
* **A server that will not start is not fatal.** Its tools are simply absent, the same way
  a missing Firecrawl key removes web research and leaves the rest working.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

ENV_VAR = "NOTEBOOKLM_MCP_SERVERS"


def connections() -> dict:
    """The configured servers, as langchain-mcp-adapters expects them."""
    raw = os.getenv(ENV_VAR, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("%s is not valid JSON, ignoring it: %s", ENV_VAR, exc)
        return {}

    if not isinstance(parsed, dict):
        logger.warning("%s must be an object of {name: connection}, ignoring it", ENV_VAR)
        return {}
    return parsed


def _run_coroutine(coroutine):
    """Run a coroutine from synchronous code, wherever that code happens to be."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)  # the ordinary case: a plain worker thread

    # Already inside a loop (an async caller): hand the work to a thread that is not.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def _make_sync(tool: BaseTool) -> BaseTool:
    """Give an MCP tool a synchronous implementation.

    MCP tools are async-only, and this application is synchronous end to end — the agents
    run under `invoke`, inside FastAPI's worker threads. Calling one without this raises
    `NotImplementedError` the moment the model reaches for it. The alternative was making
    the research path async, but `SqliteSaver` has no async methods either, so the change
    would have spread from the tool all the way to the checkpointer.
    """

    def call_sync(**kwargs):
        return _run_coroutine(tool.ainvoke(kwargs))

    async def call_async(**kwargs):
        return await tool.ainvoke(kwargs)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=call_sync,
        coroutine=call_async,
    )


def load_tools(servers: dict | None = None) -> list[BaseTool]:
    """Connect to the configured MCP servers and return their tools.

    Called once, while the agents are being built — before uvicorn owns the event loop,
    which is why ``asyncio.run`` is safe here.
    """
    servers = connections() if servers is None else servers
    if not servers:
        return []

    from langchain_mcp_adapters.client import MultiServerMCPClient

    try:
        client = MultiServerMCPClient(servers, tool_name_prefix=True)
        tools = asyncio.run(client.get_tools())
    except Exception as exc:  # an unreachable server must not stop the app from starting
        logger.warning("Could not load MCP tools from %s: %s", ", ".join(servers), exc)
        return []

    logger.info("Loaded %d MCP tool(s) from %s", len(tools), ", ".join(servers))
    return [_make_sync(tool) for tool in tools]
