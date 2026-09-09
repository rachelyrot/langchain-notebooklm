"""Borrowing tools from an MCP server, and surviving one that is not there."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

import core.mcp as mcp

STUB = Path(__file__).parent / "mcp_stub_server.py"


@pytest.fixture
def stub_connection():
    """A real MCP server, spawned over stdio with this interpreter."""
    return {
        "archive": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(STUB)],
        }
    }


def test_tools_are_loaded_from_a_live_server(stub_connection):
    tools = mcp.load_tools(stub_connection)

    names = {t.name for t in tools}
    assert len(tools) == 2, names
    assert any(name.endswith("internal_search") for name in names)
    assert any(name.endswith("read_archive") for name in names)


def test_a_borrowed_tool_runs_from_synchronous_code(stub_connection):
    """MCP tools are async-only; the app is not. The wrapper is what bridges them."""
    tools = mcp.load_tools(stub_connection)
    search = next(t for t in tools if t.name.endswith("internal_search"))

    assert "ARCH-1" in str(search.invoke({"query": "perovskite"}))
    assert "ARCH-1" in str(asyncio.run(search.ainvoke({"query": "perovskite"})))


def test_the_raw_adapter_tool_has_no_sync_path(stub_connection):
    """Guards the reason the wrapper exists, so nobody removes it as ceremony."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    raw = asyncio.run(MultiServerMCPClient(stub_connection).get_tools())

    with pytest.raises(NotImplementedError):
        raw[0].invoke({"query": "perovskite"})


def test_an_agent_calls_a_borrowed_tool_through_a_sync_invoke(stub_connection):
    """End to end through a graph, exactly as the research agent runs in production."""
    tools = mcp.load_tools(stub_connection)
    search = next(t for t in tools if t.name.endswith("internal_search"))

    script = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": search.name, "args": {"query": "perovskite"}, "id": "m1", "type": "tool_call"}
            ],
        ),
        AIMessage("The archive has ARCH-1 and ARCH-2."),
    ]

    class Scripted(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    agent = create_agent(model=Scripted(responses=script), tools=tools)
    result = agent.invoke({"messages": [{"role": "user", "content": "search"}]})

    tool_output = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    # MCP answers in content blocks rather than a bare string; `.text` flattens them
    assert "ARCH-1" in tool_output.text


def test_the_wrapper_works_when_a_loop_is_already_running(stub_connection):
    """A sync call from inside a running loop must not deadlock on it."""
    tools = mcp.load_tools(stub_connection)
    search = next(t for t in tools if t.name.endswith("internal_search"))

    async def from_inside_a_loop():
        return await asyncio.to_thread(search.invoke, {"query": "perovskite"})

    assert "ARCH-1" in str(asyncio.run(from_inside_a_loop()))


def test_tool_names_are_prefixed_by_server(stub_connection):
    """Two servers may both offer `search`; the prefix keeps them apart."""
    tools = mcp.load_tools(stub_connection)

    assert all(t.name.startswith("archive") for t in tools), {t.name for t in tools}


def test_a_server_that_will_not_start_is_not_fatal():
    """The app must boot without the tools rather than not boot."""
    tools = mcp.load_tools(
        {"broken": {"transport": "stdio", "command": sys.executable, "args": ["-c", "raise SystemExit(1)"]}}
    )

    assert tools == []


def test_no_configuration_means_no_tools(monkeypatch):
    monkeypatch.delenv(mcp.ENV_VAR, raising=False)

    assert mcp.connections() == {}
    assert mcp.load_tools() == []


def test_the_configuration_is_read_from_the_environment(monkeypatch, stub_connection):
    monkeypatch.setenv(mcp.ENV_VAR, json.dumps(stub_connection))

    assert mcp.connections() == stub_connection
    assert len(mcp.load_tools()) == 2


@pytest.mark.parametrize("garbage", ["{not json", "[]", '"a string"', "null"])
def test_malformed_configuration_is_ignored_rather_than_raised(monkeypatch, garbage):
    monkeypatch.setenv(mcp.ENV_VAR, garbage)

    assert mcp.connections() == {}
