"""A minimal MCP server, run over stdio by the MCP tests.

Not a mock: the tests spawn this as a real subprocess and speak the protocol to it, so the
loading path they exercise is the same one a filesystem or company server would take.
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("notebook-test-stub")


@server.tool()
def internal_search(query: str) -> str:
    """Search an imaginary internal archive for a query."""
    return f"Two internal documents mention '{query}': ARCH-1 and ARCH-2."


@server.tool()
def read_archive(document_id: str) -> str:
    """Read one document out of the imaginary archive."""
    return f"Full text of {document_id}."


if __name__ == "__main__":
    server.run(transport="stdio")
