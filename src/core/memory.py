"""Checkpointers: the agents' memory, kept on disk beside the notebook.

A chat thread is cheap to recreate — you ask again — but a *paused research run* is not:
it holds search results the model already paid for and a batch of proposals waiting for a
decision. A restart in the middle of choosing used to throw that away.

One SQLite file, one connection, shared by both agents. FastAPI runs sync endpoints in a
thread pool, hence ``check_same_thread=False``.
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from core.store import DATA_DIR

_saver: SqliteSaver | None = None


def checkpointer() -> SqliteSaver:
    """The process-wide checkpointer, opened on first use."""
    global _saver
    if _saver is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(DATA_DIR / "memory.sqlite", check_same_thread=False)
        _saver = SqliteSaver(connection)
        _saver.setup()
    return _saver
