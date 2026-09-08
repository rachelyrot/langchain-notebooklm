"""The CLI: ask the notebook a question without running the server.

    uv run notebooklm -s notes.md -s report.md "What changed between the two?"

Sources are given on the command line because the notebook itself lives in the server's
memory — there is nothing to inherit from a previous run.
"""

from __future__ import annotations

from netfree_unstrict_ssl import unstrict_ssl

unstrict_ssl()

from dotenv import load_dotenv

load_dotenv()

import argparse
import sys
from pathlib import Path

from agents import chat
from core.store import store


def _load(paths: list[str]) -> int:
    """Read each file into the notebook. Returns how many were indexed."""
    added = 0
    for path in paths:
        file = Path(path)
        try:
            content = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"skipping {file}: {exc}", file=sys.stderr)
            continue
        if not content.strip():
            print(f"skipping {file}: empty", file=sys.stderr)
            continue
        store.add(name=file.name, content=content)
        added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="notebooklm", description="Ask a grounded question about local source files."
    )
    parser.add_argument("question", help="the question to ask")
    parser.add_argument(
        "-s",
        "--source",
        action="append",
        default=[],
        metavar="FILE",
        help="a .md/.txt file to ground the answer in (repeatable)",
    )
    args = parser.parse_args()

    if not args.source:
        parser.error("give at least one source with -s/--source, or run the server instead")
    if not _load(args.source):
        print("No readable sources — nothing to ground an answer in.", file=sys.stderr)
        raise SystemExit(1)

    answer = chat.answer(args.question, thread_id="cli")
    print(f"\n{answer.text}\n")
    if answer.sources:
        print("sources: " + ", ".join(answer.sources))


if __name__ == "__main__":
    main()
