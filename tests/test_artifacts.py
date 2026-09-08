"""The file-shaped artifacts: what gets rendered, and what may be fetched back."""

from __future__ import annotations

import zipfile
from io import BytesIO

import pytest

import agents.artifacts as artifacts
import agents.studio as studio


@pytest.fixture(autouse=True)
def artifact_dir(tmp_path, monkeypatch):
    """Keep generated files out of the real notebook."""
    monkeypatch.setattr(artifacts, "ARTIFACT_DIR", tmp_path / "artifacts")
    return tmp_path / "artifacts"


@pytest.fixture
def infographic():
    return studio.Infographic(
        title="Perovskite efficiency in 2026",
        subtitle="Where tandem cells stand",
        stats=[studio.Stat(value="34.85%", label="tandem record", source="longi.md")],
        sections=[studio.Section(heading="Blockers", points=["Stability under heat"])],
        takeaway="Efficiency is solved; lifetime is not.",
    )


@pytest.fixture
def deck():
    return studio.Deck(
        title="Perovskite briefing",
        subtitle="Eight sources, one argument",
        slides=[
            studio.Slide(
                title="The record",
                bullets=["34.85% tandem", "Certified 2025"],
                source="longi.md",
            ),
            studio.Slide(
                title="The blocker",
                bullets=["Heat and humidity"],
                source="review.md",
            ),
        ],
    )


def test_the_infographic_is_a_self_contained_page(infographic):
    html = artifacts.render_infographic(infographic).decode()

    assert html.startswith("<!doctype html>")
    assert "34.85%" in html and "tandem record" in html
    assert "longi.md" in html, "a figure without its source is not attributable"
    assert "Efficiency is solved" in html
    assert "<style>" in html and "http://" not in html, "must not depend on external assets"


def test_the_infographic_escapes_source_text():
    """Source text comes off the open web; it must never become markup."""
    hostile = studio.Infographic(
        title="<script>alert(1)</script>",
        subtitle="ok",
        stats=[studio.Stat(value="1", label="<img onerror=x>", source="evil.md")],
        sections=[],
        takeaway="ok",
    )

    html = artifacts.render_infographic(hostile).decode()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<img onerror=x>" not in html


def test_the_deck_is_a_real_pptx_with_one_slide_per_entry(deck):
    data = artifacts.render_deck(deck)

    assert data[:2] == b"PK", "a .pptx is a zip archive"
    with zipfile.ZipFile(BytesIO(data)) as archive:
        slides = [n for n in archive.namelist() if n.startswith("ppt/slides/slide")]
        assert len(slides) == 3, "a title slide plus one per entry"
        body = archive.read(slides[1]).decode("utf-8", "ignore")
        assert "The record" in body
        assert "34.85% tandem" in body
        assert "longi.md" in body, "the attribution must reach the slide"


def test_a_stored_file_is_fetched_back_by_its_id(infographic):
    stored = artifacts.store_file(
        artifacts.render_infographic(infographic), infographic.title, ".html"
    )

    assert artifacts.resolve(stored.file_id) == stored.path
    assert stored.url == f"/api/studio/download/{stored.file_id}"
    assert stored.download_name == "perovskite-efficiency-in-2026.html"


@pytest.mark.parametrize(
    "attempt",
    ["../../.env", "..%2f.env", "nope.html", "abc.exe", "0123456789abcdef.txt", ""],
)
def test_only_generated_ids_resolve(attempt):
    assert artifacts.resolve(attempt) is None


def test_the_download_name_survives_an_awkward_title():
    stored = artifacts.store_file(b"x", 'Q3 "results" / notes: 2026!', ".pptx")

    assert stored.download_name == "q3-results-notes-2026.pptx"
    assert artifacts.resolve(stored.file_id) is not None
