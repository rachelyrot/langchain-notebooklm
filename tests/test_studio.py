"""Studio artifacts: the shapes, their rendering, and what the API does with them."""

from __future__ import annotations

import pytest

import agents.studio as studio
import api.services as services
from api.schemas import GenerateArtifactRequest


@pytest.fixture
def summary():
    return studio.Summary(
        title="Perovskite efficiency in 2026",
        overview="Where tandem cells stand today.",
        key_points=[
            studio.KeyPoint(point="LONGi reached 34.85% on a tandem cell.", source="longi.md"),
            studio.KeyPoint(point="Stability remains the blocker.", source="review.md"),
        ],
        disagreements=["longi.md and review.md report different record dates."],
        open_questions=["What is the cost per watt at scale?"],
    )


def test_a_summary_renders_every_field_with_its_attribution(summary):
    rendered = studio._render_summary(summary)

    assert summary.overview in rendered
    assert "## Key points" in rendered
    assert "LONGi reached 34.85%" in rendered
    assert "*— longi.md*" in rendered, "a claim must carry its source"
    assert "## Where the sources disagree" in rendered
    assert "## Open questions" in rendered


def test_empty_sections_are_left_out(summary):
    bare = summary.model_copy(update={"disagreements": [], "open_questions": []})

    rendered = studio._render_summary(bare)

    assert "## Where the sources disagree" not in rendered
    assert "## Open questions" not in rendered
    assert "## Key points" in rendered


def test_a_faq_renders_question_answer_and_source():
    faq = studio.Faq(
        title="Perovskite FAQ",
        items=[
            studio.FaqItem(
                question="What is the record efficiency?",
                answer="34.85% for a silicon-perovskite tandem.",
                source="longi.md",
            )
        ],
    )

    rendered = studio._render_faq(faq)

    assert "**What is the record efficiency?**" in rendered
    assert "34.85%" in rendered
    assert "*— longi.md*" in rendered


def test_an_empty_notebook_is_refused_before_calling_the_model(store):
    """No active sources means nothing to ground an artifact in — fail before spending."""
    with pytest.raises(studio.EmptyNotebook):
        studio.generate("summary")


def test_an_unknown_kind_is_rejected(store):
    store.add(name="a.md", content="something")

    with pytest.raises(KeyError):
        studio.generate("nonsense")


def test_every_advertised_artifact_is_actually_built():
    """The panel must not offer something the backend answers 501 for."""
    advertised = {a.key for a in services.list_artifacts()}

    assert advertised == set(studio.KINDS)
    assert all(a.status == "ready" for a in services.list_artifacts())


def test_an_artifact_nobody_built_still_says_coming_soon():
    with pytest.raises(services.ComingSoon):
        services.generate_artifact("hologram", "A")


def test_a_generated_artifact_is_saved_as_a_note(store, monkeypatch):
    monkeypatch.setattr(
        studio,
        "generate",
        lambda kind: studio.Artifact(title="Built title", content="# Built body"),
    )

    result = services.generate_artifact("summary", "A")

    assert result.kind == "summary"
    assert result.note.title == "Built title"
    assert result.note.content == "# Built body"
    assert result.download_url is None, "a text artifact has no file"
    assert result.note.id in {n.id for n in services.list_notes()}
    services.remove_note(result.note.id)


def test_a_file_artifact_carries_a_download_url(store, monkeypatch):
    stored = studio.artifacts.StoredFile(
        file_id="0123456789abcdef.pptx", path=None, download_name="deck.pptx"
    )
    monkeypatch.setattr(
        studio,
        "generate",
        lambda kind: studio.Artifact(title="Deck", content="outline", file=stored),
    )

    result = services.generate_artifact("powerpoint", "A")

    assert result.download_url == "/api/studio/download/0123456789abcdef.pptx"
    assert result.download_name == "deck.pptx"
    assert result.note.content == "outline", "the text still lands in the notes panel"
    services.remove_note(result.note.id)


def test_the_generate_request_defaults_its_impl():
    assert GenerateArtifactRequest(kind="summary").impl == "A"
