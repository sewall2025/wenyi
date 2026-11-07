"""Models with ``batch_output: false`` get one plain-text request per segment."""

from __future__ import annotations

import pytest
from wenyi_core.config import Config
from wenyi_core.pipeline.orchestrator import Orchestrator

from tests.fake_llm import MeteredFakeClient
from tests.precision_fixtures import precision_store

SOURCES = ("one", "two", "three")
PLAIN_LLM = {
    "preset": "fake",
    "models": {"plain": {"provider": "default", "model": "plain-mt", "batch_output": False}},
    "routes": {"translation.body": {"model": "plain"}, "polish.body": {"model": "plain"}},
}


def _config(tmp_path, mode: str) -> Config:
    return Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": PLAIN_LLM,
            "pipeline": {
                "translation_mode": mode,
                "polish": True,
                "review": False,
                "book_understanding": False,
                "annotation_alignment": False,
            },
            "paths": {"state_dir": str(tmp_path)},
        }
    )


def _source(messages) -> str:
    user = messages[-1]["content"]
    return user.split("paragraph to translate]\n", 1)[1].split("\n\n", 1)[0]


class PlainHandler:
    def __init__(self, fail_synthesis_of: str | None = None):
        self.fail_synthesis_of = fail_synthesis_of

    def __call__(self, messages, tier, json_mode):
        if json_mode:  # Glossary and other JSON operations use the tier default model.
            return "{}"
        source = _source(messages)
        if "drafts of this paragraph]" in messages[-1]["content"]:
            if source == self.fail_synthesis_of:
                raise RuntimeError("interrupted")
            return f"合:{source}"
        return f"译:{source}"


def _body_calls(client):
    return [c for c in client.calls if c["operation"] in {"translation.body", "polish.body"}]


def test_standard_translation_sends_one_plain_request_per_segment_and_skips_polish(tmp_path):
    store, config = precision_store(tmp_path, SOURCES), _config(tmp_path, "standard")
    client = MeteredFakeClient(handler=PlainHandler(), config=config.llm)
    events = []
    client.set_event_sink(lambda event, **data: events.append((event, data)))
    Orchestrator(config, client)._translation.run(store, book_synopsis="synopsis")
    assert [s.target for s in store.load_chapter(0).text_segments] == [
        "译:one",
        "译:two",
        "译:three",
    ]
    calls = _body_calls(client)
    assert [(c["operation"], c["json_mode"], _source(c["messages"])) for c in calls] == [
        ("translation.body", False, source) for source in SOURCES
    ]
    skipped = [data for event, data in events if event == "polish_skipped"]
    assert skipped and skipped[0]["reason"] == "batch_output_disabled"


def test_precision_drafts_and_synthesis_are_plain_and_resume_without_rebuying(tmp_path):
    store, config = precision_store(tmp_path, SOURCES), _config(tmp_path, "best_of_three")
    first = MeteredFakeClient(handler=PlainHandler(fail_synthesis_of="three"), config=config.llm)
    with pytest.raises(RuntimeError, match="interrupted"):
        Orchestrator(config, first)._translation.run(store, book_synopsis="synopsis")
    calls = _body_calls(first)
    assert all(not c["json_mode"] for c in calls)
    # Three drafts per segment, then synthesis in source order until the interruption.
    assert sorted(_source(c["messages"]) for c in calls if c["operation"] == "translation.body")
    assert len([c for c in calls if c["operation"] == "translation.body"]) == 9
    assert [_source(c["messages"]) for c in calls if c["operation"] == "polish.body"] == [
        "one",
        "two",
        "three",
    ]

    later = MeteredFakeClient(handler=PlainHandler(), config=config.llm)
    Orchestrator(config, later)._translation.run(store, book_synopsis="synopsis")
    # Accepted drafts and synthesized segments are reused; only the failed segment is bought.
    assert [(c["operation"], _source(c["messages"])) for c in _body_calls(later)] == [
        ("polish.body", "three")
    ]
    segments = store.load_chapter(0).text_segments
    assert [s.target for s in segments] == ["合:one", "合:two", "合:three"]
    assert [s.target_before_polish for s in segments] == ["译:one", "译:two", "译:three"]
    usage = store.load_usage()
    assert usage["by_stage"]["translation.body"]["calls"] == 9
    assert usage["by_stage"]["polish.body"]["calls"] == 4
