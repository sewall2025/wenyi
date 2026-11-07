"""Review and title translation shrink or skip requests that exceed the model context."""

from __future__ import annotations

import json
import os
import re

from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.llm.retrying import ContextLengthExceeded
from wenyi_core.models.document import Segment
from wenyi_core.models.glossary import GlossaryTerm
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.review.run_store import ReviewRunStore

from tests.fake_llm import routing_handler
from tests.pipeline_fixtures import fake_agent_config, fake_pipeline_config
from tests.sample_data import write_sample_txt


def _overflow(operation: str) -> ContextLengthExceeded:
    return ContextLengthExceeded(provider="fake", operation=operation, status_code=400, window=None)


def _numbered(user: str) -> int:
    return len(re.findall(r"^\[(\d+)\]", user, re.MULTILINE))


def _review_reply(count: int) -> str:
    issues = [{"index": 0, "type": "missing", "detail": "漏译", "suggestion": "补译"}]
    return json.dumps({"issues": issues, "reviewed_segments": count, "complete": True})


def _events(debug: ReviewRunStore) -> list[dict]:
    with open(debug.path("events.jsonl"), encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _review(handler, segments, terms, tmp_path):
    config = fake_agent_config()
    config.segment.max_tokens_per_batch = 100_000
    config.pipeline.review_concurrency = 1
    client = FakeClient(handler=handler)
    debug = ReviewRunStore(str(tmp_path))
    issues = Orchestrator(config, client=client)._review._chunks.review_chapter(
        segments, terms, chapter_index=2, debug=debug
    )
    return issues, client, debug


def test_review_chunk_over_context_is_bisected_with_chapter_offsets(tmp_path):
    def handler(messages, tier, json_mode):
        count = _numbered(messages[-1]["content"])
        if count > 1:
            raise _overflow("review.scan")
        return _review_reply(count)

    segments = [Segment(index=i, source=f"源文{i}", target=f"译文{i}") for i in range(4)]
    issues, client, debug = _review(handler, segments, [], tmp_path)
    assert [issue["index"] for issue in issues] == [0, 1, 2, 3]
    splits = [e for e in _events(debug) if e["event"] == "review_chunk_split"]
    assert [(e["start_index"], e["count"]) for e in splits] == [(0, 4), (0, 2), (2, 2)]
    assert {(e["reason"], e["trigger"]) for e in splits} == {("context_length", "context_length")}

    # Resume reuses the split leaves without any further reviewer call.
    resumed = FakeClient(handler=handler)
    config = fake_agent_config()
    config.segment.max_tokens_per_batch = 100_000
    again = Orchestrator(config, client=resumed)._review._chunks.review_chapter(
        segments, [], chapter_index=2, debug=debug
    )
    assert [issue["index"] for issue in again] == [0, 1, 2, 3]
    assert resumed.calls == []


def test_oversized_singleton_retries_with_cited_terms_then_is_skipped(tmp_path):
    terms = [
        GlossaryTerm(source="源文1", target="原文一"),
        GlossaryTerm(source="远方", target="远方"),
    ]
    glossary_sizes: list[int] = []

    def handler(messages, tier, json_mode):
        user = messages[-1]["content"]
        count = _numbered(user)
        if "[1] " not in user and "源文1" in user.split("[Paragraph pairs]")[1]:
            glossary_sizes.append(user.count(" → "))
            raise _overflow("review.scan")
        if count > 1:
            raise _overflow("review.scan")
        return _review_reply(count)

    segments = [Segment(index=10 + i, source=f"源文{i}", target=f"译文{i}") for i in range(3)]
    issues, client, debug = _review(handler, segments, terms, tmp_path)
    # Paragraph 1 is skipped; its neighbours keep their chapter-local issue indices.
    assert [issue["index"] for issue in issues] == [0, 2]
    assert glossary_sizes == [2, 1]  # Full glossary, then only the cited entry.
    skipped = [e for e in _events(debug) if e["event"] == "review_chunk_skipped"]
    assert skipped == [
        {
            **skipped[0],
            "chapter": 2,
            "start_index": 1,
            "count": 1,
            "segment": 11,
            "reason": "context_length",
            "trigger": "context_length",
        }
    ]
    resumed = FakeClient(handler=handler)
    config = fake_agent_config()
    config.segment.max_tokens_per_batch = 100_000
    Orchestrator(config, client=resumed)._review._chunks.review_chapter(
        segments, terms, chapter_index=2, debug=debug
    )
    assert resumed.calls == []


def test_review_preflight_splits_locally_before_sending(tmp_path):
    from wenyi_core.llm.configuration import LLMConfig

    llm = LLMConfig.model_validate(
        {"preset": "fake", "providers": {"default": {"kind": "fake", "context_window": 2_400}}}
    )
    config = fake_agent_config()
    config.segment.max_tokens_per_batch = 100_000
    client = FakeClient(
        handler=lambda m, t, j: _review_reply(_numbered(m[-1]["content"])), config=llm
    )
    segments = [
        Segment(index=i, source="長い原文。" * 60, target="长译文。" * 60) for i in range(2)
    ]
    debug = ReviewRunStore(str(tmp_path))
    issues = Orchestrator(config, client=client)._review._chunks.review_chapter(
        segments, [], chapter_index=0, debug=debug
    )
    assert [issue["index"] for issue in issues] == [0, 1]
    assert all(_numbered(call["messages"][-1]["content"]) == 1 for call in client.calls)
    splits = [e for e in _events(debug) if e["event"] == "review_chunk_split"]
    assert [e["trigger"] for e in splits] == ["context_window"]


def test_run_all_continues_when_every_review_request_exceeds_the_context(tmp_path):
    def handler(messages, tier, json_mode):
        if "translation reviewer" in messages[0]["content"]:
            raise _overflow("review.scan")
        return routing_handler(messages, tier, json_mode)

    txt = os.path.join(tmp_path, "novel.txt")
    write_sample_txt(txt)
    config = fake_pipeline_config(os.path.join(tmp_path, "state"))
    client = FakeClient(handler=handler)
    result = Orchestrator(config, client=client).run_all(txt)
    assert os.path.isfile(result["output"])
    assert result["review_result"]["issues"] == []
    assert any(call["operation"] == "review.scan" for call in client.calls)
    assert result["review_result"]["skipped_segments"]


def test_title_batch_over_context_is_split_and_single_title_keeps_source(tmp_path):
    def handler(messages, tier, json_mode):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "title" in system.lower() and "titles" in user.lower():
            if _numbered(user) > 1 or "放課後" in user:
                raise _overflow("translation.title")
            return json.dumps({"titles": ["第一章 相遇"]}, ensure_ascii=False)
        return routing_handler(messages, tier, json_mode)

    txt = os.path.join(tmp_path, "novel.txt")
    write_sample_txt(txt)
    config = fake_pipeline_config(os.path.join(tmp_path, "state"))
    config.pipeline.review = False
    orch = Orchestrator(config, client=FakeClient(handler=handler))
    store = orch.run(txt)
    chapters = store.load_manifest()["chapters"]
    assert [chapter.get("title_translated") for chapter in chapters] == [
        "第一章 相遇",
        " ".join(chapters[1]["title"].split()),  # Kept source title, whitespace-normalized.
    ]
    kinds = [event["event"] for event in store.list_events(limit=10_000)]
    assert "titles_translation_failed" not in kinds
    assert "titles_batch_split" in kinds
    assert "titles_kept_source" in kinds
