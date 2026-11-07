"""Batch translation prompts carry only glossary entries cited by the batch."""

from __future__ import annotations

import json

from wenyi_core.agents.polisher import Polisher
from wenyi_core.agents.prompts import render_glossary
from wenyi_core.agents.translator import Translator
from wenyi_core.config import Config
from wenyi_core.ingest.tokens import count_tokens
from wenyi_core.llm.configuration import LLMConfig
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.models.document import Segment
from wenyi_core.models.glossary import GlossaryTerm
from wenyi_core.pipeline.translate.precision import PrecisionBatchExecutor
from wenyi_core.pipeline.translate.translation_batch import BatchPlan, TranslationBatchExecutor

from tests.fake_llm import MeteredFakeClient
from tests.precision_fixtures import PrecisionHandler
from tests.precision_fixtures import precision_config as _precision_config
from tests.precision_fixtures import precision_store as _precision_store

_TERMS = [
    GlossaryTerm(source="Zed Corp", target="泽德公司", type="organization"),
    GlossaryTerm(source="Ann", target="安", type="person"),
    GlossaryTerm(source="Juniper Song", target="朱妮珀·宋", aliases=["June"], type="person"),
    GlossaryTerm(source="綾小路", target="绫小路", type="person"),
    GlossaryTerm(source="Unseen Name", target="未见", type="person"),
    GlossaryTerm(source="東京都", target="东京都", type="place"),
]


def _plan(sources: list[str], terms=_TERMS, annotations=None) -> BatchPlan:
    segments = [Segment(index=i, source=s) for i, s in enumerate(sources)]
    return BatchPlan.capture(
        0,
        0,
        segments,
        list(terms),
        "",
        "",
        "",
        "",
        annotations or [[] for _ in sources],
        "Unseen Name speaks next.",
    )


def _cited(plan: BatchPlan) -> list[str]:
    return [term.source for term in plan.terms]


def test_capture_keeps_only_cited_terms_in_snapshot_order():
    plan = _plan(["june met ZED CORP staff.", "綾小路くんが東〘とう〙京都に来た。"])
    # Alias, case-insensitive multi-word match and CJK substrings, including one split by a
    # ruby marker; the following-source reference does not admit entries.
    assert _cited(plan) == ["Zed Corp", "Juniper Song", "綾小路", "東京都"]


def test_capture_respects_word_boundaries_and_reads_annotation_sources():
    assert _cited(_plan(["Anna and Zed Corporation waited."])) == []
    annotations = [[{"target_key": "n1", "source": "Ann wrote this note."}]]
    assert _cited(_plan(["See the note."], annotations=annotations)) == ["Ann"]


def test_standard_translation_and_standalone_polish_receive_cited_terms():
    def handler(messages, tier, json_mode):
        if "literary translator" in messages[0]["content"]:
            return json.dumps({"translations": ["译"]})
        return json.dumps({"polished": ["润"]})

    config = Config.from_dict({"llm": {"preset": "fake"}})
    client = FakeClient(handler=handler)
    executor = TranslationBatchExecutor(Translator(client, config), Polisher(client, config))
    # Force the standalone polish path by dropping the translation transcript.
    executor._polisher.polish_continue = lambda *args, **kwargs: None
    executor.execute(_plan(["June left."]), polish=True)
    assert [call["operation"] for call in client.calls] == ["translation.body", "polish.body"]
    for call in client.calls:
        prompt = "\n".join(message["content"] for message in call["messages"])
        assert "Juniper Song → 朱妮珀·宋" in prompt
        assert "Zed Corp" not in prompt
        assert "Unseen Name →" not in prompt


def test_precision_drafts_and_synthesis_receive_cited_terms(tmp_path):
    store = _precision_store(tmp_path, ("June left.", "Rain fell."))
    client = MeteredFakeClient(handler=PrecisionHandler())
    plan = BatchPlan.capture(
        0,
        0,
        store.load_chapter(0).text_segments,
        list(_TERMS),
        "prior",
        "style",
        "synopsis",
        "digest",
        [[], []],
        "following",
    )
    PrecisionBatchExecutor(client, _precision_config(tmp_path)).execute(plan, store)
    operations = [call["operation"] for call in client.calls]
    assert operations == ["translation.body"] * 3 + ["polish.body"]
    for call in client.calls:
        text = "\n".join(m["content"] for m in call["messages"])
        assert "Juniper Song" in text
        assert "Zed Corp" not in text
        assert "Unseen Name" not in text


def test_book_scale_glossary_shrinks_batch_prompt_without_splitting():
    terms = [
        GlossaryTerm(
            source=f"Termword{i:04d} Phrase",
            target=f"术语{i:04d}译名",
            type="term",
            aliases=[f"Aliasword{i:04d}"],
            note="Recurring book-specific usage that the translator must keep consistent.",
        )
        for i in range(1200)
    ]
    full_tokens = count_tokens(render_glossary(terms, source_lang="en"))
    assert full_tokens > 30_000
    sources = [f"Aliasword{i:04d} walked in. Termword{i + 1:04d} Phrase followed." for i in (3, 9)]
    llm = LLMConfig.model_validate(
        {"preset": "fake", "providers": {"default": {"kind": "fake", "context_window": 8_000}}}
    )
    client = FakeClient(handler=lambda *_: json.dumps({"translations": ["译", "译"]}), config=llm)
    trims: list[str] = []
    client.set_event_sink(
        lambda event, **data: trims.append(data["step"]) if event == "context_trimmed" else None
    )
    config = Config.from_dict({"llm": {"preset": "fake"}})
    executor = TranslationBatchExecutor(Translator(client, config), Polisher(client, config))
    plan = _plan(sources, terms=terms)
    assert [term.source for term in plan.terms] == [
        "Termword0003 Phrase",
        "Termword0004 Phrase",
        "Termword0009 Phrase",
        "Termword0010 Phrase",
    ]
    assert executor.execute(plan, polish=False).targets == ("译", "译")
    assert trims == []
    assert len(client.calls) == 1
    prompt_tokens = sum(count_tokens(m["content"]) for m in client.calls[0]["messages"])
    assert prompt_tokens < full_tokens // 10
