"""Single-segment plain-text fallback chain, refused segments and their workflow semantics."""

from __future__ import annotations

import os
import re

import pytest
from wenyi_core.agents.single_segment import (
    ServiceHealth,
    TranslationServiceUnavailable,
    join_sentences,
    split_sentences,
    unwrap_reply,
)
from wenyi_core.agents.translator import Translator
from wenyi_core.config import Config
from wenyi_core.llm.configuration import LLMConfig
from wenyi_core.llm.providers.fake import FakeClient, FakeProvider
from wenyi_core.llm.retrying import EmptyResponseError, ProviderRequestError
from wenyi_core.llm.router import RoutedLLMClient
from wenyi_core.models.document import UNTRANSLATED_PLACEHOLDER
from wenyi_core.pipeline.orchestrator import Orchestrator

from tests.fake_llm import routing_handler
from tests.sample_data import write_sample_txt

REFUSED = "綾小路は小さく頷いた。何も言わなかった。"


def _http_error(status: int) -> ProviderRequestError:
    return ProviderRequestError(
        provider="fake",
        operation="translation.body",
        error_type="http_error",
        status_code=status,
        reason=f"http_{status}" if status >= 500 else None,
        resumable=status >= 500,
    )


def _plain_source(messages) -> str:
    user = messages[-1]["content"]
    return user.split("paragraph to translate]\n", 1)[1].split("\n\n", 1)[0]


def _config(**pipeline) -> Config:
    return Config.from_dict(
        {"llm": {"preset": "fake"}, "pipeline": {"align_retry_limit": 0, **pipeline}}
    )


@pytest.mark.parametrize(
    "text",
    [
        "他说：“走吧。”然后离开了……真的吗？是的！",
        'He left. "Why?" she asked... Then 3.14 stayed. Done',
        "「行こう。」と彼は言った。本当に？……はい！",
        "No terminator at all",
        "",
        "  空白。  \n下一行。",
    ],
)
def test_split_sentences_never_loses_characters(text):
    assert "".join(split_sentences(text)) == text


def test_split_sentences_keeps_closing_quotes_and_ellipses_with_the_sentence():
    assert split_sentences("他说：“走吧。”然后离开了……真的吗？") == [
        "他说：“走吧。”",
        "然后离开了……",
        "真的吗？",
    ]
    assert split_sentences('He left. "Why?" she asked... Then 3.14 stayed.') == [
        "He left. ",
        '"Why?" ',
        "she asked... ",
        "Then 3.14 stayed.",
    ]
    assert split_sentences("「行こう。」と彼は言った。はい！") == [
        "「行こう。」",
        "と彼は言った。",
        "はい！",
    ]


def test_join_sentences_spaces_only_western_targets():
    assert join_sentences(["走吧。", " 好的。"], "zh") == "走吧。好的。"
    assert join_sentences(["行こう。", "はい。"], "ja") == "行こう。はい。"
    assert join_sentences(["Go. ", "Fine."], "en-US") == "Go. Fine."


@pytest.mark.parametrize(
    "reply,source,expected",
    [
        ("  译文\n", "原文", "译文"),
        ("```text\n译文\n```", "原文", "译文"),
        ("「译文」", "原文", "译文"),
        ('"Tokyo"', "東京", "Tokyo"),
        ("「行こう」", "「行こう」", "「行こう」"),  # Dialogue keeps its own quotes.
        ("“甲”和“乙”", "x", "“甲”和“乙”"),  # Not one wrapping pair.
        ("Tokyo", "Tokyo", "Tokyo"),
    ],
)
def test_unwrap_reply_removes_only_obvious_wrapping(reply, source, expected):
    assert unwrap_reply(reply, source) == expected


def test_reply_identical_to_source_is_accepted():
    client = FakeClient(handler=lambda m, t, json_mode: "{}" if json_mode else "Tokyo")
    result = Translator(client, _config()).translate_segment("Tokyo")
    assert result.target == "Tokyo"
    assert result.tried == ["plain_text"]


def test_failed_paragraph_is_retried_sentence_by_sentence_in_order():
    def handler(messages, tier, json_mode):
        source = _plain_source(messages)
        if source == "Go now. Stay here.":
            raise EmptyResponseError("stream closed")
        return {"Go now. ": "走吧。", "Stay here.": "留下。"}[source]

    result = Translator(FakeClient(handler=handler), _config()).translate_segment(
        "Go now. Stay here."
    )
    assert result.target == "走吧。留下。"
    assert result.tried == ["plain_text", "sentences"]


def test_one_failed_sentence_refuses_the_whole_paragraph():
    def handler(messages, tier, json_mode):
        if "Stay" in _plain_source(messages):
            raise _http_error(400)
        return "走吧。"

    result = Translator(FakeClient(handler=handler), _config()).translate_segment(
        "Go now. Stay here."
    )
    assert result.target is None
    assert (result.tried, result.reason, result.provider_failed) == (
        ["plain_text", "sentences"],
        "invalid_request",
        False,
    )


def test_fallback_route_retranslates_when_the_primary_route_fails(monkeypatch):
    config = LLMConfig.model_validate(
        {
            "preset": "fake",
            "providers": {"default": {"kind": "fake", "max_retries": 0}},
            "models": {"backup": {"provider": "default", "model": "backup"}},
            "routes": {"translation.body": {"tier": "strong", "fallbacks": ["backup"]}},
        }
    )
    client = RoutedLLMClient(config)
    models = []

    def request(self, messages, model, *, json_mode, context):
        models.append(model.model)
        if model.model != "backup":
            raise EmptyResponseError("primary stream closed")
        return "备用译文"

    monkeypatch.setattr(FakeProvider, "_request", request)
    result = Translator(client, _config()).translate_segment("Go now.")
    assert result.target == "备用译文"
    assert models[-1] == "backup" and len(models) == 2
    assert result.routes[-1] == "backup"


def test_consecutive_provider_failures_stop_the_run_but_one_does_not():
    def outage(*_):
        raise _http_error(503)

    translator = Translator(FakeClient(handler=outage), _config())
    with pytest.raises(TranslationServiceUnavailable):
        translator.translate_batch(["one", "two", "three", "four"], health=ServiceHealth())

    def flaky(messages, tier, json_mode):
        if json_mode or _plain_source(messages) == "two":
            raise _http_error(503)
        return "好"

    health = ServiceHealth()
    translator = Translator(FakeClient(handler=flaky), _config())
    assert translator.translate_batch(["one", "two", "three"], health=health) == ["好", None, "好"]
    assert translator.last_refused[1].provider_failed
    assert health.failures == 0


def _refusing(calls: list[str]):
    """Refuse (HTTP 400, as moderation does) every request that contains ``REFUSED``."""

    def handler(messages, tier, json_mode):
        system = messages[0]["content"]
        if "translating one" in system:
            source = _plain_source(messages)
            calls.append(source)
            if "綾小路は小さく頷いた" in source:
                raise _http_error(400)
        elif "literary translator" in system and REFUSED in messages[-1]["content"]:
            raise _http_error(400)
        return routing_handler(messages, tier, json_mode)

    return handler


def _book_config(state_dir: str) -> Config:
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {"preset": "fake"},
            "pipeline": {"review": True, "review_autofix": False, "polish": False},
            "output": {"bilingual": True},
            "paths": {"state_dir": state_dir},
        }
    )


def test_refused_segment_completes_review_and_export_then_is_retranslated_alone(tmp_path):
    source = os.path.join(tmp_path, "novel.txt")
    write_sample_txt(source)
    config = _book_config(str(tmp_path / "state"))
    calls: list[str] = []

    result = Orchestrator(config, client=FakeClient(handler=_refusing(calls))).run_all(
        source, out_format="txt"
    )

    store = result["store"]
    assert store.pending_chapters() == []
    segment = next(
        s for c in (0, 1) for s in store.load_chapter(c).text_segments if s.source == REFUSED
    )
    assert segment.target is None
    refused = store.list_events(event_type="segment_translation_refused", limit=0)
    assert [(e["segment"], e["reason"]) for e in refused] == [(segment.index, "invalid_request")]
    assert refused[0]["tried"] == ["batch", "plain_text", "sentences"]
    review = store.list_events(event_type="review_untranslated_segments", limit=0)
    assert review[-1]["segments"] == [{"chapter": 0, "segment": segment.index}]
    mono, bilingual = sorted(result["outputs"], key=lambda path: "-bi" in path)
    mono_text = open(mono, encoding="utf-8").read()
    assert UNTRANSLATED_PLACEHOLDER in mono_text and REFUSED not in mono_text
    bilingual_text = open(bilingual, encoding="utf-8").read()
    assert bilingual_text.count(REFUSED) == 1 and UNTRANSLATED_PLACEHOLDER not in bilingual_text

    retried: list[str] = []
    Orchestrator(config, client=FakeClient(handler=_recovering(retried))).run_all(
        source, out_format="txt"
    )
    segment = next(
        s for c in (0, 1) for s in store.load_chapter(c).text_segments if s.source == REFUSED
    )
    assert segment.target == "译0"
    # Only the refused segment is requested again; translated segments are not rebought.
    assert retried == [REFUSED]
    recovered = store.list_events(event_type="segment_translation_recovered", limit=0)
    assert [e["segment"] for e in recovered] == [segment.index]
    assert (
        store.list_events(event_type="translate_run_finished", limit=0)[-1]["refused_segments"] == 0
    )
    assert UNTRANSLATED_PLACEHOLDER not in open(mono, encoding="utf-8").read()


def _recovering(batches: list[str]):
    def handler(messages, tier, json_mode):
        if "literary translator" in messages[0]["content"]:
            batches.extend(re.findall(r"^\[\d+\] (.*)$", messages[-1]["content"], re.MULTILINE))
        return routing_handler(messages, tier, json_mode)

    return handler


@pytest.mark.parametrize(
    "target,source,bilingual,expected",
    [
        (None, "原文", False, UNTRANSLATED_PLACEHOLDER),
        (None, "原文", True, "原文"),  # De-duplicated by the writers: shown once as source.
        (None, "1984", False, "1984"),
        ("", "原文", False, "原文"),
        ("译文", "原文", False, "译文"),
    ],
)
def test_export_text_never_passes_source_off_for_refused_segments(
    target, source, bilingual, expected
):
    from wenyi_core.assemble.writer_common import _bilingual_source, _seg_text
    from wenyi_core.models.document import Segment

    text = _seg_text(Segment(index=0, source=source, target=target), bilingual=bilingual)
    assert text == expected
    if target is None and bilingual:
        assert _bilingual_source(source, text) == ""


def _overflow() -> Exception:
    from wenyi_core.llm.retrying import ContextLengthExceeded

    return ContextLengthExceeded(
        provider="fake", operation="translation.body", status_code=400, window=None
    )


def _trims(client) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    client.set_event_sink(
        lambda event, **data: (
            events.append((data["step"], data["trigger"])) if event == "context_trimmed" else None
        )
    )
    return events


def test_oversized_batch_is_halved_without_counting_an_outage():
    def handler(messages, tier, json_mode):
        user = messages[-1]["content"]
        count = len(re.findall(r"^\[(\d+)\]", user, re.MULTILINE))
        if count > 2:
            raise _overflow()
        return routing_handler(messages, tier, json_mode)

    client = FakeClient(handler=handler)
    trims = _trims(client)
    translator = Translator(client, _config())
    targets = translator.translate_batch(
        ["One.", "Two.", "Three.", "Four."], health=ServiceHealth(limit=1)
    )
    assert targets == ["译0", "译1", "译0", "译1"]
    assert trims == [("split", "context_length")]
    assert translator.last_refused == {}


def test_oversized_segment_drops_optional_context_step_by_step():
    from wenyi_core.glossary import GlossaryTerm

    def handler(messages, tier, json_mode):
        if "Alice" not in _plain_source(messages):
            return "译文"
        if "Bob" in messages[-1]["content"]:  # The unmatched term is still in the glossary.
            raise _overflow()
        return "爱丽丝走了。"

    client = FakeClient(handler=handler)
    trims = _trims(client)
    result = Translator(client, _config()).translate_segment(
        "Alice left",
        glossary_terms=[
            GlossaryTerm(source="Alice", target="爱丽丝", type="person"),
            GlossaryTerm(source="Bob", target="鲍勃", type="person"),
        ],
        context="prior",
        chapter_digest="digest",
        next_source="Then rain.",
    )
    assert result.target == "爱丽丝走了。"
    assert trims == [
        ("next_source", "context_length"),
        ("context", "context_length"),
        ("glossary", "context_length"),
    ]
    assert len(client.calls) == 4


def test_configured_window_trims_a_segment_locally_before_sending():
    llm = LLMConfig.model_validate(
        {"preset": "fake", "providers": {"default": {"kind": "fake", "context_window": 1_200}}}
    )
    client = FakeClient(handler=lambda m, t, json_mode: "译文", config=llm)
    trims = _trims(client)
    result = Translator(client, _config()).translate_segment(
        "Go now. Stay here.", context="lorem ipsum " * 400
    )
    assert result.target == "译文"
    assert trims == [("context", "context_window")]
    assert len(client.calls) == 1
    assert "lorem" not in client.calls[0]["messages"][-1]["content"]


def test_sentence_still_over_the_window_is_refused_without_stopping_the_run():
    client = FakeClient(
        handler=lambda messages, tier, json_mode: (_ for _ in ()).throw(_overflow())
    )
    translator = Translator(client, _config())
    targets = translator.translate_batch(
        ["Go now. Stay here.", "Wait.", "Stop."], health=ServiceHealth(limit=1)
    )
    assert targets == [None, None, None]
    assert {outcome.reason for outcome in translator.last_refused.values()} == {"context_length"}
    assert translator.last_refused[0].tried == ["plain_text", "sentences"]
    assert not any(outcome.provider_failed for outcome in translator.last_refused.values())
