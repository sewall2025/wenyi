"""Precision validates numbered segments and re-requests only the missing ids."""

import json
from dataclasses import replace

import pytest
from wenyi_core.agents.precision import (
    PrecisionAgent,
    PrecisionError,
    PrecisionInputs,
    accepted_segments,
)
from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.llm.retrying import ProviderRequestError, TruncatedResponseError

from tests.precision_fixtures import segment_ids


def inputs(*sources, allow_empty=False):
    return PrecisionInputs(
        tuple(sources),
        (),
        "prior",
        "style",
        "synopsis",
        "digest",
        [[] for _ in sources],
        "lookahead",
        allow_empty,
    )


def agent(response=None, handler=None, retry_limit=2):
    config = Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {"preset": "fake"},
            "pipeline": {"translation_mode": "best_of_three", "align_retry_limit": retry_limit},
        }
    )
    client = FakeClient(handler=handler or (lambda *_: json.dumps(response)))
    return PrecisionAgent(client, config), client


def test_synthesis_sees_all_distinct_drafts_source_and_context_without_output_hint():
    worker, client = agent({"translations": ["综合稿"]})
    packet = inputs("original")
    for _ in range(3):
        worker.translate(packet)
    assert worker.synthesize(packet, [["甲稿"], ["乙稿"], ["丙稿"]]) == ["综合稿"]
    assert len(client.calls) == 4
    assert all(call["max_tokens"] is None for call in client.calls)
    assert all(call["messages"][:2] == client.calls[0]["messages"][:2] for call in client.calls)
    assert [call["operation"] for call in client.calls] == ["translation.body"] * 3 + [
        "polish.body"
    ]
    task = json.loads(client.calls[-1]["messages"][-1]["content"].split("Task (JSON):\n")[1])
    assert task["segments"] == [{"id": 0, "source": "original", "drafts": ["甲稿", "乙稿", "丙稿"]}]
    assert task["reference_only"]["context"] == "prior"
    assert "original" not in client.calls[-1]["messages"][1]["content"]
    assert all(
        message["role"] != "assistant" for call in client.calls for message in call["messages"]
    )
    assert "甲稿" not in str(client.calls[0]["messages"])


def test_duplicate_draws_are_not_multiple_votes():
    worker, client = agent({"translations": ["润色稿"]})
    worker.synthesize(inputs("source"), [["初稿"], ["初稿"], ["初稿"]])
    task = json.loads(client.calls[0]["messages"][-1]["content"].split("Task (JSON):\n")[1])
    assert task["segments"][0]["drafts"] == ["初稿"]


def task_in(messages):
    message = next(m for m in messages if "Task (JSON):\n" in m["content"])
    return json.loads(message["content"].split("Task (JSON):\n")[1])


def task_of(call):
    return task_in(call["messages"])


def ids_of(call):
    return segment_ids(task_of(call))


def numbered(answers, override=None):
    """Answer each requested id from ``answers``; ``override(ids)`` may replace a reply."""

    def respond(messages, tier, json_mode):
        ids = segment_ids(task_in(messages))
        if override is not None:
            reply = override(ids)
            if reply is not None:
                return reply if isinstance(reply, str) else json.dumps(reply)
        return json.dumps({"translations": {str(i): answers[i] for i in ids}})

    return respond


@pytest.mark.parametrize(
    "response", [{}, [], {"translations": []}, {"translations": [3]}, {"translations": [""]}]
)
@pytest.mark.parametrize("stage", ["translate", "synthesize"])
@pytest.mark.parametrize("retry_limit", [0, 2])
def test_single_segment_invalid_output_exhausts_shared_retry_limit(response, stage, retry_limit):
    replies = iter(range(100))

    def varied(*_):
        # A different rejected reply each time, so only the retry budget ends the requests.
        return json.dumps(response) + " " * next(replies)

    worker, client = agent(handler=varied, retry_limit=retry_limit)
    packet = inputs("source")
    if stage == "translate":
        assert worker.translate(packet) == [None]
    else:
        assert worker.synthesize(packet, [["稿"]] * 3) == [None]
    assert worker.failures == {0: "invalid_output"}
    assert len(client.calls) == retry_limit + 1
    # Every structural retry quotes the previous reply and restates the numbered contract.
    assert [len(call["messages"]) for call in client.calls] == [
        3 + 2 * index for index in range(retry_limit + 1)
    ]
    for previous, current in zip(client.calls, client.calls[1:]):
        assert current["messages"][: len(previous["messages"])] == previous["messages"]
        assert current["messages"][-2]["role"] == "assistant"
        assert "failed local validation" in current["messages"][-1]["content"]
        assert 'exactly these keys: "0"' in current["messages"][-1]["content"]


@pytest.mark.parametrize("stage", ["translate", "synthesize"])
def test_identical_single_segment_reply_stops_retrying(stage):
    worker, client = agent({"translations": {"0": ""}}, retry_limit=5)
    packet = inputs("source")
    if stage == "translate":
        assert worker.translate(packet) == [None]
    else:
        assert worker.synthesize(packet, [["稿"]] * 3) == [None]
    # The second reply repeats the first verbatim, so feedback cannot change the answer.
    assert len(client.calls) == 2
    assert worker.failures == {0: "repeated_reply"}


def test_reply_with_an_unrequested_id_is_rejected_whole_and_halved():
    reply = {"translations": {"0": "甲", "1": "乙", "2": "多余", "-1": "负", "01": "前导零"}}
    worker, client = agent(
        handler=numbered(["甲", "乙"], lambda ids: reply if len(ids) > 1 else None)
    )
    events = []
    client.set_event_sink(lambda event, **data: events.append((event, data)))
    assert worker.translate(inputs("one", "two")) == ["甲", "乙"]
    assert [ids_of(call) for call in client.calls] == [[0, 1], [0], [1]]
    assert events == [
        (
            "precision_reply_misaligned",
            {"operation": "translation.body", "ids": [0, 1], "reason": "unrequested_id"},
        )
    ]


@pytest.mark.parametrize("stage", ["translate", "synthesize"])
def test_missing_segment_is_requested_alone_with_the_same_packet(stage):
    worker, client = agent(
        handler=numbered(
            ["甲", "乙", "丙"],
            lambda ids: {"translations": {"0": "甲", "2": "丙"}} if ids == [0, 1, 2] else None,
        )
    )
    packet = inputs("one", "two", "three")
    if stage == "translate":
        assert worker.translate(packet) == ["甲", "乙", "丙"]
    else:
        drafts = [["一稿", "二稿", "三稿"], ["一稿", "二改", "三稿"], ["一稿", "二稿", "三稿"]]
        assert worker.synthesize(packet, drafts) == ["甲", "乙", "丙"]
        # Each re-requested id carries its own source and distinct drafts.
        assert task_of(client.calls[1])["segments"] == [
            {"id": 0, "source": "one", "drafts": ["一稿"]},
            {"id": 1, "source": "two", "drafts": ["二稿", "二改"]},
        ]
    # The entry just before the gap may have absorbed the missing one, so it is redone too.
    assert [ids_of(call) for call in client.calls] == [[0, 1, 2], [0, 1]]
    assert client.calls[1]["messages"][:2] == client.calls[0]["messages"][:2]
    assert all(message["role"] != "assistant" for message in client.calls[1]["messages"])


@pytest.mark.parametrize(
    ("raw", "requested"),
    [
        (
            '{"translations": {"0": "旧", "0": "甲", "5": "越界", "-1": "负", "01": "前导零"}}',
            [[0, 1], [0], [1]],
        ),
        ('{"translations": {"0": "甲", "1": "旧", "1": "乙"}}', [[0, 1], [1]]),
        ('{"translations": {"0": "甲", "1": "   "}}', [[0, 1], [1]]),
        ('{"translations": {"0": "甲", "1": 3}}', [[0, 1], [1]]),
        ('{"translations": {"0": "甲", "1": null}}', [[0, 1], [1]]),
        ('{"translations": {"0": "甲", "1": ["乙"]}}', [[0, 1], [1]]),
        ('{"translations": {"0": "甲", "1": {"src": "two", "text": "乙"}}}', [[0, 1], [1]]),
    ],
)
def test_duplicate_unknown_blank_and_non_string_entries_are_requested_again(raw, requested):
    worker, client = agent(
        handler=numbered(["甲", "乙"], lambda ids: raw if ids == [0, 1] else None)
    )
    assert worker.translate(inputs("one", "two")) == ["甲", "乙"]
    assert [ids_of(call) for call in client.calls] == requested


def test_progress_continues_rounds_on_the_missing_ids_only():
    replies = {
        (0, 1, 2, 3): {"translations": {"0": "甲", "1": "截"}},
        (1, 2, 3): {"translations": {"1": "截", "3": "丁"}},
    }
    worker, client = agent(
        handler=numbered(["甲", "乙", "丙", "丁"], lambda ids: replies.get(tuple(ids)))
    )
    assert worker.translate(inputs("a", "b", "c", "d")) == ["甲", "乙", "丙", "丁"]
    assert [ids_of(call) for call in client.calls] == [
        [0, 1, 2, 3],
        [1, 2, 3],
        [1, 2],
    ]
    # Every round quotes each id's source.
    assert task_of(client.calls[1])["segments"] == [
        {"id": 1, "source": "b"},
        {"id": 2, "source": "c"},
        {"id": 3, "source": "d"},
    ]


@pytest.mark.parametrize("failure", ["json", "truncated", "empty", "legacy_count"])
def test_zero_progress_falls_back_to_one_request_per_remaining_segment(failure):
    def broken(ids):
        if len(ids) == 1:
            return None
        if failure == "truncated":
            raise TruncatedResponseError("provider output ended")
        return {
            "json": "not JSON",
            "empty": {"translations": {}},
            "legacy_count": {"translations": ["第一段。\n\n第二段。"]},
        }[failure]

    worker, client = agent(handler=numbered(["第一段。", "第二段。"], broken))
    assert worker.translate(inputs("First.", "Second.")) == ["第一段。", "第二段。"]
    # The whole-batch round has no feedback retry; each single-segment request starts fresh.
    assert [ids_of(call) for call in client.calls] == [[0, 1], [0], [1]]
    assert all(len(call["messages"]) == 3 for call in client.calls)


def test_single_segment_feedback_retry_recovers_then_continues():
    attempts = []

    def flaky(ids):
        attempts.append(ids)
        if len(ids) > 1 or len(attempts) == 2:
            return {"translations": {}}
        return None

    worker, client = agent(handler=numbered(["甲", "乙"], flaky))
    assert worker.translate(inputs("one", "two")) == ["甲", "乙"]
    assert [ids_of(call) for call in client.calls] == [[0, 1], [0], [0], [1]]
    assert client.calls[2]["messages"][-2] == {
        "role": "assistant",
        "content": '{"translations": {}}',
    }
    assert 'exactly these keys: "0"' in client.calls[2]["messages"][-1]["content"]


def test_unresolved_segments_are_left_none_after_halving():
    worker, client = agent(
        handler=numbered(
            ["甲", "乙", "丙"],
            lambda ids: {"translations": {"0": "甲"}} if 0 in ids else {},
        )
    )
    assert worker.translate(inputs("one", "two", "three")) == ["甲", None, None]
    assert worker.failures == {1: "repeated_reply", 2: "repeated_reply"}
    # The whole round accepts nothing (0 precedes the gap), so the request is halved down to
    # single ids; ids 1 and 2 stop once their reply repeats.
    assert [ids_of(call) for call in client.calls] == [
        [0, 1, 2],
        [0],
        [1, 2],
        [1],
        [1],
        [2],
        [2],
    ]


def test_legacy_array_aligns_by_position_only_when_counts_match():
    worker, client = agent({"translations": ["甲", "乙"]})
    assert worker.translate(inputs("one", "two")) == ["甲", "乙"]
    assert len(client.calls) == 1
    worker, client = agent(
        handler=numbered(
            ["甲", "乙"], lambda ids: {"translations": ["甲", "乙", "丙"]} if len(ids) > 1 else None
        )
    )
    assert worker.translate(inputs("one", "two")) == ["甲", "乙"]
    assert [ids_of(call) for call in client.calls] == [[0, 1], [0], [1]]


def test_each_request_sends_a_strict_schema_for_its_ids():
    worker, client = agent(
        handler=numbered(
            ["甲", "乙"],
            lambda ids: {"translations": {"1": "乙"}} if len(ids) > 1 else None,
        )
    )
    worker.translate(inputs("one", "two"))
    schemas = [call["json_schema"] for call in client.calls]
    assert schemas[1] == {
        "type": "object",
        "properties": {
            "translations": {
                "type": "object",
                "properties": {"0": {"type": "string"}},
                "required": ["0"],
                "additionalProperties": False,
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }
    assert schemas[0]["properties"]["translations"]["required"] == ["0", "1"]
    assert all(call["json_mode"] is True for call in client.calls)


def test_bare_translation_array_is_accepted_like_standard_translation():
    worker, client = agent(["甲", "乙"])
    assert worker.translate(inputs("one", "two")) == ["甲", "乙"]
    assert worker.synthesize(inputs("one", "two"), [["甲", "乙"]] * 3) == ["甲", "乙"]


@pytest.mark.parametrize("status", [401, 429, 500])
def test_provider_failures_do_not_start_another_retry_loop(status):
    error = ProviderRequestError(
        provider="fake",
        operation="translation.body",
        error_type="http_error",
        status_code=status,
        reason=f"http_{status}" if status != 401 else None,
        resumable=status != 401,
    )

    def fail(*_):
        raise error

    worker, client = agent(handler=fail)
    with pytest.raises(ProviderRequestError) as caught:
        worker.translate(inputs("source"))
    assert caught.value is error
    assert len(client.calls) == 1


@pytest.mark.parametrize("requested", [False, True])
def test_recording_failures_do_not_retry_or_duplicate_paid_calls(requested):
    worker, client = agent({"translations": ["译文"]})
    error = OSError("record storage unavailable")

    def recorder(operation, messages, invoke, validate):
        if requested:
            validate(invoke())
        raise error

    worker._recorder = recorder
    with pytest.raises(OSError) as caught:
        worker.translate(inputs("source"))
    assert caught.value is error
    assert len(client.calls) == int(requested)


def test_interruption_is_not_retried():
    def interrupt(*_):
        raise KeyboardInterrupt

    worker, client = agent(handler=interrupt)
    with pytest.raises(KeyboardInterrupt):
        worker.translate(inputs("source"))
    assert len(client.calls) == 1


def test_invalid_input_drafts_do_not_trigger_model_retries():
    worker, client = agent({"translations": ["稿"]})
    with pytest.raises(PrecisionError):
        worker.synthesize(inputs("source"), [["稿"], [], ["稿"]])
    assert not client.calls


def test_truncation_halves_without_resending_or_adding_a_budget():
    def truncated(messages, *_):
        if len(segment_ids(task_in(messages))) > 1:
            raise TruncatedResponseError("provider output ended")
        return json.dumps(
            {"translations": {str(i): f"译{i}" for i in segment_ids(task_in(messages))}}
        )

    worker, client = agent(handler=truncated)
    assert worker.translate(inputs("one", "two", "three", "four")) == ["译0", "译1", "译2", "译3"]
    # Every truncated request is split at once; nothing is resent unchanged.
    assert [ids_of(call) for call in client.calls] == [
        [0, 1, 2, 3],
        [0, 1],
        [0],
        [1],
        [2, 3],
        [2],
        [3],
    ]
    assert all(call["max_tokens"] is None for call in client.calls)
    sources = ["one", "two", "three", "four"]
    assert all(
        [item["source"] for item in task_of(call)["segments"]]
        == [sources[index] for index in ids_of(call)]
        for call in client.calls
    )


def test_single_segment_truncation_is_not_resent():
    def truncated(*_):
        raise TruncatedResponseError("provider output ended")

    worker, client = agent(handler=truncated)
    assert worker.translate(inputs("one")) == [None]
    assert worker.failures == {0: "truncated"}
    assert len(client.calls) == 1


def test_protected_positions_and_intentional_blank_targets():
    worker, client = agent({"translations": ["", "123"]})
    packet = inputs("OCR", "123", allow_empty=True)
    assert worker.translate(packet) == ["", "123"]
    assert worker.synthesize(packet, [["", "123"]] * 3) == ["", "123"]
    assert worker.translate(inputs("123", "---")) == ["123", "---"]
    assert len(client.calls) == 2


@pytest.mark.parametrize("allow_empty", [False, True])
@pytest.mark.parametrize(
    ("source", "generated"),
    [
        ("10\u2005\u20059\u2005\u20058", "10 9 8"),
        ("10\u2005\u20059\u2005\u20058", ""),
        ("123", "一二三"),
        ("---", "separator"),
        ("\t\u2005\n", "added text"),
        ("", "added text"),
    ],
)
def test_protected_outputs_use_source_in_drafts_and_synthesis(source, generated, allow_empty):
    worker, client = agent({"translations": ["译文", generated]})
    packet = inputs("source", source, allow_empty=allow_empty)
    assert worker.translate(packet) == ["译文", source]
    drafts = [["甲稿", generated], ["乙稿", "changed"], ["丙稿", ""]]
    assert worker.synthesize(packet, drafts) == ["译文", source]
    task = json.loads(client.calls[-1]["messages"][-1]["content"].split("Task (JSON):\n")[1])
    assert task["segments"] == [
        {"id": 0, "source": "source", "drafts": ["甲稿", "乙稿", "丙稿"]},
        {"id": 1, "source": source, "drafts": [source]},
    ]
    assert drafts == [["甲稿", generated], ["乙稿", "changed"], ["丙稿", ""]]
    assert len(client.calls) == 2


def test_protected_positions_are_backfilled_without_requesting_them_again():
    worker, client = agent({"translations": {"0": "译文", "1": None}})
    assert worker.translate(inputs("source", "123")) == ["译文", "123"]
    assert len(client.calls) == 1


def test_book_prefix_is_stable_before_batch_context():
    worker, client = agent({"translations": ["二"]})
    packet = inputs("two")
    worker.translate(packet)
    worker.translate(replace(packet, context="new context", chapter_digest="next chapter"))
    first, second = [call["messages"][1]["content"] for call in client.calls]
    assert first.split('"chapter_digest"')[0] == second.split('"chapter_digest"')[0]


def test_synthesis_requires_three_structurally_usable_drafts():
    worker, client = agent({"translations": ["稿"]})
    with pytest.raises(PrecisionError):
        worker.synthesize(inputs("source"), [["稿"]])
    assert not client.calls


# Shapes observed from a real 36-segment batch whose drafts all failed whole-batch validation.
REAL_SOURCES = tuple(f"Paragraph {i} begins the scene where Ann waits." for i in range(36))
REAL_NEXT = "Next chapter opens on a quiet morning."


def real_reply(shape):
    entries = {str(i): f"第{i}段。" for i in range(36)}
    if shape == "early_stop":
        entries = {key: value for key, value in entries.items() if int(key) < 7}
        entries["6"] = "你当然不想现在……"
    else:
        entries["36"] = "下一章始于一个安静的早晨。"
    if shape == "loop":
        entries.update({str(i): REAL_NEXT for i in range(37, 53)})
    return {"translations": entries}


@pytest.mark.parametrize(
    ("shape", "requested"),
    [("early_stop", [list(range(36)), list(range(6, 36))])]
    # An entry for id 36 (next_source or a loop) rejects the whole reply, which is halved.
    + [
        (shape, [list(range(36)), list(range(18)), list(range(18, 36))])
        for shape in ("loop", "next_source")
    ],
)
def test_real_failure_shapes_keep_valid_entries_and_repair_only_suspicious_ids(shape, requested):
    answers = [f"第{i}段。" for i in range(36)]
    worker, client = agent(
        handler=numbered(answers, lambda ids: real_reply(shape) if len(ids) == 36 else None)
    )
    packet = replace(inputs(*REAL_SOURCES), next_source=REAL_NEXT)
    assert worker.translate(packet) == answers
    assert [ids_of(call) for call in client.calls] == requested


MAYBE = "“Maybe,” she said. “Maybe. Oh, Junie. Life is long.”"
REPAIR_SOURCES = tuple(MAYBE if i == 21 else f"Paragraph {i} waits." for i in range(36))


@pytest.mark.parametrize("stage", ["translate", "synthesize"])
def test_repairs_quote_their_sources_and_retry_with_feedback(stage):
    answers = [f"译{i}" for i in range(36)]
    replies = iter([{"translations": {"21": ""}}])

    def override(ids):
        if ids == list(range(36)):
            # Entry 21 is blank, so only that id is requested again.
            return {"translations": {str(i): "" if i == 21 else answers[i] for i in ids}}
        return next(replies, None)

    worker, client = agent(handler=numbered(answers, override))
    packet = inputs(*REPAIR_SOURCES)
    if stage == "translate":
        assert worker.translate(packet) == answers
    else:
        assert worker.synthesize(packet, [["稿"] * 36] * 3) == answers
    tasks = [task_of(call) for call in client.calls]
    # Every request names each id together with its own source.
    assert [(item["id"], item["source"]) for item in tasks[0]["segments"]] == list(
        enumerate(REPAIR_SOURCES)
    )
    repairs = [[(item["id"], item["source"]) for item in task["segments"]] for task in tasks[1:]]
    assert repairs == [[(21, MAYBE)]] * 2
    assert all(call["messages"][:2] == client.calls[0]["messages"][:2] for call in client.calls)
    if stage == "synthesize":
        assert tasks[1]["segments"][0]["drafts"] == ["稿"]
    # The lone id's blank reply is quoted back with the restated contract.
    feedback = client.calls[-1]["messages"][-1]["content"]
    assert "the source quoted with that id in task.segments" in feedback
    assert '{"translations": {"<id>": "<translation>"}}' in feedback


def test_single_segment_reply_naming_another_id_is_rejected():
    replies = {
        (0, 1): {"translations": {"0": "甲", "1": ""}},
        (1,): {"translations": {"1": "乙", "0": "多余"}},
    }
    worker, client = agent(
        handler=lambda messages, *_: json.dumps(replies[tuple(segment_ids(task_in(messages)))])
    )
    assert worker.synthesize(inputs("one", "two"), [["甲稿", "乙稿"]] * 3) == ["甲", None]
    # The lone repair answers id 0 too, so nothing of it is used; the same reply ends retries.
    assert [ids_of(call) for call in client.calls] == [[0, 1], [1], [1]]
    assert worker.failures == {1: "repeated_reply"}


def entries(texts):
    return {"translations": {str(i): text for i, text in enumerate(texts)}}


def accept(sources, data, **options):
    return accepted_segments(sources, list(range(len(sources))), data, **options)


def test_unrequested_ids_are_dropped_and_others_kept():
    sources = ("Alpha.", "Bravo.")
    data = {"translations": {"0": "甲", "1": "乙", "2": "下一段", "-1": "负", "01": "零"}}
    assert accept(sources, data) == {0: "甲", 1: "乙"}
    assert accepted_segments(sources, [1], data) == {1: "乙"}


def test_highest_id_before_a_gap_is_pending():
    sources = ("A one.", "B two.", "C three.", "D four.", "E five.")
    # Gaps follow 0 and 2; only the highest id before a gap is redone with the missing ids.
    data = {"translations": {"0": "甲", "2": "丙", "4": "戊"}}
    assert accept(sources, data) == {0: "甲", 4: "戊"}
    # A missing protected id is not a gap.
    data = {"translations": {"0": "甲", "1": "乙", "3": "丁"}}
    assert accept(("A one.", "B two.", "123", "D four."), data) == {0: "甲", 1: "乙", 3: "丁"}


def test_repeated_targets_for_different_sources_are_pending():
    sources = ("First line here.", "Second line here.", "Third line here.")
    assert accept(sources, entries(["一", "重复。", "“重复”"])) == {0: "一"}


def test_identical_targets_for_identical_short_sources_are_kept():
    sources = ("Yes—", "He nodded.", "Yes—", "123", "123")
    data = entries(["是——", "他点头。", "是——", "123", "123"])
    assert accept(sources, data) == {0: "是——", 1: "他点头。", 2: "是——", 3: "123", 4: "123"}


def test_source_copies_are_pending_only_across_languages_and_in_batches():
    sources = ("He left the room.", "She stayed.")
    data = entries(["He left the room.", "她留下了。"])
    assert accept(sources, data) == {1: "她留下了。"}
    assert accept(sources, data, copy_is_untranslated=False) == {
        0: "He left the room.",
        1: "她留下了。",
    }
    assert accepted_segments(("NASA", "x"), [0], entries(["NASA"])) == {0: "NASA"}


def test_legacy_shapes_and_non_string_entries():
    sources = ("one", "two")
    assert accept(sources, {"translations": ["甲", "乙"]}) == {0: "甲", 1: "乙"}
    assert accept(sources, ["甲", "乙"]) == {0: "甲", 1: "乙"}
    assert accept(sources, {"translations": ["甲", "乙", "丙"]}) == {}
    assert accept(sources, {"translations": ["甲"]}) == {}
    # Old ``{"src", "text"}`` objects are no longer translations.
    mixed = {"translations": {"0": "甲", "1": {"src": "two", "text": "乙"}, "2": "丙"}}
    assert accept(sources, mixed) == {0: "甲"}


def test_duplicate_keys_are_pending():
    sources = ("one", "two")
    data = entries(["甲", "乙"])
    assert accept(sources, data, duplicate_keys={"1"}) == {0: "甲"}
    assert accept(sources, data, duplicate_keys={"0"}) == {1: "乙"}
