"""Precision request budgeting, halving, per-segment fallbacks and partial-progress resume."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest
from wenyi_core.agents.precision import PrecisionAgent, PrecisionInputs
from wenyi_core.config import Config
from wenyi_core.llm.configuration import LLMConfig
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.llm.retrying import TruncatedResponseError
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.translate import precision as precision_run

from tests.fake_llm import METERED_TOTAL_TOKENS, MeteredFakeClient
from tests.precision_fixtures import PrecisionHandler as Handler
from tests.precision_fixtures import artifact as _artifact
from tests.precision_fixtures import precision_config as _config
from tests.precision_fixtures import precision_store as _store
from tests.precision_fixtures import segment_ids


def inputs(*sources):
    return PrecisionInputs(
        tuple(sources), (), "prior", "style", "synopsis", "digest", [[] for _ in sources], ""
    )


def task_in(messages):
    message = next(m for m in messages if "Task (JSON):\n" in m["content"])
    return json.loads(message["content"].split("Task (JSON):\n")[1])


def ids_of(call):
    return segment_ids(task_in(call["messages"]))


# Four ~1000-token sources: the whole request needs ~16.8K tokens with the output reserve,
# each half ~12.8K, so a 14K window admits halves only.
LONG = tuple(f"chapter {n} " + "lorem ipsum " * 500 for n in range(4))
WINDOW = 14_000


def _agent(handler, context_window=None):
    provider = {"kind": "fake"}
    if context_window is not None:
        provider["context_window"] = context_window
    llm = LLMConfig.model_validate({"preset": "fake", "providers": {"default": provider}})
    config = Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {"preset": "fake"},
            "pipeline": {"translation_mode": "best_of_three"},
        }
    )
    client = FakeClient(handler=handler, config=llm)
    return PrecisionAgent(client, config), client


def _echo(messages, tier, json_mode):
    return json.dumps({"translations": {str(i): f"译{i}" for i in segment_ids(task_in(messages))}})


def test_configured_context_window_splits_before_sending():
    worker, client = _agent(_echo, context_window=WINDOW)
    assert worker.translate(inputs(*LONG)) == ["译0", "译1", "译2", "译3"]
    assert [ids_of(call) for call in client.calls] == [[0, 1], [2, 3]]


def test_truncation_usage_teaches_the_window_for_later_requests():
    truncated = []

    def respond(messages, tier, json_mode):
        if len(segment_ids(task_in(messages))) == 4:
            error = TruncatedResponseError("cut")
            error.usage_total = WINDOW
            truncated.append(error)
            raise error
        return _echo(messages, tier, json_mode)

    worker, client = _agent(respond)
    # Truncation halves without resending; both halves succeed.
    assert worker.translate(inputs(*LONG)) == ["译0", "译1", "译2", "译3"]
    assert [ids_of(call) for call in client.calls] == [[0, 1, 2, 3], [0, 1], [2, 3]]
    assert client.context_window("translation.body") == WINDOW
    client.calls.clear()
    assert worker.translate(inputs(*LONG)) == ["译0", "译1", "译2", "译3"]
    assert [ids_of(call) for call in client.calls] == [[0, 1], [2, 3]]
    assert len(truncated) == 1


def _events(store, event_type):
    return store.list_events(event_type=event_type, limit=0)


def test_failed_synthesis_publishes_first_draft_and_marks_segments(tmp_path):
    store, config = _store(tmp_path), _config(tmp_path)
    client = MeteredFakeClient(handler=Handler(invalid=True))
    Orchestrator(config, client)._translation.run(store, book_synopsis="synopsis")
    segments = store.load_chapter(0).text_segments
    assert [s.target for s in segments] == [s.target_before_polish for s in segments]
    assert all(s.target and s.target.startswith("译") for s in segments)
    degraded = _events(store, "precision_segment_degraded")
    assert [(e["segment"], e["stage"], e["fallback"]) for e in degraded] == [
        (10, "synthesis", "draft_T1"),
        (11, "synthesis", "draft_T1"),
    ]
    assert {e["chapter"] for e in degraded} == {0}
    assert _events(store, "translate_run_finished")[-1]["precision_degraded_segments"] == 2
    key = next(
        k for k in store.list_artifacts("precision/chapters/0/") if k.endswith("/result.json")
    )
    assert [e["fallback"] for e in _artifact(store, key)["payload"]["degraded"]] == [
        "draft_T1",
        "draft_T1",
    ]
    assert store.pending_chapters() == []


def _serial_drafts(monkeypatch):
    # Drafts run T1, T2, T3 in submission order so a handler can target one candidate.
    monkeypatch.setattr(
        precision_run, "ThreadPoolExecutor", lambda max_workers: ThreadPoolExecutor(1)
    )


def test_failed_draft_segment_is_synthesized_from_the_other_drafts(tmp_path, monkeypatch):
    _serial_drafts(monkeypatch)
    store, config, base = _store(tmp_path), _config(tmp_path), Handler()
    state = {"draft": 0}

    def respond(messages, tier, json_mode):
        task = task_in(messages)
        if "drafts" not in task:
            if segment_ids(task) == [0, 1] and not isinstance(task["segments"][0], dict):
                state["draft"] += 1
            if state["draft"] == 1:
                if segment_ids(task) == [1]:
                    return '{"translations":[]}'
                return json.dumps({"translations": {"0": "甲0"}})
        return base(messages, tier, json_mode)

    client = MeteredFakeClient(handler=respond)
    Orchestrator(config, client)._translation.run(store, book_synopsis="synopsis")
    segments = store.load_chapter(0).text_segments
    assert [s.target for s in segments] == ["润1:0", "润1:1"]
    synthesis = [c for c in client.calls if c["operation"] == "polish.body"]
    assert [sum("1" in d for d in task_in(c["messages"])["drafts"]) for c in synthesis] == [2]
    assert segments[1].target_before_polish.startswith("译")
    degraded = _events(store, "precision_segment_degraded")
    assert [(e["segment"], e["stage"], e["fallback"], e["reason"]) for e in degraded] == [
        (11, "drafts", "synthesis", "repeated_reply")
    ]
    assert degraded[0]["drafts"] == ["T2", "T3"]


@pytest.mark.parametrize("standard_ok", [True, False])
def test_segment_without_drafts_uses_standard_translation_or_is_refused(tmp_path, standard_ok):
    store, config, base = _store(tmp_path), _config(tmp_path), Handler()

    def respond(messages, tier, json_mode):
        if not any("Task (JSON):\n" in m["content"] for m in messages):
            # Plain-text single-segment translation; a blank reply is refused.
            assert not json_mode
            return "「标准稿」" if standard_ok else "  "
        task = task_in(messages)
        if "drafts" not in task and 1 in segment_ids(task):
            reply = json.loads(base(messages, tier, json_mode))
            reply["translations"].pop("1")
            return json.dumps(reply) if reply["translations"] else '{"translations":[]}'
        return base(messages, tier, json_mode)

    client = MeteredFakeClient(handler=respond)
    service = Orchestrator(config, client)._translation
    service.run(store, book_synopsis="synopsis")
    segments = store.load_chapter(0).text_segments
    fallback = _events(store, "precision_segment_degraded")[-1]
    assert (fallback["segment"], fallback["stage"]) == (11, "drafts")
    # Refused or not, the chapter completes so Review and export can run.
    assert store.pending_chapters() == []
    if standard_ok:
        assert fallback["fallback"] == "standard"
        assert segments[1].target == "标准稿"
        assert service.refused_segments(store) == []
        return
    assert fallback["fallback"] == "refused"
    assert fallback["tried"] == ["synthesis", "drafts", "plain_text"]
    assert segments[0].target is not None and segments[1].target is None
    assert _events(store, "chapter_translated_with_gaps")[-1]["refused_segments"] == [11]
    refused = _events(store, "segment_translation_refused")[-1]
    assert (refused["chapter"], refused["segment"], refused["reason"]) == (0, 11, "empty_reply")
    assert [(i["chapter"], i["segment"]) for i in service.refused_segments(store)] == [(0, 11)]
    finished = _events(store, "translate_run_finished")[-1]
    assert finished["refused_segments"] == 1
    assert finished["refused"] == [{"chapter": 0, "segment": 11}]

    later = MeteredFakeClient(handler=Handler())
    service = Orchestrator(config, later)._translation
    service.run(store, book_synopsis="synopsis")
    segments = store.load_chapter(0).text_segments
    assert segments[1].target == "润1:0"
    assert service.refused_segments(store) == []
    assert [
        (e["chapter"], e["segment"]) for e in _events(store, "segment_translation_recovered")
    ] == [(0, 11)]
    assert _events(store, "translate_run_finished")[-1]["refused_segments"] == 0
    precision_calls = [
        c for c in later.calls if c["operation"] in {"translation.body", "polish.body"}
    ]
    # Only the refused segment is batched again: three drafts plus one synthesis.
    assert len(precision_calls) == 4
    assert all('"two"' in c["messages"][1]["content"] for c in precision_calls)
    assert all('"one"' not in c["messages"][1]["content"] for c in precision_calls)


def test_interrupted_synthesis_resumes_without_rebuying_accepted_segments(tmp_path):
    store, config, base = _store(tmp_path), _config(tmp_path), Handler()
    lock, synthesis = Lock(), []

    def respond(messages, tier, json_mode):
        task = task_in(messages)
        if "drafts" in task:
            with lock:
                synthesis.append(segment_ids(task))
                count = len(synthesis)
            if count == 1:
                return '{"translations":[]}'  # Zero progress: halve to single segments.
            if count == 3:
                raise RuntimeError("interrupted")
        return base(messages, tier, json_mode)

    first = MeteredFakeClient(handler=respond)
    with pytest.raises(RuntimeError, match="interrupted"):
        Orchestrator(config, first)._translation.run(store, book_synopsis="synopsis")
    assert synthesis == [[0, 1], [0], [1]]
    later = MeteredFakeClient(handler=Handler())
    Orchestrator(config, later)._translation.run(store, book_synopsis="synopsis")
    assert [ids_of(c) for c in later.calls if c["operation"] == "polish.body"] == [[1]]
    assert not [c for c in later.calls if c["operation"] == "translation.body"]
    segments = store.load_chapter(0).text_segments
    accepted = segments[0].target
    assert accepted is not None and accepted.startswith("润")
    assert segments[1].target == "润1:1"
    usage = store.load_usage()
    assert usage is not None
    assert usage["by_stage"]["translation.body"]["calls"] == 3
    assert usage["by_stage"]["polish.body"]["calls"] == 4
    total = sum(stage["calls"] for stage in usage["by_stage"].values())
    assert usage["totals"]["total_tokens"] == METERED_TOTAL_TOKENS * total


def _overflow():
    from wenyi_core.llm.retrying import ContextLengthExceeded

    return ContextLengthExceeded(
        provider="fake", operation="polish.body", status_code=400, window=None
    )


def _trims(client):
    events = []
    client.set_event_sink(
        lambda event, **data: events.append(data["step"]) if event == "context_trimmed" else None
    )
    return events


def test_estimate_calibration_splits_before_the_provider_rejects():
    # ~4.8K prompt + ~12K output reserve fits 18.5K uncalibrated, but not at 1.3x the prompt.
    worker, client = _agent(_echo, context_window=18_500)
    assert worker.translate(inputs(*LONG)) == ["译0", "译1", "译2", "译3"]
    assert [ids_of(call) for call in client.calls] == [[0, 1, 2, 3]]
    client.calls.clear()
    probe = [{"role": "user", "content": "lorem ipsum " * 500}]
    from wenyi_core.llm.base import estimate_prompt_tokens

    # The provider counted 30% more prompt tokens than cl100k on an earlier request.
    client.calibrate_prompt_tokens(
        "translation.body", probe, int(estimate_prompt_tokens(probe) * 1.3)
    )
    assert worker.translate(inputs(*LONG)) == ["译0", "译1", "译2", "译3"]
    assert [ids_of(call) for call in client.calls] == [[0, 1], [2, 3]]


def test_oversized_synthesis_keeps_two_drafts_then_adopts_one():
    drafts = [["a"], ["b"], ["c"]]

    def two_fit(messages, tier, json_mode):
        if len(task_in(messages)["drafts"]) > 2:
            raise _overflow()
        return json.dumps({"translations": {"0": "合"}})

    worker, client = _agent(two_fit)
    trims = _trims(client)
    assert worker.synthesize(inputs("Source."), drafts) == ["合"]
    assert trims == ["context", "drafts_2"]
    assert len(task_in(client.calls[-1]["messages"])["drafts"]) == 2

    def never_fits(messages, tier, json_mode):
        raise _overflow()

    worker, client = _agent(never_fits)
    trims = _trims(client)
    assert worker.synthesize(inputs("Source."), drafts) == [None]
    assert worker.failures == {0: "context_length"}
    assert trims == ["context", "drafts_2", "drafts_1"]


def test_oversized_draft_segment_drops_context_before_standard_fallback():
    def no_context(messages, tier, json_mode):
        if '"prior"' in messages[1]["content"]:
            raise _overflow()
        return _echo(messages, tier, json_mode)

    worker, client = _agent(no_context)
    trims = _trims(client)
    assert worker.translate(inputs("Source.")) == ["译0"]
    assert trims == ["context"]
