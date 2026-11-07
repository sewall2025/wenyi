"""Four-call precision execution, durable resume and guarded final publication."""

import json
import signal
from dataclasses import replace
from threading import Event, Lock

import pytest
from wenyi_core.agents.precision import PrecisionError
from wenyi_core.config import Config
from wenyi_core.pipeline import precision as precision_module
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.translate.precision import PrecisionBatchExecutor
from wenyi_core.pipeline.translate.precision_records import load_precision_call
from wenyi_core.pipeline.translate.translation import TranslationService
from wenyi_core.storage.precision_archive import PrecisionArchive

from tests.fake_llm import MeteredFakeClient
from tests.precision_fixtures import PrecisionHandler as Handler
from tests.precision_fixtures import artifact as _artifact
from tests.precision_fixtures import precision_config as _config
from tests.precision_fixtures import precision_plan as _plan
from tests.precision_fixtures import precision_store as _store
from tests.precision_fixtures import segment_ids


def test_four_calls_produce_synthesis_directly_and_keep_first_draft_for_comparison(tmp_path):
    store, config, handler = _store(tmp_path), _config(tmp_path), Handler()
    client = MeteredFakeClient(handler=handler)
    result = PrecisionBatchExecutor(client, config).execute(_plan(store), store)
    assert result.targets == ("润1:0", "润1:1")
    first = _artifact(store, f"{result.precision_key}/drafts/T1.json")["payload"]
    assert result.before_polish == tuple(PrecisionArchive(store).get(first["targets_ref"]))
    assert handler.counts == {"translate": 3, "synthesis": 1}
    assert [call["operation"] for call in client.calls] == ["translation.body"] * 3 + [
        "polish.body"
    ]
    assert all(call["max_tokens"] is None for call in client.calls)
    assert all(segment.target is None for segment in store.load_chapter(0).text_segments)
    assert _artifact(store, f"{result.precision_key}/publication.json")["status"] == "ready"


@pytest.mark.parametrize("boundary", range(1, 5))
def test_resume_each_checkpoint_reuses_completed_work(tmp_path, boundary):
    store, config, handler = _store(tmp_path), _config(tmp_path), Handler()
    client = MeteredFakeClient(handler=handler)
    writes = 0

    def interrupt():
        nonlocal writes
        writes += 1
        if writes == boundary:
            raise RuntimeError("interrupted")

    executor = PrecisionBatchExecutor(client, config)
    with pytest.raises(RuntimeError, match="interrupted"):
        executor.execute(_plan(store), store, checkpoint=interrupt)
    result = executor.execute(_plan(store), store)
    assert result.targets == ("润1:0", "润1:1")
    assert len(client.calls) == 4


def test_invalid_synthesis_falls_back_to_first_draft_without_regenerating(tmp_path):
    store, config, handler = _store(tmp_path), _config(tmp_path), Handler(invalid=True)
    client = MeteredFakeClient(handler=handler)
    executor = PrecisionBatchExecutor(client, config)
    result = executor.execute(_plan(store), store)
    # The zero-progress batch halves to single ids; each stops when its reply repeats.
    synthesis_calls = 1 + 2 * 2
    assert len(client.calls) == 3 + synthesis_calls
    first = _artifact(store, f"{result.precision_key}/drafts/T1.json")["payload"]
    assert list(result.targets) == PrecisionArchive(store).get(first["targets_ref"])
    assert result.before_polish == result.targets
    assert [entry["fallback"] for entry in result.degraded] == ["draft_T1", "draft_T1"]
    assert {entry["reason"] for entry in result.degraded} == {"repeated_reply"}
    # The degraded result is final for this fingerprint and is reused without new calls.
    handler.invalid = False
    assert executor.execute(_plan(store), store) == result
    assert len(client.calls) == 3 + synthesis_calls
    assert all(segment.target is None for segment in store.load_chapter(0).text_segments)


@pytest.mark.parametrize("stage", ["translate", "synthesis"])
def test_zero_progress_round_requests_each_segment_and_reuses_success(tmp_path, stage):
    store, config, handler = _store(tmp_path), _config(tmp_path), Handler()
    lock = Lock()
    failed_messages = []

    def fail_once(messages, tier, json_mode):
        task_message = next(m for m in messages if "Task (JSON):\n" in m["content"])
        task = json.loads(task_message["content"].split("Task (JSON):\n")[1])
        kind = "synthesis" if "drafts" in task else "translate"
        response = handler(messages, tier, json_mode)
        with lock:
            if kind == stage and not failed_messages:
                failed_messages.append(messages)
                return '{"translations":[]}'
        return response

    client = MeteredFakeClient(handler=fail_once)
    executor = PrecisionBatchExecutor(client, config)
    result = executor.execute(_plan(store), store)
    assert len(client.calls) == 6
    assert handler.counts == {
        "translate": 3 + 2 * (stage == "translate"),
        "synthesis": 1 + 2 * (stage == "synthesis"),
    }
    operation = "translation.body" if stage == "translate" else "polish.body"
    requests = [call for call in client.calls if call["operation"] == operation]
    assert all(call["max_tokens"] is None for call in requests)
    # Only the failed stage repeats, one segment per request, over the same frozen packet.
    segments = [
        segment_ids(json.loads(call["messages"][-1]["content"].split("Task (JSON):\n")[1]))
        for call in requests
    ]
    assert [ids for ids in segments if len(ids) == 1] == [[0], [1]]
    assert all(call["messages"][:2] == failed_messages[0][:2] for call in requests)
    records = [
        load_precision_call(store, key)
        for key in store.list_artifacts(f"{result.precision_key}/calls/")
        if key.endswith(".json")
    ]
    assert len(records) == 6
    failed = [record for record in records if record["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["error_category"] == "invalid_output"
    assert json.loads(failed[0]["raw_response"]) == {"translations": []}
    assert sum(record["status"] == "completed" for record in records) == 5
    assert executor.execute(_plan(store), store) == result
    assert len(client.calls) == 6
    usage = client.usage_summary()
    assert usage["totals"]["calls"] == 6
    assert usage["totals"]["total_tokens"] == 48
    assert usage["by_stage"][operation]["calls"] == len(requests)


@pytest.mark.parametrize("change", ["source", "context", "route"])
def test_incompatible_pending_checkpoints_are_not_reused(tmp_path, change):
    store, config = _store(tmp_path), _config(tmp_path)
    client = MeteredFakeClient(handler=Handler())
    executor, plan = PrecisionBatchExecutor(client, config), _plan(store)
    first = executor.execute(plan, store)
    if change == "source":
        manifest = store.load_manifest()
        manifest["source_sha256"] = "1" * 64
        store.save_manifest(manifest)
    elif change == "context":
        plan = replace(plan, context="new context")
    else:
        config.llm.models["default_strong"] = config.llm.models["default_strong"].model_copy(
            update={"model": "different-model"}
        )
    second = executor.execute(plan, store)
    assert second.precision_key != first.precision_key
    assert len(client.calls) == 8


def test_builtin_concurrency_ignores_legacy_values_and_preserves_saved_work(tmp_path, monkeypatch):
    store, config = _store(tmp_path), _config(tmp_path)
    document = config.model_dump()
    document["pipeline"]["precision_concurrency"] = 1
    config = Config.model_validate(document)
    workers = []
    original_pool = precision_module.ThreadPoolExecutor

    def pool(*, max_workers):
        workers.append(max_workers)
        return original_pool(max_workers=max_workers)

    monkeypatch.setattr(precision_module, "ThreadPoolExecutor", pool)
    client = MeteredFakeClient(handler=Handler())
    executor = PrecisionBatchExecutor(client, config)
    first = executor.execute(_plan(store), store)
    assert executor.execute(_plan(store), store) == first
    assert workers == [3]
    assert len(client.calls) == 4


def test_ctrl_c_cancels_model_work_before_joining_draft_threads(tmp_path, monkeypatch):
    store, config = _store(tmp_path), _config(tmp_path)
    started, release = Event(), Event()
    entered, lock = [], Lock()

    def blocking(messages, tier, json_mode):
        with lock:
            entered.append(True)
            if len(entered) == 3:
                started.set()
        # Bound a failing implementation's join rather than hanging the test suite.
        release.wait(timeout=5)
        return Handler()(messages, tier, json_mode)

    class CancellableClient(MeteredFakeClient):
        cancelled = False

        def cancel(self):
            self.cancelled = True
            release.set()

    client = CancellableClient(handler=blocking)
    previous = signal.getsignal(signal.SIGINT)

    def interrupt(_):
        assert started.wait(timeout=5)
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        raise AssertionError("SIGINT did not interrupt the precision batch")

    monkeypatch.setattr(precision_module, "as_completed", interrupt)
    with pytest.raises(KeyboardInterrupt):
        PrecisionBatchExecutor(client, config).execute(_plan(store), store)
    assert client.cancelled
    assert release.is_set()
    assert signal.getsignal(signal.SIGINT) is previous


def test_intentional_blanks_and_numeric_only_batches(tmp_path):
    store, config = _store(tmp_path, ("OCR", "123")), _config(tmp_path)
    result = PrecisionBatchExecutor(MeteredFakeClient(handler=Handler(blank=True)), config).execute(
        _plan(store, allow_empty=True), store
    )
    assert result.targets == ("", "123")
    numeric = _store(tmp_path / "numeric", ("123", "---"))
    client = MeteredFakeClient(handler=Handler())
    assert PrecisionBatchExecutor(client, config).execute(_plan(numeric), numeric).targets == (
        "123",
        "---",
    )
    assert not client.calls


@pytest.mark.parametrize("boundary", [1, 4])
def test_protected_sources_survive_generation_archive_and_resume(tmp_path, boundary):
    source = "10\u2005\u20059\u2005\u20058"
    store, config, handler = _store(tmp_path, ("one", source)), _config(tmp_path), Handler()

    def normalize_spaces(messages, tier, json_mode):
        response = json.loads(handler(messages, tier, json_mode))
        response["translations"]["1"] = source.replace("\u2005", " ")
        return json.dumps(response)

    client = MeteredFakeClient(handler=normalize_spaces)
    executor, plan = PrecisionBatchExecutor(client, config), _plan(store)
    writes = 0

    def interrupt():
        nonlocal writes
        writes += 1
        if writes == boundary:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        executor.execute(plan, store, checkpoint=interrupt)
    result = executor.execute(plan, store)
    assert result.targets == ("润1:0", source)
    assert result.before_polish[1] == source
    assert len(client.calls) == 4
    keys = [
        key
        for key in store.list_artifacts(f"{result.precision_key}/calls/")
        if key.endswith(".json")
    ]
    assert len(keys) == 4
    for key in keys:
        record = load_precision_call(store, key)
        assert record["status"] == "completed"
        assert record["targets"][1] == source
        assert json.loads(record["raw_response"])["translations"]["1"] == source.replace(
            "\u2005", " "
        )
    assert all(segment.target is None for segment in store.load_chapter(0).text_segments)


def test_cached_targets_restore_protected_sources(tmp_path):
    source = "10\u2005\u20059\u2005\u20058"
    store, config = _store(tmp_path, ("one", source)), _config(tmp_path)
    run = precision_module._PrecisionRun(
        MeteredFakeClient(handler=Handler()),
        config,
        _plan(store),
        store,
        checkpoint=None,
        progress=None,
    )
    assert run.targets(["译文", ""]) == ["译文", source]
    assert run.targets(["译文", "10 9 8"]) == ["译文", source]
    with pytest.raises(PrecisionError):
        run.targets(["", source])


def test_guarded_publication_keeps_manual_edits_and_recovers_marker(tmp_path, monkeypatch):
    store, config = _store(tmp_path), _config(tmp_path)
    client = MeteredFakeClient(handler=Handler())
    service = Orchestrator(config, client)._translation

    def interrupt(*_):
        raise RuntimeError("after publication")

    monkeypatch.setattr(service._precision, "mark_published", interrupt)
    with pytest.raises(RuntimeError, match="after publication"):
        service.run(store, book_synopsis="synopsis")
    key = next(
        key
        for key in store.list_artifacts("precision/chapters/0/")
        if key.endswith("/publication.json")
    )
    assert _artifact(store, key)["status"] == "ready"
    later = MeteredFakeClient(handler=Handler())
    Orchestrator(config, later)._translation.run(store, book_synopsis="synopsis")
    assert _artifact(store, key)["status"] == "published"
    assert not [
        call for call in later.calls if call["operation"] in {"translation.body", "polish.body"}
    ]
    usage = store.load_usage()
    assert usage is not None
    assert usage["by_stage"]["translation.body"]["calls"] == 3
    assert usage["by_stage"]["polish.body"]["calls"] == 1


def test_publication_refuses_manual_target_changes(tmp_path):
    store, config = _store(tmp_path), _config(tmp_path)
    plan = _plan(store)
    result = PrecisionBatchExecutor(MeteredFakeClient(handler=Handler()), config).execute(
        plan, store
    )
    chapter = store.load_chapter(0)
    chapter.segments[0].target = "人工编辑"
    store.save_chapter(chapter)
    with pytest.raises(PrecisionError, match="edited"):
        TranslationService.save_precision_batch(store, plan, result)
    assert store.load_chapter(0).segments[0].target == "人工编辑"


def test_cli_exports_and_resumes_with_four_body_calls(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from wenyi_cli.cli import app

    source, output = tmp_path / "sample.txt", tmp_path / "translated.txt"
    source.write_text("one\n\ntwo\n", encoding="utf-8")
    config, handler = _config(tmp_path / "state"), Handler()
    client = MeteredFakeClient(handler=handler)
    monkeypatch.setattr("wenyi_cli.commands.context.CommandContext.load_config", lambda *_: config)
    monkeypatch.setattr("wenyi_core.pipeline.run.runtime.build_client", lambda *_: client)
    arguments = [
        "translate",
        str(source),
        "--translation-mode",
        "best_of_three",
        "--polish",
        "--no-review",
        "--format",
        "txt",
        "--out",
        str(output),
    ]
    for _ in range(2):
        result = CliRunner().invoke(app, arguments)
        assert result.exit_code == 0, result.output
    assert "润" in output.read_text(encoding="utf-8")
    assert handler.counts == {"translate": 3, "synthesis": 1}


def test_extra_and_missing_paragraphs_are_repaired_by_id_without_pausing(tmp_path):
    sources = tuple(f"paragraph {i}" for i in range(36))
    store, config, handler = _store(tmp_path, sources), _config(tmp_path), Handler()
    requested = []

    def misaligned(messages, tier, json_mode):
        response = json.loads(handler(messages, tier, json_mode))
        translations = response["translations"]
        task = json.loads(messages[-1]["content"].split("Task (JSON):\n")[1])
        requested.append(("drafts" in task, segment_ids(task)))
        if len(segment_ids(task)) == len(sources):
            translations["36"] = "surplus"
            if "drafts" in task:
                del translations["17"]
        return json.dumps(response)

    client = MeteredFakeClient(handler=misaligned)
    result = PrecisionBatchExecutor(client, config).execute(_plan(store), store)
    assert len(result.targets) == 36 and "surplus" not in result.targets
    # Drafts with a surplus key whose entries echo their sources need no repair; synthesis
    # re-requests paragraph 17 and paragraph 16 just before the gap.
    assert len(client.calls) == 5
    assert requested[-1] == (True, [16, 17])
    assert result.targets[15] == "润1:15"
    assert result.targets[16] == "润2:16" and result.targets[17] == "润2:17"


def test_interrupted_synthesis_repair_reuses_completed_drafts(tmp_path):
    store, config, handler = _store(tmp_path), _config(tmp_path), Handler()
    interrupt = True

    def drop_then_interrupt(messages, tier, json_mode):
        response = json.loads(handler(messages, tier, json_mode))
        task = json.loads(messages[-1]["content"].split("Task (JSON):\n")[1])
        if "drafts" in task and segment_ids(task) == [0, 1]:
            del response["translations"]["1"]
        elif "drafts" in task and interrupt:
            raise KeyboardInterrupt
        return json.dumps(response)

    client = MeteredFakeClient(handler=drop_then_interrupt)
    executor = PrecisionBatchExecutor(client, config)
    with pytest.raises(KeyboardInterrupt):
        executor.execute(_plan(store), store)
    assert handler.counts == {"translate": 3, "synthesis": 2}
    interrupt = False
    result = executor.execute(_plan(store), store)
    # ponytail: mid-stage partial synthesis is not resumed; the stage restarts from its drafts.
    # Dropping id 1 makes id 0 suspicious too, so both are requested alone after each round.
    assert handler.counts == {"translate": 3, "synthesis": 5}
    assert result.targets == ("润4:0", "润5:1")
