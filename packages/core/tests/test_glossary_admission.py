"""Admission rules, the book-frequency gate and candidate-free prompt snapshots."""

from __future__ import annotations

import json
import sqlite3

import pytest
from wenyi_core.agents.glossary_extractor import GlossaryExtractor
from wenyi_core.glossary.admission import (
    MAX_CJK_CHARS,
    active_terms,
    admission_status,
    admit_candidate,
)
from wenyi_core.glossary.store import GlossaryOccurrenceMatcher, GlossaryStore, GlossaryTerm
from wenyi_core.llm.providers.fake import FakeClient

from tests.pipeline_fixtures import fake_agent_config


def _reason(source: str, type_: str = "term", batch: tuple[str, ...] = ()) -> str | None:
    return admit_candidate(GlossaryTerm(source=source, target="t", type=type_), batch)


@pytest.mark.parametrize(
    ("source", "type_", "expected"),
    [
        ("I never asked to be anyone's token.", "fixed_expression", "sentence_punctuation"),
        ("“Go home!”", "fixed_expression", "sentence_punctuation"),
        ("Why me?", "fixed_expression", "sentence_punctuation"),
        ("Wait…", "fixed_expression", "sentence_punctuation"),
        ("お前はもう死んでいる。", "fixed_expression", "sentence_punctuation"),
        ("PART ONE", "term", "all_caps_heading"),
        ("THE LAST WORD", "fixed_expression", "all_caps_heading"),
        ("NASA", "organization", None),
        ("James Tiptree Jr.", "person", None),
        ("Washington, D.C.", "place", None),
        ("Dr.", "appellation", None),
        ("綾小路清隆", "person", None),
        ("東京都", "place", None),
        ("お兄ちゃん", "appellation", None),
        ("diversity hire", "term", None),
        ("Yellowface", "term", None),
        ("one two three four five", "fixed_expression", None),
        ("one two three four five six", "fixed_expression", "too_long"),
        ("The Royal Society of London for Natural Knowledge", "organization", None),
        ("a b c d e f g h i", "organization", "too_long"),
        ("あ" * MAX_CJK_CHARS, "fixed_expression", None),
        ("あ" * (MAX_CJK_CHARS + 1), "fixed_expression", "too_long"),
        ("tired line", "statement", "unknown_type"),
        ("Athena", "other", "unknown_type"),
        ("", "term", "empty"),
        ("two\nlines", "term", "newline"),
        ("bad\x07bell", "term", "control_character"),
    ],
)
def test_admission_rules(source, type_, expected):
    assert _reason(source, type_) == expected


def test_whole_paragraph_rule_spares_single_word_names_only():
    batch = ("Athena", "Go on then", "編集者")
    assert _reason("Athena", "person", batch) is None
    assert _reason("Go on then", "fixed_expression", batch) == "whole_paragraph"
    assert _reason("go  on THEN", "term", batch) == "whole_paragraph"
    assert _reason("編集者", "term", batch) == "whole_paragraph"
    assert _reason("Go on", "fixed_expression", batch) is None


def test_frequency_gate_uses_shared_matcher_with_curly_quotes():
    phrase = GlossaryTerm(source="don't panic", target="别慌", type="fixed_expression")
    person = GlossaryTerm(source="Athena", target="雅典娜", type="person")
    once = GlossaryOccurrenceMatcher("Athena said: Don’t panic.")
    twice = GlossaryOccurrenceMatcher("Don’t panic. Later, DON'T PANIC again.")
    assert admission_status(phrase, once) == "candidate"
    assert admission_status(phrase, twice) == "ok"
    assert admission_status(person, once) == "ok"
    # Word boundaries: a longer word does not count as an occurrence.
    word = GlossaryTerm(source="token", target="象征", type="term")
    assert admission_status(word, GlossaryOccurrenceMatcher("token tokens tokenism")) == (
        "candidate"
    )


def _extractor(terms: list[dict]) -> GlossaryExtractor:
    client = FakeClient(handler=lambda *a: json.dumps({"terms": terms}, ensure_ascii=False))
    return GlossaryExtractor(client, fake_agent_config())


def test_extract_and_store_rejects_counts_and_gates_by_book_frequency(tmp_path):
    proposals = [
        {"source": "Athena", "target": "雅典娜", "type": "person"},
        {"source": "diversity hire", "target": "多元化招聘", "type": "term"},
        {"source": "She smiled at me.", "target": "她冲我笑。", "type": "fixed_expression"},
        {"source": "PART ONE", "target": "第一部", "type": "term"},
        {"source": "vibe", "target": "氛围", "type": "statement"},
    ]
    store = GlossaryStore(str(tmp_path / "g.db"))
    try:
        summary = _extractor(proposals).extract_and_store(
            store,
            "Athena was a diversity hire.\nShe smiled at me.",
            "译文",
            chapter=0,
            source_corpus="Athena was a diversity hire.\nShe smiled at me.",
        )
        assert summary["inserted"] == 2
        assert summary["rejected"] == 3
        assert summary["rejected_sentence_punctuation"] == 1
        assert summary["rejected_all_caps_heading"] == 1
        assert summary["rejected_unknown_type"] == 1
        assert summary["candidates"] == 1
        assert store.get_term("Athena").status == "ok"
        assert store.get_term("diversity hire").status == "candidate"
        assert [t.source for t in active_terms(store.all_terms())] == ["Athena"]

        # A second book occurrence (e.g. a later chapter) promotes the stored candidate.
        later = _extractor([proposals[1]])
        corpus = "Athena was a diversity hire.\nAnother diversity hire arrived."
        summary = later.extract_and_store(
            store, "Another diversity hire arrived.", "译文", chapter=1, source_corpus=corpus
        )
        assert summary["unchanged"] == 1 and summary["candidates"] == 0
        assert store.get_term("diversity hire").status == "ok"
    finally:
        store.close()


def test_candidate_stays_inactive_through_conflicts_and_is_never_deleted(tmp_path):
    store = GlossaryStore(str(tmp_path / "g.db"))
    try:
        store.upsert_term(GlossaryTerm(source="Go on", target="继续", status="candidate"))
        assert store.upsert_term(GlossaryTerm(source="Go on", target="走吧")) == "conflict"
        assert store.get_term("Go on").status == "candidate"
        assert store.upsert_term(
            GlossaryTerm(source="Go on", target="继续", status="candidate")
        ) == ("unchanged")
        assert store.get_term("Go on").status == "candidate"
        # Established entries are never demoted by a later single-occurrence proposal.
        store.upsert_term(GlossaryTerm(source="Eden", target="伊甸"))
        store.upsert_term(GlossaryTerm(source="Eden", target="伊甸", status="candidate"))
        assert store.get_term("Eden").status == "ok"
        assert len(store.all_terms()) == 2
    finally:
        store.close()


def test_legacy_database_rows_stay_active_without_migration(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """CREATE TABLE glossary (source TEXT PRIMARY KEY, target TEXT NOT NULL, reading TEXT,
        type TEXT, gender TEXT, aliases TEXT, first_chapter INTEGER, note TEXT,
        status TEXT DEFAULT 'ok', updated_at REAL);
        INSERT INTO glossary (source, target, type, status) VALUES ('Old', '旧', 'term', 'ok');
        INSERT INTO glossary (source, target, type, status) VALUES ('Clash', '冲', 'term',
        'conflict');
        INSERT INTO glossary (source, target, type) VALUES ('Bare', '裸', 'term');"""
    )
    conn.commit()
    conn.close()
    store = GlossaryStore(str(path))
    try:
        assert [t.source for t in active_terms(store.all_terms())] == ["Old", "Clash", "Bare"]
        store.upsert_term(GlossaryTerm(source="New", target="新", status="candidate"))
        assert [t.source for t in active_terms(store.all_terms())] == ["Old", "Clash", "Bare"]
    finally:
        store.close()


def test_title_requests_carry_only_cited_active_terms(tmp_path):
    from contextlib import closing

    from wenyi_core.config import Config
    from wenyi_core.models.document import Chapter, Document
    from wenyi_core.pipeline.orchestrator import Orchestrator
    from wenyi_core.storage.file import FileStorage

    config = Config.from_dict(
        {"llm": {"preset": "fake"}, "language": {"source": "en", "target": "zh"}}
    )
    store = FileStorage(str(tmp_path / "state"))
    manifest = store.stage_document(
        Document(
            title="Book",
            source_lang="en",
            target_lang="zh",
            fmt="txt",
            chapters=[Chapter(index=0, title="Athena's Desk"), Chapter(index=1, title="Go on")],
        ),
        source_hash="a" * 64,
    )
    store.save_manifest(manifest)
    with closing(GlossaryStore(store.glossary_path)) as glossary:
        glossary.upsert_term(GlossaryTerm(source="Athena", target="雅典娜", type="person"))
        glossary.upsert_term(GlossaryTerm(source="Unrelated", target="无关", type="term"))
        glossary.upsert_term(
            GlossaryTerm(source="Go on", target="继续", type="fixed_expression", status="candidate")
        )
        client = FakeClient(handler=lambda *a: json.dumps({"titles": ["一", "二"]}))
        Orchestrator(config, client)._translation._titles.run(store, glossary)
    user = client.calls[0]["messages"][-1]["content"]
    assert "Athena → 雅典娜" in user
    assert "Unrelated" not in user
    assert "继续" not in user


def test_translate_chapter_excludes_candidates_from_batch_plans(tmp_path):
    from unittest.mock import patch

    from wenyi_core.models.document import Chapter, Segment
    from wenyi_core.pipeline.orchestrator import Orchestrator
    from wenyi_core.pipeline.translate.translation_batch import BatchResult
    from wenyi_core.storage.file import FileStorage

    from tests.pipeline_fixtures import fake_agent_config as config

    store = FileStorage(str(tmp_path / "state"))
    store.save_chapter(
        Chapter(index=0, title="", segments=[Segment(index=0, source="Athena said go on.")])
    )
    store.save_manifest(
        {
            "title": "Book",
            "fmt": "txt",
            "source_lang": "en",
            "target_lang": "zh",
            "chapters": [{"index": 0, "title": "", "status": "pending"}],
        }
    )
    glossary = GlossaryStore(store.glossary_path)
    glossary.upsert_term(GlossaryTerm(source="Athena", target="雅典娜", type="person"))
    glossary.upsert_term(GlossaryTerm(source="Eden", target="伊甸", type="place"))
    glossary.upsert_term(
        GlossaryTerm(source="go on", target="继续", type="fixed_expression", status="candidate")
    )
    orch = Orchestrator(config(), client=FakeClient())
    plans = []

    def execute(plan, **kwargs):
        plans.append(plan)
        return BatchResult(("译",), (None,))

    try:
        with (
            patch.object(orch._translation._batches, "execute", side_effect=execute),
            patch.object(orch._runtime.extractor, "extract_and_store", return_value={}),
        ):
            orch._translation.translate_chapter(
                0,
                store,
                glossary,
                "",
                translation_history={},
                source_corpus="",
                annotation_context_registry=None,
            )
    finally:
        glossary.close()
    assert [term.source for term in plans[0].terms] == ["Athena"]


def test_new_candidate_does_not_change_pending_batch_identity():
    """Resume: a stored candidate never enters BatchPlan, so precision fingerprints hold."""
    from dataclasses import asdict

    from wenyi_core.models.document import Segment
    from wenyi_core.pipeline.translate.translation_batch import BatchPlan

    segments = [Segment(index=0, source="Athena said go on.")]
    base = [GlossaryTerm(source="Athena", target="雅典娜", type="person")]
    candidate = GlossaryTerm(
        source="go on", target="继续", type="fixed_expression", status="candidate"
    )

    def plan(terms):
        return BatchPlan.capture(0, 0, segments, active_terms(terms), "", "", "", "", [[]], "")

    assert asdict(plan(base)) == asdict(plan([*base, candidate]))
    promoted = GlossaryTerm(source="go on", target="继续", type="fixed_expression")
    assert [t.source for t in plan([*base, promoted]).terms] == ["Athena", "go on"]


def test_pretranslation_person_discovery_rejects_and_logs_reasons(tmp_path, monkeypatch):
    from wenyi_core.glossary.evidence_models import DiscoveryResult
    from wenyi_core.pipeline.translate.terminology import TerminologyService

    from tests.terminology_fixtures import EvidenceAgent, book_store

    store, chapters = book_store(tmp_path)
    agent = EvidenceAgent()
    monkeypatch.setattr(
        agent,
        "discover",
        lambda *args: DiscoveryResult(
            terms=[
                GlossaryTerm("Alice", "艾丽丝", type="person"),
                GlossaryTerm("Alice waited.", "艾丽丝等着。", type="person"),
                GlossaryTerm("ALICE AND CARL", "艾丽丝与卡尔", type="person"),
            ]
        ),
    )
    events = []
    monkeypatch.setattr(store, "log_event", lambda name, **data: events.append((name, data)))
    extractor = GlossaryExtractor(FakeClient(), agent.config)
    TerminologyService(store, agent, chapters).prepare_batch(
        0, chapters[0].segments, extractor=extractor, before=(0, 0)
    )
    assert store.get_term("Alice") is not None
    assert store.get_term("Alice waited.") is None
    assert "terminology_candidates_rejected" in [name for name, _ in events]
    data = dict(events)["terminology_candidates_rejected"]
    assert data == {
        "chapter": 0,
        "rejected": 2,
        "reasons": {"all_caps_heading": 1, "sentence_punctuation": 1},
    }
