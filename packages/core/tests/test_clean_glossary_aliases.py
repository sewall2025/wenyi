from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest
from wenyi_core.glossary.store import GlossaryStore
from wenyi_core.storage.locks import RUN_LOCK, exclusive_file_lock

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location(
        "clean_glossary_aliases", ROOT / "scripts/clean_glossary_aliases.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "glossary.db"
    GlossaryStore(str(path)).close()
    rows = [
        ("Athena", "雅典娜", ["Athena, Liu, Athena Liu", "Liu", "athena"]),
        ("Liu", "刘", ["Liu"]),
        ("June", "朱恩", ["Juniper"]),
    ]
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO glossary(source, target, aliases, note) VALUES (?, ?, ?, 'keep')",
            [(s, t, json.dumps(a)) for s, t, a in rows],
        )
    conn.close()
    return path


def _rows(path):
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT * FROM glossary ORDER BY source").fetchall()
    conn.close()
    return rows


def test_dry_run_reports_without_writing(script, database, capsys):
    before = _rows(database)
    assert script.main([str(database)]) == 0
    out = capsys.readouterr().out
    assert "terms with changed aliases: 2" in out
    assert "alias 'Liu' == term 'Liu'" in out
    assert _rows(database) == before
    assert not list(database.parent.glob("*.bak-*"))


def test_apply_backs_up_and_rewrites_only_aliases(script, database, capsys):
    before = _rows(database)
    assert script.main([str(database), "--apply"]) == 0
    (backup,) = database.parent.glob("glossary.db.bak-*")
    assert _rows(backup) == before
    after = {row[0]: row for row in _rows(database)}
    assert json.loads(after["Athena"][5]) == ["Liu", "Athena Liu"]
    assert json.loads(after["Liu"][5]) == []
    assert after["June"] == next(row for row in before if row[0] == "June")
    assert all(row[:5] + row[6:] == after[row[0]][:5] + after[row[0]][6:] for row in before)
    assert str(backup) in capsys.readouterr().out


def test_apply_refuses_while_run_lock_held(script, database, capsys):
    before = _rows(database)
    with exclusive_file_lock(database.parent / RUN_LOCK):
        assert script.main([str(database), "--apply"]) == 1
    assert "close Desktop" in capsys.readouterr().out
    assert _rows(database) == before
    assert not list(database.parent.glob("*.bak-*"))
