"""Re-normalize glossary aliases written before normalize_aliases existed.

Works on both local glossary stores: the CLI run's glossary.db and a Desktop
project's state.sqlite3 share the same `glossary` table. Dry-run by default;
`--apply` takes the run lock, backs the database up next to itself, and rewrites
only the aliases column in one transaction. Identity conflicts (an alias that is
another term's source) are reported, never resolved.

    uv run python scripts/clean_glossary_aliases.py PATH/TO/state.sqlite3
    uv run python scripts/clean_glossary_aliases.py PATH/TO/state.sqlite3 --apply
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

from wenyi_core.glossary.store import normalize_aliases
from wenyi_core.storage.locks import RUN_LOCK, StorageBusyError, exclusive_file_lock


def _key(value: str) -> str:
    # Same folding as normalize_aliases uses for alias identity.
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def plan(conn: sqlite3.Connection) -> tuple[list[tuple[str, list[str], list[str]]], list[tuple]]:
    """Return (changes, conflicts); changes are (source, old, new) in source order."""
    rows = conn.execute("SELECT source, aliases FROM glossary ORDER BY source").fetchall()
    sources = {_key(source): source for source, _ in rows}
    changes = []
    conflicts = []
    for source, raw in rows:
        old = json.loads(raw or "[]")
        new = normalize_aliases(source, old)
        if new != old:
            changes.append((source, old, new))
        for alias in new:
            other = sources.get(_key(alias))
            if other is not None and other != source:
                conflicts.append((source, alias, other))
    return changes, conflicts


def _short(value: str, width: int = 60) -> str:
    # Some extracted "terms" are whole sentences; keep the report scannable.
    return value if len(value) <= width else value[: width - 1] + "…"


def _report(changes, conflicts, limit: int) -> None:
    print(f"terms with changed aliases: {len(changes)}")
    for source, old, new in changes[:limit]:
        print(f"  {_short(source)}\n    old: {json.dumps(old, ensure_ascii=False)}")
        print(f"    new: {json.dumps(new, ensure_ascii=False)}")
    if len(changes) > limit:
        print(f"  ... {len(changes) - limit} more")
    print(f"suspected identity conflicts (alias is another term's source): {len(conflicts)}")
    for source, alias, other in conflicts:
        print(f"  {_short(source)!r} alias {alias!r} == term {_short(other)!r}")


def apply(path: Path, limit: int) -> int:
    try:
        # Every local writer (CLI and Desktop runs) holds this lock beside its database.
        with exclusive_file_lock(path.parent / RUN_LOCK, blocking=False):
            conn = sqlite3.connect(path, timeout=0, isolation_level=None)
            try:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as error:
                    print(f"database is busy ({error}); stop running tasks or close Desktop")
                    return 1
                changes, conflicts = plan(conn)
                _report(changes, conflicts, limit)
                if not changes:
                    conn.execute("ROLLBACK")
                    print("nothing to write")
                    return 0
                backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
                # A backup through the writing connection stalls inside its own
                # transaction; a second reader copies the same committed state while
                # BEGIN IMMEDIATE keeps every other writer out.
                reader = sqlite3.connect(path)
                target = sqlite3.connect(backup)
                try:
                    reader.backup(target)
                finally:
                    target.close()
                    reader.close()
                conn.executemany(
                    "UPDATE glossary SET aliases=? WHERE source=?",
                    [(json.dumps(new, ensure_ascii=False), source) for source, _, new in changes],
                )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
    except StorageBusyError:
        print(f"{path.parent / RUN_LOCK} is held; stop running tasks or close Desktop")
        return 1
    print(f"updated {len(changes)} terms; backup: {backup}")
    print(f"restore: close Wenyi, then copy the backup over {path} and delete its -wal/-shm files")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("database", type=Path, help="glossary.db or Desktop state.sqlite3")
    parser.add_argument("--apply", action="store_true", help="back up, then write changes")
    parser.add_argument("--limit", type=int, default=20, help="examples to print")
    args = parser.parse_args(argv)
    if not args.database.is_file():
        parser.error(f"not a file: {args.database}")
    if args.apply:
        return apply(args.database, args.limit)
    uri = f"{args.database.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        changes, conflicts = plan(conn)
    finally:
        conn.close()
    _report(changes, conflicts, args.limit)
    print("dry-run: nothing written; rerun with --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
