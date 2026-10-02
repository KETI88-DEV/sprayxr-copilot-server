"""Import the legacy Co-Pilot SQLite database without overwriting DGX rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


LEGACY_TABLES = (
    "api_logs",
    "sequence_data",
    "session_completions",
    "training_sessions",
    "users",
)


@dataclass(frozen=True)
class ImportPlan:
    source_db: Path
    target_db: Path
    source_sha256: str
    source_counts: dict[str, int]
    conflicts: dict[str, int]
    backup_dir: Path


@dataclass(frozen=True)
class ImportReport:
    inserted: dict[str, int]
    conflicts: dict[str, int]
    archived: int
    backup_path: Path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _count_key_conflicts(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    *,
    source_table: str,
    source_columns: tuple[str, ...],
    target_table: str,
    target_columns: tuple[str, ...],
) -> int:
    target_keys = {
        tuple(row)
        for row in target.execute(
            f"SELECT {', '.join(target_columns)} FROM {target_table}"
        ).fetchall()
    }
    return sum(
        tuple(row[column] for column in source_columns) in target_keys
        for row in source.execute(f"SELECT {', '.join(source_columns)} FROM {source_table}")
    )


def plan_import(source_db: Path, target_db: Path, *, backup_dir: Path | None = None) -> ImportPlan:
    source = _connect_readonly(source_db)
    target = _connect_readonly(target_db)
    try:
        counts = {
            table: int(source.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in LEGACY_TABLES
        }
        conflicts = {
            "sequences": _count_key_conflicts(
                source,
                target,
                source_table="sequence_data",
                source_columns=("session_id", "sequence_id"),
                target_table="sequences",
                target_columns=("session_key", "sequence_id"),
            ),
            "sessions": _count_key_conflicts(
                source,
                target,
                source_table="training_sessions",
                source_columns=("session_id",),
                target_table="sessions",
                target_columns=("session_id",),
            ),
            "users": _count_key_conflicts(
                source,
                target,
                source_table="users",
                source_columns=("user_id",),
                target_table="users",
                target_columns=("user_id",),
            ),
        }
        completion_conflicts = _count_key_conflicts(
            source,
            target,
            source_table="session_completions",
            source_columns=("report_id",),
            target_table="completions",
            target_columns=("report_id",),
        )
        if completion_conflicts:
            conflicts["completions"] = completion_conflicts
    finally:
        source.close()
        target.close()
    return ImportPlan(
        source_db=source_db.resolve(),
        target_db=target_db.resolve(),
        source_sha256=_file_sha256(source_db),
        source_counts=counts,
        conflicts=conflicts,
        backup_dir=(backup_dir or target_db.parent / "backups").resolve(),
    )


def _json_value(raw: Any, fallback: Any) -> Any:
    if raw is None:
        return fallback
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {"legacy_raw": str(raw)}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _backup_database(target_db: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_dir / f"{target_db.stem}-{stamp}{target_db.suffix}.bak"
    source = sqlite3.connect(target_db)
    destination = sqlite3.connect(backup)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return backup


def _archive_rows(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    source_sha256: str,
) -> tuple[int, dict[str, set[str]]]:
    target.execute(
        """
        CREATE TABLE IF NOT EXISTS legacy_import_records (
            source_sha256 TEXT NOT NULL,
            source_table  TEXT NOT NULL,
            source_pk     TEXT NOT NULL,
            row_json      TEXT NOT NULL,
            imported_at   TEXT NOT NULL,
            PRIMARY KEY (source_sha256, source_table, source_pk)
        )
        """
    )
    imported_at = datetime.now(timezone.utc).isoformat()
    archived = 0
    new_keys: dict[str, set[str]] = {table: set() for table in LEGACY_TABLES}
    for table in LEGACY_TABLES:
        for row in source.execute(f"SELECT * FROM {table} ORDER BY id"):
            source_pk = str(row["id"])
            payload = _canonical_json(dict(row))
            cursor = target.execute(
                "INSERT OR IGNORE INTO legacy_import_records"
                " (source_sha256, source_table, source_pk, row_json, imported_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (source_sha256, table, source_pk, payload, imported_at),
            )
            if cursor.rowcount:
                archived += 1
                new_keys[table].add(source_pk)
    return archived, new_keys


def apply_import(plan: ImportPlan) -> ImportReport:
    if _file_sha256(plan.source_db) != plan.source_sha256:
        raise ValueError("source database changed after import planning")
    backup_path = _backup_database(plan.target_db, plan.backup_dir)
    source = _connect_readonly(plan.source_db)
    target = sqlite3.connect(plan.target_db)
    target.row_factory = sqlite3.Row
    inserted = {"completions": 0, "sequences": 0, "sessions": 0, "users": 0}
    try:
        with target:
            archived, new_keys = _archive_rows(source, target, plan.source_sha256)

            for row in source.execute("SELECT * FROM users ORDER BY id"):
                if str(row["id"]) not in new_keys["users"]:
                    continue
                cursor = target.execute(
                    "INSERT OR IGNORE INTO users"
                    " (user_id, name, skill_level, dominant_hand, age, gender, created_at)"
                    " VALUES (?, ?, ?, 'unknown', ?, ?, ?)",
                    (
                        row["user_id"],
                        row["name"],
                        row["skill_level"] or "unknown",
                        row["age"],
                        row["gender"],
                        row["created_at"] or "legacy-unknown",
                    ),
                )
                inserted["users"] += int(bool(cursor.rowcount))

            for row in source.execute("SELECT * FROM training_sessions ORDER BY id"):
                if str(row["id"]) not in new_keys["training_sessions"]:
                    continue
                cursor = target.execute(
                    "INSERT OR IGNORE INTO sessions"
                    " (session_id, user_id, scenario_id, session_status, started_at, ended_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        row["session_id"],
                        row["user_id"],
                        row["scenario_id"],
                        row["status"] or "UNKNOWN",
                        row["started_at"],
                        row["ended_at"],
                    ),
                )
                inserted["sessions"] += int(bool(cursor.rowcount))

            for row in source.execute("SELECT * FROM sequence_data ORDER BY id"):
                if str(row["id"]) not in new_keys["sequence_data"]:
                    continue
                body = {
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "scenario_id": row["scenario_id"],
                    "zone_id": row["zone_id"],
                    "sequence_id": row["sequence_id"],
                    "sequence_setting": _json_value(row["sequence_setting"], {}),
                    "sequence_info": _json_value(row["sequence_info"], []),
                    "sequence_result": _json_value(row["sequence_result"], {}),
                }
                body_json = _canonical_json(body)
                cursor = target.execute(
                    "INSERT OR IGNORE INTO sequences"
                    " (session_key, sequence_id, body_sha256, frames, received_at, body)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        row["session_id"],
                        row["sequence_id"],
                        hashlib.sha256(body_json.encode("utf-8")).hexdigest(),
                        len(body["sequence_info"]) if isinstance(body["sequence_info"], list) else 0,
                        row["created_at"] or "legacy-unknown",
                        body_json,
                    ),
                )
                inserted["sequences"] += int(bool(cursor.rowcount))

            for row in source.execute("SELECT * FROM session_completions ORDER BY id"):
                if str(row["id"]) not in new_keys["session_completions"]:
                    continue
                payload = _canonical_json(
                    {
                        "legacy_request_id": row["request_id"],
                        "user_id": row["user_id"],
                        "scenario_id": row["scenario_id"],
                        "heatmap_summary": _json_value(row["heatmap_summary"], None),
                        "error_message": row["error_message"],
                        "completed_at": row["completed_at"],
                    }
                )
                cursor = target.execute(
                    "INSERT OR IGNORE INTO completions"
                    " (report_id, session_id, report_status, requested_at, payload)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        row["report_id"],
                        row["session_id"],
                        row["status"],
                        row["created_at"] or "legacy-unknown",
                        payload,
                    ),
                )
                inserted["completions"] += int(bool(cursor.rowcount))
    except Exception:
        target.close()
        source.close()
        shutil.copy2(backup_path, plan.target_db)
        raise
    target.close()
    source.close()
    return ImportReport(
        inserted=inserted,
        conflicts=dict(plan.conflicts),
        archived=archived,
        backup_path=backup_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_db", type=Path)
    parser.add_argument("target_db", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_import(args.source_db, args.target_db, backup_dir=args.backup_dir)
    payload: dict[str, Any] = asdict(plan)
    if args.apply:
        payload["report"] = asdict(apply_import(plan))
    print(json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
