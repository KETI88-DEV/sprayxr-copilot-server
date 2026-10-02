from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from deploy.spark.scripts.import_legacy_copilot import apply_import, plan_import


def _legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
            skill_level TEXT, age INTEGER, gender TEXT, created_at TEXT
        );
        CREATE TABLE training_sessions (
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, user_id TEXT NOT NULL,
            scenario_id TEXT NOT NULL, status TEXT, started_at TEXT, ended_at TEXT
        );
        CREATE TABLE sequence_data (
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, user_id TEXT NOT NULL,
            scenario_id TEXT NOT NULL, zone_id TEXT NOT NULL, sequence_id TEXT NOT NULL,
            sequence_setting TEXT, sequence_info TEXT, sequence_result TEXT, created_at TEXT
        );
        CREATE TABLE session_completions (
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, user_id TEXT NOT NULL,
            scenario_id TEXT NOT NULL, report_id TEXT NOT NULL, request_id TEXT NOT NULL,
            status TEXT NOT NULL, heatmap_summary TEXT, error_message TEXT,
            created_at TEXT, completed_at TEXT
        );
        CREATE TABLE api_logs (
            id INTEGER PRIMARY KEY, event_id TEXT NOT NULL, endpoint TEXT NOT NULL,
            method TEXT NOT NULL, client_ip TEXT, status_code INTEGER NOT NULL,
            request_payload TEXT, response_payload TEXT, latency_ms REAL NOT NULL,
            created_at TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "U-CONFLICT", "legacy-name", "expert", 40, "M", "2026-01-01T00:00:00Z"),
            (2, "U-NEW", "new-name", "beginner", 30, "F", "2026-01-02T00:00:00Z"),
        ],
    )
    conn.executemany(
        "INSERT INTO training_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "S-CONFLICT", "U-CONFLICT", "1", "COMPLETED", "start", "end"),
            (2, "S-NEW", "U-NEW", "3", "COMPLETED", "start2", "end2"),
        ],
    )
    frames = json.dumps([{"index": 1}, {"index": 2}])
    conn.executemany(
        "INSERT INTO sequence_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "S-CONFLICT", "U-CONFLICT", "1", "1", "Q-CONFLICT", "{}", frames, "{}", "t1"),
            (2, "S-NEW", "U-NEW", "3", "2", "Q-NEW", "{}", frames, "{}", "t2"),
        ],
    )
    conn.execute(
        "INSERT INTO session_completions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "S-NEW", "U-NEW", "3", "R-NEW", "REQ-1", "completed", "{}", None, "t3", "t4"),
    )
    conn.execute(
        "INSERT INTO api_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "E-1", "/api", "POST", "127.0.0.1", 200, "{}", "{}", 1.2, "t5"),
    )
    conn.commit()
    conn.close()


def _target_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY, name TEXT, skill_level TEXT NOT NULL,
            dominant_hand TEXT NOT NULL, age INTEGER, gender TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, scenario_id TEXT NOT NULL,
            session_status TEXT NOT NULL, started_at TEXT, ended_at TEXT
        );
        CREATE TABLE sequences (
            session_key TEXT NOT NULL, sequence_id TEXT NOT NULL, body_sha256 TEXT NOT NULL,
            frames INTEGER NOT NULL, received_at TEXT NOT NULL, body TEXT,
            PRIMARY KEY (session_key, sequence_id)
        );
        CREATE TABLE completions (
            report_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            report_status TEXT NOT NULL, requested_at TEXT NOT NULL, payload TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("U-CONFLICT", "dgx-name", "advanced", "right", None, None, "existing"),
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
        ("S-CONFLICT", "U-CONFLICT", "DGX", "ACTIVE", "existing", None),
    )
    conn.execute(
        "INSERT INTO sequences VALUES (?, ?, ?, ?, ?, ?)",
        ("S-CONFLICT", "Q-CONFLICT", "existing-hash", 9, "existing", '{"existing":true}'),
    )
    conn.commit()
    conn.close()


def test_import_preserves_target_conflicts_and_archives_every_legacy_row(tmp_path: Path) -> None:
    source = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    _legacy_db(source)
    _target_db(target)

    plan = plan_import(source, target)
    assert plan.source_counts == {
        "api_logs": 1,
        "sequence_data": 2,
        "session_completions": 1,
        "training_sessions": 2,
        "users": 2,
    }
    assert plan.conflicts == {"sequences": 1, "sessions": 1, "users": 1}

    report = apply_import(plan)

    assert report.inserted == {"completions": 1, "sequences": 1, "sessions": 1, "users": 1}
    assert report.conflicts == {"sequences": 1, "sessions": 1, "users": 1}
    assert report.archived == 8
    assert report.backup_path.exists()

    conn = sqlite3.connect(target)
    assert conn.execute("SELECT name FROM users WHERE user_id='U-CONFLICT'").fetchone() == ("dgx-name",)
    assert conn.execute("SELECT name, dominant_hand FROM users WHERE user_id='U-NEW'").fetchone() == (
        "new-name",
        "unknown",
    )
    assert conn.execute("SELECT scenario_id FROM sessions WHERE session_id='S-CONFLICT'").fetchone() == ("DGX",)
    assert conn.execute("SELECT body_sha256 FROM sequences WHERE session_key='S-CONFLICT'").fetchone() == (
        "existing-hash",
    )
    assert conn.execute("SELECT COUNT(*) FROM legacy_import_records").fetchone() == (8,)
    assert conn.execute(
        "SELECT source_table, source_pk FROM legacy_import_records ORDER BY source_table, source_pk"
    ).fetchall()[0] == ("api_logs", "1")
    conn.close()


def test_reimport_is_idempotent_and_does_not_duplicate_live_or_archive_rows(tmp_path: Path) -> None:
    source = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    _legacy_db(source)
    _target_db(target)

    first = apply_import(plan_import(source, target))
    second = apply_import(plan_import(source, target))

    assert first.archived == 8
    assert second.archived == 0
    assert second.inserted == {"completions": 0, "sequences": 0, "sessions": 0, "users": 0}
    conn = sqlite3.connect(target)
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone() == (2,)
    assert conn.execute("SELECT COUNT(*) FROM legacy_import_records").fetchone() == (8,)
    conn.close()


def test_import_can_write_the_automatic_backup_to_an_operations_directory(tmp_path: Path) -> None:
    source = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    backup_dir = tmp_path / "operations-backups"
    _legacy_db(source)
    _target_db(target)

    report = apply_import(plan_import(source, target, backup_dir=backup_dir))

    assert report.backup_path.parent == backup_dir
    assert report.backup_path.exists()
