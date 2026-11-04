"""Allowlisted management writes with previews, conflict checks, backups and audit."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.demo_api.db_admin_auth import DBAdminAuth, fail
from trainer.demo_api.report_jobs import hold_idle_reports

TABLES = {"users": "훈련자", "sessions": "훈련 세션", "sequences": "시퀀스 기록",
          "query_turns": "질문·답변", "completions": "최종 리포트", "api_calls": "API 호출 이력"}
FIELDS = {
    "users": {
        "name": {"label": "훈련자 이름", "type": "text", "nullable": True, "max_length": 100},
        "skill_level": {"label": "숙련도", "choices": {"novice": "일반 훈련자", "expert": "전문가"}},
        "dominant_hand": {"label": "주 사용 손", "choices": {"right": "오른손", "left": "왼손", "unknown": "미기록"}},
        "age": {"label": "나이", "type": "number", "nullable": True, "min": 0, "max": 120},
        "gender": {"label": "성별 코드", "type": "text", "nullable": True, "max_length": 30},
    },
    "sessions": {"creation_origin": {"label": "데이터 출처", "choices": {"unknown": "출처 미기록", "real": "일반 API 연동", "test": "모의 테스트"}}},
}
DISPLAY = ("session_id", "user_id", "name", "scenario_id", "creation_origin", "session_status", "sequence_id",
           "report_id", "report_status", "skill_level", "dominant_hand", "age", "gender", "prompt", "path", "created_at", "received_at")
RELATIONS = (("sequence_analysis_results", "session_key"), ("sequences", "session_key"),
             ("query_turns", "session_key"), ("completions", "session_id"), ("api_calls", "session_id"))


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class DBAdminStore:
    def __init__(self, path: Path, auth: DBAdminAuth) -> None:
        self.path, self.auth = path.resolve(), auth

    @contextmanager
    def connection(self, *, write: bool = False):
        if not self.path.is_file():
            fail(503, "관리할 데이터베이스가 아직 없습니다.")
        conn = sqlite3.connect(f"file:{self.path.resolve()}?mode={'rw' if write else 'ro'}", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if write:
                conn.commit()
        except sqlite3.OperationalError:
            fail(409, "데이터베이스를 사용 중입니다. 잠시 후 다시 시도하세요.")
        finally:
            conn.close()

    @staticmethod
    def columns(conn, table):
        return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}

    def checked_table(self, conn, table):
        if table not in TABLES or not self.columns(conn, table):
            fail(422, "관리할 수 없는 테이블입니다.")

    def row(self, conn, table, rowid):
        self.checked_table(conn, table)
        row = conn.execute(f'SELECT rowid AS __rowid,* FROM "{table}" WHERE rowid=?', (rowid,)).fetchone()
        if row is None:
            fail(404, "해당 기록을 찾을 수 없습니다.")
        return dict(row)

    @staticmethod
    def values(row):
        return {key: row[key] for key in DISPLAY if key in row}

    def records(self, table: str, q: str, page: int) -> dict:
        with self.connection() as conn:
            self.checked_table(conn, table)
            columns = [name for name in DISPLAY if name in self.columns(conn, table)]
            where = "1"
            params = []
            if q:
                where = "(" + " OR ".join(f'CAST("{name}" AS TEXT) LIKE ? ESCAPE \'\\\'' for name in columns) + ")"
                needle = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                params = [needle] * len(columns)
            total = conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {where}', params).fetchone()[0]
            page = min(page, max(1, (total + 24) // 25))
            selected = ",".join('"' + name + '"' for name in columns)
            rows = conn.execute(f'SELECT rowid AS __rowid,{selected} FROM "{table}" WHERE {where} ORDER BY rowid DESC LIMIT 25 OFFSET ?', [*params, (page - 1) * 25])
            return {"table": table, "label": TABLES[table], "columns": columns, "total": total, "page": page,
                    "rows": [{"rowid": row["__rowid"], "values": self.values(dict(row))} for row in rows],
                    "tables": {name: label for name, label in TABLES.items() if self.columns(conn, name)}}

    def record(self, table, rowid):
        with self.connection() as conn:
            row = self.row(conn, table, rowid)
            fields = {key: value for key, value in FIELDS.get(table, {}).items() if key in row}
            values = {key: (row[key] or "unknown") if key == "creation_origin" else row[key] for key in fields}
            return {"table": table, "rowid": rowid, "version": digest(row), "values": values,
                    "fields": fields, "summary": self.values(row)}

    def backup(self, conn) -> str:
        directory = self.path.parent / "admin_backups"
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12] + ".sqlite3"
        target = directory / name
        # BEGIN IMMEDIATE holds other writers; a separate reader can still copy the
        # last committed WAL snapshot without backing up a connection in a write txn.
        try:
            directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            with sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True) as source, sqlite3.connect(target) as dest:
                source.backup(dest)
            return name
        except Exception:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            fail(503, "백업을 만들지 못해 변경을 중단했습니다.")

    @staticmethod
    def audit(conn, actor, action, table, rowid, details, backup):
        conn.execute("""CREATE TABLE IF NOT EXISTS admin_change_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,actor TEXT NOT NULL,
            action TEXT NOT NULL,table_name TEXT NOT NULL,record_rowid INTEGER NOT NULL,
            details TEXT NOT NULL,backup_name TEXT NOT NULL)""")
        conn.execute("INSERT INTO admin_change_log(created_at,actor,action,table_name,record_rowid,details,backup_name) VALUES(?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), actor, action, table, rowid, json.dumps(details, ensure_ascii=False), backup))

    def update(self, table, rowid, version, fields, actor):
        specs = FIELDS.get(table, {})
        if not fields or not specs or set(fields) - set(specs):
            fail(422, "이 항목은 수정할 수 없습니다. 식별자와 측정 원문은 보존합니다.")
        changes = dict(fields)
        for key, value in changes.items():
            spec = specs[key]
            if value is None and spec.get("nullable"):
                continue
            if "choices" in spec:
                if not isinstance(value, str) or value not in spec["choices"]:
                    fail(422, f'{spec["label"]} 값이 올바르지 않습니다.')
            elif spec["type"] == "number":
                if type(value) is not int or not spec["min"] <= value <= spec["max"]:
                    fail(422, f'{spec["label"]} 범위를 확인하세요.')
            elif not isinstance(value, str) or len(value) > spec["max_length"]:
                fail(422, f'{spec["label"]} 입력 길이를 확인하세요.')
        with self.connection(write=True) as conn:
            row = self.row(conn, table, rowid)
            if not hmac.compare_digest(version.encode(), digest(row).encode()):
                fail(409, "기록이 변경되었습니다. 다시 불러온 뒤 수정하세요.")
            if set(changes) - set(row):
                fail(422, "현재 DB에 없는 항목입니다.")
            if table == "sessions":
                changes["creation_origin"] = None if changes["creation_origin"] == "unknown" else changes["creation_origin"]
                changes["creation_path"] = "admin:manual-classification"
            backup = self.backup(conn)
            assignments = ",".join(f'"{key}"=?' for key in changes)
            conn.execute(f'UPDATE "{table}" SET {assignments} WHERE rowid=?', [*changes.values(), rowid])
            self.audit(conn, actor, "update", table, rowid,
                       {"before": {key: row.get(key) for key in changes}, "after": changes}, backup)
        return {"message": "수정했습니다.", "backup_name": backup}

    def plan(self, conn, table, rowid):
        row = self.row(conn, table, rowid)
        if table == "users" and "user_id" in row:
            if conn.execute("SELECT COUNT(*) FROM sessions WHERE user_id=?", (row["user_id"],)).fetchone()[0]:
                fail(409, "연결된 훈련 세션이 있습니다. 세션을 먼저 정리한 뒤 훈련자를 삭제하세요.")
        targets = [(table, "rowid=?", [rowid])]
        if table == "sessions":
            for child, column in RELATIONS:
                if column in self.columns(conn, child):
                    targets.insert(0, (child, f'"{column}"=?', [row["session_id"]]))
        elif table == "sequences" and self.columns(conn, "sequence_analysis_results"):
            targets.insert(0, ("sequence_analysis_results", "session_key=? AND sequence_id=?", [row["session_key"], row["sequence_id"]]))
        identities = {}
        counts = {}
        for name, where, params in targets:
            identities[name] = [digest(dict(item)) for item in conn.execute(f'SELECT rowid AS __rowid,* FROM "{name}" WHERE {where} ORDER BY rowid', params)]
            if identities[name]:
                counts[name] = len(identities[name])
        sid = row.get("session_id", row.get("session_key"))
        reports = [item[0] for item in conn.execute("SELECT report_id FROM completions WHERE session_id=?", (sid,))] if sid and self.columns(conn, "completions") else []
        confirmation = str(row["session_id"] if table == "sessions" else row["user_id"] if table == "users" else rowid)
        return {"row": row, "counts": counts, "targets": targets, "reports": reports,
                "confirmation": confirmation, "revision": digest({"row": row, "identities": identities})}

    def preview_delete(self, table, rowid):
        with self.connection() as conn:
            plan = self.plan(conn, table, rowid)
        stamp = int(time.time())
        signed = {"table": table, "rowid": rowid, "revision": plan["revision"], "at": stamp}
        return {"counts": plan["counts"], "confirmation": plan["confirmation"], "summary": self.values(plan["row"]),
                "preview_token": f'{stamp}.{self.auth.sign(signed)}', "expires_seconds": 300,
                "notice": "삭제 전 DB 백업과 변경 이력을 저장합니다. 원본 CSV·이미지 파일은 보존됩니다."}

    def delete(self, table, rowid, token, confirmation, actor):
        try:
            timestamp, signature = token.split(".", 1)
            timestamp = int(timestamp)
        except (ValueError, AttributeError):
            fail(422, "먼저 삭제 범위를 확인하세요.")
        if not 0 <= time.time() - timestamp <= 300:
            fail(409, "삭제 확인이 만료되었습니다. 삭제 범위를 다시 확인하세요.")
        with self.connection(write=True) as conn:
            plan = self.plan(conn, table, rowid)
            signed = {"table": table, "rowid": rowid, "revision": plan["revision"], "at": timestamp}
            if not hmac.compare_digest(signature.encode(), self.auth.sign(signed).encode()):
                fail(409, "삭제 대상이나 연결 기록이 변경되었습니다. 삭제 범위를 다시 확인하세요.")
            if confirmation != plan["confirmation"]:
                fail(422, "삭제 확인값을 정확히 입력하세요.")
            with hold_idle_reports(Path(str(self.path) + ".reports"), plan["reports"]):
                backup = self.backup(conn)
                for name, where, params in plan["targets"]:
                    conn.execute(f'DELETE FROM "{name}" WHERE {where}', params)
                self.audit(conn, actor, "delete", table, rowid, {"counts": plan["counts"], "summary": self.values(plan["row"])}, backup)
                # A waiting job must see the committed deletion before it can enter
                # ReportJobs.run and read the completion through another connection.
                conn.commit()
        return {"message": "삭제했습니다.", "counts": plan["counts"], "backup_name": backup}

    def history(self):
        with self.connection() as conn:
            if not self.columns(conn, "admin_change_log"):
                return {"rows": []}
            rows = conn.execute("SELECT id,created_at,actor,action,table_name,record_rowid,details,backup_name FROM admin_change_log ORDER BY id DESC LIMIT 30")
            return {"rows": [dict(row) for row in rows]}
