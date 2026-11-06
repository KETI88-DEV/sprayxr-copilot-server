"""Read-only row parameter details and the shared accessible detail dialog."""
from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

from trainer.demo_api.db_report_status import report_progress
from trainer.demo_api.db_provenance import record_provenance, provenance_meta

PRIVATE_TABLES = frozenset({"admin_change_log"})

DETAIL_CSS = """
[data-detail-url]{cursor:pointer}tr[data-detail-url]:focus{outline:2px solid #6688ef;outline-offset:-2px}
.record-button{padding:5px 9px;background:#eef2fd;color:#3157d5;font-size:11px;white-space:nowrap}
.detail-modal{border:1px solid #d9e2f2;border-radius:16px;padding:0;width:min(1100px,calc(100vw - 30px));max-width:1100px;max-height:90vh;color:#17243b;box-shadow:0 24px 80px #10264d55}
.detail-modal::backdrop{background:#111f3e88;backdrop-filter:blur(3px)}.detail-header{position:sticky;top:0;z-index:2;background:white;padding:19px 23px;border-bottom:1px solid #e4eaf4;display:flex;align-items:center;justify-content:space-between;gap:15px}
.detail-header h2{margin:0;font-size:18px}.detail-header p{font-size:12px;color:#73829a;margin:5px 0 0}.detail-close{background:#eef2f9;color:#40516c;font-size:19px;padding:6px 13px}
.detail-body{padding:20px 23px;overflow:auto;max-height:calc(90vh - 90px)}.detail-loading{padding:40px;text-align:center;color:#6c7c95}.detail-notice{padding:12px;border-radius:8px;background:#fff4db;color:#7b5d1b;font-size:12px;line-height:1.7;margin:0 0 18px}
.detail-meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:0 0 20px}.detail-meta div{background:#f4f7fc;border-radius:8px;padding:11px;font-size:12px;overflow-wrap:anywhere}.detail-meta dt{color:#7b89a0;margin-bottom:5px}.detail-meta dd{margin:0;font-weight:650}
.detail-section{margin:12px 0;border:1px solid #e2e8f2;border-radius:9px;overflow:hidden}.detail-section>summary{padding:12px;background:#f3f6fc;color:#405d99;font-size:13px;font-weight:700;cursor:pointer}.detail-parameters{padding:12px;font-family:ui-monospace,Consolas,monospace;font-size:12px;line-height:1.7;overflow-wrap:anywhere}.json-branch{margin:3px 0 3px 12px;border-left:1px solid #dde6f3;padding-left:10px}.json-branch>summary{color:#415b85;cursor:pointer}.json-key{color:#405786;font-weight:650}.json-null{color:#8894a8}.json-value{white-space:pre-wrap;color:#215f46}.json-number{color:#3157d5}.json-leaf{margin:4px 0}.json-more{margin:8px;background:#edf2fb;color:#3157d5;font-size:11px;padding:5px 10px}.detail-error{color:#a12a2a;padding:20px}.detail-copy{background:#eef2fd;color:#3157d5;font-size:12px;padding:7px 10px;margin-left:6px}
@media(max-width:650px){.detail-meta{grid-template-columns:1fr 1fr}.detail-body{padding:14px}.detail-header{padding:15px}.detail-header h2{font-size:15px}}
"""

DETAIL_HTML = """<dialog id="db-detail-modal" class="detail-modal" aria-labelledby="db-detail-title">
<div class="detail-header"><div><h2 id="db-detail-title">상세 파라미터</h2><p>읽기 전용 · Esc 키로 닫기</p></div><div><button type="button" class="detail-copy" id="db-detail-copy">JSON 복사</button><button type="button" class="detail-close" aria-label="상세 창 닫기">×</button></div></div>
<div class="detail-body" id="db-detail-content"></div></dialog><script defer src="/admin/assets/db_sequence_heatmap.js"></script><script defer src="/admin/assets/db_viewer.js"></script>"""


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"type": "binary", "size_bytes": len(value)}
    if isinstance(value, dict):
        return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def decoded(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return json_safe(json.loads(value, parse_constant=lambda token: token))
        except (ValueError, TypeError):
            pass
    return json_safe(value)


def record_detail(conn: sqlite3.Connection, table: str, rowid: int) -> dict[str, Any] | None:
    if table in PRIVATE_TABLES:
        return None
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    if table not in names:
        return None
    quoted = '"' + table.replace('"','""') + '"'
    row = conn.execute(f"SELECT * FROM {quoted} WHERE rowid=?", (rowid,)).fetchone()
    if row is None:
        return None
    meta = [{"label":"테이블","value":table},{"label":"레코드 번호","value":rowid}]
    meta.extend(provenance_meta(record_provenance(conn, table, row)))
    if table == "completions":
        progress = report_progress(row)
        meta.extend({"label":label,"value":value} for label,value in (("처리 상태",progress['label']),("대기 경과",progress['elapsed']),("최근 처리 시각",progress['latest_at'] or '미기록'),("상태 사유",progress['reason'])))
    return {"title": f"레코드 상세 · {table}",
            "meta": meta,
            "sections": [{"title":"레코드 파라미터","data":{key:decoded(row[key]) for key in row.keys()}}]}
