"""Read existing feedback into an update-review bundle without changing its state.

Run in the provisioned container:
python -m trainer.scripts.query_feedback_review --db /output/eval_results/sequences.db
Redirect stdout to retain the JSON report. This is not a training-data exporter;
submitted suggestions remain untrusted, and existing administrator review remains
the authority for corrections/approval. No network requests or migrations occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any


def _context(raw: object) -> tuple[dict | None, bool]:
    if raw is None or raw == "":
        return None, False
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None, True
    return (parsed, False) if isinstance(parsed, dict) else (None, True)


def _provenance(row: dict, context: dict | None) -> dict[str, Any]:
    fields = ("query_turn_id", "user_id", "session_key", "conversation_key", "origin_server",
              "caller_source", "created_at", "updated_at", "reviewed_by", "server_version",
              "status_updated_by", "status_updated_at", "page")
    result = {key: row[key] for key in fields if key in row}
    for key in ("answer_source", "task", "response_mode", "record", "session_id", "worker_id", "conversation_id"):
        if context and key in context:
            result[key] = context[key]
    return result


def build_review_bundle(db_path: str | Path, *, include_closed: bool = False) -> dict:
    """Read one consistent SQLite snapshot; never initialize missing tables.

    Pending/approved Query ratings and new/reviewing user opinions are included
    by default. Rejected/done entries require include_closed. Approval here stays
    a recorded review status, not a claim of training eligibility or application.
    """
    path = Path(db_path).resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    items = []
    missing = []
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("query_answer_feedback", "user_feedback"):
            if table not in tables:
                missing.append(table)
                continue
            for raw in conn.execute("SELECT * FROM " + table):
                row = dict(raw)
                query_rating = table == "query_answer_feedback"
                status = row.get("review_status" if query_rating else "status") or ("pending" if query_rating else "new")
                if not include_closed and status == ("rejected" if query_rating else "done"):
                    continue
                context, invalid = _context(row.get("query_context" if query_rating else "answer_context"))
                saved = context or {}
                items.append({
                    "kind": "query_rating" if query_rating else "user_feedback",
                    "feedback_id": row.get("feedback_id" if query_rating else "submission_id"),
                    "status": status, "rating": row.get("rating"),
                    "question": row.get("question") if query_rating else saved.get("question"),
                    "answer": row.get("answer") if query_rating else saved.get("answer"),
                    "message": row.get("comment" if query_rating else "message"),
                    "suggested_answer": row.get("suggested_answer") if query_rating else saved.get("correct_answer"),
                    "corrected_answer": row.get("corrected_answer") if query_rating else None,
                    "review_note": row.get("review_note"), "resolution": row.get("resolution"),
                    "context": context, "context_parse_error": invalid,
                    "provenance": {"source_table": table, **_provenance(row, context)},
                })
    items.sort(key=lambda item: (str(item["provenance"].get("created_at") or ""), item["kind"],
                                 str(item["feedback_id"] or "")), reverse=True)
    return {
        "schema_version": 1, "purpose": "update_review_not_training_data", "automatic_training": False,
        "generated_at": datetime.now(timezone.utc).isoformat(), "source_database": str(path),
        "include_closed": include_closed, "missing_tables": missing,
        "summary": {"items": len(items), "by_kind": dict(Counter(item["kind"] for item in items)),
                    "by_status": dict(Counter(item["kind"] + ":" + str(item["status"]) for item in items))},
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="Existing feedback SQLite database (opened mode=ro)")
    parser.add_argument("--include-closed", action="store_true", help="Also include rejected ratings and done opinions")
    args = parser.parse_args(argv)
    try:
        bundle = build_review_bundle(args.db, include_closed=args.include_closed)
    except (OSError, sqlite3.Error) as error:
        print("Feedback review read failed: " + type(error).__name__, file=sys.stderr)
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
