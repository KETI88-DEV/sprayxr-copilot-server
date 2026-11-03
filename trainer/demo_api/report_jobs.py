"""Durable report inputs and restartable, mutually exclusive report execution."""
from __future__ import annotations

import hashlib
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from trainer.demo_api.v1_store import V1Store

CSV_UPLOAD_LIMIT = 64 * 1024 * 1024
CSV_INFLATED_LIMIT = 256 * 1024 * 1024
PNG_UPLOAD_LIMIT = 16 * 1024 * 1024
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="session-report")
_GUARD = threading.Lock()
_ACTIVE: set[str] = set()


class InputTooLarge(ValueError):
    pass


def save_artifact(directory: Path, raw: bytes) -> str:
    """Commit the full file before a database row can reference it."""
    name = hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    temporary = directory / (uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        # Also persist the directory entry on the Linux production filesystem.
        if os.name != "nt":
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)
    return name


def read_artifact(directory: Path, name: str, limit: int) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{64}", name):
        raise ValueError("invalid report artifact reference")
    with (directory / name).open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise InputTooLarge("report artifact exceeds size limit")
    if hashlib.sha256(raw).hexdigest() != name:
        raise ValueError("report artifact checksum mismatch")
    return raw


class ReportJobs:
    def __init__(self, store: V1Store, analyze: Callable[..., None]) -> None:
        self.store = store
        self.analyze = analyze

    def run(self, report_id: str) -> None:
        key = str(self.store.artifact_dir / report_id)
        with _GUARD:
            if key in _ACTIVE:
                return
            _ACTIVE.add(key)
        lock = None
        try:
            self.store.artifact_dir.mkdir(parents=True, exist_ok=True)
            # Safe server-issued UUID only; no client identifiers enter paths.
            if not re.fullmatch(r"RPT-[0-9a-f]{12}", report_id):
                return
            lock = (self.store.artifact_dir / (report_id + ".lock")).open("a+b")
            if os.name == "nt":
                import msvcrt
                lock.write(b"0")
                lock.flush()
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    return
            else:
                import fcntl
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
            row = self.store.get_completion(report_id)
            if row is None or row.report_status != "pending":
                return
            refs = row.payload.get("artifacts")
            if not isinstance(refs, dict) or not refs.get("telemetry"):
                return  # Legacy input/progress is unknown; never infer a failure.
            self.store.mark_completion_processing(report_id)
            raw = read_artifact(self.store.artifact_dir, refs["telemetry"], CSV_UPLOAD_LIMIT)
            image = read_artifact(self.store.artifact_dir, refs["height_map"], PNG_UPLOAD_LIMIT) if refs.get("height_map") else None
            self.analyze(
                report_id, session_id=row.session_id, user_id=row.payload["user_id"],
                scenario_id=row.payload["scenario_id"], csv_bytes=raw,
                height_map=image, height_map_name=row.payload.get("height_map_name", "height_map.png") if image is not None else None,
                target_thickness_um=row.payload.get("target_thickness_um"),
            )
        except Exception:
            import logging
            logging.getLogger(__name__).exception("report job failed report=%s", report_id)
            self.store.finish_completion(report_id, status="failed", analysis={"error": "report_input_unavailable"})
        finally:
            if lock is not None:
                lock.close()  # OS releases the lock even after a worker/process crash.
            with _GUARD:
                _ACTIVE.discard(key)

    def recover_pending(self) -> int:
        rows = self.store.pending_completions()
        for row in rows:
            if row.payload.get("artifacts"):
                _POOL.submit(self.run, row.report_id)
        return sum(bool(row.payload.get("artifacts")) for row in rows)
