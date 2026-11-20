"""Read-only exam information views and isolated, memory-only sample extraction.

Stored record features are never presented as a new analysis. The optional admin
job calls the existing CSV extractor in a separate bounded process; it does not
build/import records, analyze coating again, or write any dataset/DB/output file.
"""
from __future__ import annotations

import csv
import copy
import gzip
import io
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from trainer.demo_api.db_admin_auth import DBAdminAuth
from trainer.demo_api.training_data_catalog import TrainingDataCatalog


MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
PREVIEW_BYTES = 128 * 1024
MAX_ZONES = 256
WORKER_EXTRA_MEMORY_BYTES = 768 * 1024 * 1024
ID = re.compile(r"^[0-9a-f]{32}$")
NO_STORE = {"Cache-Control": "no-store"}
_METRICS = (("distance_mm", "거리 (입력)", "mm"), ("distance_real_mm", "거리 (실측)", "mm"),
            ("speed_mms", "속도 (입력)", "mm/s"), ("speed_real_mms", "속도 (실측)", "mm/s"))
_STATS = (("mean", "평균"), ("median", "중앙값"), ("p25", "25백분위"), ("p75", "75백분위"))
_COMPARE_FIELDS = ("frame_count", "duration_seconds", "spraying_frame_count", "sequence_count",
                   "mapped_spraying_frame_count", "unmapped_spraying_frames", "zone_id_available",
                   "zone_order", "map_order", "global_metrics", "zones", "sequence_heatmap_count",
                   "sequence_heatmap_invalid_count")
_METADATA = ("dataset_id", "dataset_name", "worker_id", "worker_name", "role", "qualification", "process",
             "scenario_id", "session_id", "content_type", "mode_label", "acquired_at", "quality_status", "source_kind")


def _error(status: int, reason: str) -> None:
    raise HTTPException(status, reason, headers=NO_STORE)


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _text(value, limit=300):
    return str(value)[:limit] if value is not None else None


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _items(value):
    return value if isinstance(value, list) else []


def _safe_value(value, depth=0):
    """Bound stored display data and never forward embedded binary payloads/URLs."""
    if depth > 8:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (float, int)):
        return _number(value)
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, list):
        return [_safe_value(item, depth + 1) for item in value[:MAX_ZONES]]
    if isinstance(value, dict):
        return {str(key)[:100]: _safe_value(item, depth + 1) for key, item in list(value.items())[:128]
                if str(key).casefold() not in {"data", "sequence_heatmaps", "image_url", "thickness_image_url"}}
    return None


def _metrics(distributions) -> list[dict]:
    distributions = distributions if isinstance(distributions, dict) else {}
    output = []
    for group, label, unit in _METRICS:
        stats = distributions.get(group)
        stats = stats if isinstance(stats, dict) else {}
        for key, statistic in _STATS:
            value = _number(stats.get(key))
            output.append({"key": f"{group}_{key}", "group": group, "label": f"{label} {statistic}",
                           "value": value, "unit": unit, "status": "available" if value is not None else "unavailable",
                           "sample_count": _number(stats.get("count")),
                           "description": "원본에 저장된 유효한 분사 프레임 통계입니다."})
    return output


def _coating_metrics(summary) -> list[dict]:
    return [{"key": "coating_" + key, "group": "stored_coating", "label": "저장 도막 " + label,
             "value": _number(summary.get(key)), "unit": "µm",
             "status": "available" if _number(summary.get(key)) is not None else "unavailable"}
            for key, label in (("mean_um", "평균"), ("median_um", "중앙값"), ("p25_um", "25백분위"), ("p75_um", "75백분위"))]


def _zone_key(value) -> str:
    face = value.get("map_num") if value.get("map_num") is not None else value.get("map_label")
    return f"{face if face is not None else 'unknown'}:{value.get('zone_id', 'unknown')}"


def _source_file(root: Path, relative) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("source_unavailable")
    candidate = Path(relative.replace("\\", "/"))
    windows = PureWindowsPath(relative)
    if candidate.is_absolute() or windows.drive or windows.root or ".." in candidate.parts:
        raise ValueError("source_unavailable")
    result = (root / candidate).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError("source_unavailable")
    return result


def _fingerprint(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "device": stat.st_dev, "inode": stat.st_ino}


def _preview(path: Path) -> dict:
    empty = {"status": "unavailable", "columns": [], "rows": [], "total_columns": 0, "sampled_rows": 0,
             "reason": "source_preview_unavailable", "omitted_columns": [], "truncated": False}
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rb") as source:
            raw = source.read(PREVIEW_BYTES + 1)
        truncated = len(raw) > PREVIEW_BYTES
        text = raw[:PREVIEW_BYTES].decode("utf-8-sig", errors="replace")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        columns = next(reader, [])
        if not columns or len(columns) > 512:
            return {**empty, "reason": "source_columns_unavailable"}
        omitted = [i for i, name in enumerate(columns) if re.search(r"heatmap|base64|image|texture.data", name, re.I)]
        priority = ["시간(초)", "time", "분사", "zone id", "ray texture map number", "distance (input)",
                    "distance (real)", "speed (input)", "speed (real)", "ray point x", "ray point y", "ray point z"]
        selected = sorted((i for i in range(len(columns)) if i not in omitted),
                          key=lambda i: (priority.index(columns[i].strip().casefold())
                                         if columns[i].strip().casefold() in priority else len(priority), i))[:24]
        rows = []
        try:
            for row in reader:
                if not row:
                    continue
                if len(rows) >= 8:
                    truncated = True
                    break
                cells = []
                for i in selected:
                    cell = row[i] if i < len(row) else ""
                    cells.append("[큰 페이로드 생략]" if len(cell) > 240 or re.search(r"base64|iVBORw0KGgo", cell, re.I) else cell[:120])
                rows.append(cells)
        except csv.Error:
            truncated = True
        return {"status": "available", "columns": [columns[i][:100] for i in selected], "rows": rows,
                "total_columns": len(columns), "sampled_rows": len(rows), "reason": None,
                "omitted_columns": [columns[i][:100] for i in omitted][:64], "truncated": truncated}
    except (OSError, ValueError, csv.Error, EOFError):
        return empty


def _worker_main() -> None:
    """Fixed subprocess entry point. No application startup or filesystem output."""
    try:
        request = json.loads(sys.stdin.buffer.read(65537))
        import resource
        # Apply a virtual-memory growth budget before importing NumPy/Pillow.
        virtual = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        resource.setrlimit(resource.RLIMIT_AS, (virtual + WORKER_EXTRA_MEMORY_BYTES, virtual + WORKER_EXTRA_MEMORY_BYTES))
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ[name] = "1"
        path = _source_file(Path(request["root"]), request["relative"])
        if _fingerprint(path) != request["fingerprint"]:
            raise ValueError("source_changed")
        if path.name.endswith(".gz"):
            total = 0
            with gzip.open(path, "rb") as stream:
                while block := stream.read(1024 * 1024):
                    total += len(block)
                    if total > request["max_source_bytes"]:
                        raise ValueError("source_expanded_too_large")
        from trainer.demo_api.training_data_build import _features
        features = _features(path, request["process"], request["scenario"],
                             legacy_expert=request["legacy"], content_type=request["content_type"])
        if _fingerprint(path) != request["fingerprint"]:
            raise ValueError("source_changed")
        # Actual numeric extraction is retained; binary heatmaps are not exam JSON output.
        features.pop("sequence_heatmaps", None)
        payload = json.dumps({"features": features}, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(payload) > request["max_result_bytes"]:
            raise ValueError("result_too_large")
        sys.stdout.buffer.write(payload)
    except MemoryError:
        sys.stdout.write('{"reason":"extraction_memory_limit"}')
    except BaseException as error:
        permitted = {"source_changed", "source_expanded_too_large", "result_too_large"}
        reason = str(error) if isinstance(error, ValueError) and str(error) in permitted else "feature_extraction_unavailable"
        sys.stdout.write(json.dumps({"reason": reason}))


class TrainingDataInspection:
    def __init__(self, catalog=None, *, max_source_bytes=MAX_SOURCE_BYTES, max_result_bytes=MAX_RESULT_BYTES,
                 job_ttl_seconds=600, job_timeout_seconds=120, clock=time.time):
        self.catalog = catalog or TrainingDataCatalog()
        self.max_source_bytes = min(MAX_SOURCE_BYTES, max(1, int(max_source_bytes)))
        self.max_result_bytes = min(MAX_RESULT_BYTES, max(1, int(max_result_bytes)))
        self.job_ttl_seconds = min(600, max(1, float(job_ttl_seconds)))
        self.job_timeout_seconds = min(180, max(.001, float(job_timeout_seconds)))
        self.clock = clock
        self._lock = threading.RLock()
        self._jobs = {}
        self._active = None
        self._process = None
        self._thread = None
        self._closed = False

    def _record(self, dataset_id):
        if not isinstance(dataset_id, str) or not ID.fullmatch(dataset_id):
            _error(404, "dataset_not_found")
        row = next((item for item in self.catalog._index().get("records", [])
                    if isinstance(item, dict) and item.get("dataset_id") == dataset_id), None)
        if row is None:
            _error(404, "dataset_not_found")
        if not isinstance(row.get("json_bytes"), int) or row["json_bytes"] > MAX_ARTIFACT_BYTES:
            _error(413, "inspection_record_too_large")
        try:
            artifact = self.catalog._detail_row(row)  # Existing single-record integrity check.
            if artifact.get("dataset_id") != dataset_id or not isinstance(artifact.get("features"), dict):
                raise ValueError("invalid_record")
            return artifact
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            _error(404, "dataset_unavailable")

    def _source(self, artifact):
        source = _mapping(_mapping(artifact.get("provenance")).get("source_csv"))
        try:
            path = _source_file(self.catalog.sources, source.get("relative_path"))
            snapshot = _fingerprint(path)
            if source.get("bytes") is not None and source["bytes"] != snapshot["bytes"]:
                return None, None, "source_changed"
            return path, snapshot, None
        except (OSError, ValueError, TypeError, AttributeError):
            return None, None, "source_unavailable"

    def inspection(self, dataset_id):
        artifact = self._record(dataset_id)
        metadata = _mapping(artifact.get("metadata"))
        metadata = {key: _safe_value(metadata[key]) for key in _METADATA if key in metadata}
        features = artifact["features"]
        path, fingerprint, source_reason = self._source(artifact)
        source = _mapping(_mapping(artifact.get("provenance")).get("source_csv"))
        preview = _preview(path) if path else {"status": "unavailable", "columns": [], "rows": [], "total_columns": 0,
                   "sampled_rows": 0, "reason": source_reason, "omitted_columns": [], "truncated": False}
        process = metadata.get("process")
        coating = {"basis": "stored", "status": "not_applicable" if process in {"blasting", "grinding"} else "unavailable",
                   "reason": "coating_not_applicable_to_process" if process in {"blasting", "grinding"} else "process_not_recorded", "maps": []}
        if process == "spray":
            stored_coating = _mapping(artifact.get("coating"))
            coating = {"basis": "stored", "status": _text(stored_coating.get("status") or "unavailable"),
                       "semantics": _text(stored_coating.get("semantics")), "maps": []}
            for i, face in enumerate(_items(stored_coating.get("maps"))[:32]):
                if not isinstance(face, dict):
                    continue
                displayed = _safe_value(face)
                displayed["image_url"] = f"/admin/api/training-data/{dataset_id}/images/{i}"
                displayed["thickness_image_url"] = (displayed["image_url"] + "?view=thickness") if face.get("thickness_image_url") else None
                coating["maps"].append(displayed)
        order = {_zone_key(item): i + 1 for i, item in reversed(list(enumerate(_items(features.get("zone_order"))))) if isinstance(item, dict)}
        zones = []
        for zone in _items(features.get("zones"))[:MAX_ZONES]:
            if not isinstance(zone, dict):
                continue
            key = _zone_key(zone)
            zone_coating = {"basis": "stored", "status": "not_applicable" if process in {"blasting", "grinding"} else "unavailable", "summary": {}, "metrics": []}
            for face in coating["maps"]:
                if face.get("map_num") != zone.get("map_num"):
                    continue
                matching = next((item for item in _items(face.get("zones")) if isinstance(item, dict) and str(item.get("zone_id")) == str(zone.get("zone_id"))), None)
                if matching:
                    summary = _mapping(matching.get("summary"))
                    zone_coating = {"basis": "stored", "status": face.get("status", "unavailable"),
                                    "summary": summary, "metrics": _coating_metrics(summary)}
                    break
            zones.append({"key": key, "map_num": _safe_value(zone.get("map_num")), "map_label": _text(zone.get("map_label")),
                          "zone_id": _text(zone.get("zone_id")), "frame_count": _number(zone.get("frame_count")),
                          "metrics": _metrics(zone), "sequence": {"first_visit_index": order.get(key)}, "coating": zone_coating})
        unavailable_pose = {"status": "unavailable", "reason": "stored_features_do_not_include_pose_scores"}
        unavailable_direction = {"status": "unavailable", "reason": "stored_features_do_not_include_direction_angles"}
        summary = {key: _number(features.get(key)) for key in ("frame_count", "duration_seconds", "spraying_frame_count", "mapped_spraying_frame_count", "unmapped_spraying_frames")}
        summary.update(spray_interval_count=_number(features.get("sequence_count")),
                       zone_count=len(features["zones"]) if isinstance(features.get("zones"), list) else None,
                       metrics=_metrics(features.get("global_metrics")), pose=unavailable_pose, direction=unavailable_direction)
        reason = source_reason or ("source_too_large" if fingerprint["bytes"] > self.max_source_bytes else None)
        result = {"dataset_id": dataset_id, "metadata": metadata, "source_preview": preview, "summary": summary, "zones": zones, "coating": coating,
                  "pipeline": [
                      {"key": "source_csv", "label": "원천 CSV", "status": "available" if path else "unavailable", "description": "등록된 원천의 일부 행만 미리 봅니다.",
                       "values": {"filename": _text(source.get("filename")), "bytes": fingerprint["bytes"] if fingerprint else None}},
                      {"key": "stored_features", "label": "저장된 특징 JSON", "status": "available", "description": "이미 구축된 기록별 수치 특징을 조회합니다. 새 분석 실행이 아닙니다.",
                       "values": {"schema_version": _text(artifact.get("schema_version")), "frame_count": summary["frame_count"]}},
                      {"key": "stored_coating", "label": "저장된 도막 분석", "status": coating["status"], "description": "저장 결과이며 자세 채점이나 새로운 도막 계산을 수행하지 않습니다.", "values": {"map_count": len(coating["maps"])}}],
                  "links": {"json": f"/admin/api/training-data/{dataset_id}", "json_download": f"/admin/api/training-data/{dataset_id}/download", "motion": f"/motion-viewer?dataset_id={dataset_id}"},
                  "scope": {"result_kind": "stored_record_features", "raw_frames_included": False, "full_motion_requires_source": True,
                            "reconversion_writes_registered_data": False, "zones_truncated": len(_items(features.get("zones"))) > MAX_ZONES,
                            "notes": ["훈련 한 회의 CSV와 연결 자료를 한 기록으로 봅니다. Zone·분사 구간을 별도 획득 건수로 세지 않습니다.",
                                      "품질 완료는 숙련자 자격이나 AI 정확도, 모델 학습 완료를 뜻하지 않습니다.", "시퀀스 수는 분사 Off→On 구간 수이며 공식 sequence_id가 아닙니다."]},
                  "limits": {"max_source_bytes": self.max_source_bytes, "max_result_bytes": self.max_result_bytes, "preview_bytes": PREVIEW_BYTES,
                             "preview_rows": 8, "preview_columns": 24, "max_zones": MAX_ZONES, "job_ttl_seconds": self.job_ttl_seconds, "max_retained_jobs": 2},
                  "reconvert": {"available": reason is None, "reason": reason, "source_bytes": fingerprint["bytes"] if fingerprint else None, "max_source_bytes": self.max_source_bytes}}
        if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > MAX_RESPONSE_BYTES:
            _error(413, "inspection_response_too_large")
        return result

    def _purge(self):
        now = self.clock()
        for ident, job in list(self._jobs.items()):
            if ident != self._active and job["expires_at"] <= now:
                self._jobs.pop(ident)

    def _public_job(self, job):
        return copy.deepcopy({key: value for key, value in job.items() if not key.startswith("_")})

    def start(self, dataset_id, owner):
        with self._lock:
            if self._closed:
                _error(503, "inspection_service_closed")
            self._purge()
            if self._active:
                _error(409, "reconversion_already_running")
        artifact = self._record(dataset_id)
        metadata = _mapping(artifact.get("metadata"))
        if metadata.get("process") not in {"spray", "blasting", "grinding"}:
            _error(409, "source_process_unsupported")
        path, fingerprint, reason = self._source(artifact)
        if reason:
            _error(409, reason)
        if fingerprint["bytes"] > self.max_source_bytes:
            _error(413, "source_too_large")
        ident = uuid.uuid4().hex
        with self._lock:
            if self._closed:
                _error(503, "inspection_service_closed")
            if self._active:
                _error(409, "reconversion_already_running")
            while len(self._jobs) >= 2:
                self._jobs.pop(next(iter(self._jobs)))
            job = {"job_id": ident, "dataset_id": dataset_id, "status": "queued", "result_available": False,
                   "download_url": None, "reason": None, "expires_at": self.clock() + self.job_ttl_seconds, "comparison": None,
                   "steps": [{"key": "source", "label": "등록 원천 확인", "status": "completed"},
                             {"key": "extract", "label": "CSV 범위 확인·특징 재추출", "status": "pending"},
                             {"key": "compare", "label": "저장 특징값과 비교", "status": "pending"},
                             {"key": "result", "label": "시험 표본 JSON 준비", "status": "pending"}], "_owner": owner, "_payload": None}
            self._jobs[ident] = job
            self._active = ident
            response = self._public_job(job)
            self._thread = threading.Thread(target=self._run, args=(ident, artifact, path, fingerprint), daemon=True)
            self._thread.start()
        return response

    def _run(self, ident, artifact, path, fingerprint):
        process = None
        try:
            with self._lock:
                if self._closed:
                    raise ValueError("extraction_cancelled")
                job = self._jobs[ident]
                job["status"] = "running"; job["steps"][1]["status"] = "running"
            metadata = _mapping(artifact.get("metadata"))
            source = _mapping(_mapping(artifact.get("provenance")).get("source_csv"))
            relative = source["relative_path"]
            request = {"root": str(self.catalog.sources), "relative": relative, "fingerprint": fingerprint,
                       "process": metadata["process"], "scenario": str(metadata.get("scenario_id") or ""),
                       "content_type": metadata.get("content_type") or "learning",
                       "legacy": metadata.get("source_kind") != "converted" and relative.startswith("raw/"),
                       "max_source_bytes": self.max_source_bytes, "max_result_bytes": self.max_result_bytes}
            process = subprocess.Popen([sys.executable, "-B", "-m", "trainer.demo_api.training_data_inspection", "--extract-worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            with self._lock:
                self._process = process
                if self._closed:
                    process.kill()
                    raise ValueError("extraction_cancelled")
            try:
                output, _ = process.communicate(json.dumps(request).encode(), timeout=self.job_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill(); process.communicate()
                raise ValueError("extraction_timeout")
            if process.returncode:
                raise ValueError("extraction_worker_failed")
            if len(output) > self.max_result_bytes:
                raise ValueError("result_too_large")
            parsed = json.loads(output)
            if parsed.get("reason"):
                raise ValueError(parsed["reason"])
            fresh = parsed["features"]
            if _fingerprint(path) != fingerprint:
                raise ValueError("source_changed")
            with self._lock:
                job["steps"][1]["status"] = "completed"
                job["steps"][2]["status"] = "running"
            stored = artifact["features"]
            common = [key for key in _COMPARE_FIELDS if key in stored and key in fresh]
            comparison = {"basis": "feature_values", "compared_fields": common,
                          "changed_fields": [key for key in common if stored[key] != fresh[key]],
                          "fresh_only_fields": [key for key in _COMPARE_FIELDS if key in fresh and key not in stored],
                          "stored_only_fields": [key for key in _COMPARE_FIELDS if key in stored and key not in fresh],
                          "note": "1.1·1.2 및 현재 추출기 차이가 있을 수 있어 파일 해시 일치나 AI 정확도로 판정하지 않습니다."}
            with self._lock:
                job["steps"][2]["status"] = "completed"
                job["steps"][3]["status"] = "running"
            result = {"schema_version": "exam-record-features-1.0", "result_kind": "exam_sample_fresh_features", "dataset_id": job["dataset_id"],
                      "metadata": {key: _safe_value(metadata[key]) for key in _METADATA if key in metadata}, "features": fresh,
                      "comparison": comparison, "source": {"kind": "registered_csv", "bytes": fingerprint["bytes"]},
                      "stored_analysis": {"coating": {"status": "stored_only", "reason": "coating_not_recomputed"},
                                          "pose": {"status": "unavailable", "reason": "pose_not_computed_by_feature_extractor"}},
                      "scope": {"sequence_heatmap_payloads": "omitted", "raw_motion_frames": "not_included",
                                "registered_record_replaced": False, "model_training_performed": False}}
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode()
            if len(payload) > self.max_result_bytes:
                raise ValueError("result_too_large")
            with self._lock:
                job.update(status="completed", result_available=True, comparison=comparison, _payload=payload,
                           download_url=f"/admin/api/training-data/{job['dataset_id']}/reconvert/{ident}/download")
                for step in job["steps"]:
                    step["status"] = "completed"
        except BaseException as error:
            allowed = {"source_changed", "source_expanded_too_large", "result_too_large", "feature_extraction_unavailable",
                       "extraction_memory_limit", "extraction_timeout", "extraction_cancelled", "extraction_worker_failed"}
            reason = str(error) if isinstance(error, ValueError) and str(error) in allowed else "feature_extraction_unavailable"
            with self._lock:
                job = self._jobs.get(ident)
                if job:
                    job.update(status="failed", reason=reason)
                    for step in job["steps"]:
                        if step["status"] != "completed":
                            step["status"] = "failed"
                            break
        finally:
            if process is not None and process.poll() is None:
                process.kill(); process.wait()
            with self._lock:
                self._process = None
                if self._active == ident:
                    self._active = None

    def status(self, dataset_id, job_id, owner):
        with self._lock:
            self._purge()
            job = self._jobs.get(job_id) if ID.fullmatch(job_id) and ID.fullmatch(dataset_id) else None
            if job is None or job["dataset_id"] != dataset_id or job["_owner"] != owner or job["expires_at"] <= self.clock():
                _error(404, "reconversion_job_unavailable")
            return self._public_job(job)

    def download(self, dataset_id, job_id, owner):
        with self._lock:
            self.status(dataset_id, job_id, owner)
            job = self._jobs[job_id]
            if not job["result_available"]:
                _error(409, "reconversion_result_not_ready")
            return job["_payload"]

    def close(self):
        with self._lock:
            self._closed = True
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)


class ReconvertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def create_training_inspection_router(auth: DBAdminAuth, service=None) -> APIRouter:
    service = service or TrainingDataInspection()
    router = APIRouter(prefix="/admin/api/training-data", include_in_schema=False)
    router.add_event_handler("shutdown", service.close)

    @router.get("/{dataset_id}/inspection")
    def inspection(dataset_id: str):
        return JSONResponse(service.inspection(dataset_id), headers=NO_STORE)

    @router.post("/{dataset_id}/reconvert", status_code=202)
    def reconvert(dataset_id: str, request: Request, body: ReconvertRequest | None = Body(default=None)):
        session = auth.require(request, write=True)
        return JSONResponse(service.start(dataset_id, session.username), status_code=202, headers=NO_STORE)

    @router.get("/{dataset_id}/reconvert/{job_id}")
    def job_status(dataset_id: str, job_id: str, request: Request):
        session = auth.require(request)
        return JSONResponse(service.status(dataset_id, job_id, session.username), headers=NO_STORE)

    @router.get("/{dataset_id}/reconvert/{job_id}/download")
    def job_download(dataset_id: str, job_id: str, request: Request):
        session = auth.require(request)
        return Response(service.download(dataset_id, job_id, session.username), media_type="application/json",
                        headers={**NO_STORE, "Content-Disposition": f'attachment; filename="exam-sample-{job_id}.json"'})

    return router


if __name__ == "__main__" and sys.argv[1:] == ["--extract-worker"]:
    _worker_main()
