#!/usr/bin/env python3
"""벤더 세션 샘플(요청 기록 JSON + 원시 CSV)을 v2.5 API에 **그대로** 다시 보낸다.

샘플은 한 세션의 요청을 발생 순서대로 담는다(2026-10-08 샘플 `notes.구성`):
`sessions/start` → `copilot/analyze-sequence` N건 → `sessions/end` → `sessions/complete`.
이 스크립트는 그 순서로 보내고, 본문은 **고치지 않는다** — 벤더가 실제로 보낸 것이 서버에서 어떻게
처리되는지 보는 것이 목적이다. 본문에 필드를 덧붙이면(예: `mode_id`) 서버가 다르게 판단한다.

옵션으로만 바꾸는 것:

* `--query` — 시퀀스마다 `copilot/query`(`task_type=pose_recomm`, `context=SEQ-…`)를 보낸다.
  벤더 샘플에는 질의가 없다.
* `--fill-averages` — complete에 빠진 `avg_speed_mms`·`avg_distance_mm`(명세서 p.12는 필수)를
  CSV의 분사 구간 평균(`Speed (Real)`·`Distance (Real)`)으로 채운다. 이 서버는 a2fbe37부터 둘을
  선택으로 받으므로 없어도 접수되지만, 필수로 받는 서버를 시험하거나 값을 실어 보려면 쓴다.
  기본은 벤더 그대로 — 채우지 않는다.
* `--skip SEQ-002,SEQ-004` — 그 시퀀스를 업로드하지 않는다(질의 없이 끝난 세션·전송 누락 재현).

complete가 접수되면 리포트가 `pending`을 벗어날 때까지 폴링한다. 결과는 `--out` 아래에
`<태그>_requests.csv`(요청별 상태), `<태그>_report.json`(리포트 상세)으로 남긴다.

예::

    docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/replay_session_sample.py \\
        --api http://127.0.0.1:8121 --sample-dir /data/sources/new_format/sample_2026-10-08 \\
        --query --fill-averages --out /output/sample_2026-10-08 --tag full
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

if __package__ in (None, ""):  # `python3 trainer/scripts/replay_session_sample.py`로 부를 때
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trainer.demo_api.session_analysis import SPRAY_COLUMN, SPRAY_ON, read_csv  # noqa: E402

V1_PREFIX = "/api/v1"
COMPLETE_ENDPOINT = "/sessions/complete"
ANALYZE_ENDPOINT = "/copilot/analyze-sequence"
AVERAGE_COLUMNS = {"avg_speed_mms": "Speed (Real)", "avg_distance_mm": "Distance (Real)"}
FILE_PARTS = ("telemetry_csv", "height_map_image")
POLL_SEC = 2.0
POLL_LIMIT_SEC = 600.0


@dataclass(frozen=True)
class Sample:
    requests: list[dict[str, Any]]
    directory: Path


def load_sample(directory: Path) -> Sample:
    """샘플 폴더의 요청 기록 JSON(하나)을 읽는다."""
    candidates = sorted(directory.glob("*.json"))
    if len(candidates) != 1:
        raise SystemExit(f"{directory}: 요청 기록 JSON이 하나여야 한다 ({len(candidates)}개)")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise SystemExit(f"{candidates[0]}: `requests` 목록이 없다")
    return Sample(requests=requests, directory=directory)


def endpoint_path(endpoint: str) -> str:
    """`"POST /sessions/start"` → `"/api/v1/sessions/start"`."""
    method, _, path = endpoint.partition(" ")
    if method.upper() != "POST" or not path.startswith("/"):
        raise ValueError(f"지원하지 않는 요청: {endpoint}")
    return path if path.startswith(V1_PREFIX) else V1_PREFIX + path


def part_filename(description: str) -> str:
    """`"3_11000346_1.csv.gz (8,764,114 bytes, 별도 첨부)"` → `"3_11000346_1.csv.gz"`."""
    return description.split(" (", 1)[0].strip()


def spray_averages(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    """분사(`분사`=On) 프레임의 속도·거리 평균. 값이 하나도 없는 열은 뺀다."""
    spraying = [row for row in rows if row.get(SPRAY_COLUMN) == SPRAY_ON]
    out: dict[str, float] = {}
    for field, column in AVERAGE_COLUMNS.items():
        values: list[float] = []
        for row in spraying:
            try:
                values.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                continue
        if values:
            out[field] = round(statistics.fmean(values), 2)
    return out


def _complete_parts(
    body: Mapping[str, Any], directory: Path, *, fill_averages: bool
) -> tuple[dict[str, str], dict[str, tuple[str, bytes, str]], dict[str, float]]:
    raw_parts = body.get("_parts")
    parts: Mapping[str, Any] = raw_parts if isinstance(raw_parts, Mapping) else {}
    form = {key: str(value) for key, value in body.items() if key != "_parts"}
    files: dict[str, tuple[str, bytes, str]] = {}
    for field in FILE_PARTS:
        if field in parts:
            name = part_filename(str(parts[field]))
            files[field] = (name, (directory / name).read_bytes(), "application/octet-stream")
    filled: dict[str, float] = {}
    if fill_averages and "telemetry_csv" in files:
        averages = spray_averages(read_csv(files["telemetry_csv"][1]))
        filled = {k: v for k, v in averages.items() if k not in form}
        form.update({k: str(v) for k, v in filled.items()})
    return form, files, filled


def _pose_context(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json().get("data") or {}
    except ValueError:
        return {}
    context = data.get("pose_context")
    return context if isinstance(context, dict) else {}


def replay(
    sample: Sample,
    *,
    api: str,
    query: bool = False,
    fill_averages: bool = False,
    skip: Iterable[str] = (),
    transport: httpx.BaseTransport | None = None,
    poll_sec: float = POLL_SEC,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """요청을 순서대로 보낸다. (요청별 행, 리포트 상세 또는 None)."""
    skipped = set(skip)
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] | None = None
    with httpx.Client(base_url=api.rstrip("/"), timeout=300, transport=transport) as client:
        for entry in sample.requests:
            endpoint = str(entry.get("endpoint", ""))
            raw_body = entry.get("body")
            body: dict[str, Any] = raw_body if isinstance(raw_body, dict) else {}
            path = endpoint_path(endpoint)
            row: dict[str, Any] = {
                "요청": endpoint, "시퀀스": body.get("sequence_id", ""), "구역": body.get("zone_id", ""),
                "HTTP": "", "질의 상태": "", "사유": "", "유사도": "", "비고": "",
            }
            rows.append(row)
            if path.endswith(ANALYZE_ENDPOINT) and body.get("sequence_id") in skipped:
                row["비고"] = "업로드 생략(--skip)"
                continue
            if path.endswith(COMPLETE_ENDPOINT):
                form, files, filled = _complete_parts(body, sample.directory, fill_averages=fill_averages)
                response = client.post(path, data=form, files=files)
                if filled:
                    row["비고"] = "CSV로 채움: " + ", ".join(f"{k}={v}" for k, v in filled.items())
            else:
                response = client.post(path, json=body)
            row["HTTP"] = response.status_code

            if path.endswith(ANALYZE_ENDPOINT) and query:
                answer = client.post(f"{V1_PREFIX}/copilot/query", json={
                    "session_id": body.get("session_id"), "user_id": body.get("user_id"),
                    "scenario_id": body.get("scenario_id"), "task_type": "pose_recomm",
                    "prompt": "이번 분사 자세를 평가해 주세요.", "context": body.get("sequence_id"),
                })
                context = _pose_context(answer)
                row["질의 상태"] = context.get("analysis_status") or f"HTTP {answer.status_code}"
                row["사유"] = context.get("reason_code") or ""
                score = context.get("similarity_score")
                row["유사도"] = "" if score is None else score

            if path.endswith(COMPLETE_ENDPOINT) and response.status_code == 200:
                data = response.json()["data"]
                report = _poll_report(client, data["session_id"], data["report_id"], poll_sec=poll_sec)
    return rows, report


def _poll_report(client: httpx.Client, session_id: str, report_id: str, *, poll_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + POLL_LIMIT_SEC
    while time.monotonic() < deadline:
        status = client.get(f"{V1_PREFIX}/sessions/{session_id}/reports/{report_id}").json()["data"]["report_status"]
        if status != "pending":
            break
        time.sleep(poll_sec)
    return client.get(f"{V1_PREFIX}/reports/{report_id}").json()["data"]


def write_outputs(rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any] | None, out: Path, tag: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{tag}_requests.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / f"{tag}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def summarize(rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any] | None) -> str:
    requests = Counter(f"{r['요청'].split()[-1]} {r['HTTP'] or '생략'}" for r in rows)
    lines = ["요청: " + ", ".join(f"{k}×{v}" for k, v in requests.items())]
    queried = Counter((r["질의 상태"], r["사유"]) for r in rows if r["질의 상태"])
    if queried:
        lines.append("질의: " + ", ".join(f"{state}{'/' + reason if reason else ''}×{n}" for (state, reason), n in queried.items()))
    if report is None:
        lines.append("리포트: 없음 (complete가 접수되지 않았다)")
    else:
        lines.append(
            f"리포트: {report.get('report_status')} · pose_report="
            + json.dumps(report.get("pose_report"), ensure_ascii=False)
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="벤더 세션 샘플을 v2.5 API에 그대로 재생한다.")
    parser.add_argument("--api", default="http://127.0.0.1:8120")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--query", action="store_true", help="시퀀스마다 pose_recomm 질의")
    parser.add_argument("--fill-averages", action="store_true", help="complete의 빠진 평균 필드를 CSV로 채움")
    parser.add_argument("--skip", default="", help="업로드하지 않을 시퀀스(쉼표 구분)")
    parser.add_argument("--out", type=Path, default=None, help="결과 CSV·JSON 폴더")
    parser.add_argument("--tag", default="replay")
    args = parser.parse_args(argv)

    sample = load_sample(args.sample_dir)
    rows, report = replay(
        sample, api=args.api, query=args.query, fill_averages=args.fill_averages,
        skip=[s.strip() for s in args.skip.split(",") if s.strip()],
    )
    print(summarize(rows, report))
    if args.out is not None:
        write_outputs(rows, report, args.out, args.tag)
        print(f"저장: {args.out / (args.tag + '_requests.csv')}, {args.out / (args.tag + '_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
