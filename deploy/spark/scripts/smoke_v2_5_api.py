#!/usr/bin/env python3
"""배포된 v2.5 공개 API 15개를 순서대로 두드려 계약대로 답하는지 확인한다.

표준 라이브러리만 쓴다. 프로젝트 의존성이 없는 Spark 호스트의 python3(3.10+)에서
그대로 돈다(`bundle_manifest.py`와 같은 전제). `smoke_answer_contract.py`는
`/copilot/query`의 지식 답변 하나만 보므로 그 둘은 겹치지 않는다.

판정은 "200이 왔다"가 아니라 **계약이 약속한 값**을 본다. 예를 들어 시퀀스 수집은
`sequence_data_ack.stored`와 `frames_accepted == frames_received`까지 본다 —
2026-10-01에 그 둘이 false/0인데도 HTTP는 200이었다.

매 실행마다 `--run-id`(기본: 타임스탬프)로 식별자를 새로 만든다. 같은
`(session_id, sequence_id)`에 다른 본문을 보내면 계약상 충돌이라 저장되지 않으므로,
재실행이 직전 실행과 부딪히지 않게 하기 위한 것이다.

    python3 smoke_v2_5_api.py --base-url http://127.0.0.1:58000
    python3 smoke_v2_5_api.py --sequence-file ~/seq-telemetry-valid.json
    python3 smoke_v2_5_api.py --skip-llm          # knowledge_search 생략(느리다)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

TIMEOUT = 30.0
LLM_TIMEOUT = 180.0


class Outcome:
    """한 단계의 결과. `ok`가 False면 종료 코드가 1이 된다."""

    def __init__(self, name: str, status: int | None, ok: bool, detail: str) -> None:
        self.name = name
        self.status = status
        self.ok = ok
        self.detail = detail


def _request(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = TIMEOUT,
) -> tuple[int, Any, str]:
    """(HTTP 상태, 파싱된 JSON 또는 None, 원문 앞부분). 연결 실패는 상태 0이다."""
    req = urllib.request.Request(url, data=body, method=method)
    if content_type is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"
    text = raw.decode("utf-8", "replace")
    try:
        return status, json.loads(text), text[:400]
    except json.JSONDecodeError:
        return status, None, text[:400]


def _post_json(url: str, payload: Any, *, timeout: float = TIMEOUT) -> tuple[int, Any, str]:
    return _request(
        url,
        method="POST",
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        content_type="application/json",
        timeout=timeout,
    )


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    """`/sessions/complete`는 multipart/form-data다(명세서 p.12)."""
    boundary = f"----smoke{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    for key, (filename, data, mime) in files.items():
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        parts.append(head + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _envelope_ok(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("status") == "SUCCESS"


def _data(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _sample_sequence(run_id: str, session_id: str, user_id: str, path: str | None) -> dict[str, Any]:
    """실물 본문이 있으면 식별자만 바꿔 쓰고, 없으면 최소 계약 본문을 만든다.

    최소 본문의 `sequence_setting`/`sequence_result`는 빈 객체다 — 명세서 p.4가 요구하는
    것은 **키의 존재**이고, 벤더 실물 샘플의 `sequence_result`도 `{}`다.
    """
    if path:
        with open(path, encoding="utf-8") as fh:
            body = json.load(fh)
        body["session_id"] = session_id
        body["user_id"] = user_id
        body["sequence_id"] = f"SEQ-{run_id}"
        return body
    return {
        "session_id": session_id,
        "user_id": user_id,
        "scenario_id": "1-A",
        "zone_id": "12",
        "sequence_id": f"SEQ-{run_id}",
        "sequence_setting": {},
        "sequence_result": {},
        "sequence_info": [
            {
                "index": i + 1,
                "time": "2026-08-07T14:05:00.000Z",
                "tool_pos_rot": {"pos": [0.1 * i, 1.2, 0.8], "rot": [0.0, 0.0, 0.0]},
                "gun_metrics": {"speed": 300.0, "distance": 250.0, "spraytime": 100 * i},
            }
            for i in range(5)
        ],
    }


def run(args: argparse.Namespace) -> list[Outcome]:
    base = args.base_url.rstrip("/")
    api = f"{base}/api/v1"
    run_id = args.run_id
    user_id = f"W-{run_id}"
    session_id = f"1_{run_id}_1"
    results: list[Outcome] = []

    def step(name: str, status: int | None, ok: bool, detail: str = "") -> None:
        results.append(Outcome(name, status, ok, detail))

    # --- 0. 배포 확인 ---------------------------------------------------
    status, spec, raw = _request(f"{base}/openapi.json")
    paths = sorted(spec["paths"]) if isinstance(spec, dict) and "paths" in spec else []
    step(
        "openapi 경로 14개",
        status,
        len(paths) == 14,
        f"{len(paths)}개" + ("" if paths else f" / {raw[:120]}"),
    )

    # --- 1. 사용자 ------------------------------------------------------
    status, payload, raw = _post_json(
        f"{api}/users",
        {"user_id": user_id, "name": "스모크", "skill_level": "novice", "dominant_hand": "right"},
    )
    step("POST /users", status, status == 200 and _envelope_ok(payload), raw[:120])

    status, payload, raw = _request(f"{api}/users/{user_id}")
    step(
        "GET /users/{user_id}",
        status,
        status == 200 and _data(payload).get("user_id") == user_id,
        raw[:120],
    )

    status, payload, raw = _request(f"{api}/users")
    # 이 엔드포인트의 `data` 는 객체가 아니라 프로필 배열이다.
    rows = payload.get("data") if isinstance(payload, dict) else None
    step(
        "GET /users",
        status,
        status == 200 and isinstance(rows, list) and any(r.get("user_id") == user_id for r in rows),
        f"{len(rows)}명" if isinstance(rows, list) else raw[:120],
    )

    # --- 2. 세션 시작 ---------------------------------------------------
    status, payload, raw = _post_json(
        f"{api}/sessions/start",
        {
            "session_id": session_id,
            "user_id": user_id,
            "scenario_id": "1-A",
            "started_at": "2026-10-01T06:00:00.000Z",
            "worker": {"skill_level": "novice", "handedness": "right"},
        },
    )
    step("POST /sessions/start", status, status == 200 and _envelope_ok(payload), raw[:120])

    # --- 3. 시퀀스 수집 -------------------------------------------------
    body = _sample_sequence(run_id, session_id, user_id, args.sequence_file)
    frames = len(body.get("sequence_info") or [])
    status, payload, raw = _post_json(f"{api}/telemetry/sequence", body)
    ack = _data(payload).get("sequence_data_ack") or {}
    step(
        "POST /telemetry/sequence (저장)",
        status,
        status == 200 and ack.get("stored") is True and ack.get("frames_accepted") == frames,
        f"stored={ack.get('stored')} {ack.get('frames_accepted')}/{ack.get('frames_received')}"
        if ack
        else raw[:200],
    )

    status, payload, _ = _post_json(f"{api}/telemetry/sequence", body)
    ack = _data(payload).get("sequence_data_ack") or {}
    step(
        "POST /telemetry/sequence (같은 본문 재전송 → duplicate)",
        status,
        status == 200 and ack.get("duplicate") is True,
        f"duplicate={ack.get('duplicate')}",
    )

    conflicting = dict(body)
    conflicting["zone_id"] = "13" if body.get("zone_id") != "13" else "14"
    status, payload, _ = _post_json(f"{api}/telemetry/sequence", conflicting)
    ack = _data(payload).get("sequence_data_ack") or {}
    step(
        "POST /telemetry/sequence (같은 키·다른 본문 → 저장 안 함)",
        status,
        status == 200 and ack.get("stored") is False and ack.get("duplicate") is False,
        f"stored={ack.get('stored')} duplicate={ack.get('duplicate')}",
    )

    status, payload, raw = _post_json(f"{api}/copilot/analyze-sequence", {**body, "mode_id": 1})
    step("POST /copilot/analyze-sequence", status, status == 200 and _envelope_ok(payload), raw[:120])

    # --- 4. 질의 -------------------------------------------------------
    status, payload, raw = _post_json(
        f"{api}/copilot/query",
        {
            "user_id": user_id,
            "session_id": session_id,
            "task_type": "pose_recomm",
            "prompt": "자세를 평가해 줘",
            "context": f"SEQ-{run_id}",
        },
    )
    data = _data(payload)
    pose = data.get("pose_context") if data.get("pose_context") is not None else data.get("context_status")
    step(
        "POST /copilot/query (pose_recomm)",
        status,
        status == 200,
        json.dumps(pose, ensure_ascii=False)[:200] if pose is not None else raw[:200],
    )

    if args.skip_llm:
        step("POST /copilot/query (knowledge_search)", None, True, "--skip-llm 으로 생략")
    else:
        status, payload, raw = _post_json(
            f"{api}/copilot/query",
            {
                "user_id": user_id,
                "task_type": "knowledge_search",
                "prompt": "에폭시 프라이머의 재도장 간격은?",
            },
            timeout=LLM_TIMEOUT,
        )
        answer = _data(payload).get("answer")
        step(
            "POST /copilot/query (knowledge_search)",
            status,
            status == 200 and isinstance(answer, str) and bool(answer.strip()),
            (answer or raw)[:200].replace("\n", " "),
        )

    # --- 5. 테스트 라우트와 SSE ------------------------------------------
    events: list[str] = []

    def _listen() -> None:
        try:
            req = urllib.request.Request(f"{api}/stream/anomalies/{session_id}")
            with urllib.request.urlopen(req, timeout=args.sse_wait + 5) as resp:
                deadline = time.monotonic() + args.sse_wait
                while time.monotonic() < deadline:
                    line = resp.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", "replace").strip()
                    if text.startswith("data:"):
                        events.append(text[5:].strip())
                        return
        except Exception as exc:  # 연결 자체가 안 되면 아래에서 실패로 잡힌다
            events.append(f"__error__ {type(exc).__name__}: {exc}")

    listener = threading.Thread(target=_listen, daemon=True)
    listener.start()
    time.sleep(1.0)  # 구독이 붙은 뒤에 쏜다

    for case in ("normal", "abnormal-vertical", "abnormal-horizontal"):
        status, payload, raw = _post_json(f"{api}/test/sequence/{case}?session_id={session_id}", {})
        step(f"POST /test/sequence/{case}", status, status == 200 and _envelope_ok(payload), raw[:100])

    listener.join(timeout=args.sse_wait + 6)
    got = events[0] if events else ""
    step(
        "GET /stream/anomalies/{session_id} (SSE)",
        None,
        bool(got) and not got.startswith("__error__"),
        got[:200] if got else f"{args.sse_wait}초 안에 이벤트 없음",
    )

    # --- 6. 세션 마감과 리포트 -------------------------------------------
    status, payload, raw = _post_json(
        f"{api}/sessions/end",
        {
            "session_id": session_id,
            "user_id": user_id,
            "scenario_id": "1-A",
            "ended_at": "2026-10-01T06:30:00.000Z",
        },
    )
    step("POST /sessions/end", status, status == 200 and _envelope_ok(payload), raw[:120])

    csv_bytes = b"index,time,speed\n1,2026-10-01T06:00:00Z,300\n"
    data_bytes, content_type = _multipart(
        {
            "session_id": session_id,
            "user_id": user_id,
            "scenario_id": "1-A",
            "vr_score": "88",
            "is_pass": "true",
            "avg_thickness_um": "120.5",
            "avg_speed_mms": "300.0",
            "avg_distance_mm": "250.0",
        },
        {"telemetry_csv": ("telemetry.csv", csv_bytes, "text/csv")},
    )
    status, payload, raw = _request(
        f"{api}/sessions/complete", method="POST", body=data_bytes, content_type=content_type
    )
    report_id = _data(payload).get("report_id")
    step(
        "POST /sessions/complete (multipart)",
        status,
        status == 200 and bool(report_id),
        f"report_id={report_id}" if report_id else raw[:200],
    )

    if report_id:
        status, payload, raw = _request(f"{api}/sessions/{session_id}/reports/{report_id}")
        step(
            "GET /sessions/{id}/reports/{rid}",
            status,
            status == 200 and _envelope_ok(payload),
            json.dumps(_data(payload).get("report_status"), ensure_ascii=False),
        )
        status, payload, raw = _request(f"{api}/reports/{report_id}")
        step("GET /reports/{report_id}", status, status == 200 and _envelope_ok(payload), raw[:120])
    else:
        step("GET /sessions/{id}/reports/{rid}", None, False, "report_id가 없어 건너뜀")
        step("GET /reports/{report_id}", None, False, "report_id가 없어 건너뜀")

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:58000")
    parser.add_argument("--run-id", default=time.strftime("%m%d%H%M%S"))
    parser.add_argument("--sequence-file", default=None, help="실물 시퀀스 본문(JSON). 식별자만 교체해 쓴다")
    parser.add_argument("--skip-llm", action="store_true", help="knowledge_search 생략")
    parser.add_argument("--sse-wait", type=float, default=15.0)
    parser.add_argument("--json", action="store_true", help="표 대신 JSON으로 낸다")
    args = parser.parse_args(argv)

    results = run(args)

    if args.json:
        print(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "run_id": args.run_id,
                    "ok": all(r.ok for r in results),
                    "steps": [
                        {"name": r.name, "status": r.status, "ok": r.ok, "detail": r.detail}
                        for r in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        width = max(len(r.name) for r in results)
        for r in results:
            mark = "OK  " if r.ok else "FAIL"
            code = f"{r.status}" if r.status is not None else "-"
            print(f"{mark} {r.name.ljust(width)}  {code:>4}  {r.detail}")
        failed = [r.name for r in results if not r.ok]
        print()
        print(
            f"{len(results) - len(failed)}/{len(results)} 통과"
            + (f" | 실패: {', '.join(failed)}" if failed else "")
        )

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
