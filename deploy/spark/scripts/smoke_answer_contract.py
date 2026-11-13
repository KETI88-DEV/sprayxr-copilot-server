#!/usr/bin/env python3
"""S4 스모크: 배포된 /api/v1/copilot/query 의 지식 답변 계약을 실물에서 검사한다.

저장소 루트에서 실행한다(``trainer`` 가 import 되어야 한다).

    cd <repo> && python3 deploy/spark/scripts/smoke_answer_contract.py off
    cd <repo> && python3 deploy/spark/scripts/smoke_answer_contract.py on

판정은 ``live_response_contract.is_contract_shaped`` 와 ``document_ids`` 를 **그대로
불러서** 한다. 그 모듈은 표준 라이브러리만 쓰므로 프로젝트 의존성이 없는 Spark 호스트
python3 에서도 돈다. 검사 규칙을 여기에 복제하지 않는 이유는, 복제본은 결국 구현과
어긋나면서도 계속 통과하기 때문이다.

지연은 **답변 길이로 정규화해서** 본다. 응답마다 출력 길이가 다르고 지연은
출력 토큰 수에 거의 비례하므로(§11.4: 디코드 99.6%), 원시 지연을 조건 간에 비교하면
길이 분산이 효과를 덮는다. 실제로 MTP 1.450배가 원시 지연 비교에서는 사라졌다.

게이트(설계 2026-09-29 §11 S4):
  - 계약 형태 전부 통과
  - 본문의 모든 문서 id 가 citations 안에 있음
  - use_rag="N" 응답에 문서 id 가 없음
같은 질문 묶음을 ``off`` 와 ``on`` 으로 두 번 돌려 비교하면 제약이 실제로 바꾼 것이 보인다.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from trainer.scripts.live_response_contract import document_ids, is_contract_shaped  # noqa: E402

DEFAULT_BASE = "http://127.0.0.1:58000"

# 앞 8건은 문서 근거를 쓰고, 뒤 2건은 use_rag="N" 으로 문서 id 누출을 본다.
QUESTIONS = (
    "저도막의 원인과 보수 방법은?",
    "블라스팅이 무엇인가?",
    "도막 두께 측정은 어떻게 하는가?",
    "에폭시 프라이머의 용도는?",
    "도장 전 표면처리 등급은 어떻게 구분하는가?",
    "과도막이 생기면 어떤 문제가 있는가?",
    "도장 작업 시 온습도 조건은?",
    "핀홀의 발생 원인은?",
    "스프레이 건과 피도면의 적정 거리는?",
    "방청 도료의 역할은 무엇인가?",
)
_RAG_COUNT = 8


def ask(url: str, label: str, index: int, question: str, use_rag: str) -> tuple[dict, int]:
    body = json.dumps(
        {
            "user_id": "smoke",
            "request_id": f"SMOKE-{label}-{index:03d}",
            "prompt": question,
            "task_type": "knowledge_search",
            "use_rag": use_rag,
        }
    ).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    return payload, int((time.monotonic() - started) * 1000)


def main(argv: list[str]) -> int:
    label = argv[1] if len(argv) > 1 else "run"
    base = argv[2] if len(argv) > 2 else DEFAULT_BASE
    url = base.rstrip("/") + "/api/v1/copilot/query"

    rows: list[dict] = []
    for index, question in enumerate(QUESTIONS, start=1):
        use_rag = "Y" if index <= _RAG_COUNT else "N"
        try:
            payload, latency_ms = ask(url, label, index, question, use_rag)
        except Exception as exc:  # noqa: BLE001 - 한 건이 실패해도 나머지를 계속 잰다
            rows.append({"i": index, "question": question, "use_rag": use_rag, "error": repr(exc)})
            print(f"{index:2d} ERROR {exc!r}")
            continue

        data = payload.get("data") or {}
        answer = str(data.get("answer", ""))
        citations = [str(c) for c in (data.get("citations") or [])]
        body_text = answer.split("\n[근거]")[0]
        chars = len(answer)
        row = {
            "i": index,
            "question": question,
            "use_rag": use_rag,
            "latency_ms": latency_ms,
            "answer": answer,
            "citations": citations,
            "answer_chars": chars,
            "ms_per_100_chars": round(latency_ms / chars * 100, 1) if chars else None,
            "shaped": is_contract_shaped(answer),
            "ids_within_citations": set(document_ids(answer)) <= set(citations),
            "body_lines": body_text.count("\n") + 1,
        }
        rows.append(row)
        ok = row["shaped"] and row["ids_within_citations"]
        print(
            f"{index:2d} {'OK  ' if ok else 'FAIL'} rag={use_rag} "
            f"lines={row['body_lines']} cites={len(citations)} "
            f"{latency_ms:>6}ms {chars:>4}자 {row['ms_per_100_chars']:>6}ms/100자"
        )

    done = [r for r in rows if "error" not in r]
    out = Path(f"smoke-{label}.json")
    if not done:
        # 전부 실패한 실행이 직전의 성공 결과를 덮어쓰지 않게 한다.
        print(f"응답 없음 — {out} 를 덮어쓰지 않았다")
        return 1
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    shaped = sum(1 for r in done if r["shaped"])
    within = sum(1 for r in done if r["ids_within_citations"])
    multiline = sum(1 for r in done if r["body_lines"] > 1)
    rag_rows = [r for r in done if r["use_rag"] == "Y"]
    norag = [r for r in done if r["use_rag"] == "N"]
    leaked = sum(1 for r in norag if document_ids(r["answer"]))
    latencies = sorted(r["latency_ms"] for r in done)

    print("=" * 58)
    print(f"label={label}  응답 {len(done)}/{len(rows)}")
    print(f"계약 형태             {shaped}/{len(done)}   (게이트: 전부)")
    print(f"본문 id in citations  {within}/{len(done)}   (게이트: 전부)")
    print(f"여러 줄 본문          {multiline}/{len(done)}  (0이면 다중 줄 패턴 미확인 유지)")
    if rag_rows:
        counts = sorted(len(r["citations"]) for r in rag_rows)
        print(f"use_rag=Y citations   min={counts[0]} median={counts[len(counts)//2]} max={counts[-1]}")
    print(f"use_rag=N 문서id 누출  {leaked}/{len(norag)}  (게이트: 0)")
    print(
        f"지연 p50={latencies[len(latencies)//2]}ms "
        f"p90={latencies[min(len(latencies)-1, int(len(latencies)*0.9))]}ms max={latencies[-1]}ms"
    )
    # 조건 간 비교는 이 줄로 한다. 원시 지연은 답변 길이 분산에 지배된다.
    norm = sorted(r["ms_per_100_chars"] for r in done if r["ms_per_100_chars"] is not None)
    total_ms = sum(r["latency_ms"] for r in done)
    total_chars = sum(r["answer_chars"] for r in done)
    print(
        f"길이 정규화  p50={norm[len(norm)//2]}ms/100자  "
        f"전체 {total_ms/total_chars*100:.1f}ms/100자 ({total_chars}자)"
    )
    print(f"저장: {out}")
    return 0 if (shaped == len(done) and within == len(done) and leaked == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
