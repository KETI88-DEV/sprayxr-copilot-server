# Live 데모 레이턴시 예산 설계 (RAG 30s TTFT / Tool 180s)

## 1. 목표

원격 웹 데모에서 Qwen3.6-27B v6.1 파이프라인이 다음 예산을 지키게 한다.

| 경로 | 측정 대상 | 예산 |
|---|---|---|
| RAG 전용 (도구 호출 없음) | TTFT (첫 답변 토큰) | ≤ 30초 |
| 도구 호출 (tools ≥ 1) | 전체 완성 | ≤ 180초 |

동시 사용자는 **1명(시연용)** 을 전제한다. 배칭·큐잉 최적화는 범위 밖이다.

## 2. 확정된 요구사항 해석

브레인스토밍에서 확정한 정의이며, 구현·검증은 전부 이 정의를 따른다.

- **TTFT의 "첫 토큰"은 답변 토큰만 인정한다.** SSE `session`·`status`·`tool_result` 이벤트는 TTFT 계산에 포함하지 않는다. `message_delta` 첫 조각이 도달한 시각이 TTFT다.
- **RAG 전용 30초는 TTFT 기준**, **도구 호출 180초는 전체 완성 기준**이다. 두 경로의 측정 대상이 서로 다르다.
- 경로 판정은 실행 결과의 도구 호출 수로 한다. `tool_calls`가 비어 있으면 RAG 전용, 1건 이상이면 도구 호출 경로다. 라우터 이름(`fast`/`training_realtime`/`training_review`)으로 판정하지 않는다. 실측에서 `training_realtime` 라우트에도 도구 미호출 케이스가 다수 존재하기 때문이다.

## 3. 측정된 기준선 (2026-07-25, n=240 question-runs)

어댑터 `qwen36_27b_shipcoat_lora_v6_1`, A40 46GB, `backend: hf`, `load_in_4bit: true`.

| 경로 | n | p50 | p90 | max | 예산 | 준수 | 위반 |
|---|---|---|---|---|---|---|---|
| RAG 전용 (tools=0) | 154 | 31.4s | 59.1s | 93.0s | ≤30s | 43.5% | 87건 |
| 도구 호출 (tools≥1) | 86 | 124.6s | 158.3s | 191.2s | ≤180s | 95.3% | 4건 |

도구 경로 위반 4건: AQ021(191.2s, 188.4s), AQ011(185.4s), AQ004(182.3s).

관측 처리량 중앙값 **7.3 chars/s ≈ 4.6 tok/s**.

### 3.1 위 수치의 중요한 한계

위 값은 전부 **전체 완성 시간**이다. 현재 파이프라인에 토큰 스트리밍이 없어 TTFT를 측정할 수단 자체가 없다. 따라서 "RAG 전용 87건 위반"은 TTFT 위반이 아니라 완성 시간 위반이며, **RAG 경로의 TTFT는 아직 미지수다.**

RAG 전용 경로는 LLM 라운드가 1개이므로 `TTFT = prefill + 첫 토큰 1개`다. 시스템 프롬프트 1059토큰에 RAG 컨텍스트를 더한 prefill이 지배적이며, 완성 시간(31.4초 p50)보다 현저히 짧을 것으로 예상된다. 이 설계의 1차 목적은 그 값을 **추정이 아니라 실측**하는 것이다.

### 3.2 발견된 성능 저하 원인

모든 실행 로그 첫 부분에 다음이 기록되어 있다.

```
The fast path is not available because one of the required library is not installed.
Falling back to torch implementation.
```

Qwen3.6은 hybrid Gated-DeltaNet 아키텍처인데 `flash-linear-attention`·`causal-conv1d`가 설치되지 않아 linear-attention 경로가 순수 torch fallback으로 동작한다. 4.6 tok/s라는 낮은 처리량의 주원인으로 판단된다. 다만 본 설계에서는 이를 **조건부 후속 조치**로 두며, 이유는 5절에 기술한다.

## 4. 설계 범위

### 포함

1. 최종 종합(synthesis) 생성의 토큰 스트리밍
2. TTFT·전체 완성 시간 계측 및 SSE 노출
3. 도구 루프의 wall-clock 데드라인 가드
4. 라운드 종류별 `max_new_tokens` 차등 적용
5. 실제 HTTP/SSE 경로를 구동하는 레이턴시 벤치 하네스
6. 품질 회귀 방지 게이트(agentic 점수 유지 확인)

### 제외

- vLLM 백엔드 전환 및 MTP speculative decoding
- `flash-linear-attention`·`causal-conv1d` 설치
- 4B/27B 하이브리드 모델 라우팅
- 동시 다중 사용자 배칭·큐잉
- v7 SFT 품질 트랙 (별도 프로젝트)

### 제외 항목의 재도입 조건

6절의 검증 게이트가 실패할 때만 착수한다. 우선순위는 `flash-linear-attention` 설치 → vLLM 전환 → 하이브리드 라우팅 순이다. 근거 없이 선제적으로 착수하지 않는다.

## 5. 접근 전략과 그 근거

**스트리밍 + 데드라인 가드를 먼저 적용하고, 실측 후 필요할 때만 백엔드를 손댄다.**

이 순서를 택한 이유는 세 가지다.

1. TTFT 기준이므로 RAG 목표는 스트리밍 도입만으로 달성될 가능성이 높다. 한 번도 측정하지 않은 값을 근거로 의존성 리스크가 큰 커널 설치나 백엔드 교체를 먼저 하는 것은 순서가 틀렸다.
2. 도구 경로는 이미 95.3% 통과이며, 남은 4건은 시간 가드로 결정적으로 해결된다. 처리량 개선 없이도 100%가 된다.
3. 동시 사용자가 1명이므로 배칭·처리량 최적화의 실익이 작다. 이 저장소는 `unsloth`·`FlagEmbedding`·`vLLM`에서 이미 의존성 충돌 전례가 있어 리스크 대비 이득이 낮다.

## 6. 아키텍처

### 6.1 컴포넌트 변경

**A. 스트리밍 실행 경로 — `trainer/scripts/agent_pipeline.py`**

기존 `run()`은 완성 문자열을 반환하며 평가 스크립트가 이에 의존하므로 **변경하지 않는다.** 새 메서드를 추가한다.

```
AgentPipeline.run_streaming(...) -> AsyncIterator[PipelineEvent]
```

`PipelineEvent`는 `(kind, payload)` 형태이며 `kind ∈ {status, tool_result, sequence_update, delta, complete, error}`이다. 스트리밍은 **최종 종합 생성에만** 적용한다. 도구 결정 라운드의 출력은 사용자에게 노출되는 답변 텍스트가 아니므로 스트리밍 대상이 아니다.

HuggingFace `TextIteratorStreamer`를 사용하고, 생성은 별도 스레드에서 돌린 뒤 이터레이터를 async로 소비한다.

**B. 계측 — `agent_pipeline.py` + `trainer/demo_api/chat_service.py`**

- 요청 진입 시각 `t0` 기록
- 첫 `delta` 방출 시각에 `ttft_ms` 확정
- 종합 생성 시작 시각을 별도로 잡아 종료 시 `synthesis_ms` 확정
- 종료 시 `total_ms` 확정
- 세 값을 `complete` SSE 이벤트의 `data`에 포함하고 INFO 로그로도 남긴다

`StreamEvent` 계약의 `complete` 이벤트 `data`에 `ttft_ms`, `synthesis_ms`, `total_ms`, `tool_call_count`, `rag_chunk_count`를 담는다. 기존 필드는 유지하므로 프런트 Zod 스키마는 확장만 필요하고 파괴적 변경이 없다.

`synthesis_ms`는 단순 관측용이 아니라 C절의 데드라인 기본값을 실측으로 확정하는 데 쓰이는 필수 입력이다.

**C. 데드라인 가드 — `trainer/scripts/tool_calling_adapter.py`**

새 설정 `llm.tool_deadline_sec`(기본 150)을 도입한다. `run_tool_loop`가 매 라운드 시작 전 경과 시간을 확인하고, 초과 시 추가 라운드를 중단한 뒤 지금까지 확보한 도구 결과로 강제 종합에 들어간다.

기본값을 180이 아니라 150으로 잡은 이유는 데드라인 이후에도 종합 생성 시간이 추가로 필요하기 때문이다.

다만 종합 단계의 소요 시간은 **아직 단독으로 계측된 바 없다.** 현재 보유 데이터는 도구 미호출 1라운드 전체 완성 p50 31.4초뿐이며, 종합 단계는 이보다 컨텍스트가 길어 더 걸릴 수 있다. 따라서 150이라는 값은 초기 추정치이며, 구현 시 다음을 반드시 수행한다.

1. 종합 단계 소요 시간을 별도 계측 필드(`synthesis_ms`)로 기록한다.
2. 1차 벤치 결과의 `synthesis_ms` p95를 확인한다.
3. `tool_deadline_sec = 180 - synthesis_ms_p95` 로 재조정하고 그 값을 설정 기본값으로 확정한다.

즉 150은 1차 실행용 잠정값이고, 최종값은 실측으로 결정한다. 이 재조정을 거치지 않으면 180초 예산 준수를 보장할 수 없다.

데드라인으로 중단할 때는 `status` 이벤트에 `stage: "deadline_forced_synthesis"`를 실어 사용자와 로그 양쪽에 사유를 남긴다. 조용히 잘라내지 않는다.

**D. 라운드별 토큰 상한 — `tool_calling_adapter.py` + 27B 설정**

- `generation.max_new_tokens_tool` (기본 256) — 도구 결정 라운드
- `generation.max_new_tokens` (1024 유지) — 최종 종합

도구 호출 JSON은 대략 100~200토큰이면 충분한데 현재 1024를 균일 적용한다. 이는 모델이 태그를 내기 전에 장황한 서술을 붙일 여지를 주며, 실제로 AQ022에서 관측된 실패 양상(후보 도구를 산문으로 서술하고 `<tool_call>` 태그를 생략)과 정확히 일치한다.

다만 이 변경은 품질에 양방향 영향이 가능하다. 산문 여지를 줄여 태그 방출을 앞당길 수도 있고, 반대로 태그 방출 직전에 잘릴 수도 있다. **가정하지 않고 6.3의 게이트로 측정한다.** 설정 플래그로 제어하며 기본값은 활성이되, 게이트 실패 시 즉시 되돌린다.

**E. 벤치 하네스 — `trainer/scripts/bench_latency.py` (신규)**

파이프라인을 직접 호출하지 않고 **FastAPI 데모 API의 SSE 엔드포인트를 구동**한다. 사용자가 실제로 겪는 경로(HTTP + SSE + 서비스 계층)를 그대로 재도록 하기 위함이다.

- `--profile rag` — 도구 미호출 예상 문항
- `--profile tools` — 도구 호출 예상 문항
- `--profile mixed` — 30문항 전체
- `--repeat N` — 반복 횟수

산출물은 `/output/eval_results/latency_bench/<timestamp>/`에 `report.json`과 `SUMMARY.md`로 저장하며 다음을 담는다.

- TTFT·`synthesis_ms`·전체 완성의 p50/p90/p95/p99/max
- 경로별(RAG 전용 / 도구 호출) 예산 준수율과 위반 목록
- 문항별 표 (q_id, 실제 도구 호출 수, ttft_ms, synthesis_ms, total_ms, deadline_hit)
- `deadline_hit` 건수, `degraded`(스트리머 폴백) 건수, MCP 타임아웃 건수

`synthesis_ms` p95는 C절의 `tool_deadline_sec` 재조정에 직접 사용되므로 리포트 필수 항목이다.

프로파일 분류는 사전 라벨이 아니라 **실행 결과의 실제 도구 호출 수**로 사후 판정한다. 사전 예상과 실제가 다를 수 있으므로 리포트에 양쪽을 모두 남긴다.

### 6.2 데이터 흐름

```
bench_latency.py
  └─ POST /api/chat (SSE)
       └─ FastAPI demo_api
            └─ ChatService (live runner)
                 └─ AgentPipeline.run_streaming()
                      ├─ RAG 검색        → status         (t ≈ 1~3s)
                      ├─ 도구 라운드      → tool_result    (데드라인 가드 적용)
                      └─ 종합 스트리밍    → delta          ← 첫 delta 시각 = TTFT
                                          → complete      ← ttft_ms / synthesis_ms / total_ms 포함
```

### 6.3 오류 처리

| 상황 | 동작 |
|---|---|
| 데드라인 초과 | 추가 라운드 중단, `status{stage:"deadline_forced_synthesis"}` 방출 후 기존 도구 결과로 종합 |
| 스트리머 초기화 실패 | 기존 비스트리밍 `run()`으로 폴백, 전체 답변을 단일 `delta`로 방출, WARNING 로그, 벤치는 해당 실행을 `degraded`로 표시 |
| MCP 타임아웃 | 기존 동작 유지. 벤치가 별도 카운터로 기록해 레이턴시 통계를 오염시키지 않는다 |
| 생성 중 예외 | `error` 이벤트 방출, `total_ms`는 기록, `ttft_ms`는 `null` |

### 6.4 검증 게이트

세 가지를 모두 통과해야 완료로 인정한다.

| 게이트 | 기준 |
|---|---|
| RAG 전용 TTFT | p90 ≤ 30초, 30초 초과 0건 |
| 도구 경로 전체 완성 | p99 ≤ 180초, 180초 초과 0건 |
| 품질 회귀 없음 | agentic total ≥ 6.60 (기준선 6.744 ± 0.139 밴드 하단 이상) |

품질 게이트를 함께 두는 이유는 토큰 상한 축소와 데드라인 가드가 답변 품질을 떨어뜨릴 수 있기 때문이다. 레이턴시를 얻고 정확도를 잃는 교환을 무의식적으로 수용하지 않기 위한 장치다.

## 7. 테스트

**단위 테스트**
- 데드라인 가드: 가짜 시계로 경계 직전·직후 동작 확인, 강제 종합 진입 검증
- 토큰 상한: 도구 라운드에 `max_new_tokens_tool`, 종합에 `max_new_tokens`가 각각 전달되는지 검증
- TTFT 계측: 가짜 스트리머로 첫 delta 시각이 정확히 기록되는지 검증

**통합 테스트**
- FastAPI SSE가 `delta` 이벤트를 점진적으로 방출하는지 확인한다. 단일 대형 delta가 아니라 2건 이상이 시간 간격을 두고 도착해야 한다.
- `complete` 이벤트에 `ttft_ms`·`synthesis_ms`·`total_ms`가 포함되는지 확인한다.

**벤치(수동, GPU 필요)**
- v6.1 어댑터로 `--profile mixed --repeat 2` 실행 후 게이트 판정

기존 Python 47건·Vitest 16건·Playwright 2건은 전부 통과 상태를 유지해야 한다.

## 8. 프런트엔드 영향

`complete` 이벤트에 필드가 추가되므로 `demo-ui/src/contracts.ts`의 Zod 스키마를 확장한다. 기존 필드는 유지되므로 파괴적 변경이 아니다.

상태바에 마지막 응답의 TTFT를 표시해 시연 중에도 예산 준수를 눈으로 확인할 수 있게 한다. 이는 선택 사항이며 게이트 대상은 아니다.

## 9. 결정 기록

- TTFT의 첫 토큰은 답변 토큰만 인정한다. 진행 표시 이벤트는 제외한다.
- RAG 전용은 TTFT 30초, 도구 경로는 전체 완성 180초로 측정 대상이 서로 다르다.
- 경로 판정은 실제 도구 호출 수로 하며 라우터 이름으로 하지 않는다.
- 스트리밍은 최종 종합에만 적용한다. 도구 결정 라운드는 사용자 노출 텍스트가 아니다.
- 데드라인 기본값은 150초이며, 종합 생성 여유 30초를 더해 180초 예산을 만족시킨다.
- 백엔드 교체·커널 설치는 게이트 실패를 실증했을 때만 착수한다.
- 기존 `run()`은 평가 스크립트 호환을 위해 유지하고 `run_streaming()`을 병행 도입한다.

## 10. 완료 조건

- `run_streaming()`과 TTFT 계측이 동작하고 SSE로 노출된다.
- 데드라인 가드와 라운드별 토큰 상한이 설정으로 제어된다.
- `bench_latency.py`가 실제 HTTP/SSE 경로를 구동해 리포트를 생성한다.
- 6.4의 세 게이트를 모두 통과한다.
- 기존 자동 테스트가 모두 통과한다.
- 결과가 `PATCH_NOTE.md`와 `README.md`에 반영된다.
