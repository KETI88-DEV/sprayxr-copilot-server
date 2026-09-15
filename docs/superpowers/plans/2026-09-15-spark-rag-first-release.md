# Spark RAG 우선 릴리스 계획 (v6.1 병합 → FP8 → vLLM API → copilot/query)

- 작성: 2026-09-15
- 근거 조사: 같은 날 병렬 조사 2회(병합 구조·수치, vLLM API 전환 범위, Spark vLLM 호환, 지연 분해, RAG 우선 경로). 수치의 출처는 각 항목에 적는다.
- 상위 계획: `.omo/plans/dgx-spark-qwen-coaching-agent.md`(7월 v3.1). 이 문서가 그 계획의 T14·T16·T19 전제 가운데 아래 "결정"과 충돌하는 부분을 대체한다.

## 1. 결정

| # | 결정 | 근거 |
|---|---|---|
| D1 | **Spark 서빙 산출물은 FP8 block-128 (e4m3, dynamic activation)이다. BF16 병합본은 마스터이자 폴백으로 보관한다.** | 속도: Spark 디코딩은 메모리 대역폭(273GB/s)이 묶는다. 단일 스트림 상한 추정은 BF16 약 5.3 tok/s, FP8 약 10 tok/s이고, 둘 다 MTP로 추가 이득이 있다(추정, Spark 실측 전). 이번에 지적된 문제가 응답 지연이다. 정확도: 층 출력 기준 원본 대비 상대 오차는 채널별 FP8 2.6%, 현재 라이브 서빙인 NF4 9.2%다(층 단위 측정). 현재 기준 점수 6.744도 NF4에서 나왔다. block-128 실측(층 2·3·34·35·62·63, 39텐서): 가중치와 층 출력 오차가 모두 0.0264~0.0266이고, NF4는 0.092~0.105로 비율 0.25~0.29다(`quantize_fp8_block.py --report`, 사전에 정한 합격 규칙 ≤0.5를 통과). FP8은 부동소수 격자라 블록 크기와 무관하게 상대 오차가 거의 일정하므로, block-128이 채널별보다 더 정확한 것은 아니다. 형식은 공식 `Qwen/Qwen3.6-27B-FP8`과 같게 맞춘다(`quantization_config` 동일, `weight_scale_inv` 407개). 그러면 vLLM이 공식 체크포인트와 같은 경로로 로드한다. |
| D2 | v2.5 `POST /api/v1/copilot/query`에서 `knowledge_search`만 실제로 답한다. 나머지 `trajectory_recomm`, `pose_recomm`, `error_analysis`는 MCP가 완성될 때까지 **HTTP 200 + status SUCCESS + `context_status: "not_available"`** 로 응답하고 MCP를 호출하지 않는다. | 사용자 결정(2026-09-15). v2.5 스키마를 그대로 지켜 클라이언트 파싱이 깨지지 않는다. |
| D3 | `task_type`이 경로를 정한다(`knowledge_search` → `force_route="fast"`). 라우터의 의도 파악은 쓰지 않는다. | 사용자 요구(지연 단축). 라우터 비용 자체는 약 10µs라, 실제 절감은 LLM 라운드를 없애는 데서 나온다(라이브 로그: 도구 경로 약 92초 중 1차 라운드 약 62초). |
| D4 | 품질 게이트는 **Δ=0.2, 단계적 게이트: 각 8회 → 95% Welch CI 규칙(pass: 하한>−Δ, fail: 상한<−Δ, 그 외 각 18회로 확장 후 재판정)**이다(Δ는 agentic 30문항 총점 기준 허용 하락폭). NF4 기준선은 A40에서 라이브 API를 멈추고 다시 측정한다. | 사용자 승인(측정 창). Δ와 단계 규칙은 Spark FP8 후보 결과를 처음 보기 전까지만 바꿀 수 있고, 그 뒤에는 고정한다. 단계 규칙은 사용자 결정(2026-09-15)으로 결과 전에 사전등록했다(`deploy/spark/scripts/quality_gate.py`, `deploy/spark/README.md` §17). |
| D5 | 병합은 safetensors 샤드 단위로 직접 한다. transformers `merge_and_unload`/`save_pretrained`는 쓰지 않는다. | ForCausalLM 경로는 `mtp.*`와 visual을 잃는다. ForConditionalGeneration + PeftModel은 어댑터 키가 0/992 매칭된다(실측). |
| D6 | vLLM 호출은 텍스트 `<tool_call>` JSON 프로토콜을 유지한다. `tools` 파라미터를 보내지 않고 서버 tool parser도 쓰지 않는다. 요청마다 `chat_template_kwargs.enable_thinking=false`, temperature 0.3, top_p 0.8, top_k 20, repetition_penalty 1.15, max_tokens를 명시한다. | v6 SFT는 JSON `<tool_call>` 383줄, XML 0줄이다. native tools를 켜면 6.744 → 6.367로 떨어졌다. vLLM 기본 샘플링은 generation_config의 temperature 1.0 / top_p 0.95다. |
| D7 | 병합 산출물 쓰기, MCP import 수정(`simulator_csv`), 토크나이저 기반 프롬프트 예산 수정은 승인됐다. | 사용자 승인(2026-09-15). |

## 2. 작업 순서

```
T1 MCP import 수정 ─┐
T2 토큰 예산 수정 ───┼─► T5 NF4 기준선 재측정(8 seed, 라이브 API 중단 창, 코드 스냅숏 고정)
T3 BF16 샤드 병합 ──┴─► T4 FP8 block-128 변환(공식 형식)
T2 ─► T6 OpenAI 호환 백엔드 + /api/ready ─► T7 copilot/query(knowledge_search, not_available)
T4·T6·T7 ─► T8 Spark 검증(로딩 → 단일 응답 → 스트리밍 → 지연 → FP8 품질 게이트 → 재기동·소크)
```

- T5는 T1·T2 커밋 이후의 코드 스냅숏(git worktree)에서 돌린다. CPU를 많이 쓰는 T3·T4와 겹치지 않게 한다. CPU 경합은 HF generate를 느리게 해 `tool_deadline_sec` 초과로 점수를 오염시킨다(9/15 관찰: 같은 질문이 42초에서 100초로 늘었다).

## 3. 태스크별 완료 기준

- **T1** 새 프로세스에서 `import mcp_servers.main` 성공, 회귀 테스트(수정 전 실패), 전체 pytest 통과.
- **T2** 실제 토큰 수 기반 예산을 쓴다. 평가셋에서 기존 추정기와 트리밍 결과를 비교한 표를 남긴다. `llm.max_model_len`은 vLLM `--max-model-len`과 같은 값이며 근거를 주석에 적는다.
- **T3** `/output/merged/qwen36_27b_shipcoat_v6_1_bf16/`. 키 1199개와 샤드 배치가 원본과 같고, 바뀐 텐서는 정확히 496개이며 전부 재계산과 비트 일치한다. 나머지 703개는 바이트가 같다. `merge_manifest.json`(sha256)을 남긴다.
- **T4** `/output/merged/qwen36_27b_shipcoat_v6_1_fp8/`. 양자화 대상은 공식 FP8 index의 `weight_scale_inv` 대상 407개 키와 같다(mtp 포함, in_proj_a/b·visual·embed·lm_head·norm 제외). `quantization_config`도 공식과 같다. 역양자화 오차를 층별로 기록하고, NF4 대비 크게 작아야 한다. manifest를 남긴다. 품질 판정은 A40이 FP8을 지원하지 않으므로 T8에서 한다.
- **T5** NF4 + v6.1, 현재 설정, seed 42–49. 평균, SD, 문항별 표, tool_not_called, deadline 초과 수, 코드 커밋 해시를 남긴다.
- **T6** `llm.backend: openai`(httpx, vllm 패키지 불필요). D6 요청 규약을 지키고, 가짜 OpenAI 서버로 도구 루프·스트리밍·취소·오류 매핑을 테스트한다. `/api/ready`는 vLLM `/health`, `/v1/models`, RAG를 확인한다. 기존 HF 경로와 `run()` 평가 호환은 유지한다.
- **T7** v2.5 요청/응답 모델(`extra="forbid"`)과 표준 래퍼를 구현한다. 요구 사항은 다음과 같다.
  - `use_rag` 'Y'/'N'
  - `resolved_intents=[task_type]`
  - `session_id` 요청값 그대로 반환
  - `latency_ms`는 핸들러 벽시계 시간(세마포어 대기 포함)
  - GPU 세마포어 공유
  - MCP 접근 불가 상태에서도 knowledge_search 200을 확인하는 회귀 테스트
- **T8** 인수 기준 초안은 조사 결과 AC-1~AC-6이다. 추가 항목은 다음과 같다.
  - SM121 장시간 소크(vLLM #37431 계열 크래시 확인)
  - MTP 수락률
  - HF 토큰 ID와 vLLM `/tokenize` 동일성
  - FP8 30문항 품질 게이트가 D4 단계 규칙으로 pass(각 8회, inconclusive면 각 18회로 확장 후 재판정)

## 4. 열린 항목

- Spark 이미지 태그 확정: `vllm/vllm-openai:v0.29.0` arm64와 `nvcr.io/nvidia/vllm:26.08-py3` 중에서 고른다.
- citations 형식(청크 ID 그대로 / 문서 단위 / SPEC 매핑), answer의 `[근거]` 꼬리 제거 여부, `use_rag` 형식 불일치(v2.5 'Y'/'N' vs 계약 문서 boolean)를 KETI와 확인한다.
- `latency_ms` 목표: v2.5 예시의 180~210ms는 27B 생성으로는 불가능하므로 KETI와 다시 합의한다.
- 체감 지연 개선(실제 토큰 스트리밍, fast 경로 생성 토큰 축소)은 T7 이후 별도 태스크로 한다.
- `resolved_intents`는 `task_type` 원문(예 `trajectory_recomm`)을 그대로 돌려준다. v2.5 예시는 `trajectory_recommendation`처럼 풀어 쓴 문자열이므로 KETI 파싱과 맞는지 확인한다(리뷰 LOW, 6ca602f).
- 잠재 위험(현재 동작 영향 없음, 2026-09-15 확인): `agent_pipeline`은 `tool_calling_adapter`를 짧은 이름으로 import하고, `openai_backend`는 `trainer.scripts.tool_calling_adapter`에서 `ToolInfrastructureError`를 상속한다. 스크립트를 경로로 실행하면 `issubclass(OpenAIBackendUnavailableError, <짧은 이름>.ToolInfrastructureError)`가 False다(재현함). 지금은 이 타입으로 예외를 잡는 런타임 코드가 없다. 컨텍스트 초과와 백엔드 불가를 잡는 곳은 모두 `trainer.scripts.openai_backend` 이름을 쓴다. 앞으로 도구 루프에서 `except ToolInfrastructureError`를 추가하려면 먼저 import 이름을 하나로 통일한다.
- **T5 결과(2026-09-15 11:45Z~18:05Z, 스냅숏 965549e):** NF4 + v6.1, seed 42–49 → **total 6.704 ± 0.215**(최소 6.433, 최대 7.000). 차원별 평균: tool_selection 0.383, parameter 0.642, reasoning 1.788, context 2.000, structure 1.892. 도구 미호출은 seed당 17~20/30이고, 12개 문항은 8 seed 모두 미호출이었다. 7월 V1(nt=F, do_sample=T, 3 seed)은 6.744 ± 0.139에 미호출 17/18/17이었으므로 회귀가 아니다. 결과 경로는 `/output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e/`(`RUN_INFO.txt`에 입력 sha256 기록). 7월 보고서의 `ds`는 `generation.do_sample`을 뜻한다(`TOOL_ADAPTER_FORCE_SYNTHESIS`가 아니다). 이번 측정은 설정의 `do_sample: true`와 기본 강제 합성을 썼다.
- **D4 검정력 재계산:** 실측 SD 0.215에서 Δ=0.2를 α=0.05(양측), 검정력 0.8로 가려내려면 팔당 n ≈ 2·(1.96+0.84)²·σ²/Δ² ≈ **18**이 필요하다. 8 seed로 가려낼 수 있는 최소 차이는 약 0.30이다. 사용자 결정이 필요하다: (a) 두 팔 모두 약 18회로 늘린다(NF4 추가 10 seed ≈ 9시간, Spark FP8 18회), (b) Δ=0.3을 받아들이고 8회를 유지한다. 참고로 vLLM 쪽 반복은 seed가 요청에 전달되지 않아 독립 반복이다. 사용자 결정(2026-09-15): 8회로 먼저 보고 결정 → 위 단계 규칙으로 사전등록
- T5 기록: 평가 하네스는 42ddd16(07-27)부터 경로 실행 시 `trainer` import에 실패했다(cf95389에서 수정). 기준선은 `PYTHONPATH=<스냅숏>`으로 돌렸고 `RUN_INFO.txt`에 기록했다.
