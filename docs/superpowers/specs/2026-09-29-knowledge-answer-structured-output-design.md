# knowledge_search 답변 구조화 출력 설계 (2026-09-29)

- 대상: `POST /api/v1/copilot/query`의 `task_type="knowledge_search"` 응답에서 `answer` 문자열과 `citations` 배열을 **구조적으로 보장**하는 설계
- 문제 제기: 지식 검색 답변이 정규화되어 있지 않음 → guided decoding 필요 여부 판단
- 선행: `docs/ai_copilot_v2_5_deployment_contract.md` §18.8, `docs/superpowers/plans/2026-09-17-v25-contract-migration.md`
- 범위 밖: 다른 4개 `task_type`(현재 `not_available`), 텔레메트리 5개 분석 인터페이스, 자세/시퀀스 설계(별도 문서 `2026-09-29-sprayxr-runtime-alignment-design.md`)
- 표기: 수치는 실측이며 표본을 밝힌다. 추정은 `[추정]`.

## 0. 설계 원칙

1. **JSON 봉투는 대상이 아니다.** `CopilotResponse`/`CopilotQueryData`는 서버가 Pydantic으로 직렬화한다. 모델은 봉투를 한 글자도 쓰지 않으므로 여기에 제약 생성을 걸 대상이 없다. 제약의 대상은 `answer` **문자열 내부**다.
2. **학습 분포를 떠나지 않는다.** v6.1 LoRA는 `[답변]`/`[근거]` 텍스트 스캐폴드로 SFT됐다. 같은 구조를 강제하는 문법을 쓰면 분포 이동이 거의 없다. JSON으로 바꾸면 이동이 크다(§3.1).
3. **사후 필터보다 디코드 시점 차단을 우선한다.** 다만 요청마다 문법을 다시 컴파일해야 하는 차단은 지연이 측정되기 전까지 채택하지 않는다. 지연은 이미 제품 지적사항이다.
4. **정규화 함수는 전역 함수(total function)여야 한다.** 파싱 실패를 원문 통과로 처리하면, 정규화가 필요한 경우에 정확히 아무 일도 하지 않는다.
5. **`citations`는 "검색한 것"이 아니라 "근거로 쓴 것"이다.** 두 집합을 한 필드에 섞지 않는다.

## 1. 측정

표본: `/output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e/seed_{42..49}/agentic_responses.jsonl`, **240건**(8시드 × 30문항), v6.1 NF4, `top_k_rerank=8`, 코퍼스 청크 10,829개(`/data/chunked/chunks.jsonl`).

**주의:** 이 240건은 도구 경로의 5섹션 시스템 프롬프트로 생성됐고, 패스트 경로의 `fast_system`(2섹션)이 아니다. 따라서 아래 수치는 "명시된 출력 구조에 대한 모델의 일반적 준수율"과 "인용 행동"을 재며, 패스트 경로를 그대로 측정한 것은 아니다. 패스트 경로 재측정은 §5의 게이트가 담당한다.

### 1.1 구조 준수율

| 형태 | 건수 | 비율 |
|---|---|---|
| 규범 스캐폴드 일치 | 196 | 81.7% |
| 섹션 헤딩은 있으나 비규범 | 35 | 14.6% |
| 섹션 헤딩 자체가 없음 | 9 | 3.8% |

- **44/240 = 18.3%**가 프롬프트가 지시한 구조를 벗어난다.
- 그중 **23건**은 `[근거]`를 두 번 쓴다. 프롬프트가 "각 항목은 정확히 1회"라고 명시한 항목이다.
- 도구를 쓰지 않은 부분집합(패스트 경로에 가까운 148건)만 봐도 **25/148 = 16.9%**가 벗어난다.
- 상위 라벨 시퀀스: `분석|판정|원인|조치|근거` 169, `분석|판정|조치|근거` 24, `분석|판정|원인|조치|근거|근거` 13.

### 1.2 인용 정합

| 항목 | 실측 |
|---|---|
| 답변 본문에 DOC id가 1개 이상 등장 | 240/240 (100%) |
| 본문의 DOC id 개수 | 1개 141건, 2개 33건, 3개 66건 (중앙값 1) |
| `citations` 배열 길이 | 8 → 216건, 9 → 12건, 10 → 12건 |
| 검색 증거 밖의 id를 인용 | **24/240 = 10.0%** (1개 12건, 2개 12건) |
| 코퍼스에 존재하지도 않는 id | 1/240 = 0.4% (`DOC-doc_0036-C0277-C0003`) |

`citations` 길이가 항상 8 이상인 것은 `top_k_rerank=8`의 검색 결과 전체가 무조건 들어가기 때문이다. 본문이 실제로 기대는 id는 중앙값 1개이므로 **`citations`는 약 5.6배 과보고**한다.

## 2. 결함 확정

| # | 위치 | 내용 | 증거 |
|---|---|---|---|
| **D1** | `trainer/scripts/live_response_contract.py:48-52` | `normalize_fast_knowledge_response`가 응답이 *이미* 규범 형태일 때만 섹션을 걷어내고, 그 외에는 원문을 반환한다. `_split_sections`가 `None`이거나 `_is_canonical_diagnostic_scaffold`가 거짓이면 그대로 나간다 | §1.1의 18.3%가 무가공으로 클라이언트에 도달 |
| **D2** | `trainer/scripts/agent_pipeline.py:630-637` | `_extract_citations`가 본문 id와 **검색 결과 전체**를 합집합한다. 사용 여부와 무관하게 8개가 항상 들어간다 | `citations` 길이 8/9/10, 본문 id 중앙값 1 |
| **D3** | 같은 함수 632행 | 본문에서 정규식으로 뽑은 id를 **검색 집합 소속 검사 없이** 앞쪽에 넣는다. LoRA가 기억한 id가 증거에 없어도 계약 응답에 실린다 | 증거 밖 인용 10.0% |
| **D4** | `trainer/demo_api/copilot_router.py:153` | `use_rag="N"`이면 `citations=[]`이지만 **본문 텍스트는 걸러지지 않는다.** 문서를 하나도 주지 않은 요청의 답변이 `DOC-xxx-Cxxxx`를 제시할 수 있다 | 코드 경로. 라이브 재현은 §7 T1의 시험이 담당 |
| **D5** | 계약 §18.8 마지막 항 | "`citation_id`는 최종 Query 응답의 `citations`와 동일한 값"이 D2·D3와 양립하지 않는다. 검색 수준 집합과 인용 수준 집합을 같은 이름으로 묶었다 | 문서 대조 |

D1은 지적하신 "정규화되지 않음"의 직접 원인이고, D2~D4는 그 아래에 있던 계약 위반이다.

## 3. 설계

### 3.1 제약 방식: JSON이 아니라 문법(EBNF)

두 후보를 비교한다.

| | JSON 스키마 (`response_format: json_schema`) | 텍스트 문법 (`guided_grammar`, EBNF) |
|---|---|---|
| 구조 보장 | 있음 | 있음 |
| v6.1 SFT 분포와의 거리 | **큼** — 학습 타깃은 `[답변]`/`[근거]` 평문이었고 JSON 이스케이프·중괄호는 학습되지 않았다 | **거의 없음** — 학습 타깃의 형태를 그대로 문법으로 옮긴다 |
| 파싱 | `json.loads` | 문법이 보장하므로 정규식 분해가 전역 함수가 된다 |
| 계약 `answer`로의 변환 | 서버가 렌더링 | 거의 그대로 사용 |
| 한국어 리터럴 토큰 정렬 | 무관 | **확인 필요** — `[답변]`은 다중바이트이고 xgrammar의 토큰 마스크가 이를 쪼갤 수 있다 |

**채택: 텍스트 문법.** 원칙 2에 따른다. JSON은 폴백으로만 남긴다(`structured_output: json`).

### 3.2 문법 정의

증거가 있는 경우(`use_rag="Y"`이고 히트 ≥ 1):

```ebnf
root        ::= answer_sec "\n" evidence_sec
answer_sec  ::= "[답변] " body
body        ::= (* 1..600자, "\n"과 "[" 제외 *)
evidence_sec::= "[근거] " cite (", " cite){0,2}
cite        ::= "DOC-" doc_id "-C" digit digit digit digit
```

증거가 없는 경우(`use_rag="N"` 또는 히트 0):

```ebnf
root        ::= answer_sec
answer_sec  ::= "[답변] " body
```

증거 불충분 분기는 `body`의 대안으로 고정 문구를 넣는다.

```ebnf
body        ::= "제공된 문서에 해당 정보가 없습니다." | free_body
```

설계 귀결:

- `[근거]` 중복(§1.1의 23건)이 **문법적으로 불가능**해진다.
- 섹션 누락·순서 뒤바뀜·추가 섹션(`[판정]` 등)이 불가능해진다. 패스트 경로에서 판정을 지어내는 것을 프롬프트가 아니라 문법이 막는다.
- `use_rag="N"`에서 `cite`가 문법에 없으므로 **D4가 구조적으로 사라진다.**
- 두 문법은 정적이므로 **프로세스 기동 시 1회 컴파일**하고 캐시한다. 요청당 컴파일 비용이 없다.

### 3.3 인용 소속 검사 (D3)

문법은 id의 *형태*만 보장한다. 소속은 서버가 판단한다.

```
cited        = 문법이 보장한 [근거] 파싱 결과 (0..3개, 순서 유지)
retrieved    = 이번 요청의 rerank 히트 chunk_id 집합 (|retrieved| ≤ 8)
validated    = [c for c in cited if c in retrieved]      # 순서 유지, 중복 제거
dropped      = [c for c in cited if c not in retrieved]
```

- `dropped`가 비어 있지 않으면 **요청당 WARNING 1건**으로 `request_id`, `dropped`, `|retrieved|`를 남긴다. 이 비율이 §1.2의 10.0%에 대한 라이브 관측치가 된다.
- `validated`가 비었고 `retrieved`가 비어 있지 않으면 → 답변은 그대로 두고 `citations=[]`로 응답한다. 답변을 버리지 않는다. 근거 없는 답변임을 `citations`의 공집합이 이미 표현한다.
- `answer` 문자열의 `[근거]` 줄은 **서버가 `validated`로 다시 렌더한다.** 본문과 배열이 불일치할 수 없다.
- `retrieved`가 비어 있으면(`use_rag="N"` 또는 히트 0) `[근거]` 줄을 붙이지 않는다.

**선택 사항(측정 후 결정):** `cite`를 `retrieved`의 8개 id 열거로 바꾸면 증거 밖 인용이 디코드 시점에 불가능해진다. 대가는 요청당 문법 컴파일이다. 원칙 3에 따라 **컴파일 p95가 30ms 미만으로 측정될 때만** 채택한다. 측정 없이는 채택하지 않는다.

### 3.4 정규화 함수의 전역화 (D1)

`normalize_fast_knowledge_response`를 대체한다. 새 함수는 반드시 `(answer, citations, diagnostics)`를 반환하며 **원문 통과 분기를 갖지 않는다.**

```
parse_knowledge_answer(raw, retrieved) -> KnowledgeAnswer
  1. 문법 보장 경로: [답변]/[근거]로 분해. 성공 → validated 렌더.
  2. 문법 미적용/실패 경로(structured_output: off, 또는 문법 지원 없는 서버):
     2a. 섹션 분해 성공 → [답변]|[분석]을 본문으로, [판정]/[원인]/[조치] 폐기,
         [근거] 중복은 병합, 본문의 DOC id는 제거하고 validated로 재렌더
     2b. 섹션 분해 실패 → 전체를 본문으로 보고 "[답변] " 접두를 붙인다.
         본문의 DOC id는 제거한다.
  3. 어떤 경로든 반환값은 정확히 "[답변] ...(\n[근거] ...)?" 형태다.
  4. 2a/2b로 내려간 건수는 diagnostics.repair_path에 기록하고 WARNING으로 남긴다.
```

2b가 있기 때문에 이 함수는 어떤 입력에도 계약 형태를 반환한다. **정규화 실패는 이제 존재하지 않고, 대신 복구 경로 통계가 남는다.**

### 3.5 설정 스위치와 폴백

`trainer/configs/rag_config_spark_vllm.yaml`:

```yaml
llm:
  structured_output: grammar    # grammar | json | off
  structured_output_fallback: on # 서버가 문법을 거부(400)하면 off로 1회 재시도
```

- `off`는 §3.4의 2a/2b 복구 경로만 쓴다. 지금 동작보다 엄격하되 문법 없이 돈다.
- 배포된 vLLM 이미지가 문법을 지원하지 않는 상황(`VLLM_IMAGE`는 외부 지정이다)에서 서비스가 죽지 않아야 한다. 그래서 폴백은 기본 켜짐이고, 폴백 발동은 **기동 시 1회 ERROR 로그 + `/api/ready`의 `llm.structured_output` 필드**로 드러낸다.
- §5의 게이트가 실패하면 `off`로 되돌린다. 이 스위치가 되돌릴 수단이다.

### 3.6 vLLM 배선

`OpenAIBackend.build_payload`(`trainer/scripts/openai_backend.py:207`)에 선택적 필드를 추가한다. 기존 키는 건드리지 않는다.

```python
if grammar is not None:
    payload["extra_body"] = {"guided_grammar": grammar, "guided_decoding_backend": "xgrammar"}
```

호출자(`agent_pipeline._generate_direct`)가 경로와 `retrieved`에 따라 두 정적 문법 중 하나를 고른다. 도구 경로는 이번 범위가 아니므로 문법을 주지 않는다.

**기동 전 실물 확인 항목** (지금 단정할 수 없다 — 이미지 버전이 `VLLM_IMAGE` 환경변수로 외부 지정이다):

1. `guided_grammar`(또는 당해 버전의 등가 필드명)를 `/v1/chat/completions`가 받는지.
2. `[답변]`·`[근거]` 같은 다중바이트 리터럴이 토크나이저 경계와 충돌하지 않는지. 충돌하면 문법을 바이트 단위로 재작성해야 한다.
3. `chat_template_kwargs.enable_thinking=false`와 동시 사용 시 템플릿이 `<think>`를 전혀 내지 않는지. 낸다면 문법 `root`가 선행 `<think>...</think>`를 허용해야 한다. 패스트 경로는 `enable_thinking=false`이므로 기본 가정은 "내지 않는다"이고, 이는 **확인 항목이다.**
4. `repetition_penalty=1.15`와의 상호작용. 문법이 토큰을 1개로 강제하는 구간에서는 패널티가 무력하고 자유 `body` 구간에서만 작용한다. 형태상 문제는 없으나 본문 품질은 §5가 잰다.
5. 문법 컴파일 p95 — §3.3의 동적 문법 채택 여부를 결정한다.

## 4. 계약 개정안

| 조항 | 현행 | 개정 | 영향 |
|---|---|---|---|
| §18.8 마지막 항 | "`citation_id`는 최종 Query 응답의 `citations`와 동일한 값" | 검색 인터페이스의 `result.citations`는 **검색 집합**(≤8), Query 응답의 `citations`는 **답변이 인용한 집합**(⊆ 검색 집합, 0~3개)이다. 포함 관계만 보장한다 | SIM이 받는 배열이 평균 8개에서 1~3개로 **줄어든다.** 키 집합은 불변 |
| §18.8 | 없음 | `use_rag="N"`인 Query의 `citations`는 항상 `[]`이고, `answer` 본문에도 문서 id가 나타나지 않는다 | D4 봉인의 계약 명문화 |
| Query 응답 `answer` | "자연어 답변" | `knowledge_search`의 `answer`는 정확히 `[답변] <본문>` 또는 `[답변] <본문>\n[근거] <id>[, <id>]*` 형태다 | SIM이 렌더 규칙을 고정할 수 있다 |
| `CopilotQueryData` | 필드 고정 | **변경 없음.** 문법 적용 여부·복구 경로는 로그와 `/api/ready`로만 노출한다 | 계약 표면을 넓히지 않는다 |

`citations` 축소는 SIM 화면에 보이는 변화이므로 **사용자가 SIM 측에 통지해야 하는 항목**이다. 기술적으로는 답변이 기대지 않은 문서 7개를 근거로 표시하던 것을 멈추는 것이다.

## 5. 품질 게이트

게이트는 `trainer/scripts/run_rag_eval.py`(50문항, `/data/eval/eval_set.jsonl`)를 쓴다. 이 평가 세트는 문항마다 `required_chunks`를 갖고 있어 **인용 정확도를 직접 잴 수 있다.** 현재 `run_rag_eval.py`는 이 지표를 계산하지 않으므로 추가한다.

| 지표 | 성격 | 기준 |
|---|---|---|
| `manual_score` (5축 평균, `run_rag_eval.py:232`) | 단계형 통계 게이트 | Welch 95% CI, Δ=0.2. CI 하한 > −Δ면 통과, 상한 < −Δ면 실패, 그 외 8→18회 확장. FP8 때와 같은 절차 |
| `rouge_l`, `bertscore_f1` | 참고 | 보고만 |
| **구조 준수율** | 경성 게이트 | `off` 경로에서도 100%. 문법 경로에서 100% 미만이면 배선 결함이다 |
| **증거 밖 인용률** | 경성 게이트 | 0%. §1.2의 10.0%가 0으로 가는 것이 이 작업의 정의다 |
| **인용 재현율** (`validated ∩ required_chunks`) | 신설·보고 | 기준선 먼저 측정. 축소된 `citations`가 정답 청크를 놓치는지 본다 |
| p50/p90 지연 | 경성 게이트 | 기준선 대비 +5% 이내. 문법 마스크는 디코드 시점 비용이므로 [추정] 무시 가능하지만 측정으로 확인한다 |
| `completeness`(길이비 0.5~3.0, `run_rag_eval.py:223`) | 감시 | 문법이 본문을 짧게 만들면 이 축이 먼저 떨어진다. 하락 시 `body` 상한(600자)을 조정한다 |

측정 시 같은 컨테이너에서 다른 CPU 부하를 돌리지 않는다(과거 100.2s vs 42s 오염 사례).

## 6. 실패 모드

| 실패 | 징후 | 대응 |
|---|---|---|
| 배포 vLLM이 문법 미지원 | 첫 요청 400 | `structured_output_fallback`이 `off`로 1회 재시도. ERROR 1건 + `/api/ready`에 표시. 서비스는 계속된다 |
| 다중바이트 리터럴 토큰 경계 충돌 | 생성이 즉시 멈추거나 깨진 글자 | 문법을 바이트 단위 재작성. §7 T3의 시험이 먼저 잡는다 |
| `<think>` 누출로 문법 위반 | 생성 0토큰 또는 400 | `root`에 선행 think 허용 분기 추가 |
| 문법이 본문을 과도하게 짧게 만듦 | `completeness` 하락 | `body` 상한 조정 후 재측정 |
| 통계 게이트 실패 | CI 상한 < −0.2 | `structured_output: off`. §3.4의 복구 경로만 남는다. **그래도 D1~D4는 고쳐진 상태다** |

마지막 행이 이 설계의 중요한 성질이다. 문법이 실패해도 결함 수정은 남는다.

## 7. 구현 순서

문법과 무관하게 옳은 수정(T1·T2)이 먼저다. 문법(T3~T5)이 게이트에서 떨어져도 T1·T2는 유지된다.

| # | 작업 | 산출 | 시험 게이트 |
|---|---|---|---|
| **T1** | `_extract_citations`를 `retrieved` 소속 검사 기반으로 분리: `cited_validated`와 `retrieved_all`을 별도 반환. `citations`는 전자 | `agent_pipeline.py` | 단위: 본문 id ⊄ 검색집합 → 드롭·경고. `use_rag=False` → 본문 DOC id 제거·`citations=[]` |
| **T2** | `parse_knowledge_answer` 전역 함수로 `normalize_fast_knowledge_response` 대체(§3.4). 라우터 배선 | `live_response_contract.py`, `copilot_router.py` | 단위: §1.1의 9개 라벨 시퀀스 전부를 입력으로 넣어 **모두** 계약 형태 반환. 중복 `[근거]` 병합. 섹션 없는 입력도 `[답변]` 접두 |
| **T3** | 문법 문자열 2종 + 컴파일 캐시. vLLM 실물에서 §3.6의 5개 항목 확인 | `trainer/scripts/answer_grammar.py` | 실물 스모크: 문법 적용 요청 10건이 100% 계약 형태. 컴파일 p95 기록 |
| **T4** | `build_payload` 선택적 `extra_body` + config 스위치 + 폴백 + `/api/ready` 노출 | `openai_backend.py`, `rag_config_spark_vllm.yaml`, `readiness.py` | 단위: 스위치 3값 각각의 payload 동등성. 400 응답 → 1회 폴백. 기존 payload 키 불변 |
| **T5** | `run_rag_eval.py`에 구조 준수율·증거 밖 인용률·인용 재현율 추가 | `run_rag_eval.py` | 단위: 세 지표의 경계 사례 |
| **T6** | 게이트 실행: `off` vs `grammar`, 8회 → 필요시 18회 | 보고서 | §5 표 전체 |
| **T7** | 통과 시 `grammar` 기본값. 계약 문서 §4 개정 반영 | `docs/ai_copilot_v2_5_deployment_contract.md` | `test_spark_deploy_contract.py` 갱신 |

T1·T2는 순수 Python이고 컨테이너 pytest로 끝난다. T3만 실물 vLLM이 필요하며, DGX Spark 또는 개발 서버의 기동된 서버를 쓴다. x86 호스트에서 Spark 이미지를 빌드하지 않는다.

## 8. 미확인 전제

1. **패스트 경로의 실제 이탈률.** §1의 18.3%는 도구 프롬프트 표본이다. `fast_system`(2섹션)은 지시가 더 단순하므로 이탈률이 낮을 [추정]. T6이 기준선으로 측정한다. 낮게 나와도 D1~D4는 코드 사실이므로 작업은 유효하다.
2. **v6.1 LoRA가 `[답변]`/`[근거]` 2섹션 형태로 SFT됐는지.** 학습 데이터에서 확인해야 한다. 5섹션만 학습됐다면 2섹션 문법도 약한 분포 이동이며, §5의 Δ=0.2가 이를 잡는다.
3. **배포 vLLM의 문법 지원 여부와 필드명.** §3.6의 확인 항목 1.
4. **`citations` 축소를 SIM이 감당하는지.** 계약 개정 통지 대상(§4).
5. **동적 문법의 컴파일 비용.** §3.3의 선택 사항은 이 측정 없이 채택하지 않는다.
