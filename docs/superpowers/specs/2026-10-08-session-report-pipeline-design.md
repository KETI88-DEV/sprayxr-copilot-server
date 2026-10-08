# 세션 종료 리포트 처리 구조 설계

**상태:** 초안
**작성일:** 2026-10-08
**적용 범위:** `POST /api/v1/sessions/complete`, `GET /api/v1/sessions/{session_id}/reports/{report_id}`, `GET /api/v1/reports/{report_id}`
**정본:** `docs/XR Painting Simulator AI Co-Pilot Service API 정의서 - 대화형 API 명세서.pdf` p.11-16
**선행 문서:** `2026-08-07-session-report-response-contract-design.md` — 응답 **모양**은 그 문서가 정한다. 이 문서는 그 모양을 **채우는 파이프라인**을 정한다.

## 1. 왜 쓰는가

엔드포인트 3개는 있고 응답 계약도 명세서와 맞는데, **그 사이를 잇는 처리기가 없다.** 코드에서 확인한 현재 상태:

| 선행 문서 §2가 요구하는 것 | 현재 구현 |
|---|---|
| ① 세션 입력 CSV를 원천 데이터의 정본으로 보관 | **버린다.** `v1_router.py` 가 `len(await telemetry_csv.read())` 로 **바이트 수만 세고** 내용을 쓰지 않는다 |
| ② 시퀀스별 판정을 파생 레코드로 기록 | 없다. `sequences` 테이블은 본문만 들고 판정 결과 열이 없다 |
| ③ 실시간 Query가 파생 레코드 사용 | 없다 |
| ④ 같은 알고리즘 버전의 파생 레코드를 집계 | 없다 |
| ⑤ 레코드가 없으면 정본 CSV에서 재분석 | 없다 |

그리고 **상태 전이가 없다.** `ReportStatus = Literal["pending","complete","failed"]` 인데 `"pending"` 으로 쓰는 곳이 한 군데뿐이고 다른 값으로 바꾸는 코드가 없다. 한 번 만들어진 리포트는 영원히 `pending` 이다.

이것 자체는 아직 계약 위반이 아니다. 명세서 p.12가 그 상태를 직접 정의한다.

> 현재 `pending` 은 요청과 heatmap 요약이 저장되었지만 실제 최종 분석 결과는 아직 생성되지 않았음을 뜻합니다. **실제 분석 작업이 연결되기 전에는 `complete` 로 표시하지 않습니다.**

즉 지금은 "정의된 미구현"이다. 이 문서는 그 작업을 연결한다.

## 2. 범위 — 구조는 전부, 구현은 자세부터

세 영역 중 **지금 실제로 분석 가능한 것은 `pose_report` 하나**다(§8). 그래도 구조는 세 영역을 모두 전제하고 설계한다. 파생 레코드 스키마와 실행 모델은 영역 종류와 무관하게 같은 뼈대를 쓰므로, 나중에 나눠 설계하면 두 번 고치게 된다.

* **설계한다** — 업로드 보관, 실행 모델, 파생 레코드, 집계, 상태 전이, 실패·재기동, 폴링, 세 영역 전부
* **먼저 구현한다** — `pose_report` 경로
* **자리만 만든다** — `coating_report`, `sequence_report` 는 설계대로 `unavailable` + 안정적 `reason_code`. 엔진이 생기면 §7의 같은 자리에 꽂는다

## 3. 바꿀 수 없는 제약

설계 선택지를 좁히는 사실들이다. 전부 실물에서 확인했다.

**C1 — 쓸 수 있는 디렉터리는 하나다.** `deploy/spark/compose.yaml` 의 agent-api 마운트 중 쓰기 가능한 것은 `${RESULTS_DIR}:/output/eval_results` 뿐이고 나머지(`/models/*`, `/output/indexes`, `/output/merged`)는 전부 `:ro` 다. 업로드 파일과 DB는 **반드시** 그 아래 있어야 한다. `sequences.db` 기본 경로가 이미 `/output/eval_results/sequences.db` 다.

**C2 — 단일 워커 + SQLite(WAL).** 이관 계획 `2026-09-17-v25-contract-migration.md` E7의 결정이다.

> 영속 저장소는 SQLite(표준 라이브러리, WAL, 단일 워커)로 한다. 세션, 시퀀스, 이상 이벤트, 질의, 리포트를 저장한다. **재기동하면 `pending` 리포트를 다시 큐에 넣는다.**

재기동 복구는 이미 결정돼 있다(§10에서 구체화).

**C3 — 같은 프로세스가 27B 추론을 돌린다.** 라이브 모드의 생성 슬롯과 CPU를 나눠 쓴다. 분석이 이벤트 루프를 막으면 `/copilot/query` 의 TTFT가 직접 나빠진다.

**C4 — 업로드 크기.** 코퍼스 605세션 CSV의 중앙값 14.2 MB, 최대 155.3 MB. 실물 세션 CSV도 이 자릿수로 본다.

**C5 — 폴링 간격은 명세서가 정했다.** p.14: "비동기로 생성되는 종합 리포트의 상태를 주기적으로(**2~5초 간격**) 조회".

## 4. 전체 구조

```
POST /sessions/complete
  │  (요청 스레드에서 끝낸다 — 분석을 기다리지 않는다)
  ├─ 업로드를 디스크에 쓴다           → /output/eval_results/completions/<report_id>/
  ├─ completions 행을 pending 으로 넣는다 (경로·sha256·크기 포함)
  └─ 200 + report_status="pending" (계약이 200이다. 202가 아니다)

[리포트 워커]  단일 데몬 스레드. completions 테이블이 큐다.
  │
  ├─ pending 행을 하나 집는다 (claimed_at 으로 선점)
  ├─ 세션의 시퀀스를 모은다 (sequences 테이블)
  ├─ 영역별 분석기를 돈다
  │     pose     : 파생 레코드가 있으면 그대로, 없으면 지금 채점해 레코드를 남긴다
  │     coating  : 입력 없음 → unavailable
  │     sequence : 엔진 없음 → unavailable
  ├─ 집계 결과를 저장한다
  ├─ llm_briefing 생성 (available 영역만 근거로)
  └─ report_status = complete | failed

GET /reports/{report_id}         저장된 결과를 읽어 ReportDetail 로 돌려준다
GET /sessions/{id}/reports/{id}  상태만 돌려준다 (2~5초 폴링)
```

## 5. 업로드 보관 (R1)

**R1-a 위치.** `/output/eval_results/completions/<report_id>/telemetry.csv`, 선택적으로 `height_map.png`. C1 때문에 다른 선택지가 없다. `report_id` 는 서버가 만드는 불투명 식별자(`RPT-<hex12>`)라 경로에 그대로 쓸 수 있다 — 클라이언트가 준 문자열을 경로에 넣지 않는다.

**R1-b 스트리밍 저장.** `await telemetry_csv.read()` 로 전부 메모리에 올리지 않는다. C4의 최대 155 MB를 한 번에 올리면 27B 추론과 메모리를 다툰다. 청크로 읽어 쓰면서 sha256을 같이 계산한다.

**R1-c 크기 상한.** 256 MB. 관측 최대(155.3 MB)보다 넉넉하되 유한하다. 넘으면 쓰던 파일을 지우고 계약의 ERROR 래퍼로 413을 돌려준다 — **행을 만들지 않는다.** 반쯤 쓰인 업로드로 리포트를 만들면 무엇을 분석한 것인지 알 수 없다.

**R1-d 기록.** `completions` 에 열을 더한다.

| 열 | 형 | 뜻 |
|---|---|---|
| `csv_path` | TEXT | 저장 경로. **상대 경로**로 둔다 — 마운트 지점이 바뀌어도 레코드가 썩지 않는다 |
| `csv_sha256` | TEXT | 저장 직후 계산. 재분석 전에 다시 확인한다 |
| `csv_bytes` | INTEGER | 응답의 `telemetry_csv_bytes` 와 같은 값 |
| `image_path` | TEXT NULL | `height_map_image` 는 선택이다(p.12 "레거시") |

**R1-e 보존.** 삭제 정책은 **이번 범위 밖**이다. 지우는 코드를 넣지 않는다 — 원천 데이터를 지우는 일은 운영자가 정할 일이고, 잘못 지우면 §8.1의 재분석이 불가능해진다. 대신 디스크 사용량을 로그로 남겨 운영자가 보게 한다.

## 6. 실행 모델 (R2)

**R2-a 큐는 테이블이다.** 별도 큐 미들웨어를 넣지 않는다. `completions` 의 `report_status='pending'` 행이 곧 대기열이다. 이렇게 하면 C2의 "재기동하면 pending 리포트를 다시 큐에 넣는다"가 **공짜로** 성립한다 — 재기동 후 워커가 테이블을 보면 그 행들이 그대로 있다. 큐를 따로 두면 큐와 테이블이 어긋날 수 있고, 그 불일치를 맞추는 코드가 또 필요해진다.

**R2-b 전용 스레드 한 개.** `asyncio.create_task` 로 이벤트 루프에서 돌리지 않는다. 분석은 CSV 파싱과 채점이라 CPU를 쓰고, C3 때문에 루프를 막으면 질의 지연이 직접 나빠진다. 스레드 하나면 C2의 단일 워커 전제와도 맞고 SQLite 쓰기가 자연히 직렬화된다.

**R2-c 선점.** 열 세 개를 더한다.

| 열 | 형 | 뜻 |
|---|---|---|
| `claimed_at` | TEXT NULL | 워커가 집은 시각(ISO-8601 UTC, 예 `2026-10-08T02:30:00Z`) |
| `attempts` | INTEGER | 시도 횟수. 기본 0 |
| `failed_reason` | TEXT NULL | `failed` 일 때의 안정적 코드 |

집을 때 `UPDATE ... SET claimed_at=?, attempts=attempts+1 WHERE report_id=? AND claimed_at IS NULL` 로 한다. 단일 워커라도 이 조건을 쓰는 이유는 **재기동 직후 유령 선점**을 막기 위해서다(§10).

**R2-d 주기.** 워커는 2초마다 대기열을 본다. C5의 클라이언트 폴링(2~5초)보다 빠르면 충분하다. 큐가 비면 그대로 잔다 — 바쁜 대기를 하지 않는다.

## 7. 파생 레코드 (R3)

선행 문서 §2②가 요구하는 "시퀀스별 판정"을 담는다. 지금은 어디에도 저장하지 않는다.

**새 테이블 `sequence_findings`**

| 열 | 형 | 뜻 |
|---|---|---|
| `session_key` | TEXT | `sequences` 와 같은 키 |
| `sequence_id` | TEXT | 〃 |
| `area` | TEXT | `pose` \| `coating` \| `sequence` |
| `algorithm_version` | TEXT | 예 `v3-arm-6/2026-09-15`. **버전이 다르면 재사용하지 않는다**(선행 문서 §2⑤) |
| `analysis_status` | TEXT | `available` \| `unavailable` |
| `reason_code` | TEXT NULL | `unavailable` 일 때 |
| `metrics` | TEXT | JSON. 영역별 수치(자세면 `similarity_score`, `reach_m`, `reference_band`, `support`) |
| `corrections` | TEXT | JSON 배열 |
| `created_at` | TEXT | ISO-8601 UTC |

PRIMARY KEY `(session_key, sequence_id, area, algorithm_version)`. 같은 시퀀스를 다른 버전으로 다시 재면 **덮어쓰지 않고 행이 늘어난다** — 어느 버전으로 낸 점수인지 추적할 수 있어야 한다.

**R3-a 누가 쓰는가.** 두 경로가 같은 테이블에 쓴다.

1. `POST /copilot/query` 의 `pose_recomm` — 훈련 중 사용자가 물어볼 때. 이미 채점하고 있으므로 결과를 버리지 말고 남긴다.
2. 리포트 워커 — 레코드가 없거나 버전이 다른 시퀀스를 그 자리에서 채점한다(선행 문서 §2⑤).

`POST /telemetry/sequence` 는 **쓰지 않는다.** 그 경로는 올리기만 한다(명세서 p.4) — 2026-10-01에 확정한 배치다.

**R3-b 멱등.** 워커가 같은 리포트를 두 번 처리해도 같은 키에 같은 값이 들어간다. 재시도가 레코드를 부풀리지 않는다.

## 8. 영역별 분석기

### 8.1 `pose_report` — 구현한다

명세서 p.16이 요구하는 필드는 `analysis_status`, `average_similarity_score`(0.0~1.0), `guideline` 셋이다.

* **입력** — 세션의 시퀀스 전부. `sequence_findings` 에 `area='pose'` 이고 버전이 현행인 행이 있으면 그걸 쓰고, 없으면 MCP `/pose_estimation/sequence` 로 채점해 남긴다.
* **집계** — `average_similarity_score` = `analysis_status='available'` 인 시퀀스의 `similarity_score` **산술 평균**. 분모는 available 인 것만이다. `unavailable` 시퀀스를 0점으로 세면 **자세를 못 잰 것과 못한 것이 섞인다** — 선행 문서 §3의 "값을 추정하지 않는다"와 어긋난다.
* **available 이 하나도 없으면** → 영역 전체가 `unavailable`, `reason_code` 는 가장 많이 나온 시퀀스 사유를 그대로 올린다(`tool_not_held` 등). 새 사유를 만들지 않는다.
* **`guideline`** — 집계된 교정에서 만든다. 빈도 상위 교정 축을 사람 말로 적는다. **LLM이 수치를 새로 만들지 않는다**(선행 문서 §3).
* **근거** — 이 경로는 이미 돈다. 2026-10-01 측정: CSV 재생 두 세션 6구간에서 저장 6/6, `analysis_status` available 6/6, `similarity_score` 0.5278~0.7654, 교정 문장 생성 확인.

### 8.2 `coating_report` — 입력이 없다

명세서 p.12가 계산 방법을 명시한다.

> `height_map_image`: 과거 최종 Height Map 업로드 호환용. 신규 방식에서는 사용하지 않으며, **서버는 각 시퀀스의 `cur_sequence_heatmap` 을 누적하여 최종 도막 상태를 판단**

그런데 **그 필드가 오지 않는다.** 실물 벤더 샘플(`docs/SprayXR_CoPilot_샘플_2026-09-21.json`)의 `sequence_result` 는 `{}` 이고, `cur_sequence_heatmap` 은 우리 코드 어디에서도 수집하지 않는다.

따라서 `analysis_status="unavailable"`, `reason_code="heatmap_absent"`. **벤더 확인 항목이다**(§12). 누적기 자체는 입력이 오면 바로 붙일 수 있게 자리를 둔다 — `sequence_findings` 의 `area='coating'` 행에 시퀀스별 heatmap 요약을 적고 워커가 합산한다.

`coating_distribution_index`(표준 도막 분포 지수)의 **정의가 명세서에 없다**. 입력이 오면 그때 정의를 받아야 한다. 우리가 지어내지 않는다.

### 8.3 `sequence_report` — 엔진이 없다

필요한 것은 `adherence_rate`, `direction_adherence_rate` 두 비율이다. 선행 문서 §2가 이미 적었다.

> 현재 `sequence_validation` MCP는 다음 권장 zone을 계산하지만 **순서 이탈 유형과 세션 준수율을 계산하지 않는다.**

그 상태 그대로다. `analysis_status="unavailable"`, `reason_code="engine_unavailable"`.

구현하려면 ① 시나리오의 `recommended_flow` 정본, ② 시퀀스별 `zone_id` 방문 순서 정규화, ③ 방향 판정 기준이 필요하다. ②는 선행 문서 §4.1이 이미 설계했고 `sequences` 본문에 `zone_id` 가 있으므로 재료는 확보돼 있다.

### 8.4 `llm_briefing`

* `analysis_status="available"` 인 영역만 근거로 쓴다(선행 문서 §3).
* available 영역이 없으면 `null`. 빈 문자열이 아니다 — 계약이 `str | None` 이다.
* **수치·enum·처방 코드를 바꾸거나 새로 만들지 않는다.** 구조화된 결과를 문장으로 옮기기만 한다.
* 생성 실패(모델 미기동 등)는 **리포트를 실패시키지 않는다.** `llm_briefing=null` 로 두고 나머지를 낸다. 총평이 없다고 측정값을 버릴 이유가 없다.

## 9. 상태 전이와 실패

```
pending ──(워커가 집음)──> pending(claimed) ──┬─> complete
                                              └─> failed
```

**R4-a `complete` 의 뜻은 "리포트가 생성됐다"이지 "세 영역이 다 됐다"가 아니다.** 영역별 가용성은 각 영역의 `analysis_status` 가 말한다. 자세만 나와도 `complete` 다 — 그래야 클라이언트가 폴링을 멈추고 결과를 본다. 세 영역이 다 될 때까지 `pending` 으로 두면 엔진이 생기기 전까지 영원히 끝나지 않는다.

**R4-b 다만 지금(= pose 구현 전)은 `complete` 로 가지 않는다.** 명세서 p.12의 "실제 분석 작업이 연결되기 전에는 `complete` 로 표시하지 않습니다"를 지킨다. §8.1을 구현하는 커밋이 `complete` 전이를 함께 켠다.

**R4-c `failed` 는 파이프라인이 못 돈 경우만이다.** 영역이 `unavailable` 인 것은 실패가 아니다.

| `failed_reason` | 언제 |
|---|---|
| `source_data_missing` | `csv_path` 의 파일이 없다 |
| `source_data_corrupt` | sha256이 기록과 다르다 |
| `max_attempts_exceeded` | `attempts` 가 3을 넘었다 |
| `timeout` | 집은 지 15분이 지나도 안 끝났다 |

`source_data_missing` 은 선행 문서 §5가 이미 쓰는 코드명이다 — 새로 만들지 않는다.

**R4-d 재시도.** 예외로 끝나면 `claimed_at=NULL` 로 되돌려 다시 집히게 한다. `attempts` 가 3이 되면 `failed`. 무한 재시도는 같은 입력으로 같은 예외를 반복할 뿐이고 그동안 뒤의 리포트가 막힌다.

## 10. 재기동 복구 (R5)

C2가 요구한다. 워커가 뜰 때 한 번:

```sql
UPDATE completions SET claimed_at = NULL
 WHERE report_status = 'pending' AND claimed_at IS NOT NULL
```

프로세스가 죽으면 선점이 남는데 그걸 집을 워커는 이미 없다. 단일 워커이므로 기동 시점에 선점된 행은 전부 유령이다. 이 한 줄이 R2-c의 `claimed_at IS NULL` 조건을 다시 의미 있게 만든다.

**타임아웃 회수는 별도다.** 돌고 있는 워커가 `claimed_at` 이 15분 넘은 행을 보면 `timeout` 으로 `failed` 시킨다 — 되돌리지 않는다. 15분을 넘긴 작업은 재시도해도 같을 가능성이 높고, 되돌리면 큐가 영원히 돈다.

## 11. 검증 계획

게이트는 컨테이너 안의 `pytest trainer mcp_servers -q` 다(AGENTS.md). 아래는 **구현 전에 실패해야** 하는 단언들이다.

| 무엇 | 어떻게 |
|---|---|
| 업로드가 디스크에 남는다 | `csv_path` 의 파일 sha256이 올린 내용과 같다 |
| 상한을 넘기면 행이 안 생긴다 | 257 MB 업로드 → 413, `completions` 행 수 변화 없음, 파일 없음 |
| 워커가 pending 을 집어 끝낸다 | 큐에 넣고 한 바퀴 돌려 `complete` 와 `pose_report.analysis_status` 확인 |
| 평균의 분모가 available 뿐이다 | available 2건(0.6, 0.8) + unavailable 1건 → **0.7**. 0.466…이 아니다 |
| 파생 레코드를 재사용한다 | 같은 버전 레코드가 있으면 MCP를 **부르지 않는다**(호출 수 0) |
| 버전이 다르면 재사용하지 않는다 | 버전을 바꾸면 다시 부른다(호출 수 1) — 위 단언의 대조군 |
| 재기동이 유령 선점을 푼다 | `claimed_at` 을 박아두고 워커를 새로 띄우면 집힌다 |
| 재시도가 3에서 멈춘다 | 항상 터지는 분석기로 `attempts=3`, `failed_reason=max_attempts_exceeded` |
| 총평 실패가 리포트를 안 죽인다 | 생성이 터져도 `complete` + `pose_report` 는 살아 있다 |
| 엔진 없는 영역이 수치를 안 만든다 | `coating_report.coating_distribution_index is None` 이고 `analysis_status="unavailable"` |

**비종결 루프 주의.** 리포트 워커는 멈추지 않는 루프다. Starlette `TestClient` 로 몰지 않는다(AGENTS.md). 워커의 **한 바퀴**를 함수로 떼어 직접 부른다 — 데몬을 돌리지 않고도 전이를 시험할 수 있어야 한다.

## 12. 벤더 확인 항목

1. **`cur_sequence_heatmap`** — 명세서 p.12가 이걸 누적해 도막을 판정하라고 적었지만 실물 샘플의 `sequence_result` 는 `{}` 다. 보낼 계획인지, 형식이 무엇인지.
2. **`coating_distribution_index` 의 정의** — "표준 도막 분포 지수"의 계산식이 명세서에 없다.
3. **시나리오 `recommended_flow`** — `sequence_report` 의 준수율을 내려면 정본 순서가 필요하다.
4. (기존) **이동하며 분사한 global 캡처** — 자세 채점의 좌표 정합. 이 문서의 범위 밖이지만 `pose_report` 의 실물 품질을 정한다.

## 13. 작업 순서

각 단계가 독립적으로 커밋 가능하고 검증 게이트를 통과해야 한다.

1. `completions` 열 추가(R1-d, R2-c) + 이행. **기존 DB 파일에 대한 `ALTER TABLE` 을 반드시 넣는다** — 2026-10-01에 `sequences.body` 를 열만 추가하고 이행을 빠뜨려 배포가 500으로 깨졌다.
2. 업로드 스트리밍 저장과 상한(R1-a~c).
3. `sequence_findings` 테이블(R3).
4. 리포트 워커 뼈대(R2) — 집기·전이·재시도·재기동 복구. 분석기는 전부 `unavailable` 을 돌려주는 자리만.
5. `pose_report` 분석기와 집계(§8.1). 이 단계에서 `complete` 전이를 켠다(R4-b).
6. `pose_recomm` 질의가 파생 레코드를 남기게 한다(R3-a①).
7. `llm_briefing`(§8.4).
8. `GET /reports/{report_id}` 가 저장된 결과를 읽게 한다(지금은 `_unavailable_report` 고정).

1~4는 영역 엔진과 무관해서 지금 바로 할 수 있다. 5 이후가 자세 경로다.

## 14. 하지 않는 것

* **보존·삭제 정책** — R1-e. 운영자 결정이다.
* **작업 큐 미들웨어** — R2-a. 테이블이 큐다.
* **여러 워커** — C2가 단일 워커다. 늘리려면 E7부터 다시 본다.
* **`coating`·`sequence` 엔진** — §12가 풀리기 전에는 입력이 없다.
* **응답 모양 변경** — 선행 문서와 명세서 p.15-16이 정본이다. 현재 `ReportDetail` 은 이미 그 모양이라 바꿀 것이 없다.
