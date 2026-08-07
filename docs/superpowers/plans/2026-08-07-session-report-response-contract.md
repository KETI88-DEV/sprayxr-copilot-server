# 세션 종료 리포트 응답 계약 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docs/ai_copilot_response_contract.md`의 세션 종료 리포트를 CSV 정본·파생 판정 하이브리드 방식과 크기가 제한된 순서·direction·자세 집계 계약으로 교정한다.

**Architecture:** 세션 입력 CSV를 재분석 가능한 정본으로 두고 시퀀스별 판정을 버전이 있는 파생 레코드로 기록한다. 최종 응답은 개별 순서 이탈 사례를 노출하지 않고 zone 단위 집계만 제공하며, 순서와 direction 준수율을 독립적으로 계산한다. 도막·순서·자세 context는 각각 하나의 `guideline`을 제공하고 `llm_briefing`이 available context 전체를 종합한다.

**Tech Stack:** Markdown, JSON, Python 3.11 표준 `json` 모듈, Git

## Global Constraints

- 설계 정본은 `docs/superpowers/specs/2026-08-07-session-report-response-contract-design.md`다.
- 구현 대상은 `docs/ai_copilot_response_contract.md` 하나다. 런타임 Python·TypeScript 코드는 변경하지 않는다.
- 세션 입력 CSV가 원천 데이터의 정본이며 시퀀스별 판정은 `algorithm_version`을 가진 파생 레코드다.
- `MonitorRecorder`의 500건 링버퍼와 `SimSessionStore.anomaly_count`는 리포트 영속 저장소가 아니다.
- `sequence_data_ack.stored=true`는 CSV 또는 프레임 원본 저장 성공만 뜻한다.
- 순서 deviation enum은 `out_of_order | zone_skipped | zone_revisited | unexpected_zone`이다.
- 최종 리포트에 개별 순서 deviation 사례, `sequence_id` 목록 또는 무제한 `deviations[]`를 포함하지 않는다.
- direction enum은 `horizontal | vertical | diagonal | curved_irregular | stationary`다.
- 순서 `adherence_rate`와 `direction_adherence_rate`를 서로 섞지 않는다.
- 자세 집계는 현재 MCP의 `similarity_score`, `corrections`, `roll | pitch | yaw`만 사용한다.
- `priority_corrections`는 최대 3개다.
- `coating_report`, `sequence_report`, `pose_report`는 각각 하나의 `guideline`을 가진다.
- `llm_briefing`은 도막·순서·direction·자세의 available 결과를 모두 반영한다.
- 위치 ID는 `A-1`, `A-2`, `A-3`만 사용한다.
- `docs/Interface.pdf`는 검증 자료일 뿐 stage하거나 커밋하지 않는다.
- 모든 Python 검증은 저장소 루트에서 `ship-paint-trainer` 컨테이너로 실행한다.

---

### Task 1: 리포트 원천 데이터와 공통 생성 규칙 교정

**Files:**
- Modify: `docs/ai_copilot_response_contract.md:796`

**Interfaces:**
- Consumes: 설계 §2 현재 구현과 목표 계약의 경계, §3 최종 리포트 공통 규칙
- Produces: Task 2·3이 사용하는 CSV 정본, 파생 판정, `analysis_status`, `source_artifact_id`, `algorithm_version`, context별 `guideline` 규칙

- [ ] **Step 1: 기존 리포트 경계의 충돌 문구 식별**

`docs/ai_copilot_response_contract.md` §5.1~§5.4에서 다음 기존 문구를 교체 대상으로 확인한다.

```text
- 도막 수치와 분류는 콘텐츠센터, 순서 판정은 순서 검증 MCP, 자세 판정은 자세 추정 MCP의 결과를 그대로 사용한다.
- pose_report: 현재 구현은 누적 자세 평가를 지원하지 않으므로 ...
- sequence_report: ... deviations ...
```

현재 MCP의 단건 결과와 목표 세션 집계 계약을 구분하지 않는 위 문구를 유지하지 않는다.

- [ ] **Step 2: 하이브리드 원천 데이터 규칙 작성**

§5.1 생성 규칙에 다음 의미를 명시한다.

```text
세션 입력 CSV는 리포트 재분석의 정본이다. Trigger ON → OFF 구간마다 sequence_id, zone_id, 도막·순서·direction·자세 판정과 algorithm_version을 파생 레코드로 기록한다. 리포트는 동일 버전의 파생 레코드를 집계하고, 누락되었거나 버전이 다르면 CSV에서 다시 분석한다.
```

이어서 현재 구현 상태를 다음처럼 분리한다.

```text
현재 MonitorRecorder는 최대 500건의 메모리 관측 버퍼이고 SimSessionStore는 anomaly_count만 유지하므로 어느 것도 리포트의 영속 정본이 아니다.
```

- [ ] **Step 3: `sequence_data_ack.stored` 의미 고정**

§3.2 또는 §5.1에 다음 불변 규칙을 추가한다.

```text
sequence_data_ack.stored=true는 해당 시퀀스의 CSV 또는 프레임 원본 저장 성공을 의미하며, anomaly payload 영속 저장이나 최종 리포트 생성 완료를 의미하지 않는다.
```

정상·이상 판정 전체를 파생 레코드에 보관해야 평균과 준수율의 분모를 재현할 수 있다고 명시한다.

- [ ] **Step 4: 완료 리포트 공통 필드 표 교정**

§5.3에서 세 하위 리포트의 공통 필드를 다음처럼 정의한다.

| 필드 | 조건 | 설명 |
|---|---|---|
| `analysis_status` | 항상 | `available`, `not_available` |
| `reason_code` | 항상 | 분석 가능 시 `null`, 불가능 시 안정적인 사유 코드 |
| `algorithm_version` | `available` | 집계 알고리즘 버전 |
| `source_artifact_id` | `available` | CSV 또는 최종 히트맵 artifact ID |
| `guideline` | 항상 | 해당 context의 구조화 결과만 반영한 안내 |

`llm_briefing`은 모든 available context를 종합하되 구조화된 값을 재계산하지 않는다고 명시한다.

- [ ] **Step 5: Task 1 문서 검사**

Run:

```bash
GIT_MASTER=1 git diff --check -- docs/ai_copilot_response_contract.md
```

Expected: exit code 0, no output.

- [ ] **Step 6: Task 1 원자적 커밋**

```bash
GIT_MASTER=1 git add docs/ai_copilot_response_contract.md
GIT_MASTER=1 git commit -m "docs(api): 세션 리포트 원천 데이터 규칙 추가"
```

### Task 2: 순서 deviation과 direction 집계 교정

**Files:**
- Modify: `docs/ai_copilot_response_contract.md:833`

**Interfaces:**
- Consumes: Task 1의 CSV 정본·파생 판정·공통 하위 리포트 규칙
- Produces: 제한된 크기의 `sequence_report`, 네 deviation enum, 독립된 `direction_summary`, 순서 context `guideline`

- [ ] **Step 1: 방문 정규화 규칙 작성**

§5에 다음 예시와 규칙을 추가한다.

```text
원본 zone 흐름: A-1, A-1, A-3, A-2, A-1
평가 방문 흐름: A-1, A-3, A-2, A-1
최초 방문 흐름: A-1, A-3, A-2
```

연속된 동일 zone 버스트는 한 방문으로 압축하고, 다른 zone으로 이동한 뒤 돌아온 경우에만 재방문으로 센다. `first_visit_flow`는 각 zone의 최초 방문만 담아 길이를 zone 수 이하로 제한한다.

- [ ] **Step 2: deviation 4종과 우선순위 작성**

다음 표를 계약에 추가한다.

| `type` | 판정 조건 |
|---|---|
| `out_of_order` | 권장 흐름에 있지만 현재 기대 zone과 다른 순서로 최초 진입 |
| `zone_skipped` | 세션 종료까지 권장 zone을 한 번도 방문하지 않음 |
| `zone_revisited` | 다른 zone으로 이동한 뒤 이미 방문·완료한 zone으로 재진입 |
| `unexpected_zone` | `recommended_flow`에 없는 zone으로 진입 |

방문당 주 deviation 우선순위는 `unexpected_zone → zone_revisited → out_of_order`이고, `zone_skipped`는 종료 시 미방문 zone마다 추가한다고 명시한다.

- [ ] **Step 3: 순서 집계와 준수율 공식 작성**

기존 `visited_flow`, 무제한 `deviations` 중심 설명을 다음 필드로 교체한다.

```text
recommended_flow
first_visit_flow
evaluated_visit_count
on_track_visit_count
skipped_zone_count
adherence_rate
deviation_count
deviation_summary
zone_summary
```

공식과 불변식을 정확히 작성한다.

```text
adherence_rate
= on_track_visit_count / (evaluated_visit_count + skipped_zone_count)

on_track_visit_count + deviation_count
= evaluated_visit_count + skipped_zone_count

sum(deviation_summary.*.count)
= deviation_count
```

분모가 0이면 순서 결과를 추정하지 않고 `not_available`로 반환한다.

- [ ] **Step 4: direction 하위 상태와 집계 작성**

`sequence_report.direction_summary`에 다음 필드를 정의한다.

```text
analysis_status
reason_code
evaluated_sequence_count
matched_sequence_count
direction_error_count
direction_adherence_rate
affected_zone_ids
zone_summary
```

zone별 항목은 `zone_id`, `expected_direction`, 평가·일치·오류 건수, `direction_adherence_rate`, 다섯 enum을 모두 키로 가진 `observed_direction_counts`를 제공한다.

```text
direction_adherence_rate
= matched_sequence_count / evaluated_sequence_count

matched_sequence_count + direction_error_count
= evaluated_sequence_count
```

순서 집계가 available이어도 direction 기준이 없으면 `direction_summary.analysis_status="not_available"`로 반환할 수 있다고 명시한다. `distance_speed_error`, `trajectory_deviation`은 direction 준수율에 포함하지 않는다.

- [ ] **Step 5: 완료 응답의 `sequence_report` 교체**

설계 §9의 `sequence_report` JSON을 그대로 반영한다. 특히 다음 값을 유지한다.

```json
{
  "evaluated_visit_count": 37,
  "on_track_visit_count": 29,
  "skipped_zone_count": 0,
  "adherence_rate": 0.784,
  "deviation_count": 8,
  "direction_summary": {
    "analysis_status": "available",
    "reason_code": null,
    "evaluated_sequence_count": 32,
    "matched_sequence_count": 24,
    "direction_error_count": 8,
    "direction_adherence_rate": 0.75
  }
}
```

응답에서 개별 `sequence_id`, deviation 대표 사례와 `deviations[]`를 제거한다. `sequence_report.guideline`은 순서와 direction 결과만 반영한다.

- [ ] **Step 6: Task 2 JSON과 수치 검사**

Run:

```bash
docker exec -w /workspace ship-paint-trainer python3 -c 'import json,re,pathlib; text=pathlib.Path("docs/ai_copilot_response_contract.md").read_text(encoding="utf-8"); docs=[json.loads(x) for x in re.findall(r"```json\n(.*?)\n```", text, re.S)]; reports=[x["data"]["report"] for x in docs if isinstance(x,dict) and isinstance(x.get("data"),dict) and isinstance(x["data"].get("report"),dict)]; report=reports[-1]; seq=report["sequence_report"]; direction=seq["direction_summary"]; assert seq["on_track_visit_count"] + seq["deviation_count"] == seq["evaluated_visit_count"] + seq["skipped_zone_count"]; assert sum(x["count"] for x in seq["deviation_summary"].values()) == seq["deviation_count"]; assert sum(x["visit_count"] for x in seq["zone_summary"]) == seq["evaluated_visit_count"]; assert direction["matched_sequence_count"] + direction["direction_error_count"] == direction["evaluated_sequence_count"]; assert all(sum(z["observed_direction_counts"].values()) == z["evaluated_sequence_count"] for z in direction["zone_summary"]); print("sequence_direction=ok")'
```

Expected: `sequence_direction=ok`.

- [ ] **Step 7: Task 2 원자적 커밋**

```bash
GIT_MASTER=1 git add docs/ai_copilot_response_contract.md
GIT_MASTER=1 git commit -m "docs(api): 순서와 방향 리포트 집계 교정"
```

### Task 3: 자세 누적 리포트와 종합 평 복원

**Files:**
- Modify: `docs/ai_copilot_response_contract.md:840`

**Interfaces:**
- Consumes: Task 1 공통 하위 리포트 규칙, 현재 `pose_estimation`의 `similarity_score`와 `corrections`
- Produces: zone별 자세 평균·문제 관절·우선 교정, `pose_report.guideline`, 모든 context를 반영한 `llm_briefing`

- [ ] **Step 1: 현재 구현 한계와 목표 집계 구분**

`pose_report`가 무조건 `not_available`이라는 문구를 제거한다. 대신 현재 MCP가 단일 자세 fixture 비교만 제공한다는 사실과, 세션 리포트 분석 계층이 CSV의 시퀀스별 단건 결과를 집계한다는 목표를 함께 명시한다.

다음 결과는 자세 리포트에서 생성하지 않는다.

```text
손떨림
몸통 기준 이동 방향
시간에 따른 자세 안정성
```

- [ ] **Step 2: 자세 평균 규칙 작성**

다음 규칙을 추가한다.

```text
zone.average_similarity_score
= 해당 zone의 분석 가능한 sequence similarity_score 단순 평균

pose_report.average_similarity_score
= 모든 분석 가능한 sequence similarity_score 단순 평균
```

분석 불가능한 시퀀스는 평균에서 제외하고 `unavailable_sequence_count`에 포함한다. 전체 평가·분석 불가 건수는 zone별 건수 합과 같아야 한다.

- [ ] **Step 3: 문제 관절과 우선 교정 필드 작성**

`problem_joints`에는 `angle_corrections`가 비어 있지 않은 관절만 포함한다고 명시하고 다음 필드를 정의한다.

```text
joint
label
occurrence_count
occurrence_rate
average_distance
angle_issues
correction_guidance
```

`angle_issues[].axis`는 `roll | pitch | yaw`, `direction`은 `excess | insufficient`다. 축과 direction별로 나누어 `average_diff_deg`를 계산해 서로 다른 부호가 상쇄되지 않게 한다.

`priority_corrections`는 발생률 내림차순 최대 3개이며, 동률이면 권장 zone 순서와 관절 코드로 정렬한다.

- [ ] **Step 4: 완료 응답의 `pose_report` 교체**

설계 §9의 `pose_report`를 반영한다. 다음 전체 수치와 zone 수치를 유지한다.

```json
{
  "evaluated_sequence_count": 32,
  "unavailable_sequence_count": 5,
  "average_similarity_score": 0.789,
  "zone_similarity_inputs": {
    "A-1": [10, 0.86],
    "A-2": [12, 0.72],
    "A-3": [10, 0.8]
  }
}
```

`zone_similarity_inputs`는 계산 설명용이며 실제 응답에는 넣지 않는다. 실제 응답은 설계의 `zone_summary`, `problem_joints`, `priority_corrections`, `guideline` 구조를 사용한다.

- [ ] **Step 5: context별 guideline과 `llm_briefing` 규칙 작성**

세 guideline의 근거 범위를 표로 명시한다.

| 필드 | 근거 |
|---|---|
| `coating_report.guideline` | 도막 분포·최악 zone·회복 가능성·처방 |
| `sequence_report.guideline` | 순서 deviation과 direction 오류 집계 |
| `pose_report.guideline` | zone별 유사도·반복 문제 관절·우선 교정 |

`llm_briefing`은 잘된 점, 가장 큰 도막 위험, 순서·direction 문제, 자세 문제, 다음 훈련 우선 조치 순으로 available context를 종합한다. unavailable context의 수치나 조치를 생성하지 않는다.

- [ ] **Step 6: Task 3 자세 수치 검사**

Run:

```bash
docker exec -w /workspace ship-paint-trainer python3 -c 'import json,re,pathlib; text=pathlib.Path("docs/ai_copilot_response_contract.md").read_text(encoding="utf-8"); docs=[json.loads(x) for x in re.findall(r"```json\n(.*?)\n```", text, re.S)]; reports=[x["data"]["report"] for x in docs if isinstance(x,dict) and isinstance(x.get("data"),dict) and isinstance(x["data"].get("report"),dict)]; pose=reports[-1]["pose_report"]; assert sum(z["evaluated_sequence_count"] for z in pose["zone_summary"]) == pose["evaluated_sequence_count"]; assert sum(z["unavailable_sequence_count"] for z in pose["zone_summary"]) == pose["unavailable_sequence_count"]; weighted=sum(z["average_similarity_score"]*z["evaluated_sequence_count"] for z in pose["zone_summary"])/pose["evaluated_sequence_count"]; assert round(weighted,3) == pose["average_similarity_score"]; assert len(pose["priority_corrections"]) <= 3; print("pose_report=ok")'
```

Expected: `pose_report=ok`.

- [ ] **Step 7: Task 3 원자적 커밋**

```bash
GIT_MASTER=1 git add docs/ai_copilot_response_contract.md
GIT_MASTER=1 git commit -m "docs(api): 자세 리포트와 종합 평 복원"
```

### Task 4: 전체 계약 정합성 검증

**Files:**
- Verify: `docs/ai_copilot_response_contract.md`

**Interfaces:**
- Consumes: Task 1~3의 최종 응답 계약
- Produces: JSON 파싱, 수치 불변식, enum, 응답 크기, context 책임 분리 검증 증거

- [ ] **Step 1: 모든 JSON과 SSE payload 파싱**

Run:

```bash
docker exec -w /workspace ship-paint-trainer python3 -c 'import json,re,pathlib; text=pathlib.Path("docs/ai_copilot_response_contract.md").read_text(encoding="utf-8"); blocks=re.findall(r"```json\n(.*?)\n```", text, re.S); [json.loads(x) for x in blocks]; sse=[json.loads(line.removeprefix("data: ")) for line in text.splitlines() if line.startswith("data: {")]; print(f"json_blocks={len(blocks)} sse_payloads={len(sse)} parse=ok")'
```

Expected: exit code 0 and `parse=ok`.

- [ ] **Step 2: 최종 리포트 불변식 통합 검사**

Run:

```bash
docker exec -w /workspace ship-paint-trainer python3 -c 'import json,re,pathlib; text=pathlib.Path("docs/ai_copilot_response_contract.md").read_text(encoding="utf-8"); docs=[json.loads(x) for x in re.findall(r"```json\n(.*?)\n```", text, re.S)]; reports=[x["data"]["report"] for x in docs if isinstance(x,dict) and isinstance(x.get("data"),dict) and isinstance(x["data"].get("report"),dict)]; report=reports[-1]; seq=report["sequence_report"]; direction=seq["direction_summary"]; pose=report["pose_report"]; assert seq["on_track_visit_count"] + seq["deviation_count"] == seq["evaluated_visit_count"] + seq["skipped_zone_count"]; assert sum(v["count"] for v in seq["deviation_summary"].values()) == seq["deviation_count"]; assert sum(z["visit_count"] for z in seq["zone_summary"]) == seq["evaluated_visit_count"]; assert direction["matched_sequence_count"] + direction["direction_error_count"] == direction["evaluated_sequence_count"]; assert all(sum(z["observed_direction_counts"].values()) == z["evaluated_sequence_count"] for z in direction["zone_summary"]); assert sum(z["evaluated_sequence_count"] for z in pose["zone_summary"]) == pose["evaluated_sequence_count"]; assert sum(z["unavailable_sequence_count"] for z in pose["zone_summary"]) == pose["unavailable_sequence_count"]; weighted=sum(z["average_similarity_score"]*z["evaluated_sequence_count"] for z in pose["zone_summary"])/pose["evaluated_sequence_count"]; assert round(weighted,3) == pose["average_similarity_score"]; assert len(pose["priority_corrections"]) <= 3; assert all("guideline" in report[k] for k in ("coating_report","sequence_report","pose_report")); assert report["llm_briefing"]; print("report_invariants=ok")'
```

Expected: `report_invariants=ok`.

- [ ] **Step 3: enum과 금지 구조 검사**

다음 사항을 문서 전체와 완료 응답 JSON에서 확인한다.

```text
deviation_summary keys = out_of_order, zone_skipped, zone_revisited, unexpected_zone
observed_direction_counts keys = horizontal, vertical, diagonal, curved_irregular, stationary
priority_corrections length <= 3
sequence_report has no deviations array
sequence_report has no individual sequence_id examples
```

검증용 Python은 완료 응답을 로드해 정확한 key 집합을 비교한다.

```bash
docker exec -w /workspace ship-paint-trainer python3 -c 'import json,re,pathlib; text=pathlib.Path("docs/ai_copilot_response_contract.md").read_text(encoding="utf-8"); docs=[json.loads(x) for x in re.findall(r"```json\n(.*?)\n```", text, re.S)]; reports=[x["data"]["report"] for x in docs if isinstance(x,dict) and isinstance(x.get("data"),dict) and isinstance(x["data"].get("report"),dict)]; seq=reports[-1]["sequence_report"]; assert set(seq["deviation_summary"]) == {"out_of_order","zone_skipped","zone_revisited","unexpected_zone"}; assert "deviations" not in seq; expected={"horizontal","vertical","diagonal","curved_irregular","stationary"}; assert all(set(z["observed_direction_counts"]) == expected for z in seq["direction_summary"]["zone_summary"]); print("report_enums=ok")'
```

Expected: `report_enums=ok`.

- [ ] **Step 4: 문서 공백과 작업 범위 검사**

Run:

```bash
GIT_MASTER=1 git diff --check
GIT_MASTER=1 git status --short
```

Expected: `git diff --check` exit code 0. Task 4에서 수정이 없으면 계약 문서는 나타나지 않고 `docs/Interface.pdf`만 untracked로 남는다. 수정이 있으면 계약 문서와 `docs/Interface.pdf`만 나타난다.

- [ ] **Step 5: 최종 검토 결과가 수정되었을 때만 커밋**

Task 4 검증 과정에서 문서 수정이 발생한 경우에만 실행한다.

```bash
GIT_MASTER=1 git add docs/ai_copilot_response_contract.md
GIT_MASTER=1 git commit -m "docs(api): 세션 리포트 계약 정합성 보강"
```

수정이 없으면 빈 커밋을 만들지 않는다.
