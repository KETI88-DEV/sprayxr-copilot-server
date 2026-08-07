# 세션 종료 리포트 응답 계약 설계

**상태:** 승인됨
**작성일:** 2026-08-07
**적용 문서:** `docs/ai_copilot_response_contract.md`

## 1. 목표

세션 입력 CSV와 시퀀스별 판정 결과를 바탕으로 도막·순서·방향·자세 결과를 재현 가능하게 집계한다. 최종 응답은 시퀀스 수가 증가해도 무제한으로 길어지지 않아야 하며, 각 context의 구조화된 결과와 독립된 `guideline`, 모든 context를 반영한 `llm_briefing`을 제공한다.

## 2. 현재 구현과 목표 계약의 경계

현재 구현은 상세 이상탐지 결과를 영속 저장하지 않는다.

- `MonitorRecorder`는 최대 500건의 입출력 이벤트를 보관하는 단일 프로세스 메모리 링버퍼다.
- `SimSessionStore`는 현재 단일 세션 snapshot과 `anomaly_count`만 소유한다.
- 모니터 이벤트와 세션 snapshot은 프로세스 재시작 시 사라지며 최종 리포트의 정본이 아니다.
- 현재 `sequence_validation` MCP는 다음 권장 zone을 계산하지만 순서 이탈 유형과 세션 준수율을 계산하지 않는다.
- 현재 `pose_estimation` MCP는 단일 자세를 fixture와 비교하며 세션 누적 집계를 직접 제공하지 않는다.

목표 계약은 다음 하이브리드 방식을 사용한다.

1. 세션 입력 CSV를 원천 데이터의 정본으로 보관한다.
2. Trigger ON → OFF 구간마다 `sequence_id`, `zone_id`, 도막·순서·방향·자세 판정과 `algorithm_version`을 파생 레코드로 기록한다.
3. 실시간 Query는 최신 파생 레코드를 사용한다.
4. 최종 리포트는 동일 알고리즘 버전의 파생 레코드를 집계한다.
5. 파생 레코드가 누락되었거나 버전이 다르면 정본 CSV에서 다시 분석한다.

`sequence_data_ack.stored=true`는 해당 시퀀스의 CSV 또는 프레임 원본이 저장되었다는 뜻이다. 이상 알림 저장이나 최종 리포트 생성 완료를 뜻하지 않는다. anomaly SSE payload만 따로 정본으로 저장하지 않으며, 정상·이상 판정 전체를 기록해 평균과 준수율의 분모를 보존한다.

## 3. 최종 리포트 공통 규칙

- `coating_report`, `sequence_report`, `pose_report`는 항상 존재한다.
- 각 하위 리포트는 `analysis_status`, `reason_code`를 필수로 가진다.
- 분석 가능한 하위 리포트는 `algorithm_version`, `source_artifact_id`, 구조화된 집계와 `guideline`을 제공한다.
- 분석 불가능한 하위 리포트는 값을 추정하지 않고 `analysis_status="not_available"`과 안정적인 `reason_code`를 반환한다.
- `guideline`은 해당 하위 리포트의 구조화된 결과만 근거로 작성한다.
- `llm_briefing`은 모든 `analysis_status="available"` context를 반영한다.
- LLM은 구조화된 수치, enum, 분류와 처방 코드를 변경하거나 새로 계산하지 않는다.
- zone 배열은 시나리오의 `recommended_flow` 순서로 정렬한다.

## 4. 순서 리포트

### 4.1 방문 정규화

같은 zone에서 연속된 분사 버스트는 정상 반복일 수 있으므로 하나의 방문으로 압축한다.

```text
원본 zone 흐름: A-1, A-1, A-3, A-2, A-1
평가 방문 흐름: A-1, A-3, A-2, A-1
최초 방문 흐름: A-1, A-3, A-2
```

- `evaluated_visit_count`는 연속 중복을 제거한 평가 방문 수다.
- `first_visit_flow`는 각 zone의 최초 방문만 남기므로 길이가 전체 zone 수를 넘지 않는다.
- 다른 zone으로 이동한 뒤 기존 zone으로 돌아온 경우에만 재방문으로 센다.

### 4.2 순서 deviation enum

| `type` | 판정 조건 |
|---|---|
| `out_of_order` | 권장 흐름에 있는 zone이지만 현재 기대 zone과 다른 순서로 최초 진입 |
| `zone_skipped` | 세션 종료까지 권장 zone을 한 번도 방문하지 않음 |
| `zone_revisited` | 다른 zone으로 이동한 뒤 이미 방문·완료한 zone으로 다시 진입 |
| `unexpected_zone` | `recommended_flow`에 없는 zone으로 진입 |

평가 방문 하나에는 하나의 주 deviation만 부여한다. 우선순위는 `unexpected_zone`, `zone_revisited`, `out_of_order` 순이다. `zone_skipped`는 세션 종료 시 미방문 zone마다 추가한다.

### 4.3 순서 준수율

```text
adherence_rate
= on_track_visit_count / (evaluated_visit_count + skipped_zone_count)
```

- 범위는 0~1이다.
- 분모가 0이면 `analysis_status="not_available"`로 반환한다.
- `deviation_count`는 네 deviation 유형의 `count` 합과 같아야 한다.
- 최종 응답에 개별 `sequence_id`나 무제한 `deviations[]`를 포함하지 않는다.
- 개별 판정은 내부 파생 레코드에만 보관한다.

### 4.4 제한된 크기의 순서 집계

`sequence_report`는 다음 집계만 외부에 제공한다.

- `recommended_flow`
- `first_visit_flow`
- `evaluated_visit_count`, `on_track_visit_count`, `skipped_zone_count`
- `adherence_rate`, `deviation_count`
- 네 enum별 `count`, `affected_zone_ids`를 가진 `deviation_summary`
- zone별 `visit_count`, `revisit_count`, `deviation_count`를 가진 `zone_summary`

응답 크기는 시퀀스 수가 아니라 zone 수에 비례한다. 개별 deviation 대표 사례도 최종 응답에는 포함하지 않는다.

## 5. direction 집계

방향 오류는 trajectory 판정이지만 최종 사용자 context에서는 구역 흐름과 함께 확인할 수 있도록 `sequence_report.direction_summary`에 둔다. 순서 준수율과 방향 준수율은 서로 섞지 않는다.

### 5.1 direction enum

허용값은 다음 다섯 값이다.

- `horizontal`
- `vertical`
- `diagonal`
- `curved_irregular`
- `stationary`

### 5.2 방향 준수율

```text
direction_adherence_rate
= matched_sequence_count / evaluated_sequence_count
```

- `direction_summary`는 `direction_error` 판정만 집계한다.
- `direction_summary`는 독립된 `analysis_status`, `reason_code`를 가진다. 순서 집계가 가능해도 direction 기준이 없으면 direction만 `not_available`로 반환할 수 있다.
- `direction_error_count = evaluated_sequence_count - matched_sequence_count`가 성립해야 한다.
- 거리·속도 오류와 일반 궤적 이탈은 방향 준수율에 포함하지 않는다.
- zone별로 `expected_direction`, 평가·일치·오류 건수, 방향 준수율과 다섯 direction의 관측 건수를 제공한다.
- 다섯 direction의 `observed_direction_counts` 합은 해당 zone의 `evaluated_sequence_count`와 같아야 한다.
- 분석 가능한 direction 시퀀스가 없으면 `direction_summary.analysis_status="not_available"`로 반환하고 방향 수치를 추정하지 않는다.

## 6. 자세 리포트

CSV의 Trigger ON → OFF 구간을 시퀀스 단위로 분석하고, 현재 자세 비교 결과인 `similarity_score`와 `corrections`를 집계한다.

### 6.1 평균 유사도

- zone 평균은 해당 zone의 분석 가능한 시퀀스 `similarity_score`를 단순 산술평균한다.
- 전체 평균은 모든 zone의 분석 가능한 시퀀스 점수를 시퀀스 수로 가중한 산술평균이다.
- 분석 불가능한 시퀀스는 평균에서 제외하고 `unavailable_sequence_count`에 포함한다.
- 전체 `evaluated_sequence_count`는 zone별 분석 가능 시퀀스 수의 합과 같아야 한다.
- 전체 `unavailable_sequence_count`는 zone별 분석 불가 시퀀스 수의 합과 같아야 한다.

### 6.2 문제 관절 집계

zone별 `problem_joints`는 다음 필드를 제공한다.

| 필드 | 설명 |
|---|---|
| `joint`, `label` | MCP의 관절 코드와 표시 이름 |
| `occurrence_count` | 해당 관절 교정이 발생한 시퀀스 수 |
| `occurrence_rate` | `occurrence_count / zone.evaluated_sequence_count` |
| `average_distance` | 해당 관절 교정 발생 시퀀스의 위치 거리 평균 |
| `angle_issues` | 축별 `average_diff_deg`, `direction` 집계 |
| `correction_guidance` | 집계된 차이를 설명하는 교정 안내 |

`problem_joints`에는 현재 MCP의 `angle_corrections`가 비어 있지 않은 관절만 포함한다. `occurrence_count`와 `average_distance`도 해당 조건을 만족한 시퀀스만 집계한다.

`angle_issues[].axis`는 `roll`, `pitch`, `yaw`만 사용한다. `direction`은 `excess`, `insufficient`를 사용한다. 서로 다른 부호의 차이를 상쇄하지 않도록 축과 direction별로 나누어 평균한다.

`priority_corrections`는 전체 세션에서 발생률이 높은 교정 항목을 최대 3개 제공한다. 동률이면 `recommended_flow`의 zone 순서, 관절 코드 순서로 정렬한다.

현재 `EP-001` 계열 기준 자세는 실제 숙련자 데이터가 아니라 fixture다. 설계된 집계는 현재 MCP가 산출하는 위치 거리와 `roll`, `pitch`, `yaw` 차이만 사용한다. 손떨림, 몸통 기준 이동 방향, 시간 안정성을 자세 결과로 추정하지 않는다.

## 7. context별 guideline

| 필드 | 사용 가능한 근거 |
|---|---|
| `coating_report.guideline` | 도막 분포, 최악 zone, 회복 가능성, 구조화된 처방 |
| `sequence_report.guideline` | 순서 deviation 집계와 direction 오류 집계 |
| `pose_report.guideline` | zone별 평균 유사도, 반복 문제 관절, 우선 교정 항목 |

각 guideline은 다른 context의 수치를 가져오지 않는다. 구조화된 결과가 `not_available`이면 해당 context의 원인만 설명하며 근거 없는 조치를 생성하지 않는다.

## 8. 종합 평

`llm_briefing`은 세 guideline을 단순 연결하지 않고 다음 순서로 available context를 종합한다.

1. 전체적으로 잘된 점
2. 가장 큰 도막 품질 위험
3. 순서와 direction 문제
4. 자세 문제
5. 다음 훈련에서 먼저 수행할 조치

일부 context가 `not_available`이면 나머지 available context로 종합 평을 작성하고, 분석하지 못한 context를 명시한다. 모든 context가 `not_available`이면 판정이나 우선 조치를 만들지 않고 분석 근거 부족을 안내한다.

## 9. 완료 리포트 응답 샘플

아래 예시는 실제 계약에 사용할 유효한 JSON이다. 필드 설명은 앞 절의 규칙을 따른다.

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Session report completed",
  "data": {
    "request_id": "req-report-0001",
    "session_id": "SESSION-001",
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-07T14:40:18Z",
    "latency_ms": 18120,
    "report": {
      "report_id": "report-SESSION-001",
      "report_status": "complete",
      "coating_report": {
        "analysis_status": "available",
        "reason_code": null,
        "algorithm_version": "coating-session-report-1.0",
        "source_artifact_id": "heightmap-SESSION-001-final",
        "coating_distribution_index": 72.4,
        "zone_summary": {
          "under2": 0.31,
          "under1": 0.22,
          "normal": 0.24,
          "over1": 0.12,
          "over2": 0.07,
          "over3": 0.04
        },
        "distribution": {
          "below": 151,
          "in_band": 17,
          "above": 5
        },
        "reachability": {
          "reachable": 81,
          "needs_process": 75
        },
        "worst_zones": [
          {
            "zone_id": "A-2",
            "map_num": 2,
            "uv_bin": [2, 3],
            "thickness_um": 56,
            "status": "under_app",
            "zone_class": "under2",
            "cause": "이동속도 과다·이격 과다",
            "action": "increase_dwell",
            "action_reachable": true
          }
        ],
        "root_cause": {
          "dominant": "operator_behavior",
          "behavior_r2": 0.44,
          "process_r2": 0.12,
          "note": "저도막의 주원인은 속도·거리·체류 시간이며 일부 구역은 공정 설정 조정이 필요합니다."
        },
        "guideline": "A-2 부족 구역의 체류 시간을 늘리고, 작업자 동작만으로 회복하기 어려운 구역은 공정 설정을 함께 조정하세요."
      },
      "sequence_report": {
        "analysis_status": "available",
        "reason_code": null,
        "algorithm_version": "sequence-session-report-1.0",
        "source_artifact_id": "telemetry-SESSION-001.csv",
        "recommended_flow": ["A-1", "A-2", "A-3"],
        "first_visit_flow": ["A-1", "A-3", "A-2"],
        "evaluated_visit_count": 37,
        "on_track_visit_count": 29,
        "skipped_zone_count": 0,
        "adherence_rate": 0.784,
        "deviation_count": 8,
        "deviation_summary": {
          "out_of_order": {
            "count": 3,
            "affected_zone_ids": ["A-2", "A-3"]
          },
          "zone_skipped": {
            "count": 0,
            "affected_zone_ids": []
          },
          "zone_revisited": {
            "count": 5,
            "affected_zone_ids": ["A-1", "A-2"]
          },
          "unexpected_zone": {
            "count": 0,
            "affected_zone_ids": []
          }
        },
        "zone_summary": [
          {
            "zone_id": "A-1",
            "visit_count": 14,
            "revisit_count": 3,
            "deviation_count": 3
          },
          {
            "zone_id": "A-2",
            "visit_count": 12,
            "revisit_count": 2,
            "deviation_count": 4
          },
          {
            "zone_id": "A-3",
            "visit_count": 11,
            "revisit_count": 0,
            "deviation_count": 1
          }
        ],
        "direction_summary": {
          "analysis_status": "available",
          "reason_code": null,
          "evaluated_sequence_count": 32,
          "matched_sequence_count": 24,
          "direction_error_count": 8,
          "direction_adherence_rate": 0.75,
          "affected_zone_ids": ["A-2", "A-3"],
          "zone_summary": [
            {
              "zone_id": "A-1",
              "expected_direction": "horizontal",
              "evaluated_sequence_count": 10,
              "matched_sequence_count": 10,
              "direction_error_count": 0,
              "direction_adherence_rate": 1.0,
              "observed_direction_counts": {
                "horizontal": 10,
                "vertical": 0,
                "diagonal": 0,
                "curved_irregular": 0,
                "stationary": 0
              }
            },
            {
              "zone_id": "A-2",
              "expected_direction": "horizontal",
              "evaluated_sequence_count": 12,
              "matched_sequence_count": 8,
              "direction_error_count": 4,
              "direction_adherence_rate": 0.667,
              "observed_direction_counts": {
                "horizontal": 8,
                "vertical": 3,
                "diagonal": 1,
                "curved_irregular": 0,
                "stationary": 0
              }
            },
            {
              "zone_id": "A-3",
              "expected_direction": "vertical",
              "evaluated_sequence_count": 10,
              "matched_sequence_count": 6,
              "direction_error_count": 4,
              "direction_adherence_rate": 0.6,
              "observed_direction_counts": {
                "horizontal": 2,
                "vertical": 6,
                "diagonal": 0,
                "curved_irregular": 1,
                "stationary": 1
              }
            }
          ]
        },
        "guideline": "A-1 이후 A-2, A-3 순서로 이동하고, A-2에서는 가로 패스, A-3에서는 세로 패스를 유지하세요."
      },
      "pose_report": {
        "analysis_status": "available",
        "reason_code": null,
        "algorithm_version": "pose-session-report-1.0",
        "source_artifact_id": "telemetry-SESSION-001.csv",
        "evaluated_sequence_count": 32,
        "unavailable_sequence_count": 5,
        "average_similarity_score": 0.789,
        "zone_summary": [
          {
            "zone_id": "A-1",
            "evaluated_sequence_count": 10,
            "unavailable_sequence_count": 1,
            "average_similarity_score": 0.86,
            "problem_joints": [
              {
                "joint": "wrist_r",
                "label": "오른쪽 손목",
                "occurrence_count": 3,
                "occurrence_rate": 0.3,
                "average_distance": 0.041,
                "angle_issues": [
                  {
                    "axis": "yaw",
                    "average_diff_deg": 4.2,
                    "direction": "excess"
                  }
                ],
                "correction_guidance": "오른쪽 손목 yaw를 기준 자세에 가깝게 줄이세요."
              }
            ]
          },
          {
            "zone_id": "A-2",
            "evaluated_sequence_count": 12,
            "unavailable_sequence_count": 3,
            "average_similarity_score": 0.72,
            "problem_joints": [
              {
                "joint": "elbow_r",
                "label": "오른쪽 팔꿈치",
                "occurrence_count": 8,
                "occurrence_rate": 0.667,
                "average_distance": 0.084,
                "angle_issues": [
                  {
                    "axis": "pitch",
                    "average_diff_deg": 5.8,
                    "direction": "excess"
                  }
                ],
                "correction_guidance": "오른쪽 팔꿈치 pitch를 평균 약 5.8° 줄이세요."
              }
            ]
          },
          {
            "zone_id": "A-3",
            "evaluated_sequence_count": 10,
            "unavailable_sequence_count": 1,
            "average_similarity_score": 0.8,
            "problem_joints": [
              {
                "joint": "shoulder_r",
                "label": "오른쪽 어깨",
                "occurrence_count": 5,
                "occurrence_rate": 0.5,
                "average_distance": 0.057,
                "angle_issues": [
                  {
                    "axis": "roll",
                    "average_diff_deg": 4.7,
                    "direction": "insufficient"
                  }
                ],
                "correction_guidance": "오른쪽 어깨 roll을 평균 약 4.7° 높이세요."
              }
            ]
          }
        ],
        "priority_corrections": [
          {
            "priority": 1,
            "zone_id": "A-2",
            "joint": "elbow_r",
            "label": "오른쪽 팔꿈치",
            "occurrence_rate": 0.667,
            "correction_guidance": "오른쪽 팔꿈치 pitch를 줄이세요."
          },
          {
            "priority": 2,
            "zone_id": "A-3",
            "joint": "shoulder_r",
            "label": "오른쪽 어깨",
            "occurrence_rate": 0.5,
            "correction_guidance": "오른쪽 어깨 roll을 높이세요."
          }
        ],
        "guideline": "A-2에서 반복된 오른쪽 팔꿈치 pitch 과다를 우선 교정하고, A-3에서는 오른쪽 어깨 roll을 기준 자세에 맞추세요."
      },
      "llm_briefing": "도막 정상 비율은 24%로 낮고 A-2 저도막이 가장 큰 품질 위험입니다. 구역 순서 준수율은 78.4%이며 A-2와 A-3에서 순서 이탈이 발생했습니다. 방향 준수율은 75%로 A-2의 가로 패스와 A-3의 세로 패스를 안정적으로 유지해야 합니다. 자세 평균 유사도는 78.9%이며 A-2의 오른쪽 팔꿈치 pitch 교정이 가장 자주 필요했습니다. 다음 훈련에서는 A-2 체류 시간을 먼저 늘린 뒤 A-1→A-2→A-3 순서와 구역별 권장 방향을 유지하고 오른쪽 팔꿈치 각도를 교정하세요."
    }
  }
}
```

## 10. 수치 불변 규칙

샘플과 실제 응답은 다음 관계를 만족해야 한다.

```text
sum(sequence_report.deviation_summary.*.count)
= sequence_report.deviation_count

sum(sequence_report.zone_summary.*.visit_count)
= sequence_report.evaluated_visit_count

sequence_report.on_track_visit_count
+ sequence_report.deviation_count
= sequence_report.evaluated_visit_count
+ sequence_report.skipped_zone_count

sequence_report.direction_summary.matched_sequence_count
+ sequence_report.direction_summary.direction_error_count
= sequence_report.direction_summary.evaluated_sequence_count

sum(pose_report.zone_summary.*.evaluated_sequence_count)
= pose_report.evaluated_sequence_count

sum(pose_report.zone_summary.*.unavailable_sequence_count)
= pose_report.unavailable_sequence_count
```

샘플의 자세 전체 평균은 다음과 같다.

```text
(0.86 × 10 + 0.72 × 12 + 0.80 × 10) / 32 = 0.789
```

## 11. 오류와 분석 불가 처리

- CSV가 없거나 읽을 수 없으면 `source_data_missing`을 사용한다.
- 기준 순서 또는 기준 direction이 없으면 해당 집계를 추정하지 않고 `reference_data_missing`을 사용한다.
- 자세 기준 fixture 또는 향후 전문가 기준이 없으면 `pose_report.reason_code="reference_data_missing"`을 사용한다.
- 유효 시퀀스가 없는 축은 `quality_insufficient`을 사용한다.
- 분석 도구 호출 실패는 `tool_unavailable`을 사용한다.
- 한 context의 실패가 다른 context의 정상 집계를 제거하지 않는다.
- `llm_briefing`은 실패한 context의 수치나 조치를 만들어내지 않는다.

## 12. 검증 기준

- 설계와 응답 계약의 모든 `json` 코드 블록이 파싱되어야 한다.
- 순서 deviation enum은 `out_of_order`, `zone_skipped`, `zone_revisited`, `unexpected_zone` 네 값만 사용한다.
- 최종 리포트에는 개별 순서 deviation 사례 또는 무제한 `deviations[]`가 없어야 한다.
- direction enum은 `horizontal`, `vertical`, `diagonal`, `curved_irregular`, `stationary` 다섯 값만 사용한다.
- 순서와 direction 준수율은 독립된 필드와 공식을 사용해야 한다.
- `direction_summary`는 순서 리포트와 별도로 분석 가능 상태를 표현해야 한다.
- 세 하위 리포트에 각각 `guideline`이 있어야 한다.
- `llm_briefing`은 도막·순서·direction·자세의 available 결과를 모두 반영해야 한다.
- 자세 전체 평균과 zone 평균은 분석 가능한 시퀀스의 단순 산술평균 규칙을 따라야 한다.
- `priority_corrections`는 최대 3개여야 한다.
- 모니터 링버퍼 또는 `anomaly_count`를 리포트의 영속 정본으로 설명하지 않아야 한다.
- `source_artifact_id`와 `algorithm_version`으로 재분석 근거와 파생 결과 버전을 추적할 수 있어야 한다.
