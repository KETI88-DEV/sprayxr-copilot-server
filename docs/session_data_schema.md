# 시뮬레이터 ➔ 에이전트 (런타임 메시지 3종)

> **이 문서는 향후 Unity 실시간 연동 계약이다.** 현재 취득된 전문가 데이터는 이 경로가 아니라
> 오프라인 CSV 일괄 export로 들어왔다. 실제로 무엇이 확보됐는지는 [`dataset_spec.md`](dataset_spec.md),
> 항목별 미확보 상태는 [`data_requirements_pose_sequence_mcp.json`](data_requirements_pose_sequence_mcp.json) v3.0을 본다.
>
> 2026-07-29 실측 기준으로 아래 표 중 **오프라인 원본에 이미 존재함이 확인된** 필드:
> `frames[].skeleton`(20관절), `frames[].gun_metrics`, `frames[].surface_hit`(CSV의 Ray point/UV 컬럼),
> `sequence_data.burst_meta`(분사 트리거 On/Off), `active_gun_pose`(위치 분산 판별로 유도), `thickness_texture_refs`(Height PNG).
>
> 반대로 **런타임 계약에도 아직 없어서 추가돼야 하는** 것: rig root의 월드 변환(REQ-14),
> `conventions.rotation.order`(REQ-05), `skeleton_def.hierarchy`(REQ-08), `worker.handedness`(REQ-03).

## ① 세션 시작 (`session_start`)
| Key | 상세 | 형식 | 예시 | 가능여부 |
| :--- | :--- | :--- | :--- | :--- |
| `session_id` | 세션 식별자 | String | `"S-20260722-042"` | 필수 |
| `user_id` | 작업자 식별자 | String | `"W-031"` | 필수 |
| `level_id` | 대상 레벨 정보 | Number (또는 String) | `1` | 필수 |
| `model` | 모델 지오메트리 정보 | String | `"Exam"` | 필수 |
| `worker.skill_level` | 작업자 숙련도 | String | `"novice"` | 필수 |
| `worker.handedness` | 주 사용 손 | String | `"right"` | 필수 |
| `spec.nominal_um` | 도막 두께 기준값 | Number | `100.0` | 필수 |
| `spec.tolerance_rule` | 도막 두께 허용 규칙 | String | `"90/10"` | 필수 |
| `skeleton_def` | 관절 계층 및 기본 포즈 | Object | `{"joint_count": 20, "joints": [...]}` | 옵션 (협의시 생략) |
| `recorded_at` | 세션 시작 시각 | String (ISO 8601) | `"2026-07-22T02:40:11Z"` | 옵션 (권장) |
| `worker.height_mm` | 작업자 키 | Number | `1682` | 옵션 (권장) |
| `conventions` | 좌표/단위계 규칙 | Object | `{"position_unit": "m", ...}` | 옵션 (협의시 생략) |

## ② 이벤트 (`event`)
| Key | 상세 | 형식 | 예시 | 가능여부 |
| :--- | :--- | :--- | :--- | :--- |
| `session_id` | 세션 식별자 | String | `"S-20260722-042"` | 필수 |
| `user_id` | 작업자 식별자 | String | `"W-031"` | 필수 |
| `event_id` | 이벤트 식별자 | String | `"E-0007"` | 필수 |
| `sequence_data.burst_meta` | 분사 구간 정보 | Object | `{"definition": "spray_trigger_on_burst", ...}` | 필수 |
| `sequence_data.frames[]` | 시계열 프레임 배열 | Array of Objects | `[{ "t_sec": 12.10, ... }, ...]` | 필수 |
| ↳ `frames[].t_sec` | 상대 시간 (초) | Number | `12.10` | 필수 |
| ↳ `frames[].spray_on` | 분사 여부 | Boolean | `true` | 필수 |
| ↳ `frames[].skeleton` | 상반신 관절 데이터 (결측 금지) | Object | `{"root_pos_m": [...], "joints_rot_euler_deg": {...}}` | 필수 |
| ↳ `frames[].active_gun_pose`| 활성화된 스프레이 건 위치/회전 | Object | `{"key": "gun_hangil_short", "pos_m": [...]}` | 필수 |
| ↳ `frames[].gun_metrics` | 실제 분사 거리/속도 등 | Object | `{"distance_real_mm": 612.4, ...}` | 필수 |
| ↳ `frames[].surface_hit` | 스프레이 도달 영역 정보 | Object | `{"hit_pos_m": [...], "face_id": 124}` | 필수 (`spray_on=true` 시) |
| `user_request` | 사용자의 코칭 질의 텍스트 | String | `"지금 분사 상태 괜찮아?"` | 옵션 (권장) |
| `sequence_data.target_region_summary` | 타겟 영역 요약 | Object | `{"region_id": "R-UP-01", ...}` | 옵션 (협의시 생략) |

## ③ 세션 완료 (`session_complete`)
| Key | 상세 | 형식 | 예시 | 가능여부 |
| :--- | :--- | :--- | :--- | :--- |
| `session_id` | 세션 식별자 | String | `"S-20260722-042"` | 필수 |
| `user_id` | 작업자 식별자 | String | `"W-031"` | 필수 |
| `dft_measurements[]` | 건조 도막 두께(DFT) 측정점 배열 | Array of Objects | `[{"value_um": 82.0, "zone_id": "R-UP-01"}, ...]` | 필수 |
| `thickness_texture_refs` | 전체 두께맵 텍스처 파일 참조 경로 | String (또는 Array) | `"results/S-20260721-001_dft.exr"` | 필수 |

---

# 에이전트 ➔ 시뮬레이터 (Agent Response 데이터)

## 공통 응답 메타데이터 (모든 응답에 포함)
| Key | 상세 | 형식 | 예시 | 가능여부 |
| :--- | :--- | :--- | :--- | :--- |
| `session_id` | 세션 식별자 | String | `"S-20260722-042"` | 필수 |
| `user_id` | 작업자 식별자 | String | `"W-031"` | 필수 |
| `request_id` | 요청 추적용 식별자 | String | `"req-9f3c1a55"` | 필수 |
| `generated_at` | 응답 생성 시각 | String (ISO 8601) | `"2026-07-22T02:40:12Z"` | 필수 |
| `latency_ms` | 응답 처리 소요 시간 (ms) | Number | `142` | 필수 |
| `tool_calls_made` | 내부 사용 툴 내역 | Array of Strings | `["sequence_validation:guide"]` | 필수 |

## ① 세션 시작 응답 (`response_to_session_start`)
| Key | 상세 | 형식 | 예시 | 가능여부 |
| :--- | :--- | :--- | :--- | :--- |
| `sequence_guide` | 권장 순서 안내 정보 | Object | `{"flow_source": "expert", "recommended_flow": ["R-UP-01", ...], "guidance": "..."}` | 필수 |
| `spec_summary` | 인지한 도막 규격 요약 정보 | Object | `{"nominal_um": 100.0, "tolerance_rule": "90/10"}` | 필수 |

## ② 이벤트 응답 (`response_to_event`)
| Key | 상세 | 형식 | 예시 | 가능여부 |
| :--- | :--- | :--- | :--- | :--- |
| `event_id` | 이벤트 식별자 | String | `"E-0007"` | 필수 |
| `route` | 처리된 라우팅 경로 | String | `"training_realtime"` | 필수 |
| `text_guide` | 텍스트 코칭 내용 | Object | `{"summary": "...", "detail": "...", "priority": "high"}` | 필수 |
| `anomaly` | 이상 탐지 발생 여부 및 내역 | Object | `{"detected": true, "items": [...]}` | 필수 |
| `sequence_data_ack` | 프레임 데이터 수신/저장 확인 | Object | `{"stored": true, "frames_stored": 119}` | 필수 |
| `citations` | 참고한 RAG 문서 목록 | Array of Strings | `["DOC-0421-C0007"]` | 옵션 |
| `dft_result` | 실시간 도막 판정 결과 | Object 또는 Null | `null` | 옵션 |

## ③ 세션 완료 응답 (`response_to_session_complete`)
| Key | 상세 | 형식 | 예시 | 가능여부 |
| :--- | :--- | :--- | :--- | :--- |
| `route` | 처리된 라우팅 경로 | String | `"training_review"` | 필수 |
| `summary_report` | 최종 요약 리포트 객체 (하위 항목 포함) | Object | `{...}` (아래 항목들 포함) | 필수 |
| ↳ `overall_score` | 전체 평가 점수 | Number | `0.63` | 필수 |
| ↳ `dft_result` | 합격/불합격 여부 및 통계 | Object | `{"verdict": "fail", "mean_um": 103.3, ...}` | 필수 |
| ↳ `pose_analysis` | 자세 유사도 수치/관절별 교정 | Object | `{"mean_similarity": 0.74, "regions": [...]}` | 필수 |
| ↳ `deficiency_distribution`| 도막 부족 부위 분포/결함 분석 | Object | `{"ranked_deficient_regions": [...]}` | 필수 |
| ↳ `sequence_adherence` | 권장 순서 준수율 및 이탈 내역 | Object | `{"followed_ratio": 0.83, "deviations": [...]}` | 필수 |
| ↳ `top_corrections` | 가장 중요한 주요 교정 사항 | Array of Objects | `[{"joint": "RightUpperArm", "impact": "..."}]` | 필수 |
| ↳ `text_report` | 텍스트 형태의 종합 강평 | String | `"이번 세션은 상부..."` | 필수 |
| `citations` | 리포트 생성에 참고된 가이드 문서 | Array of Strings | `["DOC-0421-C0007"]` | 옵션 |

---

# 응답 필드 중 현재 데이터로 산출 불가한 것

2026-07-29 실측 기준. 계약을 지금 고치자는 뜻이 아니라, 납품 시점이 아니라 지금 합의해야 할 항목이다.

| 응답 필드 | 상태 | 사유 |
|---|---|---|
| `pose_analysis.mean_similarity` | **부분 가능** | 스켈레톤끼리 비교는 가능. 단 팔 관절 1~2% 프레임의 ~180° 불연속(REQ-15) 때문에 이상치 트리밍이 전제다 |
| `pose_analysis.regions[]` | **불가** | region 라벨이 아직 저작되지 않았다. 학습모드는 레벨당 맵이 1개다 |
| `top_corrections[].joint` + 각도 | **불가** | `rotation.order`(REQ-05)와 `hierarchy`(REQ-08)가 null이라 해부학적 축·방향을 특정할 수 없다 |
| `sequence_adherence.followed_ratio` | **불가** | 학습모드는 region이 1개라 순서 개념이 성립하지 않는다. UV 공간 도포 순서로 재정의해야 한다 |
| `deficiency_distribution` | **부분 가능** | Height PNG로 µm 산출은 검증됐으나, 세션 최종 상태인지 누적 버퍼인지 미확인(REQ-11) |
| 숙련자 대비 훈련생 비교 전반 | **불가** | 훈련생 데이터가 0건이다(REQ-16). 현재 "기존 자세"는 숙련자 본인의 편차 프레임에서 온다 |
