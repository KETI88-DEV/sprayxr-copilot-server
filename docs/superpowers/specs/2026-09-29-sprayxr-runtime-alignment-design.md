# SprayXR 실물 입력 정합 설계 (2026-09-29)

- 대상: v2.5 계약을 실물 SIM 입력(`docs/SprayXR_CoPilot_샘플_2026-09-21.json`)에 맞추는 설계
- 선행: `docs/sprayxr_v25_sample_delta_2026-09-28.md`(차이 목록), `docs/superpowers/plans/2026-09-17-v25-contract-migration.md`(작업 분해)
- 사용자 결정(2026-09-29): `/copilot/query`의 `session_id` 부재는 **SIM 측 실수**로 간주한다. 구역 키 해석과 마스크 납품은 사용자가 벤더에 요청 중이다. 그 외 항목은 이 문서가 설계한다.
- 표기: 수치는 실측이며 표본을 밝힌다. 추정은 `[추정]`.

## 0. 설계 원칙

1. **실물 샘플이 정본이다.** 계약과 다르면 샘플을 택하고 이유를 남긴다. 예외는 샘플이 명백히 빠뜨린 것(`session_id`, `handedness`, `request_id`)이며, 이때도 부재를 422로 만들지 않는다.
2. **판정 불가와 계약 위반을 분리한다.** 정의·자산·단위가 없어 판정을 못 하는 것은 200 + `analysis_status="not_available"` + `reason_code`다. 422는 보낸 쪽이 틀렸을 때만 쓴다.
3. **마스크가 없는 세계와 있는 세계의 응답 키 집합은 동일하다.** 값만 달라진다. 마스크가 도착해도 SIM 클라이언트는 바뀌지 않는다.
4. **미확정은 주석이 아니라 타입과 설정으로 봉인한다.** 판정 코드가 만지면 예외가 난다.

## 1. 측정이 뒤집은 전제 (앞선 델타 문서 정정)

| 델타 문서의 서술 | 실측 | 결론 |
|---|---|---|
| S5 "어깨 사이 0.056m, 발끝→머리 0.796m는 스케일 모순" | 601개 세션·작업자 6인에서 19개 뼈 길이가 **유일값 1개**(어깨 사이 0.0567m 고정). CSV-FK 28,335프레임의 발끝↔머리 p50 1.127m, **&lt;0.8m가 27.25%** | **이상이 아니다.** 고정 아바타 리그이고 `shoulder`는 쇄골 기점, `head`는 목뼈 기점이다. 사람 해부 임계(어깨 0.25~0.45m)를 쓰면 코퍼스 **100%**가 거부된다 |
| B22 "회전순서 미확정 → `spray_angle_deg` 산출 불가" | 샘플 프레임에서 후보 72개 중 12개가 잔차 4.08°로 동률이고, 그 안에 우리 CSV 실측 규약(YXZ·+1·`GUN_FORWARD=(0,1,0)`, 63,280프레임에서 중앙값 4.0°, 차선 20.7°)이 포함된다 | **산출 가능하다.** `angle_source="inferred_csv"` 라벨을 달고 내보내고, 라이브에서 `R(rot)·forward`와 `raypoint−tool`의 각 차이 ≤15°를 계속 검증한다 |
| — | `facing_yaw`가 월드 Y-up 입력에서 실제 8.8° 회전을 **50.7°**로 보고한다(`world_axes` 기준 −80.6°) | **현행 코드 결함.** 자세 정규화가 요를 못 지우고 높이를 수평에 섞는다 |

**실제 결함으로 확인된 것:** `head` 전 시퀀스 고정(range 0.00000, 같은 구간 `pelvis` 0.2086·`righthand` 0.9830), SEQ-001의 건이 손에서 4.961m 떨어짐, 오른팔 IK 퇴화(`RightUpperArm→RightHand` 비율 p10=p50=1.000).

## 2. 공개 계약 설계

### 2.1 식별자
- `session_id`는 **불투명 문자열**이다. 파싱하지 않는다(샘플 `"1_1001642_1"`의 규칙이 `user_id`와 불일치해 확정 불가).
- `zone_id`는 `-1` 또는 1~99 정수 문자열. `"01"`→`"1"` 정규화. 상한을 39가 아니라 99로 두어 새 라벨이 정상 데이터를 죽이지 않게 한다. 실재 여부는 마스크가 판단하고 결과는 `not_available`이다.
- 구역 복합 키 `ZoneKey = (map_num, zone_id)`. 최상위 요청에 `map_num`이 없으므로 프레임 `raytexmapnum`의 최빈값으로 유도하고, 응답 `zone_key`에 우리 해석을 노출한다. 후보가 2개 이상이면 `sequence_spans_multiple_maps`.
- `scenario_id`는 SIM 값(`"1"`)이 정본이고 `TR-L01`은 내부 설정 키로 격하한다. 매핑은 `trainer/configs/scenarios/index.yaml`에 두고 미등록 값은 **404가 아니라** `scenario_status="unmapped"`로 답한다. 텔레메트리는 언제나 저장한다.

### 2.2 엄격성 정책 (가장 큰 변경)
현행 `extra="forbid"`를 전면 적용하면 **샘플 시퀀스 요청 4건, 즉 정상 데이터 100%가 422**다. 원인은 다리 8관절, `sequence_setting.month`, 히트맵 크롭 객체다.

| 레이어 | 정책 | 이유 |
|---|---|---|
| 요청 최상위, `worker` | **forbid** | 오타(`sesion_id`)가 조용히 무시되면 엉뚱한 세션에 쓴다 |
| `sequence_setting`, `sequence_info[]` 및 하위, `sequence_result`, 히트맵 크롭 | **allow + 보고** | 명중/미명중이 이미 키 집합이 다르다. 프레임 레벨 새 키 하나가 세션 전체를 죽이면 안 된다 |
| multipart 파트 이름 | allow + 보고 | 파트 하나로 세션 리포트를 날리는 손실이 비대칭 |
| 모든 응답 모델 | **forbid** | 우리가 저자다. Zod 동등성 시험이 닫힌 집합을 요구한다 |

미지 필드는 JSON path로 모아 `data.meta.unknown_fields`(최대 50)에 싣고, **요청당 WARNING 1건**으로 로그한다(120FPS 세션에서 프레임마다 찍으면 실제 경고가 묻힌다). `gun_metrics` 하위에 레벤슈타인 거리 1인 미지 키가 있으면 `possible_typo` 경고를 붙여 `raytexUV` 같은 오타로 명중이 미명중이 되는 것을 잡는다.

### 2.3 미확정 봉인
```python
Unitful(raw, unit)   # unit=="unconfirmed" 면 .si 접근이 UnitUnconfirmedError
EulerRaw(values)     # to_quaternion/to_matrix 메서드가 아예 없다
```
단위 미확정 값(`pressure`, `viscosity`, `thinning_ratio`, `nozzle_size`, `month`)은 저장·에코만 하고 임계 비교에 넣지 않는다. 해당 판정은 `unit_unconfirmed:<field>`. 회전은 원문 저장·동일성 검사에만 쓰고, 각도 산출은 §5.5의 `inferred_csv` 경로로만 한다.

### 2.4 `session_id` 폴백
```
있음 → 조회. 없으면 404. ended 여도 허용(질의는 읽기 전용, 종료 직후 복기가 제품 가치다)
없음 → user_id 의 active 세션
        1개  → 사용, session_source="fallback_latest_active"
        0개  → 200 + no_active_session
        2개↑ → 200 + ambiguous_active_session (최신을 고르지 않는다)
어떤 경우에도 session_id 부재만으로 422/400을 내지 않는다
```
`data.session_id`에 **실제 사용한 세션**을 반환하는 것이 폴백의 검증 수단이다.

### 2.5 `/copilot/analyze-sequence`
본문이 `telemetry/sequence`와 sha256 동일하고 같은 시퀀스를 두 번 보낸다. 하나의 핸들러에 두 경로를 등록하고 멱등 별칭으로 처리한다.

| 상황 | 동작 |
|---|---|
| 최초 | 분석·저장, `ack{stored:true, duplicate:false}`, 이상이면 SSE |
| 동일 hash·완료 | 저장된 `data` 그대로, `duplicate:true`, **SSE 재발행 없음** |
| 동일 hash·진행 중 | 최대 5초 대기 후 완료면 위와 동일, 타임아웃 409 |
| 상이 hash | 409, 저장된 판정 불변 |

응답은 두 경로에서 바이트 동일하다. 경로 구분을 응답에 넣으면 SIM이 그 차이에 의존해 나중에 통합할 수 없게 된다.

### 2.6 경로 prefix
`/api/v1` 유무를 벤더가 확정하기 전까지 **두 경로를 동시 서빙**한다(`V1_PATH_MODE=both`, bare는 `include_in_schema=False`). 주의: 현행 `copilot_router.py:44`는 오류 래핑 여부를 **경로 문자열**로 판단하므로 bare 경로에서 v2.5 래퍼가 빠진다. 판정을 **라우트 태그 소속**으로 바꾼다.

### 2.7 오류 규칙 요약
- **400** 형식 오류(JSON/multipart 파싱 실패, Content-Type 불일치)
- **404** 세션·리포트 없음
- **409** 종료 후 시퀀스, 중복 키 + 상이 payload, 진행 중 중복, `user_id`/`scenario_id`가 저장된 세션과 불일치
- **422** 필수 식별자 누락, forbid 레이어 미지 필드, `zone_id` 형식 위반(`"A-1"` 포함), `index` 비순증가, 히트맵 base64·크기 상한 위반
- **503** LLM 백엔드 불가(**query 전용**), 저장소 쓰기 불가
- **200 + not_available**: `zone_id="-1"`, 마스크 미납, `neck` 결측, `sequence_result={}`, 단위·회전순서 미확정, 프레임 부족, 미지 히트맵 버전, `scenario_id` 미등록, `head` 고정

`telemetry/sequence`는 **모델 부재로 503을 내지 않는다.** 결정론 엔진이 판정하고 LLM은 설명만 하므로 모델이 죽어도 수집·판정·저장은 계속되어야 한다. 그 텔레메트리가 학습 데이터다.

### 2.8 `sessions/*` 정리
- `sessions/end`의 `worker`는 **삭제하지 않고 optional·무시**한다. 최상위 forbid 하에서 필드를 지우면 계약을 그대로 구현한 SIM 빌드가 종료에 실패해 세션이 영구 active로 남고, 그러면 query 폴백이 `ambiguous`로 오염된다.
- `sessions/complete`는 실물 3파트(`session_id`, `telemetry_csv` gzip, `height_map_image`)만 필수. `final_score`·`is_pass`는 요청에서 빼고 **서버 소유**로 옮기되 숫자를 만들지 않는다(`score_status="not_available"`). `avg_thickness_um`과 paint 지표 5종은 `sim_metrics`에 원문 보관하고 판정에 쓰지 않는다.
- 멱등 키가 없으므로 `report_id = f"REPORT-{session_id}"`, `request_id = form.request_id or f"complete-{session_id}"`, payload sha256으로 중복·충돌을 가른다. 세션당 리포트는 1개다.

## 3. 히트맵 어댑터 설계

### 3.1 디코더
신규 순수 모듈 `trainer/demo_api/heightmap_codec.py`(표준 라이브러리 `zlib`·`struct`만). 4계층: 청크 → 픽셀 → 포맷 문자열 → 두께 필드.

- **IHDR만 읽고 크기 상한을 먼저 거른다.** inflate 비용 0으로 zip bomb과 2048² 전면을 막는다.
- 행 반전(`row = img_height - 1 - y`)은 `decode_heatmap_cell` **한 곳**에서만. 이후 모든 인덱싱은 텍스처 방향(아래→위)이다.
- 두께는 16bit 정수 코드로 보관한다(2B/px). 정확한 정수 차분이 가능하고 메모리가 작다.
- Up 필터(행 필터 2)만 정수 레인 트릭으로 벡터화한다. 실물에서 filter 2가 42~79%를 차지해 1.44~1.54배 빨라지고, 출력 바이트 동일성을 두 실물에서 확인했다.

### 3.2 버전 분기
선언(`format` 문자열·`tEXt RealSPRAY-HeightFormat`)이 1순위, 픽셀 증거가 2순위다.

| 조건 | 처리 |
|---|---|
| `v2` 선언 + 필수 키 | 정상 |
| 미지 버전 / `format` 부재 | 200 + `heatmap_format_unsupported` |
| `v2` 선언인데 전 픽셀 B=0 | 200 + `heatmap_alpha_all_zero` |
| `v2` 선언인데 G=255 존재 | 200 + `heatmap_encoding_out_of_range` |

**기존 코퍼스(1,362장)는 다른 버전이다.** `tEXt`가 없고 B 전부 0이며, R=0·R=1 구간에 **G=255가 존재**해 base-255 규약을 위반한다(밑수 255/256에 따라 최댓값 222.07 vs 220.95로 갈림). 기본 거부하고 배치 재처리에서만 `legacy_v1.enabled`로 opt-in하며 결과에 `encoding_verified: false`를 붙인다. 런타임 경로에서는 설정과 무관하게 항상 거부한다.

### 3.3 두께 파이프라인
```
디코드 → 유효 마스크(B > alpha_threshold, 기본 0) → DeadVolume 차감 → 통계
```
- DeadVolume 기본 25µm, `mode="clamp"`(음수→0, 픽셀은 `coated_px`에 남고 `effective_px`에서 빠짐).
- **차감 후 유효 픽셀 0은 정상 경로다.** `analysis_status="available"`, `coating_status="no_deposit"`. 샘플 최대 17.22µm이므로 실제로 발생한다.
- p95는 nearest-rank(보간 없음). 실측 11.0727이 이 정의의 값이다.
- 면적은 `coated_texels`가 1차 산출물이고 mm²는 텍셀 밀도표가 생긴 뒤에만(`level_mesh`의 UV 면적비로 오프라인 산출, 균일성 게이트 실패 시 `null`).

### 3.4 벤더 값과 재계산값 분리
벤더 `avg_thickness_um=5.26`은 **재현되지 않는다.** 자연스러운 집계 규칙 전수 탐색에서 B&gt;0 평균 5.1369(−2.3%)가 가장 가깝고, 정확히 맞는 규칙은 임계 87이라는 인위적 값뿐이다.

- 두 값을 **다른 키**(`thickness.vendor_reported` / `thickness.recomputed`)에 `source` 라벨과 함께 싣는다.
- **판정은 `recomputed`만으로 한다.** `judge_coating` 시그니처가 벤더 값을 받지 않는 것이 집행 지점이다.
- `agreement` 블록에 비교 기준(`basis`)과 차이를 기록하되 판정을 막지 않는다. 벤더 값은 DeadVolume 미차감이므로 `pre_dead_volume`과 비교한다.

### 3.5 누적 → 증분
`cur_sequence_heatmap`이 누적이고 `acc_`는 오지 않는다. 내부 이름에 `cur`/`acc`를 쓰지 않고 `heatmap_semantics: cumulative_at_frame_end` 설정으로 표현한다.

- 저장은 **16bit 코드 + alpha를 64×64 텍셀 타일**로. 3B/텍셀, 면당 12.58MB, **시퀀스 수와 무관**하다. 크롭 원본을 시퀀스마다 보관하면 1,000시퀀스에서 524MB로 터진다.
- 증분 = `max(0, 현재 − 직전)`. 크롭 영역이 달라도 텍셀 공간에서 정렬된다.
- 누적 감소 비율 &gt;1%면 `heatmap_non_monotone`(원인 후보: 벤더 리셋, 세션 혼선, `map_num` 혼동).
- 크롭 테두리에서 증분&gt;0 비율 &gt;2%면 `heatmap_crop_clipped`(크롭이 분사 영역을 다 못 덮었다는 신호).
- 커밋은 **전부-아니면-전무**다. 두 번째 크롭이 422인데 첫 크롭이 붙으면 다음 증분이 조용히 틀린다.
- 예산 초과 축출 시 `delta_unavailable_evicted`로 증분 기반 판정을 하지 않는다.

### 3.6 구역 집계 (마스크 2세계)
```python
def aggregate_by_zone(field, mask: ZoneMask | None, *, spec, ...) -> ZoneAggregation
```
`mask=None`이 정상 입력이다. 기본 인자를 두지 않아 호출자가 마스크 부재를 명시하게 한다.

| 마스크 | `zones` | `overall` | reason_code |
|---|---|---|---|
| 없음 | `()` | **정상 산출** | `zone_mask_missing` |
| 있음·일치 | 라벨별 통계 | 정상 | `null` |
| 있음·불일치 | SIM 값 기준 주소 지정 + 양쪽 값 보고 | 정상 | 경고 `zone_mismatch`(409 아님) |

스케일&gt;1 크롭은 **픽셀 중심 텍셀의 라벨**을 쓰고, 경계 모호 비율이 10%를 넘으면 경고한다.

**마스크 납품 형식이 성능을 가른다(실측).** indexed(PLTE)+filter 0은 7.8ms, RGB8은 디코드 21.9ms + RGB→라벨 변환 **1,301ms**다. 오프라인 변환 스텝(`build_zone_masks.py` → `.labels.u8`)을 두어 요청 경로에서 변환을 없앤다. 메모리는 면당 4.19MB, 캐시 64MiB로 16면 상주.

### 3.7 성능 예산 (실측 기반)
| 대상 | 픽셀 | 디코드 |
|---|---|---|
| 샘플 크롭 64×108 | 6,912 | **1.49~2.14ms** |
| 2048² 전면 | 4,194,304 | **943~1,453ms** |

최악 단가 0.553µs/px(레인 최적화+전 픽셀 유효+구역 집계) → 보수적으로 0.60µs/px. 계약 §14의 p95 500ms 중 히트맵 몫 150ms를 배정하면 **요청당 총 262,144픽셀(면의 6.25%)**이 상한이다.

초과 시 **거부**한다(다운샘플은 디코드가 선행되어 비용이 안 줄고 평균이 바뀌어 출처 규약이 깨진다. 비동기는 계약의 동기 판정 요구와 충돌한다). 단 히트맵만 `heatmap_too_large`로 비우고 pose·sequence 분석은 살린다.

배치 경로(코퍼스 재처리, `height_map_image`)는 같은 코드에 상한만 바꿔 쓴다(2048² 943ms × 1,362장 = 21.4분, 4병렬 5.4분 `[추정]`).

## 4. 골격 정규화와 입력 품질 게이트

### 4.1 관절 어휘 3자 통일
신규 `trainer/scripts/joint_names.py`. 계약 어휘를 **런타임 어휘와 동일하게** 통일하고 `right_elbow` 류 snake_case와 현행 `mcp_servers/pose_estimation.py`의 `wrist_r`·`elbow_r` 6개는 폐기 별칭으로 흡수한다.

- `to_code`/`to_runtime`은 모르는 이름에 **예외 대신 `None`**을 돌려준다(`zone_resolver` 규약과 동일).
- 값은 3원소 배열과 `{pos, rot}` 객체를 **둘 다** 받고 `rot`은 보관만 한다.
- 다리 8관절은 받아서 보관하되 채점·교정에서 제외하고 `excluded_joint_groups: ["legs"]`, `exclusion_reason: "no_tracker"`로 명시한다(IK 다리가 해부 검사를 모두 통과하는 프레임이 37.1%뿐).

### 4.2 `neck` 결측 — 채점 집합 재선언
현행 `SCORING_JOINTS`는 12종에 `Neck`을 포함하지만 실효는 11종이다(없으면 `continue`). 이 괴리를 플래그로 영구화하지 않는다.

- `SCORING_JOINTS_V2` **10종**(Neck 삭제, Head 분리), `CONDITIONAL_JOINTS_V2=("Head",)`, `MIN_SCORED_JOINTS=8`.
- `Neck` 영구 삭제 근거: 런타임·CSV 모두에 없고, BVH의 `Neck`은 `body_proportions.py`가 비율로 삽입한 표시용 관절이다. 측정량이 아닌 값을 채점에 넣으면 σ의 출처가 데이터가 아니라 우리 상수가 된다.
- `similarity_score`는 σ² 평균의 지수라 **분모가 관절 수**다. 응답에 `scoring_set_id`와 `scored_joints`를 항상 싣고, 레퍼런스의 `scoring_set_id`가 다르면 `expert_percentile=null`로 강제한다.
- 채점 관절이 8종 미만이면 점수를 만들지 않는다. **`0.0`이 아니라 `null`이다.**

### 4.3 입력 품질 게이트 (`trainer/scripts/skeleton_gate.py`)

| 규칙 | 임계 | 근거(실측) | 판정 |
|---|---|---|---|
| R0 좌표 부정 | NaN/inf/3원소 아님/필수 관절 누락 | 계약 불변식 | reject |
| R1 뼈 길이 세션 내 일관성 | ±5% 또는 3mm | 리그는 강체. CSV 601세션 유일값 1개, 샘플 변동 ≤1e-4m | reject |
| R1b 등록 리그 대조 | ±10% | 샘플 리그가 CSV와 3뼈에서 다름(×0.72·×1.885·×7.48) → 거부 근거로 못 씀 | **degrade** |
| R2 어깨 너비 | 0.045~0.070m | **리그 기준 0.0567m.** 사람 해부 기준은 코퍼스 100% 거부 | reject |
| R3 키(발끝↔머리) | 0.35~1.50m | CSV-FK 28,335프레임 min 0.411 max 1.435, 체인합 1.4645. **1.0m 하한은 47.71%, 1.5m는 100% 거부** | reject |
| R4a 전체 동결 | 채점 관절 전부 range&lt;1e-4 **그리고** 도구 변위 ≥30mm | 샘플 SEQ-002는 도구도 정지 → 정상 | reject |
| R4b 관절별 동결 | 한 관절만 range&lt;1e-4 | 90Hz 11프레임에서 Spine 0.12%, Head·팔 0.00% | degrade(해당 관절 제외) |
| R4c 교차 시퀀스 동결 | 두 시퀀스에서 같은 관절 동결 | 샘플 `head`만 0.00000. CSV 교차창 **0/473** | degrade(영구 제외) |
| R5 손↔건 거리 | 0.05~1.30m | CSV p50 0.519 max 0.860, &gt;1.30m가 **0/957**. 샘플 SEQ-002 0.359 통과 / SEQ-001 **4.961 거부** | reject |
| R6 프레임 수 | 지표 ≥3, 자세 ≥1 | PCA 주축에 3점 필요 | reject(지표만) |
| R7 프레임 간격 | 4~40ms, max/min ≤3 | 샘플 8·9ms(120FPS), CSV 11.111ms(90FPS) | degrade |
| R8 오른팔 퇴화 | 신전 비율 ≥0.98 | CSV-FK p10=p50=**1.000**(항상 완전 신전) | info(교정에서 제외) |

**코퍼스 오탐률 실측(957창):** PASS 99.79%, 관절 동결 강등 0.21%, **하드 거부 0.00%**. 세션 단위로는 605개 중 4개(0.66%)가 빈 파일로 걸린다.

### 4.4 축 규약
공간이 둘이다. **월드는 Y-up**(착탄점·도구·런타임 골격), **리그는 Z-up**(FK 출력·BVH·자세 레퍼런스). `world_axes`를 정본으로 `Space` 타입을 도입하고 `pose_similarity`는 **기본값 없는 공간 인자**를 받는다(기본값이 있으면 호출자가 안 적고, 안 적힌 쪽이 지금의 결함이다).

`pass_segmentation` 수정 지점: 독스트링(9행), `elevation`(195행, 주축이 아니라 **변위**에서 계산 — 주축은 PCA라 부호가 임의), `heading`(196–199행), `mid_height`(200행), 높이 임계(36–37행 → 상대 높이로 전환).

**재빌드 범위:** 궤적 레퍼런스, 자세 산포 레퍼런스, 백분위 사다리는 **필수**. `skeleton_mounts.json`은 불필요. 재빌드 전후 확인:

| 항목 | 기대 |
|---|---|
| `elevation_band` 분포 | Z-up 균등(35/33/32%) → Y-up **horizontal 60.9% / diagonal 26.9% / vertical 12.2%** |
| 버킷 변경률 | **64.33%**(13,774패스 중 8,861). 훨씬 작으면 수정이 안 먹은 것 |
| `height_band` 실효 버킷 수 | 1.000(전부 high) → 1보다 커져야 함 |
| 뼈 길이 | 전후 차이 0(축 수정은 라벨을 바꾸지 길이를 안 바꾼다) |
| held-out 자기유사도 | 올라가야 함. 내려가면 정규화 프레임 통일이 잘못된 것 |

## 5. 시퀀스 지표

### 5.1 시간 기반 상수
```
MIN_BURST_DURATION_S = 0.111   # 기존 10프레임 @90FPS와 비트 동일
MIN_PASS_DURATION_S  = 0.111
SMOOTH_WINDOW_S      = 0.056   # 상한 9프레임
MIN_SAMPLES          = 3
```
기존 상수는 `frames_for(duration, 90.0)`으로 재정의해 90FPS 코퍼스에서 같은 값이 나오게 한다(평가 재현성 보존). 현재 `MIN_BURST_FRAMES=10` 때문에 **코퍼스 버스트의 7.82%가 통째로 버려지고** 샘플 7프레임 시퀀스는 분할 자체가 안 된다.

분할 정책: `duration < 2 × MIN_PASS_DURATION_S`면 분할하지 않고 패스 1개로 낸다. 83ms 안에 반환점은 없다.

### 5.2 방향 5종
```
stationary        : 변위 < 30mm (프레임 수와 무관하게 먼저 판정)
null              : n < 5 또는 변위 < 100mm
curved_irregular  : rms_ratio > 0.25 또는 path_ratio > 2.00
horizontal/diagonal/vertical : elevation ≤20 / 20~70 / ≥70도
confidence=high   : 변위 ≥400mm 그리고 n ≥11
```
임계는 13,774패스 스윕에서 골랐다(rms 0.25/path 2.00에서 horizontal 46.0%·curved 25.5%로 한 종이 압도하지 않고 짧은 창 일치율이 가장 높음). **스무딩을 짧은 창에 적용하지 않는다**(3점 이동평균을 넣으면 5프레임 일치율이 47.5%→38.7%로 떨어진다).

**짧은 창의 한계(핵심 수치):** 패스 앞 w프레임 분류가 전체와 일치하는 비율은 w=5에서 **47.5%**, w=8에서 58.7%, w=11에서 69.6%다. 프레임 수보다 **변위가 지배한다**. 120FPS 11프레임(83ms)에서 착탄 속도 p50이면 변위 242mm로 `low` 신뢰도 방향을 낼 수 있지만, 건 속도 기준이면 48mm로 `stationary`가 된다. 이것이 "7·11프레임이 잘린 샘플인가"라는 벤더 질의가 **기술적으로 가장 비싼 질문**인 이유다.

### 5.3 속도 — 기준이 두 개고 배율이 5배다
23,231패스에서 같은 패스의 두 속도를 동시에 계산한 결과:

| 값 | 출처 | 대응 |
|---|---|---|
| 400~520 mm/s | 벤더 notes | **건 속도**(우리 p10 322 ~ p50 580) |
| 2,111 mm/s | 기존 보고 | 착탄점 속도 |
| 2,800 mm/s | 계약 목표 | **착탄점 속도**(우리 p50 2,919와 4% 차이) |

p50 비율 **5.03배**. 세 값은 모순이 아니라 기준이 다르다. `avg_speed_mms`는 **착탄점 속도**로 확정한다(도막 두께를 지배하는 것은 손목이 아니라 착탄점 체류 시간이고 계약 목표가 이미 그 기준이다). 벤더 `gun_metrics.speed`는 다른 필드명으로 보존한다.

모든 속도·거리는 `{value, basis, source}` 3종 세트로만 나가고, **다른 basis끼리 연산하면 `ValueError`**다.

### 5.4 거리
벤더 `distance` 372.06mm와 `|tool−raypoint|` 454.25mm의 차 82.19mm는 노즐 오프셋으로 보인다`[추정, n=1]`. **벤더 값이 정본**이고 우리는 검증용으로만 재계산하며 오프셋은 설정 키로 둔다.

### 5.5 `spray_angle_deg`
§1의 측정에 따라 산출한다. `angle_source="inferred_csv"` 라벨을 달고, 라이브에서 각 차이 ≤15° 단언을 계속 돌린다. `rot`이 90° 배수만 오면 순서를 영구히 확정할 수 없으므로 이 점을 벤더에 묻는다.

## 6. `pose_analysis` 출력
`analysis_status`(available/degraded/not_available), `reason_code`(닫힌 집합), `scoring_set_id`, `scored_joints`, `excluded_joints`, `reference_quality`, `support`, `similarity_score`(불가 시 **null**), `expert_percentile`, `correction_mode="joint_only"`, `corrections[]`(joint·label·severity·`angle_delta_deg: null`·`direction: null`·`evidence{sigma,distance_m,support,expected_gain}`), `input_quality`.

**레퍼런스 품질:**

| 상황 | `reference_quality` | `analysis_status` |
|---|---|---|
| 버킷 없음 / n&lt;20 | `unavailable` | not_available |
| coarse 강등으로 n≥20 | `coarse_bucket` | degraded |
| 작업자 1인, n≥20 | `single_worker` | **degraded** |
| 작업자 ≥3인, n≥100 | `multi_worker` | available |

`single_worker`를 `available`로 두지 않는다. 1인 산포로 만든 σ는 "숙련자 통상 범위"가 아니라 "그 사람의 습관"이다. coarse 강등 사다리는 `cell`(uv8→uv2) → `height` 제거 → `heading` 제거 → `band`만 순이고, 어느 키로 맞췄는지 `reference_id`에 접미어로 남긴다.

**샘플 두 시퀀스의 판정:**

| | SEQ-001 | SEQ-002 |
|---|---|---|
| 게이트 | R5 `tool_not_held`(4.961m) | R4c `skeleton_joint_frozen`(head) |
| status | **not_available** | **degraded** |
| direction | `null`(착탄점 없음) | `stationary`(변위 0) |

## 7. 계약 개정 목록

| 조항 | 개정 |
|---|---|
| §4.1 | 구역을 `(map_num, zone_id)` 복합 키로. `A-1`~`A-3` 열거 삭제. `-1` 정상값 |
| §5 | `/copilot/analyze-sequence` 멱등 별칭 추가, prefix 정책 명시 |
| §6.1 | `handedness` optional, `scenario_id` 패턴 비강제 |
| §7.1 | 120FPS, `index` 1-base, `gun_metrics` 이름·optional, `raytexuv` 텍셀 좌표, `sequence_setting` 실물 이름, 관절 3원소 배열 허용·`neck` 결측 허용·다리 8관절 |
| §7.3 | 세 feedback에 `analysis_status`+`reason_code` 균일 배치, `zone_key` 승격, `anomaly` 객체를 HTTP 응답에도 포함 |
| §9.1 | `session_id`·`scenario_id` optional + 폴백 규정, `context`를 객체로 |
| §10 | `worker` optional·무시 |
| §11.1 | 파트 3개 필수, 나머지 optional 또는 서버 소유 |
| §14 | 히트맵 몫 150ms, 크롭 262,144픽셀 상한 각주 |
| §18.4 | 히트맵 절 전면 교체(PNG 크롭 형식, B=알파, base-255 코드와 G≤254, `cur_`=누적·`acc_` 없음, DeadVolume 차감 주체, 벤더/재계산 분리, 크기 상한, 행 필터 0 권고) |
| §18.5 | 관절 어휘 통일, `neck` 삭제, 각도·방향 `null` 고정, `reference_quality` 4단계 |
| §18.6 | 방향 5종 판정식과 최소 요건, 속도 basis 명시, 목표값을 시나리오 설정에서 |

## 8. 현행 코드 결함 (구현 시 함께 고칠 것)

1. `copilot_contracts.py:31` — `context: str | None`. 계약과 SIM 모두 객체를 보낸다. 지금 형태로는 계약 준수 요청이 422다.
2. `copilot_router.py:44` — 오류 래핑을 경로 문자열로 판단. bare 경로에서 v2.5 래퍼가 빠져 SIM 파서가 깨진다.
3. `pose_similarity.py:283, 300` / `mcp_servers/pose_estimation.py:228` — 각도 교정과 "올리세요/내리세요" 문구를 생성·노출. **현재 배포본이 계약 §18.5 위반**이다.
4. `pose_similarity.facing_yaw` — 월드 Y-up 입력에서 요를 못 지운다(실제 8.8°를 50.7°로 보고).
5. `SCORING_JOINTS` 선언 12종 vs 실효 11종.

## 9. 벤더 질의 추가·수정

**철회:** 델타 §5의 4번(골격 스케일 이상) — 고정 리그로 확인됨.

**추가:**
- 리그 버전: `pelvis→spine`(×1.885), `chest→shoulder`(×7.48), `pelvis→thigh`(×0.72) 세 오프셋만 CSV 리그와 다르다. 바인드 오프셋 표를 납품할 수 있는가?
- `gun_metrics.distance`와 `|tool−raypoint|`의 차 82.19mm는 노즐 오프셋인가? 고정값인가?
- `rot`이 전 프레임 90° 배수다. 실제 작업 프레임도 그런가?
- 히트맵·마스크 PNG를 **행 필터 0**으로 인코딩해 줄 수 있는가? 디코딩이 100배 빨라진다.
- 구역 마스크를 **indexed(PLTE)** PNG로 줄 수 있는가? RGB8이면 변환이 1,301ms/면이다.
- 크롭 총 픽셀 상한을 **262,144(면의 6.25%)**로 합의할 수 있는가? 2048² 전면은 943~1,453ms로 500ms를 못 지킨다.
- v2에서 G의 유효 범위가 0~254인가? 기존 코퍼스에는 G=255가 있다.
- 기존 코퍼스 PNG의 인코딩 밑수는 255인가 256인가?
- `avg_thickness_um=5.26`의 집계 규칙(재현 실패, 최근접 5.1369).
- 한 시퀀스가 여러 `map_num`을 걸칠 수 있는가?

**보강:** 속도 질의는 "벤더 400~520 = 건 속도, 계약 2,800 = 착탄점 속도로 확인됐다(같은 패스 p50 비율 5.03). `gun_metrics.speed`가 건 팁 기준인지만 확인해 달라"로 좁힌다. 프레임 수 질의는 "7·11프레임 창은 전체 패스 방향과 47~70%만 일치한다. 답이 `direction` 산출 가능 여부를 직접 결정한다"를 덧붙인다.

## 10. 구현 순서

1. `joint_names.py` + 매핑 왕복 테스트 — 나머지가 전부 이름에 의존한다.
2. `world_axes` 확장(`Space`, `elevation_deg`) + 축 테스트를 **먼저 실패시킨다**.
3. `pose_similarity` 공간 인자화 + 회전·각도 경로 제거.
4. `pass_segmentation` 시간 기반 상수 + 축 수정.
5. `skeleton_gate.py` + 임계 경계 테스트(임계값 근거를 상수 주석에).
6. `heightmap_codec.py` + 좌표 모듈 + 두께 파이프라인.
7. v1 계약 모델(`trainer/demo_api/v1/contracts.py`) + 라우터·오류 핸들러.
8. 타일 누적기와 구역 집계 인터페이스.
9. 레퍼런스 재빌드(궤적 → 자세 → 사다리)와 §4.4 전후 대조. **이 단계 전까지 `expert_percentile`은 `null`이 정답이다.**
10. 계약 개정문 작성 + `mcp_servers/pose_estimation.py`의 각도 누출 제거.
