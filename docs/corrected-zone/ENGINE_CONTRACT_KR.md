# SprayXR 보정 Zone 엔진 구현 계약

작성일: 2026-11-06

계약 버전: `sprayxr-corrected-zone/v2-draft-20261106`

엔진 버전: `sprayxr-corrected-zone-engine/1.0.0`

## 1. 구현 범위

`trainer.zone_engine`은 SprayXR 이벤트와 Height PNG를 받아 Zone별 보정 도막 평가와 보수적인 행동 코칭 후보를 만드는 별도 CPU HTTP 서비스다. 서비스는 다음을 직접 수행한다.

- Height PNG의 실제 바이트와 SHA-256 검증
- 정확 컬러 Zone 마스크와 면 번호 검증
- TSB 회차 및 누적 의미에 맞는 도막 계산
- Zone 재작업 revision과 세션 집계 생성
- 행동 관측 근거, 원인 후보, 조절 방향의 구조화
- SQLite 이벤트 ledger에 이벤트·상태·평가를 한 트랜잭션으로 영속화

이 엔진의 Zone 대표값은 **관측 범위의 유효 도포 픽셀 평균**이다. 공급자 공식 평균은 별도 필드로 보존하며, 이 엔진이 공급자 공식을 재현한다고 주장하지 않는다.

## 2. 실행

```bash
ZONE_ENGINE_DB=/state/engine.db \
ZONE_ASSET_ROOT=/zone-assets \
ZONE_ENGINE_HOST=0.0.0.0 \
ZONE_ENGINE_PORT=8122 \
ZONE_ENGINE_MAX_BODY_BYTES=67108864 \
python -m trainer.zone_engine.service
```

필수 런타임은 Python 3.11, NumPy, Pillow, FastAPI, Pydantic, Uvicorn이다. 기본 내부 포트는 `8122`다. 인메모리 SQLite는 거부하며 `/state/engine.db` 같은 영속 경로를 사용해야 한다.

## 3. HTTP API

### 3.1 경로

| 메서드 | 경로 | 의미 |
|---|---|---|
| `POST` | `/internal/v1/events` | 이벤트 수신 및 동기 처리 |
| `GET` | `/internal/v1/sessions/{session_id}` | 세션 ledger와 최신 평가 상태 조회 |
| `GET` | `/internal/v1/assessments/{assessment_id}` | Zone 또는 세션 평가 조회 |
| `GET` | `/internal/v1/health` | 버전, asset, ledger 상태 조회 |

POST 본문의 공통 외곽 형식은 다음과 같다.

```json
{
  "event_id": "unique-event-id",
  "event_type": "session_start",
  "session_id": "session-id",
  "payload": {}
}
```

이벤트 종류는 `session_start`, `sequence`, `zone_observation`, `session_end`, `session_complete`다. 모든 시각은 `Z` 또는 `+00:00`을 명시한 UTC ISO-8601이어야 한다.

### 3.2 처리 확인 응답

```json
{
  "status": "accepted",
  "event_id": "unique-event-id",
  "event_type": "zone_observation",
  "session_id": "session-id",
  "payload_hash": "canonical-json-sha256",
  "assessment_id": "za-...",
  "session_revision": 1,
  "processing_status": "processed",
  "evidence_ids": ["ev-quality-...", "ev-lineage-..."]
}
```

`session_revision`은 항상 0 이상의 정수다. 평가가 없는 이벤트에는 마지막 평가 revision 또는 0을 반환한다. 같은 `event_id`와 같은 canonical JSON은 `status=duplicate`로 기존 결과를 반환한다. 같은 ID에 다른 본문이 오면 HTTP 409 `event_id_payload_conflict`다.

## 4. 이벤트 계약

### 4.1 `session_start`

핵심 payload 필드는 다음과 같다.

```json
{
  "source_instance_id": "sprayxr-integration-api-20261106",
  "user_id": "user-id",
  "scenario_id": "1",
  "started_at": "2026-11-06T00:00:00Z",
  "evaluation_mode": "corrected_zone_v2",
  "contract_version": "sprayxr-corrected-zone/v2-draft-20261106",
  "target_thickness_um": 80.0,
  "target_source": "scenario-contract",
  "model_mapping": {
    "provider_model_key": "Learn01",
    "work_id": "work-id",
    "session_family_id": "family-id",
    "round_index": 1,
    "round_id": "round-1",
    "geometry_model_key": "Learn01"
  },
  "zone_manifest": {
    "manifest_id": "tsb-zones-20261015-v1",
    "sha256": "920aa742425cc8778d3512ad9b87b3ec07c08c8b1669696dbb47cbd64a39d044",
    "model_key": "Learn01",
    "map_nums": [1]
  },
  "expected_zones": [
    {"face_id": "face-1", "map_num": 1, "zone_ids": ["1"]}
  ]
}
```

목표 두께를 모르면 `target_thickness_um`과 `target_source`를 모두 `null`로 보낸다. 엔진은 임의 목표를 넣지 않는다. 같은 `source_instance_id`와 `session_family_id` 안에서 모델·작업·geometry·manifest 정체성은 바뀔 수 없으며 `round_index`와 `round_id`는 중복될 수 없다.

### 4.2 `sequence`

```json
{
  "sequence_id": "sequence-1",
  "face_id": "face-1",
  "map_num": 1,
  "round_id": "round-1",
  "captured_at": "2026-11-06T00:00:09Z",
  "sequence_started_at": null,
  "frames": [
    {
      "timestamp": "2026-11-06T00:00:08Z",
      "relative_time_s": null,
      "zone_id": "1",
      "spray_on": true,
      "speed_mm_s": 300.0,
      "distance_mm": 250.0,
      "raytexmapnum": 1,
      "raytexuv": {"x": 1074.0464, "y": 246.4934}
    }
  ],
  "cumulative_crop": null
}
```

`raytexuv`는 0..1 정규화 좌표가 아니라 **atlas 픽셀 단위의 Unity 좌하단 원점 좌표**다. 엔진은 PNG 좌상단 원점 행으로 정확히 반전하며 범위 밖 좌표를 clamp하지 않는다. `zone_id`가 있으면 해당 면의 정확 컬러 마스크에 실제로 존재해야 한다. UV와 명시 Zone이 함께 있으면 둘이 일치해야 한다.

상대 시각 프레임은 `timestamp=null`, `relative_time_s>=0`, `sequence_started_at`을 함께 사용한다. 한 프레임은 절대 시각과 상대 시각 중 정확히 하나만 가져야 한다.

누적 crop을 함께 보낼 때 형식은 다음과 같다.

```json
{
  "snapshot_id": "cumulative-face-1",
  "snapshot_revision": 2,
  "height_png": {
    "data_base64": "...",
    "sha256": "64-lowercase-hex",
    "encoding": "rg16_b8",
    "basis": "cumulative_crop",
    "width": 512,
    "height": 512,
    "origin": "top_left",
    "crop": {
      "x0": 768,
      "y0": 768,
      "width": 512,
      "height": 512,
      "atlas_width": 2048,
      "atlas_height": 2048
    }
  }
}
```

### 4.3 `zone_observation`

```json
{
  "face_id": "face-1",
  "map_num": 1,
  "zone_id": "1",
  "round_id": "round-1",
  "snapshot_id": "height-1",
  "snapshot_revision": 1,
  "captured_at": "2026-11-06T00:00:10Z",
  "height_png": {
    "data_base64": "...",
    "sha256": "64-lowercase-hex",
    "encoding": "rg16_b8",
    "basis": "current_round",
    "width": 2048,
    "height": 2048,
    "origin": "top_left",
    "crop": {
      "x0": 0,
      "y0": 0,
      "width": 2048,
      "height": 2048,
      "atlas_width": 2048,
      "atlas_height": 2048
    }
  }
}
```

PNG 바이트가 본문 안에 있어야 한다. 파일 경로나 메타데이터만으로는 평가하지 않는다. 선언한 SHA-256, PNG 형식, 색상 채널, 이미지 크기, crop 크기를 모두 검증한다.

### 4.4 종료와 완료

```json
{"ended_at": "2026-11-06T00:00:20Z"}
```

```json
{
  "completed_at": "2026-11-06T00:00:21Z",
  "provider_official_summary": {
    "avg_thickness_um": 138.91,
    "source": "provider_completion_payload",
    "unit": "um",
    "progress_value": null,
    "progress_unit": null
  }
}
```

`session_complete`는 `session_end` 다음에만 허용한다. 종료 뒤 새로 촬영한 관측은 거부하지만, `captured_at<=ended_at`인 관측이 전송만 늦어진 경우에는 허용한다. 완료 뒤에는 어떤 새 이벤트도 받지 않는다.

## 5. 도막 수치 계약

### 5.1 raw 복호화와 도포 support

각 픽셀의 raw 두께는 다음 식으로 한 번 복호화한다.

```text
raw_um = (R / 255 + G / 65025) * 20000
```

- `rg16_b8`: `B > 0`인 픽셀만 공급자 도포 support다.
- `legacy_rg_positive`: `R > 0 또는 G > 0`인 픽셀만 도포 support다.

두 encoding을 섞어 추론하지 않는다. `rg16_b8`의 B 채널은 실제 PNG 바이트에 있어야 한다.

### 5.2 회차와 누적 crop

- `current_round`: 같은 면·Zone의 1..현재 회차 raw surface를 좌표 정렬 후 합산한다. 필요한 이전 회차가 없거나 같은 Zone 관측이 없으면 전체 평가를 보류한다.
- `cumulative_crop`: 절대 누적 스냅샷이다. 이벤트 순서대로 관측 픽셀을 **교체**하며 이전 값과 더하지 않는다.
- 서로 다른 atlas 크기나 마스크 크기는 거부한다.
- crop 밖 픽셀은 0 두께가 아니라 `unknown`이다. 확대·패딩·리샘플링하지 않는다.

회차 결합 뒤 보정은 픽셀마다 정확히 한 번 적용한다.

```text
corrected_um = min(raw_sum_um - 10, 1600)
painted = known AND source_painted AND corrected_um >= 1
```

`corrected_um < 1`은 미도장으로 집계한다. Zone 평균은 `painted=true`인 픽셀만의 산술 평균이며 미도장 비율은 별도로 보존한다.

### 5.3 품질 경계

목표 두께를 `T`, 도포부 평균을 `M`이라 하면 3등급은 정확히 다음과 같다.

| 조건 | `grade` |
|---|---|
| `M < T` | `below` |
| `T <= M < 1.5T` | `normal` |
| `M >= 1.5T` | `above` |

6개 bin은 아래 반개구간을 사용한다.

1. `M < 0.9T`
2. `0.9T <= M < 1.0T`
3. `1.0T <= M < 1.5T`
4. `1.5T <= M < 2.0T`
5. `2.0T <= M < 2.5T`
6. `M >= 2.5T`

목표가 없으면 `grade=null`, `six_band_counts=null`이다. 전체 Zone 픽셀이 관측되지 않았으면 `analysis_status=partial`, `statistics_scope=observed_only`이며 전체 Zone grade·평균을 보류한다. 관측 부분의 통계는 `metrics.observed_statistics`에만 남긴다.

## 6. 평가와 코칭 출력

Zone 평가 최상위 필드는 다음과 같다.

```text
assessment_id, session_id, revision, assessment_type, status,
zone_key, quality, metrics, coaching, evidence, evidence_ids,
created_at, provenance
```

각 근거는 `evidence_id`, 한국어 `statement`, `value`, `unit`, `source_event_ids`, `contract_version`, `revision`을 가진다. 각 원인 후보와 행동은 자체 ID와 `evidence_ids`를 가져 Qwen 설명이 검증된 근거만 선택할 수 있게 한다.

세션 평가에서는 Zone에서 수집한 모든 근거·원인·행동 문장에 `face_id`, `map_num`, `zone_id` 범위를 붙인다. 저도막 Zone과 과도막 Zone의 반대 조절 방향이 함께 있어도 서로 다른 관측임을 문장 자체에서 확인할 수 있다. 세션 evidence 목록의 앞부분에는 픽셀 가중 도포부 평균과 유효 픽셀 수, 공급자 공식 평균의 별도 전달값, 최신 Zone revision 계보를 순서대로 둔다. Zone 원본 ID·수치·단위·근거 연결은 바꾸지 않는다.

행동 통계는 해당 Zone의 Height 촬영 시각 이전 프레임만 사용한다. `spray_on=true`인 프레임의 `speed_mm_s`, `distance_mm`을 각각 `count`, `mean`, `minimum`, `maximum`, `median`으로 보존한다.

- 등급이 `normal`이면 문제 원인과 교정 행동을 만들지 않는다.
- `partial`, 목표 누락, 행동 근거 누락이면 코칭을 보류한다.
- `below` 또는 `above`이고 선행 행동 근거가 있을 때만 원인 **후보**와 조절 방향을 낸다.
- 원인 확정, 효과 크기, 최적값, 정밀 수치 목표는 생성하지 않는다.
- 세션 품질이 보류되면 세션 coaching의 causes/actions도 빈 배열이다.

## 7. 세션 집계

세션 완료 평가는 각 Zone의 **최신 revision만** 사용한다. 도포부 평균은 Zone 평균의 단순 평균이 아니라 모든 최신 Zone의 `painted_sum_um / painted_pixel_count`로 계산한다. 예상 Zone이 빠졌거나 최신 revision이 partial/unavailable이면 세션 전체 품질을 보류한다.

`provider_official_summary`는 `metrics`와 세션 조회에 원본 그대로 보존한다. 엔진 계산과 공식값의 동일성을 가정하거나 공식 평균을 보정하지 않는다.

## 8. 영속성과 시간 정책

SQLite는 WAL, foreign key, `synchronous=FULL`, 이벤트당 단일 트랜잭션을 사용한다. 다음 항목은 재시작 뒤에도 유지된다.

- 원본 canonical 이벤트 본문과 SHA-256
- 세션 이벤트 순서와 마지막 이벤트
- Zone별 모든 assessment revision
- 최신 Zone revision index
- 누적 crop의 교체 의미로 재구성한 상태
- 세션 평가와 공급자 공식 요약

서비스 자체는 idle 종료를 추측하지 않는다. 실서비스에서 무분사 종료 판정이 필요하면 API가 명시적인 `spray_on=false` tick을 sequence 이벤트로 보내야 한다. 세션 조회의 `idle_policy.requires_explicit_no_spray_tick`이 이를 나타낸다.

## 9. 고정 asset

엔진 manifest는 `trainer/zone_engine/assets/zone_asset_manifest.json`이며 SHA-256은 다음과 같다.

```text
920aa742425cc8778d3512ad9b87b3ec07c08c8b1669696dbb47cbd64a39d044
```

manifest는 palette와 각 label PNG의 SHA-256 및 2048×2048 geometry를 고정한다. Zone 판정은 palette 색의 픽셀별 정확 일치로 수행한다. `zone_rects.csv`는 진단용 경계 자료일 뿐 평가 마스크로 사용하지 않는다.

## 10. 검증 상태와 제한

구현 이관 및 검증 범위는 [설치·인계 안내](README_KR.md)를 따른다. 이 문서의 이벤트 예시는 엔진 내부 계약이다. 외부 `/api/v2` 본문에는 `event_type`/`payload` 봉투를 씌우지 않으며, 공개 요청의 정확한 필드는 `trainer/demo_api/engine_contracts.py`와 API 서버의 OpenAPI를 확인한다.

행동 코칭은 관측 통계와 품질 규칙을 연결한 원인 후보와 조절 방향이다. 실제 인과 효과·최적 속도·현장 훈련 효과를 검증한 모델로 해석하지 않는다.
