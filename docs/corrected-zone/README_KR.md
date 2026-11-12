# 보정 도막 평가·Zone 코칭 통합 안내

## 1. 전달 범위

공유 서버의 기존 기능을 유지하면서 선택적으로 켤 수 있는 도막 평가 경로를 추가했다. 세션 목표 두께와 실제 Height로 품질을 판정하고, 같은 구역에서 관측한 작업 행동으로 원인 후보와 다음 행동을 설명한다. 전문가 개인의 수행을 품질 정답으로 사용하지 않는다.

이 구현은 **검토·연동 시험용 v2 초안**이다. PR 병합만으로 기존 Unity의 v1 요청이 새 엔진으로 바뀌지 않는다. 기존 v1 클라이언트와 화면은 그대로이며, 새 목표/회차/Zone 관측을 전송하는 클라이언트 또는 변환 계층의 연결이 필요하다.

이관 기준은 기존 독립 통합 코드 `abb6944f131232912f41ec839438048f6779d547`이며, 공유 저장소 `main`의 `c6cc9bc99497b9dc75c0baf93ae8fd3e1a265d2f`에 적용했다. 공유 서버의 최근 대화·음성·뷰어 변경은 보존했다. 이관 중 새 경로의 API 키 검사를 보완하고, 새 원본 요청이 기존 DB 뷰어로 노출되지 않도록 API 저장 DB를 분리했다.

## 2. 처리 흐름과 변경 위치

**세션 시작 → 시퀀스 저장·관측 상태 갱신 → Zone Height 수신 → 보정 평가·행동 근거 → Qwen 안내 → 세션 종합**

| 파일/패키지 | 역할 |
|---|---|
| `trainer/demo_api/app.py` | 설정이 있을 때 v2 라우터·별도 DB·전달 작업자 등록, 종료 시 정리 |
| `trainer/demo_api/corrected_router.py` | 세션·시퀀스·Zone 관측 수신과 평가/설명 조회 |
| `trainer/demo_api/engine_contracts.py` | 단위·회차·좌표·버전·입출력 검증 |
| `trainer/demo_api/engine_store.py` | API 원본 이벤트·전달 대기 작업·평가·설명 저장 |
| `trainer/demo_api/engine_worker.py` | 엔진 전달과 설명 생성, 장애 시 재시도 |
| `trainer/demo_api/engine_gateway.py` | 내부 엔진·설명 서비스 HTTP 호출 |
| `trainer/zone_engine/` | Height 디코딩·정확 Zone 마스크·회차 합산·보정·원장·행동 분석 |
| `trainer/coating_explanation_service/` | 로컬 Qwen 모델 확인·근거 ID 선택·한국어 안내 구성 |
| `trainer/demo_api/access_control.py` | 설정된 API 키 검사를 v1과 v2에 동일 적용 |

API는 받은 요청과 전달 작업을 같은 SQLite 트랜잭션으로 저장한다. 엔진 처리 후 평가를 보존하고 Qwen 설명을 요청한다. 엔진 또는 Qwen 장애 시 해당 단계의 작업을 남겨 재시도한다. 재작업은 새 revision으로 저장하며 이전 평가는 보존한다.

## 3. 설치 전에 별도로 준비할 항목

소스 PR에 원본 훈련 데이터, 실행 DB·로그, 모델 가중치, 시험 서버의 Docker 이미지와 자산 사본을 넣지 않았다. 다음 항목은 설치 환경에 준비해야 한다.

| 항목 | 필요 조건 |
|---|---|
| 실행 환경 | 기존 공유 서버의 Python 3.11 컨테이너. 엔진에는 FastAPI·Pydantic 2·NumPy·Pillow·Uvicorn, 설명에는 httpx가 필요하다. 기존 배포 의존성을 따른다. |
| Zone 자산 | TSB의 `KETI_ZoneData_2026-10-15` 내 `palette.csv`, `scenario_map.csv`, `zone_rects.csv`, `color/*.png`. `trainer/zone_engine/assets/zone_asset_manifest.json`의 파일 해시와 일치해야 한다. |
| Qwen | Ollama의 `qwen3:4b`, 시험에 사용한 Q4_K_M 모델. 현재 설명 서비스의 접속 주소는 **`http://qwen:11434`로 고정**되어 있으므로 같은 내부 네트워크의 서비스명/별칭 `qwen`이 필요하다. |
| 상태 저장 | 기존 API DB와 별개인 corrected API DB, 엔진 DB, 설명 감사 로그를 각각 영속 저장 경로에 연결한다. 컨테이너 실행 사용자에게 쓰기 권한이 있어야 한다. |
| 기존 27B 경로 | 이 설명 서비스는 별도 Qwen3 4B 경로다. 공유 서버의 27B/vLLM/LoRA를 검증하거나 대체한 것으로 보지 않는다. 동시 GPU 사용량은 설치 서버에서 확인한다. |

Zone 자산 manifest의 기본 경로는 원래 시험 폴더를 가리키므로 새 설치에서는 **`ZONE_ASSET_ROOT`를 반드시 실제 자산 경로로 지정**한다. 엔진에 쓰는 manifest 파일의 SHA-256을 세션 시작 본문의 `zone_manifest.sha256`으로 전송한다. 이 값은 PNG나 폴더의 해시가 아니다.

엔진 health가 정상이어도 모든 PNG가 준비됐다는 뜻은 아니다. 시작 시 palette를 검증하고 라벨맵은 처음 사용할 때 해시·해상도·색상을 검증한다. 설치 전에는 manifest에 나열된 palette·label PNG·참고 CSV의 해시를 별도로 대조하고, 사용할 모든 모델/면의 마스크를 실제로 읽어 확인한다. `scenario_map.csv`·`zone_rects.csv`는 manifest의 참고 자산이며 엔진이 실행 중 직접 검증·평가에 사용한 파일로 집계하지 않는다.

공유 프로젝트의 기존 이미지 빌드가 `trainer/` 전체를 포함하므로 새 패키지도 같은 소스 버전으로 빌드할 수 있다. `deploy/spark`는 ARM64 전용이므로 설치 서버 아키텍처에 맞는 기존 배포 절차를 따른다. 이 PR은 별도 운영 Compose나 기존 라이브 서비스 설정을 자동으로 바꾸지 않는다.

## 4. 서비스 실행 설정

아래는 **각 컨테이너 내부**에서 적용할 설정과 실행 명령이다. `/zone-assets`, `/state`는 설치 시 볼륨으로 연결할 컨테이너 경로 예시다. 저장소 루트가 작업 디렉터리여야 한다. engine/explanation/qwen 포트는 내부 네트워크에서만 연결한다.

### 보정 엔진

```bash
export ZONE_ASSET_ROOT=/zone-assets
export ZONE_ENGINE_DB=/state/engine.db
python3 -m trainer.zone_engine.service
```

기본 내부 포트는 8122다. 엔진을 `engine`이라는 서비스명으로 실행하는 예시다.

### 설명 서비스

```bash
export EXPLANATION_QWEN_URL=http://qwen:11434
export EXPLANATION_MODEL=qwen3:4b
export EXPLANATION_MODEL_DIGEST=sha256:359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7
export EXPLANATION_LOG_DIR=/state/explanations
python3 -m trainer.coating_explanation_service.service
```

기본 내부 포트는 8123다. API의 `/api/ready`는 이 새 엔진·설명 서비스까지 확인하지 않으므로, 내부에서 엔진의 `/internal/v1/health`와 설명 서비스의 `/health`도 각각 확인한다. 위 digest는 실제 통합시험에 사용한 모델이다. 설치한 모델의 `/api/tags` digest가 다르면 같은 모델이라고 가정하지 말고 버전을 확인한다. 설명 서비스명은 아래 예시에서 `explanation`이다.

### 공유 API

```bash
export CORRECTED_ENGINE_URL=http://engine:8122
export CORRECTED_EXPLANATION_URL=http://explanation:8123
export CORRECTED_SOURCE_INSTANCE_ID=tsb-copilot-01
export SEQUENCE_DB_PATH=/state/sequences.db
export CORRECTED_DB_PATH=/state/corrected-api.db
python3 -m trainer.demo_api.app --host 0.0.0.0 --port 8120
```

`CORRECTED_SOURCE_INSTANCE_ID`는 필수이며 설치 인스턴스의 고정 ID로 관리한다. 시험용 ID를 기본값으로 사용하지 않는다. 기존 라이브 모드·모델·MCP 설정은 기존 설치 절차에 따르며, 위 명령은 그 설정을 대체하지 않는다. `CORRECTED_ENGINE_URL`을 설정하면 설명 서비스 주소와 `CORRECTED_DB_PATH`도 필수다. 새 DB는 기존 `SEQUENCE_DB_PATH`와 다른 실제 파일이어야 하며 같은 경로·심볼릭 링크·하드 링크는 거부한다. 설정하지 않으면 새 v2 경로를 등록하지 않는다.

격리된 시험망 밖에 공개하기 전에는 **`COPILOT_API_KEY` 또는 동등한 역방향 프록시 인증을 반드시 설정**한다. 키를 설정하면 v2도 같은 `X-API-Key` 또는 Bearer 값을 요구한다. 키 미설정 상태는 공개 운영 설정으로 사용하지 않으며 키를 소스에 넣지 않는다. 별도 corrected API DB에는 사용자 ID와 행동·이미지 요청 본문, 평가 및 질의 기록이 저장되므로 접근·보존 정책을 적용한다. 이 파일은 기존 `/admin`의 DB와 분리하고, v2 조회는 설정된 API 키 검사를 거친다. 기존 공개 뷰어에 이 파일을 `SEQUENCE_DB_PATH`로 연결하지 않는다.

### 공개 앞단의 요청 크기 제한

새 v2 API는 JSON 검증 전에 요청 크기를 제한하는 자체 미들웨어가 없다. **v2 요청을 노출하는 프록시에서 요청 본문을 최대 64MiB로 제한한 뒤 연결**한다. Nginx에서는 `/api/v2/`에 적용되는 server/location에 `client_max_body_size 64m;`을 지정한다. 내부 엔진의 64MiB 제한은 API가 본문을 읽고 저장한 뒤 적용되므로 앞단 제한을 대신하지 않는다. 배포 환경의 프록시 수정·기동은 이 PR의 실행 결과에 포함하지 않았다.

API→엔진·설명 호출은 `trust_env=False`로 운영체제의 프록시 환경변수를 사용하지 않는다.

## 5. API 연결 시 확인할 사항

계약 버전은 `sprayxr-corrected-zone/v2-draft-20261106`, 평가 모드는 `corrected_zone_v2`다. 실행 중인 서버의 `/openapi.json`과 `/docs`에서 공개 요청의 정확한 스키마를 확인한다.

| 메서드 | 경로 | 입력/용도 |
|---|---|---|
| POST | `/api/v2/sessions/start` | 사용자·세션·시각·실제 목표 두께·모델/회차/마스크 연결 |
| POST | `/api/v2/telemetry/sequence` | 시퀀스별 행동 프레임과 선택적 누적 crop |
| POST | `/api/v2/copilot/analyze-sequence` | 같은 시퀀스의 중복 방지 수신·처리 확인 |
| POST | `/api/v2/zones/observations` | Zone 완료 시 Height와 촬영 시각·회차·revision |
| POST | `/api/v2/sessions/end` | 세션 종료 |
| POST | `/api/v2/sessions/complete` | 공식 평균·진행률을 포함한 완료 정보(JSON) |
| GET | `/api/v2/sessions/{session_id}/status` | 처리 상태·최신 평가 연결 |
| GET | `/api/v2/assessments/{assessment_id}` | 해당 평가와 설명 생성 상태 |
| GET | `/api/v2/sessions/{session_id}/report` | 세션 종합과 공식 평균 |
| POST | `/api/v2/copilot/query` | 해당 세션·평가·revision에 저장된 설명 조회 |

- v2의 `sessions/complete`는 기존 v1의 multipart 파일 업로드와 다르다. JSON 본문을 사용하며 Height는 앞서 Zone 관측으로 보내야 한다.
- 대부분 수신 API의 202는 **저장·접수 성공**이다. 최종 평가·설명 완료와 구분해 상태를 조회한다. `analyze-sequence`는 제한 시간 동안 엔진 처리를 기다리며 설명 완성을 보장하지 않는다.
- 프레임 속도는 `speed_mm_s`, 거리는 `distance_mm`, 시각은 UTC다. 제공하지 못한 값은 `null`로 두며 임의로 채우지 않는다. 좌표·시간으로 계산한다면 실제 실행 좌표와 실제 실행 시간을 사용한다.
- PNG crop은 원본 해상도, 좌상단 원점이다. `map_num`과 `zone_id`를 함께 전송한다. 축소 이미지를 원래 해상도로 간주하지 않는다.
- `height_png.basis=current_round`는 해당 회차분이다. `cumulative_crop`는 회차 누적 관측이며 이전 회차와 중복 합산하지 않는다.
- 시퀀스의 `coating_progress`는 관측 상태다. 작업 대상 면적의 분모가 없을 때 진행률 백분율은 `null`로 반환한다. 기존 v1 진행률을 새 엔진이 자동으로 대체하지 않는다.
- 현재 query는 저장된 설명 조회다. 질문에 맞춰 새 추론을 수행하는 자유 대화 기능은 포함하지 않는다.

엔진 내부 이벤트 예시는 [엔진 계약](ENGINE_CONTRACT_KR.md), 계산 규칙은 [TSB 확인 규칙](TSB_THICKNESS_RULES_KR.md)을 참고한다. 내부 이벤트 봉투와 공개 v2 본문을 혼용하지 않는다.

## 6. 평가·코칭의 의미

회차별 raw Height를 픽셀 단위로 합산한 뒤 DeadVolume 10µm를 한 번 차감하고, 1600µm 상한과 1µm 유효 기준을 적용한다. 정확한 컬러 마스크 안의 도포부 평균으로 목표 미만/목표 이상~1.5배 미만/1.5배 이상을 판정한다. 미도장률은 별도로 관리한다. TSB 공식 세션 평균은 별도 보존한다.

Qwen은 저장된 근거·원인 후보·행동 중 사용할 항목을 선택한다. 허용 ID인지 검증한 뒤 한국어 안내를 조립한다. 관측되지 않은 원인, 최적 속도, 예상 개선량을 생성하지 않는다. 행동 조절 방향은 현재 규칙의 후보이며 실험으로 검증된 효과가 아니다.

목표·회차·구역 관측이 부족하면 부분 평가 또는 판단 유보 상태를 반환한다. 과거 전문가 설정값을 목표가 확인되지 않은 새 세션에 임의로 넣지 않는다.

## 7. 시험 결과와 남은 확인

- 원래 독립 통합 환경: 2026-11-06, 새 단위 54/54·기존 회귀 195/195·HTTP 확인 275/275, 실제 Qwen 생성 27회/fallback 0. 이 수치는 해당 보존 코드의 결과이며 이 PR 이관본의 새 실행 결과와 구분한다.
- 최신 공유 main 이관본 최종 검사(2026-11-11): 통합·인증·별도 DB 검사 68개와 기존 API·OpenAPI·뷰어·관리 기능 회귀 287개, **총 355/355 통과(20.64초)**. 모의 HTTP를 사용한 CPU 검사이며 이번 작업에서 실제 Qwen을 다시 실행하지 않았다. 새 요청이 별도 DB에 저장되고 기존 관리 뷰어 DB에는 남지 않는 것도 확인했다.
- 전체 Python 검사도 시도했으나 CPU 시험 이미지에 PyTorch·Matplotlib이 없어 16개 모듈 수집 단계에서 중단됐다. 수정 전 main에서도 같은 16개 모듈이 중단됨을 확인했다. 전체 테스트 통과로 해석하지 않는다.
- 기존 10월8일 개발자 자료는 목표 두께가 확인되지 않아 등급·조절 방향을 유보했다. 최종 Height를 구역 관측으로 변환한 시험이며 실제 Zone 완료 시점 데이터는 아니다.
- 실제 Unity 새 요청 전송, TSB 서버 아키텍처에서의 기동, 기존 27B와의 동시 실행, 현장 코칭 효과는 이 PR의 검증 결과가 아니다.
- 늦게 도착하거나 수정된 이전 회차가 이미 저장된 다음 회차 평가를 자동 갱신하지 않는다. 이전 회차를 먼저 확정하거나 다음 회차 관측을 새 revision으로 보내야 한다.
- 세션 설명의 후보 선택 수가 제한되므로 39개 Zone의 주요 안내를 모두 포함하는지는 추가 확인이 필요하다.
- 원본·전달 대기 작업·평가 원장에 요청이 중복 저장될 수 있다. 실제 세션의 저장 용량·보존/정리 주기·백업과 다수 Zone 부하 검증은 설치 환경에서 확인해야 하며, 요청당 크기 제한만으로 전체 디스크 사용량이 제한되는 것은 아니다.

### CPU 검사 명령

기존 프로젝트 컨테이너의 저장소 루트에서 실행한다. 아래 새 검사에는 외부 데이터·실제 Qwen·GPU가 필요하지 않다. 합성 마스크와 모의 HTTP 응답을 사용한다.

```bash
python3 -m pytest -q \
  trainer/zone_engine/tests \
  trainer/coating_explanation_service/tests \
  trainer/demo_api/tests/test_corrected_contracts.py \
  trainer/demo_api/tests/test_corrected_flow.py \
  trainer/demo_api/tests/test_corrected_store.py \
  trainer/demo_api/tests/test_corrected_access_control.py \
  trainer/demo_api/tests/test_corrected_app_wiring.py
```

같은 실행에서 기존 API·OpenAPI·뷰어·관리 회귀도 함께 확인했다. 검사 모듈 목록과 결과 구분은 [검증 요약](VALIDATION_20261111.json)에 기록했다.

프로젝트 전체 검사는 기존 지침대로 `python3 -m pytest trainer mcp_servers -q`다. 모델 관련 패키지가 없는 CPU 시험 이미지에서 전체 검사가 중단되는 경우, 전체 통과로 표현하지 않는다.

## 8. 설정 해제와 자료 보존

기존 동작만 사용할 때는 API의 두 `CORRECTED_*_URL` 설정을 제거하고 API를 재시작한다. 엔진·설명 서비스를 중지하되 corrected API DB·기존 API DB·엔진 DB·설명 감사 기록을 삭제하지 않는다. v2 전용 테이블과 내부 시퀀스 사본은 별도 corrected API DB에만 저장하며 기존 v1 자료를 덮어쓰지 않는다. 11월6일 로컬 시험은 공유 DB 방식이었지만 이 PR은 별도 DB를 요구한다. 과거 DB를 자동 이전하지 않으며 기존 시험 상태가 필요하면 별도 이전 검토가 필요하다. 실행 중인 DB를 단순 복사해 백업 완료로 판단하지 말고 기존 서버의 SQLite 백업 절차를 따른다.
