# DGX Spark Co-Pilot 통합 이관 설계

## 1. 목표와 완료 기준

기존 Ubuntu Co-Pilot 서버의 API 계약, SQLite 데이터, 전형기 반장 전문가 기준 데이터와 평가 기능을 DGX Spark의 현행 Shipcoat 서비스로 이관한다. DGX에 이미 구현된 27B RAG 질의와 자세 비교 기능은 유지한다.

완료 조건은 다음과 같다.

- 기존 텍스트 `POST /api/v1/copilot/query` 계약과 기능이 유지된다.
- 별도 `POST /api/v1/copilot/query-voice`가 WAV 음성을 받아 transcript와 기존 Query 형식의 답변을 반환한다.
- `POST /api/v1/copilot/analyze-sequence`가 전문가 Zone 기준으로 속도와 이격거리 guide를 매번 반환한다.
- 훈련모드 1~10의 Zone 순서, 속도, 이격거리, 궤적 및 높이맵 기준 데이터가 DGX에 보존되고 실제 판정에 사용된다.
- 기존 DB와 원본 프로젝트는 체크섬으로 검증한 복구 가능한 스냅샷으로 DGX에 보존된다.
- DGX 내부·재부팅·회귀 시험이 통과하기 전에는 외부 포트포워딩을 변경하지 않고 기존 서버를 폐기하지 않는다.

## 2. 이관 전략

기존 x86 프로젝트를 DGX ARM64 런타임 위에 그대로 실행하지 않는다. 기존 프로젝트 전체는 감사·복구용 원본 스냅샷으로 복사하고, 실행 기능은 DGX의 `/home/keti/shipcoat/repo` 코드와 컨테이너 구조에 선택적으로 통합한다. x86 가상환경도 원본 보존 대상이지만 DGX 런타임에서는 사용하지 않는다.

이관은 다음 순서로 수행한다.

1. 기존 서비스의 파일 목록, DB 통계, 서비스 설정과 SHA-256 manifest를 저장한다.
2. 기존 프로젝트를 DGX staging 디렉터리로 복사하고 manifest를 재검증한다.
3. 전문가 CSV·PNG·생성 프로필과 SQLite DB를 DGX 영속 볼륨 구조로 옮긴다.
4. 코드 변경은 별도 Git 브랜치/작업공간에서 테스트 우선으로 구현한다.
5. 새 이미지를 별도 태그로 빌드하고 내부 포트에서 시험한다.
6. API 계약, 전문가 판정, 자세 비교, RAG 질의, Swagger, 재부팅 복구를 검증한다.
7. 사용자에게 전환 준비 완료를 보고한 뒤 공유기 포트포워딩만 수동 변경한다.

## 3. 단일 27B 질의 정책

27B 모델 하나를 유지한다. 요청마다 모델을 적재하거나 해제하지 않는다.

- 실시간 시퀀스의 속도·거리·Zone 판정은 LLM을 호출하지 않는다.
- 간단한 지식 질의는 `fast` 정책을 사용한다: thinking 비활성화, RAG 상위 1~2건, 짧은 프롬프트, 24~40 출력 토큰, 1~2문장 답변.
- 상세 Replay 질의와 최종 리포트는 기존 상세 생성 경로를 사용하고, 실시간 질의보다 낮은 우선순위로 큐 처리한다.
- 근거가 부족하면 모델이 값을 추측하지 않고 판정 불가를 명시한다.

## 4. 음성 Query API

기존 JSON 텍스트 Query API는 변경하지 않는다. 음성 API는 입력 어댑터이며 STT 이후 동일한 Query 서비스로 위임한다.

`POST /api/v1/copilot/query-voice`는 `multipart/form-data`로 다음 필드를 받는다.

- `audio`: WAV, 16 kHz, mono, 최대 15초
- `user_id`: 필수
- `language`: `ko`, `vi`, `id`, `th`
- `session_id`, `scenario_id`, `request_id`: 선택
- `task_type`: 기본 `knowledge_search`
- `use_rag`: 기본 `Y`

응답은 기존 `CopilotResponse.data`에 `transcript`, `language`, `stt_latency_ms`를 추가한 전용 계약을 사용한다. 답변 생성은 fast 정책을 기본으로 한다.

오디오 형식·길이 오류는 422, 크기 제한 초과는 413, STT 또는 모델 비가용은 503으로 반환하며 내부 오류를 노출하지 않는다. STT 모델은 DGX ARM64/CUDA에서 네 언어 정확도와 27B 동시 실행 메모리를 측정한 뒤 가장 작은 적합 모델을 선택한다. 목표 지연시간은 warm 상태에서 전체 3~6초이며, 시험 결과와 함께 확정한다.

## 5. 시퀀스 속도·이격거리 guide

`analyze-sequence`에서 현재 Zone·mode와 전형기 전문가 프로필을 조회해 관측 속도와 이격거리를 기준 분포와 비교한다. 판정은 결정론적 평가기로 구현하며 LLM을 사용하지 않는다.

응답 `data.guide`는 항상 존재한다.

```json
{
  "guide": {
    "speed": {"status": "slow", "message": "이동 속도를 조금 높이세요.", "available": true},
    "distance": {"status": "far", "message": "스프레이건을 피도면 쪽으로 가까이 이동하세요.", "available": true}
  }
}
```

- speed 상태: `slow`, `ok`, `fast`
- distance 상태: `near`, `ok`, `far`
- 정상 시퀀스에도 `ok`를 반환한다.
- 최근 2~3개 유효 시퀀스와 경계 히스테리시스로 UI 깜빡임을 억제한다.
- Zone ID 또는 기준 프로필이 없으면 `status: null`, `available: false`, `reason: reference_unavailable`을 반환한다. 임의로 `ok`를 만들지 않는다.
- 실전모드 A-1~A-3은 Zone ID가 다시 생성될 때까지 판정 불가로 유지한다.

현재 TSB JSON 입력에는 누적 히트맵을 요구하지 않는다. 누적 상태가 필요한 평가는 서버가 세션 단위로 관리한다.

## 6. 데이터 및 호환성

전문가 원본은 읽기 전용 raw 영역에 보존하고, 런타임은 버전이 붙은 파생 프로필을 읽는다. 프로필에는 source hash, mode, Zone, 단위, 표본 수와 생성 버전을 기록한다. 높이맵 PNG는 기준 산출과 오프라인 검증에 사용하고, 운영 API 입력은 TSB 텍스트/JSON 계약을 유지한다.

기존 DB를 바로 덮어쓰지 않는다. DGX DB를 백업한 뒤 스키마별 변환 도구로 병합하고, 사용자·세션·시퀀스 충돌은 원본 ID와 출처를 보존해 해결한다.

## 7. 시험과 배포 안전장치

- 계약 테스트: 기존 Query 요청/응답이 변경되지 않는지 확인한다.
- 음성 테스트: 정상 WAV, 잘못된 형식, 15초 초과, 지원 언어별 transcript와 장애 응답을 확인한다.
- guide 테스트: slow/ok/fast, near/ok/far, 정상 복귀, 경계 안정화, 기준 부재를 확인한다.
- 전문가 데이터 테스트: 훈련모드 1~10 프로필과 raw 파일 hash를 확인한다.
- 회귀 테스트: pose_recomm, knowledge_search, telemetry, report, Swagger를 확인한다.
- 운영 테스트: 동시 fast 질의와 상세 생성, 메모리 최고점, 컨테이너 재시작과 호스트 재부팅을 확인한다.

외부 전환은 위 시험 통과 후 수행한다. 실패 시 공유기 포트포워딩을 기존 서버로 되돌리는 것이 롤백이며, 기존 서버와 백업은 안정화 기간 동안 유지한다.
