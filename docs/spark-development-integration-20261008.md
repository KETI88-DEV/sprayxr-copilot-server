# Spark·개발 서버 통합 브랜치

## 통합 범위

- 브랜치: `integration/spark-dev-20261008`.
- 원격 `main` 기준점: `6f4274d131347b671f922e6d3c0ec63923799b91`.
- 개발 서버 기준점: `672f763c317000123162758a3882c18ec568a5cf`.
- 원격 기준점에 없는 개발 서버 커밋 19개를 일반 merge로 통합한다. 개별 커밋 이력을 보존하며 squash/rebase하지 않는다.
- Spark의 음성·전문가 기준 안내·Swagger·최신 TSB 계약과, 개발 서버의 다국어 평가·리그별 자세 기준·시퀀스 자세 채점·세션 종합 분석을 함께 보존한다.
- 개인 도구 정리 커밋 `ea075a20`을 유지한다. `.agent/`와 루트 npm 도구 파일을 다시 추적하지 않는다.

## 충돌 해결

| 파일 | 통합한 동작 |
|---|---|
| `trainer/demo_api/app.py` | guided workflow·voice router 등록을 유지하고, 같은 `V1Store`와 설정된 MCP 주소를 `create_v1_router`에 함께 전달 |
| `trainer/demo_api/v1_router.py` | Spark의 Swagger 태그·예시·선택형 평균 속도/거리 필드를 유지하면서 `BackgroundTasks`와 원시 CSV를 이용한 세션 분석 연결 |

`telemetry_router.py`, 배포 Compose와 Dockerfile은 자동 병합됐지만 의미상 변경도 검토했다. 전문가 가이드와 시퀀스 내용 해시 처리를 모두 유지하고, Spark의 Whisper 설정 및 개발 서버의 리그 프로필 복사를 함께 남겼다.

통합 회귀 테스트는 shared store/MCP 연결과, 평균 속도·거리 없는 최신 TSB 완료 요청이 접수된 뒤 실제 분석 완료 상태로 전환되는 동작을 확인한다.

## 배포와 분리

이 브랜치를 올리는 것은 `main` 병합 또는 Spark 배포 승인이 아니다. 운영 컨테이너·이미지·모델·DB를 교체하지 않았다. 본 서버의 기존 작업 디렉터리에서도 checkout/reset하지 않았다.

2026-10-08 읽기 전용 확인에서 현재 Spark 번들에는 `output/expert_reference/expert_arm_reference_avatar_2026-10.json`이 없었다. 통합 Compose는 이 리그별 기준을 사용하므로 **새 번들을 검증·준비하기 전 운영 이미지만 교체하면 안 된다.** 생성·포함 절차는 [Spark 배포 문서](../deploy/spark/README.md)에 따른다.

ARM 이미지 빌드, 운영 모델을 통한 실제 음성/다국어 질의, UI 수동 검수는 별도 배포 검증 단계다. CPU 중심 자동 회귀 통과만으로 이를 완료했다고 간주하지 않는다.

## 검증 환경의 실물 샘플

전체 테스트는 Git 외부의 실물 자료에도 의존한다. `docs/SprayXR_CoPilot_샘플_2026-09-21.json`과 컨테이너의 `/data/sources/new_format/sample_2026-10-08`을 별도로 준비해야 한다. 이 자료는 이번 커밋에 포함하지 않는다.

격리 작업 공간의 첫 전체 실행에서는 앞의 JSON이 없어 3개 테스트가 `FileNotFoundError`로 실패하고 관련 MCP 테스트 29개가 생략됐다. 기존 개발 서버의 원본을 작업 공간에서 읽을 수 있도록 로컬 링크만 추가한 뒤 전체 검증을 다시 실행했다. 실패를 없애기 위해 테스트를 삭제하거나 skip 조건을 추가하지 않았다.

## 검증 결과

| 검증 | 결과 |
|---|---|
| 원격 기준점의 충돌 관련 테스트 | 95 passed |
| 통합 후 질의·시퀀스·리포트·다국어 관련 테스트 | 132 passed |
| 추가한 연결부 회귀를 포함한 앱/V1 테스트 | 39 passed |
| `trainer mcp_servers` 전체 재실행 | **1900 passed, 2 warnings**, 304.12초; 실패·생략 없음 |
| `deploy/spark/scripts` | 9 passed |
| 변경된 JavaScript 3개 | `node --check` 통과 |
| Git 충돌 표식·공백 검사 | 미해결 충돌 없음, `git diff --cached --check` 통과 |
| 병합 부위 별도 코드 리뷰 | 차단 사항 없음; 운영 배포 승인과는 별개 |

전체 테스트는 provisioned `ship-paint-trainer` 컨테이너에서 `/workspace/.omc/integration-spark-20261008`을 작업 디렉터리로 실행했다. 2개 경고는 SWIG 타입의 `__module__` 관련 기존 DeprecationWarning이다. 테스트용 샘플 링크는 추적하지 않는다.
