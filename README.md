# 선박 도장 AI Co-Pilot (SprayXR)

**[API 문서](http://keti-xr-lab.iptime.org:58000/docs) · [DB뷰어](http://keti-xr-lab.iptime.org:58000/admin) · [3D 모션뷰어](http://keti-xr-lab.iptime.org:58000/motion-viewer)**

선박 도장·블라스팅 훈련 시뮬레이터와 연동해 작업자에게 피드백과 훈련 결과를 제공하는 AI Co-Pilot 서버입니다.
전문가 기준 데이터와 훈련 기록을 비교하고 작업 자세와 도장 경로를 시각화합니다. 훈련 중 발생하는 질문에도 답변합니다.

현재 버전은 **v3.8.2**입니다. [버전별 변경 이력](CHANGELOG.md)과 [GitHub 릴리스](https://github.com/KETI88-DEV/sprayxr-copilot-server/releases)에서 추가 기능을 확인할 수 있습니다.

세션 시작 응답의 `first_zone_id`와 `first_zone_guidance`에서 첫 작업 Zone을 확인할 수 있습니다. 스프레이·블라스팅별로 등록된 기준 순서를 사용하며, 기준이 없는 실전·Unit 콘텐츠는 안내 불가 사유를 반환합니다.

## 주요 기능

| 기능 | 설명 |
|---|---|
| 훈련 세션 관리 | 훈련 시작부터 시퀀스 수집, 종료, 결과 조회까지 세션 단위로 관리합니다. |
| 시퀀스 피드백 | 입력된 시퀀스를 분석해 작업 상태와 속도·이격거리 가이드를 제공합니다. |
| 전문가 기준 비교 | 작업 자세, Zone별 도장 방식과 작업 순서를 전문가 데이터와 비교합니다. |
| 텍스트·음성 질의 | 훈련 맥락과 관련 지식을 활용해 답변합니다. 빠른 안내와 상세 분석을 선택할 수 있습니다. |
| 훈련 리포트 | 업로드된 기록과 높이맵을 바탕으로 훈련 결과를 정리합니다. |
| 3D 모션뷰어 | 작업자와 전문가의 기록을 재생하며 도장 경로, Zone, 작업 자세와 도막 분포를 확인합니다. |
| DB뷰어 | 세션 상태와 API 호출 이력을 조회합니다. 관리자 모드에서는 관리 정보 수정과 기록 삭제가 가능합니다. |

분석에 필요한 데이터나 기준 자료가 없는 경우에는 API 응답에 분석 가능 여부와 사유를 표시합니다.

## 접속 안내

현재 운영 서버에서는 다음 페이지를 사용할 수 있습니다.

| 페이지 | 용도 |
|---|---|
| [API 문서](http://keti-xr-lab.iptime.org:58000/docs) | API별 입력값과 응답 형식을 확인하고 예제로 호출해 볼 수 있습니다. |
| [DB뷰어](http://keti-xr-lab.iptime.org:58000/admin) | 훈련자, 세션, 시퀀스, 질문·답변, 리포트와 호출 이력을 조회합니다. |
| [3D 모션뷰어](http://keti-xr-lab.iptime.org:58000/motion-viewer) | 훈련 기록을 재생하고 전문가 기준과 비교합니다. |
| [3D 모션뷰어 v2](http://keti-xr-lab.iptime.org:58000/motion-viewer/v2) | 같은 기록을 큰 3D 장면과 오른쪽 고정 질문 열로 재생합니다. 기존 화면은 그대로 둡니다. |

### 모의 연동

- 시뮬레이터 없이 연동 흐름을 확인하려면 API 문서의 **0. 모의 연동**에서 **0-0부터 0-4까지** 순서대로 실행합니다.
- 테스트에 필요한 기본값이 준비되어 있고 응답에서 실제로 사용된 입력값을 확인할 수 있습니다.
- 모의 API는 **모의훈련자**(`COPILOT-DEMO-USER`) 계정을 재사용하고 새 훈련을 시작할 때 세션을 생성합니다.

### 대화 구분

- 질의는 도장 지식뿐 아니라 일반 대화와 후속 질문도 처리합니다. 도장 지식이 필요한 질문에만 `use_rag=Y`의 검색을 적용합니다.
- 모션뷰어는 채팅 창별 대화 식별자를 자동으로 보냅니다. API 클라이언트는 선택 헤더 `X-Copilot-Conversation`으로 대화를 구분할 수 있습니다(영문·숫자·`._:-`, 최대 128자).
- 헤더를 생략하면 같은 사용자·훈련 세션의 기본 대화를 이어갑니다. 훈련 세션이 없는 질의도 사용자별 대화 기록으로 저장합니다.
- 이 식별자는 인증 기능이 아닙니다.

### 호출 출처 표시

- DB뷰어는 훈련 대상 작업자와 호출 채널을 따로 표시합니다. 모션뷰어 호출은 **3D VIEWER**, 자동 검증은 **TEST**, 모의 API는 **TEST API**로 표시하고 API 호출에서 수집한 IP를 보여 줍니다.
- 선택 헤더 `X-Copilot-Source`는 `motion-viewer` 또는 `assistant-test`를 받는 출처 표시용 힌트입니다.
- 과거에 IP를 수집하지 않은 기록은 **미기록**, 파일로 수집한 기록은 **해당 없음**으로 표시합니다.
- 프록시·공유기를 거친 경우 서버에 전달된 주소가 기록되며 실제 작업자 신원을 보증하지 않습니다.

### DB뷰어 권한

- 일반 조회에는 로그인이 필요하지 않습니다. 수정·삭제 기능은 좌측 메뉴 하단의 **관리자 모드**에서 로그인한 뒤 사용할 수 있습니다.
- 삭제 전에는 연결된 기록의 범위를 확인하며 DB 백업과 변경 이력이 저장됩니다.

## 시뮬레이터 연동

기본적으로 아래 순서로 호출합니다.

```text
훈련 시작 → 시퀀스 전송(반복) → 훈련 종료 → 결과 파일 업로드 → 리포트 조회
```

| 메서드 | API | 용도 |
|---|---|---|
| `POST` | `/api/v1/sessions/start` | 훈련 세션을 시작합니다. |
| `POST` | `/api/v1/telemetry/sequence` | 시퀀스 데이터를 저장합니다. |
| `POST` | `/api/v1/copilot/analyze-sequence` | 시퀀스 데이터를 저장하고 피드백을 요청합니다. |
| `POST` | `/api/v1/copilot/query` | 텍스트로 질문합니다. |
| `POST` | `/api/v1/copilot/query-voice` | 녹음한 음성으로 질문합니다. |
| `POST` | `/api/v1/sessions/end` | 훈련을 종료합니다. |
| `POST` | `/api/v1/sessions/complete` | CSV·높이맵 등 결과 파일을 업로드하고 리포트 분석을 요청합니다. |
| `GET` | `/api/v1/reports/{report_id}` | 리포트의 처리 상태와 분석 결과를 조회합니다. |

`telemetry/sequence`와 `analyze-sequence`는 필요한 기능에 따라 선택합니다. 같은 시퀀스를 두 경로에 모두 전송할 필요는 없습니다.
질문 API는 훈련 중에도 사용할 수 있으며 세션 ID를 함께 보내면 저장된 훈련 기록을 답변에 활용할 수 있습니다.
자세한 파라미터와 예시 요청은 [API 문서](http://keti-xr-lab.iptime.org:58000/docs)를 참고하세요.

## 동작 기준

### 도막 두께와 등급

- 두께는 PR #8 도막 기준의 보정값으로 잽니다. 받은 두께에서 10µm를 한 번 빼고 1600µm에서 자릅니다. 1µm 미만은 칠하지 않은 곳으로 봅니다.
- 새로 시작하는 세션에서 목표를 생략하면 기본 목표 160µm로 판정하고 "기본 목표 160µm 기준"으로 표시합니다. 목표를 명시한 세션은 "세션 목표 Nµm 기준"으로 표시합니다.
- Zone 등급은 칠한 곳의 평균 두께로 정합니다. 목표 미만은 저도막, 목표의 1.5배 이상은 과도막입니다.
- 등급은 시퀀스 분석의 `coating_feedback`과 완료 리포트에 실립니다.
- 조절 후보(속도·이격거리)는 규칙에 따른 후보이며 원인으로 확정된 것이 아닙니다.

### 다음 Zone 안내

- 스프레이의 다음 Zone 안내는 두 조건을 모두 채울 때만 나갑니다. 공식 구역 픽셀의 95% 이상이 세션 목표 두께를 충족해야 하고 칠한 곳의 평균 두께도 목표 이상(저도막이 아님)이어야 합니다.
- 면적을 채웠어도 평균이 목표 미만이면 현재 Zone에 머물며 얇은 곳을 보완하라고 안내합니다. 과도막 Zone에는 추가 분사를 멈추라고 안내합니다.
- 부분 크롭은 최신 누적 두께로 덮어쓰고 아직 관측하지 못한 구역도 완료 면적의 분모에 포함합니다.
- 도막 자료나 공식 구역 정보가 없으면 완료로 판단하지 않습니다. 블라스팅에는 이 도막 기준을 적용하지 않습니다.

### 이상 알림

- 이상 알림 SSE로 과도막(`coating_over_app`)과 저도막(`coating_under_app`, 방금 떠난 Zone) 알림을 보냅니다.
- 서버가 떠 있는 동안 세션마다 Zone당 종류별로 한 번씩 보냅니다.

### 모션뷰어 도막 카드와 완료 리포트

- 모션뷰어의 도막 카드는 기록 옆 최종 높이맵으로 잰 목표 대비 평가를 기본으로 보여 줍니다. 한 번 누르면 전문가 최종 도막과 비교한 값으로 바뀝니다.
- 높이맵이 없는 기록은 이유를 밝히고 전문가 대비로 열립니다.
- 완료 리포트의 주 판정은 목표 대비 등급입니다. 전문가 최종 도막과 견준 판정은 보정 전 두께 기준의 참고 정보(`expert_comparison`)로만 함께 실립니다.

### 상세 분석

- 상세 분석은 Qwen3.6-27B를 유지하면서 기존 27B 빠른 답변 수준의 짧은 분석을 제공합니다.
- 핵심 답변·근거·조치를 2문장 또는 최대 2개 짧은 항목으로 정리합니다. 이미 계산된 자세·순서 평가는 추가 LLM 생성 없이 설명합니다.
- 실제 응답 시간은 질문·검색·번역·서버 부하에 따라 달라집니다.

### 도포 방향 질문

- 모션뷰어에서 "도포 방향은?"이라고 물으면 현재 재생 시점 주변의 Ray Point 궤적으로 방향을 판단해 **수평·수직·대각선·비정형**으로 답합니다.
- 같은 분사 구간 안의 인접 궤적을 사용합니다. 유효한 분사 궤적이 없거나 정지한 시점에는 판단할 수 없다고 안내합니다.
- 전체 기록이나 Zone의 대표 방향을 명시적으로 물으면 해당 범위를 유지합니다.
- 프레임 방향 자료는 서버가 계산·보관하므로, 오래된 화면에서는 재생 기록을 다시 불러와야 합니다.
- TSB의 API 요청 JSON에 새 필수 항목을 추가하지 않습니다.

### 훈련 전 기준 순서 안내

- 훈련 시작 전에도 `scenario_id`로 학습 모드를 알 수 있고 등록된 기준이 있으면 Query API가 Zone 작업 순서를 안내합니다.
- 공정이 없고 스프레이·블라스팅 기준이 모두 있으면 공정별로 구분합니다.
- 기준 안내는 실제 방문 순서·준수율·도막 완료 판정과 별개입니다. 지금 다음 Zone으로 이동할 수 있는지 묻는 질문은 저장된 시퀀스로 판단하며, 시퀀스 피드백과 같은 도막 완료 기준을 씁니다.

### 의견 보내기

- 상단의 **의견 보내기**에서 API 문서·DB뷰어·모션뷰어에 대한 개선 의견을 남길 수 있습니다.
- DB뷰어의 **사용자 의견**에서 조회합니다. 처리 상태 변경은 기존 관리자 인증으로 제한합니다.
- 의견 제출은 훈련자나 훈련 세션을 생성하지 않습니다.

## 데이터 구성

전문가 비교에 사용하는 자료와 API 운영 기록은 별도로 관리합니다.

- **전문가 기준 자료**: Zone ID가 포함된 CSV, 도막 높이맵 PNG와 이를 가공한 비교 프로필입니다. 훈련자의 작업을 평가하는 기준으로 사용합니다.
- **API 운영 기록**: 세션, 시퀀스, 질문·답변, 리포트와 API 호출 이력입니다. SQLite에 저장되며 DB뷰어에서 확인할 수 있습니다.
- **모델·검색 자료**: LLM 모델, 학습 어댑터와 RAG 검색 인덱스입니다. 지식 검색과 답변 생성에 사용합니다.
- 신규 훈련 코퍼스는 원본 출처와 Zone ID 유무를 보존하는 기록별 학습용 JSON으로 등록합니다. LLM 추가 학습이나 전문가 판단 기준 교체는 별도 작업입니다.

원본 훈련 데이터, 모델 파일, 운영 DB와 인증 정보는 이 저장소에 포함하지 않습니다. 새 환경에 배포할 때는 아래 "모델·데이터 배치"대로 따로 준비합니다.

## 모델·데이터 배치

모델 가중치(체크포인트), 색인, 운영 데이터는 git에 넣지 않습니다. 배포 PC에서는 `deploy/spark/.env`의 경로 변수로 위치를 정하고 컨테이너는 그 경로를 마운트해 읽습니다.

- 현재 상세분석용 27B는 v6.1 LoRA가 이미 병합된 BF16 모델입니다.
- 빠른 안내용 Qwen3-8B-AWQ는 별도 모델이며 Co-Pilot v6.1 LoRA 적용을 확인한 경로가 아닙니다.
- API 코드만 복제하면 모델 자산이 포함되지 않으므로, 설치할 때 병합 모델과 참조 자산을 함께 옮겨야 합니다.

| 무엇 | 넣는 위치 | 컨테이너 경로 |
|---|---|---|
| 27B 병합 체크포인트(약 52GB, safetensors 15개와 설정·토크나이저 파일) | `${BUNDLE_DIR}/output/merged/qwen36_27b_shipcoat_v6_1_bf16/` | vLLM `/models/shipcoat` |
| 임베딩·재정렬 모델 | `${BUNDLE_DIR}/models/KURE-v3/`, `${BUNDLE_DIR}/models/KURE-Reranker-nano/` | agent-api `/models/…` |
| RAG 색인(`faiss.index`, `metadata.json`) | `${BUNDLE_DIR}/output/indexes/kure3-20261113/` | agent-api `/output/indexes/kure3-20261113` |
| 자세 기준(`expert_arm_reference*.json`) | `${BUNDLE_DIR}/output/expert_reference/` | mcp `/output/expert_reference` |
| 서빙 설정(`rag_config_spark_vllm_bf16*.yaml`) | `${BUNDLE_DIR}/config/` | agent-api `/bundle-config` |
| 빠른 안내 8B 모델(Qwen3-8B-AWQ) | `${QUICK_MODEL_DIR}` | quick-vllm `/models/quick` |
| 음성 인식 모델(whisper-large-v3-turbo) | `${STT_MODEL_DIR}` | agent-api `/stt-model` |
| 전문가·작업자 기록 | `${EXPERT_REFERENCE_DIR}` | agent-api `/expert_reference_data` |
| 학습모드 FBX | `${POSE_MODELINGS_DIR}` | agent-api `/data/sources/modelings` |
| 운영 DB와 리포트(쓰기) | `${RESULTS_DIR}` | agent-api `/output/eval_results` |

- **학습 체크포인트는 배포 PC에 두지 않습니다.** LoRA 어댑터(`/output/adapters`)와 학습 중간 저장(`/output/checkpoints`)은 개발 서버에만 둡니다. 개발 서버에서 병합하고 검증한 병합본(`/output/merged/…`)만 번들로 옮깁니다.
- 배포 PC용 상세 안내(파일 목록, 확인 명령, 띄우는 순서)는 [`deploy/release/README.release.md`](deploy/release/README.release.md)에 있습니다. 같은 내용이 `release` 브랜치의 README로 들어갑니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| [`trainer/demo_api/`](trainer/demo_api/) | 시뮬레이터 연동 API, 세션·리포트 처리, DB뷰어와 3D 모션뷰어 |
| [`trainer/pose_lab/`](trainer/pose_lab/) | 자세 비교 분석을 위한 실험·검증 도구 |
| [`trainer/scripts/`](trainer/scripts/) | 기준 데이터 생성, 모델 학습과 평가 도구 |
| [`trainer/configs/`](trainer/configs/) | 모델, RAG와 학습 설정 |
| [`mcp_servers/`](mcp_servers/) | 자세 비교 등 분석 도구의 API |
| [`demo-ui/`](demo-ui/) | React·Electron 기반 개발 콘솔 |
| [`deploy/spark/`](deploy/spark/) | DGX Spark 배포 설정과 운영 절차 |
| [`deploy/release/`](deploy/release/) | `release` 브랜치를 만드는 스크립트, 포함 파일 목록, 배포 PC용 README |
| [`docs/`](docs/) | 연동 규격과 계약 문서 |

## 개발과 배포

운영 서버는 DGX Spark에서 Docker Compose로 실행합니다. 환경 준비와 자산 배치, 이미지 빌드·기동 방법은 [Spark 배포 안내](deploy/spark/README.md)를 참고하세요.
DB뷰어와 3D 모션뷰어는 FastAPI 서버에서 제공하며 `demo-ui/`의 개발 콘솔과는 별도로 사용할 수 있습니다.

Python 명령은 프로젝트 의존성이 설치된 컨테이너의 저장소 루트(`/workspace`)에서 실행합니다.

```bash
# API 검사
python3 -m pytest trainer/demo_api/tests -q

# 전체 Python 검사
python3 -m pytest trainer mcp_servers -q
```

일부 통합 검사는 모델과 기준 데이터가 준비된 환경에서 실행해야 합니다.
웹·Electron 콘솔을 수정할 때는 `demo-ui/`에서 빌드와 검사를 실행합니다.

```bash
cd demo-ui
npm ci
npm run build:web
npm test
```

## 릴리스 관리

기능 추가는 `MINOR`, 호환되는 오류 수정은 `PATCH`, 호환되지 않는 변경은 `MAJOR`를 올립니다. API 경로의 `/api/v1`과 소스 릴리스 버전은 별도로 관리합니다.

### 브랜치

- `main`은 개발 통합 브랜치입니다.
- `release` 브랜치와 버전 태그에는 API 서버(agent-api, MCP) 운영에 필요한 코드·설정·배포 정의와 배포 PC용 README만 담습니다(`deploy/release/release_files.txt`). 문서, 테스트, 학습·평가 스크립트, 웹 UI(demo-ui)는 넣지 않습니다.
- `release`는 main 이력과 끊긴 별도 이력입니다. 배포 PC는 `git clone --single-branch --branch release <저장소 주소>`로 받고 이후 `git pull --ff-only`로 갱신합니다.
- 개발 문서(설계·측정·보고 자료)는 어느 브랜치에도 올리지 않습니다([저장소 추적 범위](docs/repository-hygiene.md)).

### 릴리스 만드는 순서 (v3.0.2부터)

1. `main`에서 `trainer/demo_api/__init__.py`의 `__version__`을 올리고 `CHANGELOG.md`에 같은 버전의 변경 내용을 적은 뒤, 관련 검사를 실행하고 커밋합니다.
2. `deploy/release/make_release.sh main vX.Y.Z`를 실행합니다. 스크립트가 운영 파일만 담은 커밋과 태그를 로컬에 만들고 import 확인·push·GitHub 릴리스 게시 명령을 출력합니다.
3. 출력된 순서대로 확인하고 `release` 브랜치와 태그를 올린 뒤, `gh release create`로 게시합니다. 운영 코드에 새 모듈이나 데이터 파일이 생겼다면 먼저 `release_files.txt`에 더합니다.

소스 릴리스 게시와 운영 서버 배포는 별도 절차입니다. 태그를 올려도 운영 서버가 자동으로 재시작되지는 않습니다.

### v3.0.1까지의 방식

v3.0.1까지는 `main`의 커밋에 태그를 달고 GitHub Actions가 게시했습니다. `release` 브랜치에는 워크플로 파일이 없어 v3.0.2부터는 Actions가 돌지 않습니다.

```bash
# 예시: 다음 수정 버전이 2.7.3인 경우
python3 -m trainer.scripts.release_notes --tag v2.7.3 --output /tmp/release-notes.md
git tag -a v2.7.3 -m "v2.7.3: 변경 내용 요약"
git push --atomic origin main refs/tags/v2.7.3
```

- 게시 상태는 저장소의 **Actions → Publish release**, 게시된 내용은 **Releases**에서 확인합니다.
- 자동 게시와 재게시 대상은 이 절차가 포함된 **v2.7.0 이후**의 태그입니다. 필요하면 Actions의 **Run workflow**에서 해당 태그를 지정해 다시 게시할 수 있습니다. 최신 버전으로 지정하려면 **make_latest**를 선택합니다. 이전 버전이 최신 버전을 덮어쓰는 것은 차단합니다.
- `v2.6.0`은 소스 비교용 기준 태그이며 자동 게시 대상이 아닙니다. 태그는 이동하거나 덮어쓰지 않습니다.
- 릴리스 워크플로를 수정해 `main`에 반영하면 현재 소스 버전의 기존 태그를 다시 게시·검증합니다. 게시 결과는 Actions 요약과 `refs/notes/releases/{태그}`의 Git 메모에 남습니다.

## 보정 도막 평가·Zone 코칭 통합

선택적 v2 경로의 설치 준비, 변경 위치, API 연결, 시험 범위는 [통합 안내](docs/corrected-zone/README_KR.md)를 참고하세요.

- 선택적 v2 서비스(엔진·설명·`/api/v2`)는 기본으로 꺼져 있고 따로 켜야 합니다.
- v1 요청의 도막 판정은 같은 계산 함수(`trainer/zone_engine/numerical.py`)를 서버 안에서 직접 불러 씁니다(`trainer/demo_api/coating_assessment.py`).

## 관련 문서

- [개발 환경과 저장소 작업 규칙](AGENTS.md)
- [전문가 자세 기준 데이터](docs/arm_reference_artifact.md)
- [모델·RAG·학습 설정](trainer/configs/)
