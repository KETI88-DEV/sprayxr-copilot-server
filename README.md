# 선박 도장 AI Co-Pilot (SprayXR)

**[API 문서](http://keti-xr-lab.iptime.org:58000/docs) · [DB뷰어](http://keti-xr-lab.iptime.org:58000/admin) · [3D 모션뷰어](http://keti-xr-lab.iptime.org:58000/motion-viewer)**

선박 도장·블라스팅 훈련 시뮬레이터와 연동해 작업자에게 피드백과 훈련 결과를 제공하는 AI Co-Pilot 서버입니다.
전문가 기준 데이터와 훈련 기록을 비교하고, 작업 자세와 도장 경로를 시각화하며, 훈련 중 발생하는 질문에 답변합니다.

현재 버전은 **v2.7.0**입니다. [버전별 변경 이력](CHANGELOG.md)과 [GitHub 릴리스](https://github.com/KETI88-DEV/sprayxr-copilot-server/releases)에서 추가 기능을 확인할 수 있습니다.

## 주요 기능

| 기능 | 설명 |
|---|---|
| 훈련 세션 관리 | 훈련 시작부터 시퀀스 수집, 종료, 결과 조회까지 세션 단위로 관리합니다. |
| 시퀀스 피드백 | 입력된 시퀀스를 분석해 작업 상태와 속도·이격거리 가이드를 제공합니다. |
| 전문가 기준 비교 | 전문가 데이터를 바탕으로 작업 자세, Zone별 도장 방식과 작업 순서를 비교합니다. |
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

시뮬레이터 없이 연동 흐름을 확인하려면 API 문서의 **0. 모의 연동**에서 **0-0부터 0-4까지** 순서대로 실행하면 됩니다.
테스트에 필요한 기본값이 준비되어 있으며, 응답에서 실제로 사용된 입력값을 확인할 수 있습니다.
모의 API는 **모의훈련자** 계정을 재사용하고, 새 훈련을 시작할 때 세션을 생성합니다.

DB뷰어의 일반 조회에는 로그인이 필요하지 않습니다. 수정·삭제 기능은 좌측 메뉴 하단의 **관리자 모드**에서 로그인한 뒤 사용할 수 있습니다.
삭제 전에는 연결된 기록의 범위를 확인하며, DB 백업과 변경 이력이 저장됩니다.

## 시뮬레이터 연동

기본적인 호출 순서는 다음과 같습니다.

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
질문 API는 훈련 중에도 사용할 수 있으며, 세션 ID를 함께 보내면 저장된 훈련 기록을 답변에 활용할 수 있습니다.
자세한 파라미터와 예시 요청은 [API 문서](http://keti-xr-lab.iptime.org:58000/docs)를 참고하세요.

## 데이터 구성

전문가 비교에 사용하는 자료와 API 운영 기록은 별도로 관리합니다.

- **전문가 기준 자료**: Zone ID가 포함된 CSV, 도막 높이맵 PNG와 이를 가공한 비교 프로필입니다. 훈련자의 작업을 평가하는 기준으로 사용합니다.
- **API 운영 기록**: 세션, 시퀀스, 질문·답변, 리포트와 API 호출 이력입니다. SQLite에 저장되며 DB뷰어에서 확인할 수 있습니다.
- **모델·검색 자료**: LLM 모델, 학습 어댑터와 RAG 검색 인덱스입니다. 지식 검색과 답변 생성에 사용합니다.

원본 훈련 데이터, 모델 파일, 운영 DB와 인증 정보는 이 저장소에 포함하지 않습니다. 새 환경에 배포할 때는 필요한 자료와 저장 경로를 별도로 준비해야 합니다.

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
| [`docs/`](docs/) | 연동 규격, 설계와 평가 관련 문서 |

## 개발과 배포

운영 서버는 DGX Spark에서 Docker Compose로 실행합니다. 환경 준비와 자산 배치, 이미지 빌드·기동 방법은 [Spark 배포 안내](deploy/spark/README.md)를 참고하세요.
DB뷰어와 3D 모션뷰어는 FastAPI 서버에서 제공하며, `demo-ui/`의 개발 콘솔과는 별도로 사용할 수 있습니다.

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

1. `trainer/demo_api/__init__.py`의 `__version__`을 변경하고 `CHANGELOG.md`에 같은 버전의 변경 내용을 기록합니다.
2. 관련 검사를 실행하고, 아래 명령으로 태그와 변경 이력의 일치를 확인한 뒤 커밋합니다.
3. 해당 커밋에 버전 태그를 만들고 `main`과 태그를 올립니다. GitHub Actions가 변경 이력의 해당 버전 항목으로 릴리스를 게시합니다.

```bash
# 예시: 다음 수정 버전이 2.7.1인 경우
python3 -m trainer.scripts.release_notes --tag v2.7.1 --output /tmp/release-notes.md
git tag -a v2.7.1 -m "v2.7.1: 변경 내용 요약"
git push --atomic origin main refs/tags/v2.7.1
```

게시 상태는 저장소의 **Actions → Publish release**, 게시된 내용은 **Releases**에서 확인합니다. 자동 게시와 재게시 대상은 이 절차가 포함된 **v2.7.0 이후**의 태그입니다. 필요하면 Actions의 **Run workflow**에서 해당 태그를 지정해 다시 게시할 수 있습니다. 최신 버전으로 지정하려면 **make_latest**를 선택합니다. 이전 버전이 최신 버전을 덮어쓰는 것은 차단합니다. `v2.6.0`은 소스 비교용 기준 태그이며 자동 게시 대상이 아닙니다. 태그는 이동하거나 덮어쓰지 않습니다.
릴리스 워크플로를 수정해 `main`에 반영하면 현재 소스 버전의 기존 태그를 다시 게시·검증합니다. 게시 결과는 Actions 요약과 `refs/notes/releases/{태그}`의 Git 메모에 남습니다.
소스 릴리스 게시와 운영 서버 배포는 별도 절차입니다. 태그를 올려도 운영 서버가 자동으로 재시작되지는 않습니다.

## 관련 문서

- [개발 환경과 저장소 작업 규칙](AGENTS.md)
- [전문가 자세 기준 데이터](docs/arm_reference_artifact.md)
- [Zone 단위 비교 평가 설계](docs/zone_similarity_copilot_prd.md)
- [모델·RAG·학습 설정](trainer/configs/)
