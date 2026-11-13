# 저장소 추적 범위

2026-11-13 갱신. 이 저장소는 배포 PC가 그대로 pull한다. 실행에 필요한 코드·설정·연동 규격을 추적하고, mjkim의 내부 개발 문서는 mjkim의 개발 서버에만 둔다(원격 저장소 없음). 배포 때는 개발 문서를 뺀 내용만 올리고, 배포 PC는 그것을 받아 시험한다.

| 추적 유지 | 추적 제외 |
|---|---|
| `trainer/`, `mcp_servers/`, `demo-ui/`, `deploy/`의 실행 코드·테스트·배포 설정 | 모델 가중치, 데이터, 인덱스, 실행 결과, 캐시 |
| `README.md`, `CHANGELOG.md`, `AGENTS.md`, 연동 규격과 계약 문서 | mjkim의 내부 개발 문서 117개(`.gitignore`의 "내부 개발 문서" 절) |
| 다른 개발자가 추가한 문서(예: `docs/superpowers/`의 deploy 작성 5개, `.github/issue-batches/`의 10개와 동기화 워크플로) | `.agent/`, `.codex/`, `.claude/`, `.opencode/`, `.cache/`, `plugins/cache/` |
| `demo-ui/package.json`, `demo-ui/package-lock.json` | 루트 `package.json`, `package-lock.json`의 개인 도구 의존성 |
| 비밀값 없는 `.env.example`, `.env.*.example` | `.env`, `.env.local`, `.env.production` 등 실환경 설정 |
| `trainer/data/`의 작은 공통 자산, Spark 모델 `*.manifest.json` | 모델 디렉터리의 가중치·다운로드 파일 |

## 내부 개발 문서 분리 (2026-11-13)

- mjkim 계정(같은 메일을 쓰는 `DESKTOP-SFB8GL3\KETI`, `Meejin Kim` 포함)이 추가한 117개 파일을 Git 인덱스에서 빼고(`git rm --cached`) `.gitignore`에 정확한 경로로 적었다. 폴더 단위로 무시하지 않았으므로 다른 개발자가 같은 폴더에 둔 파일과 앞으로 추가할 파일은 영향을 받지 않는다.
  - 설계·계획(`docs/superpowers/` 중 31개), 측정(`docs/measurements/`), KPI 시험 방법(`docs/year2/`, `docs/year3/`), 보관 노트(`docs/archive/`)
  - 다국어 비교 기록(`docs/local_voice_multilingual_*.md`), PRD와 분석 노트, `trainer/docs/`
  - Task Master(`.taskmaster/`), 에이전트 스킬(`.agents/`, `skills-lock.json`), MCP 설정(`.mcp.json`), 루트 계획 파일
  - Issue 배치 2개(`20261111-detailed-knowledge-accuracy`, `20261112-pr8-coating-adoption`). 해당 GitHub Issue는 그대로 남는다.
- 같은 경로 그대로 개발 서버의 로컬 보관소로 옮겼다(main `49dca08` 기준). 코드 주석이 가리키는 `docs/...` 경로는 그 보관소에 있다.
- main과 mjkim 브랜치(`feat/glossary-rc3`)의 최신본에서 뺐다. 다른 개발자의 브랜치는 main을 병합할 때 함께 빠진다. 과거 커밋과 태그(v2.6.0~v3.0.1)에는 그대로 남는다.
- 실제 비밀값(키·비밀번호·개인 키)은 정리 시점의 추적 파일과 전체 이력 847개 커밋 모두에서 발견되지 않았다(예시용 가짜 값만 있음).

## 배포 PC 정리

0. 배포 PC는 운영 코드만 담은 `release` 브랜치를 받는다(`git clone --single-branch --branch release <저장소 주소>`). `release`에는 개발 문서와 테스트가 없다. `main`은 개발 통합 브랜치다.
1. 배포 PC의 사본에서 `git pull --ff-only`를 실행하면 git이 위 파일을 작업 트리에서 지운다. 미커밋 변경이 있으면 먼저 보존한다. 강제 checkout/reset은 하지 않는다.
2. `deploy/purge_internal_docs.sh`로 남은 사본을 확인하고, `--execute`로 지운다. 이 스크립트는 `.gitignore`의 "내부 개발 문서" 절에 적힌 경로만 지우고, 추적 파일은 건드리지 않는다.
3. 이력까지 없는 사본이 필요하면 `git clone --depth 1 --branch main <저장소 주소>`로 다시 받는다. 얕은 클론에는 지금 커밋의 파일만 들어 있다.
4. 배포 이미지(`deploy/spark/images/*.Dockerfile`)는 `trainer/`, `mcp_servers/`, `deploy/spark/scripts`, `demo-ui/`만 복사하므로 이 문서들이 들어 있지 않다. 이미지는 다시 만들 필요가 없다.

## 협업 시 주의

1. 이 커밋을 pull하면 작업 트리에서 위 문서가 지워진다. 이 저장소의 같은 경로에 다시 복사해 두는 것은 괜찮다(무시된다).
2. `git add -A` 대신 변경한 파일을 명시적으로 stage한다. `.gitignore`는 이미 추적 중인 파일을 자동으로 빼지 않는다. 새 제외 항목은 `git ls-files`와 함께 점검한다.
3. `.env.example`은 예시 변수와 가짜 값만 포함한다. 추적 허용은 비밀값 포함 허용을 뜻하지 않는다.
4. mjkim의 새 개발 문서(설계, 측정, 회의·보고 자료, 제안서)는 개발 서버에만 둔다. 개발 서버의 git hook이 개발 문서 커밋과 push를 막는다.
