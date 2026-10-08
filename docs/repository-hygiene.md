# 저장소 추적 범위

2026-10-08. 실행 코드와 팀 공통 개발 자산은 Git으로 관리하고, 개인 도구 설치본·캐시·실험 산출물은 로컬에 둔다.

| 추적 유지 | 추적 제외 |
|---|---|
| `trainer/`, `mcp_servers/`, `demo-ui/`, `deploy/`의 실행 코드·테스트·배포 설정 | 모델 가중치, 데이터, 인덱스, 실행 결과, 캐시 |
| `demo-ui/package.json`, `demo-ui/package-lock.json` | 루트 `package.json`, `package-lock.json`의 개인 Codex 도구 의존성 |
| `AGENTS.md`, `.agents/skills/frontend-design/`와 라이선스, `skills-lock.json` | `.agent/`, `.codex/`, `.claude/`, `.opencode/`, `.cache/`, `plugins/cache/` |
| `docs/superpowers/`, `docs/archive/agent-notes/` | `.omo/`, `.sisyphus/`, `.superpowers/`의 로컬 작업 상태 |
| 비밀값 없는 `.env.example`, `.env.*.example` | `.env`, `.env.local`, `.env.production` 등 실환경 설정 |
| `trainer/data/`의 작은 공통 자산, Spark 모델 `*.manifest.json` | 모델 디렉터리의 가중치·다운로드 파일 |

## 이번 정리

- `.agent/`의 도구 설치본 43개와 루트 npm 도구 매니페스트 2개를 Git 인덱스에서 제거했다. 정리 작업 공간에서는 `git rm --cached`로 파일 자체를 보존했다.
- 기존 계획·실험 기록 5개는 [문서 보관소](archive/agent-notes/README.md)로 옮기고 참조를 갱신했다. 동일한 `.sisyphus` 계획 사본은 중복 추적하지 않는다.
- 앱의 소스·의존성·이미지·런타임 설정은 변경하지 않았다. `.mcp.json`과 Task Master 관리 데이터도 변경하지 않았다.

## 협업 시 주의

1. 다른 개발자가 이 커밋을 pull하면 과거에 추적하던 도구 파일이 작업 트리에서 제거될 수 있다. 필요한 개인 도구 파일은 먼저 저장소 밖에 백업하고, pull 후 사용자별 도구 경로나 무시된 폴더에 복원한다. 미커밋 변경이 있으면 먼저 보존하고 강제 checkout/reset하지 않는다.
2. `git add -A` 대신 변경한 코드·문서를 명시적으로 stage한다. `.gitignore`는 이미 추적 중인 파일을 자동으로 제거하지 않는다. 새 제외 항목은 `git ls-files`와 함께 점검한다.
3. `.env.example`은 예시 변수와 가짜 값만 포함한다. 추적 허용은 비밀값 포함 허용을 뜻하지 않는다.
4. 이번 변경은 새 커밋에서 추적을 정리한 것이며 과거 이력을 삭제하지 않았다. 비밀정보가 과거에 올라갔다면 자격 증명 폐기·교체와 별도 이력 정리가 필요하다. 이번 작업은 전체 Git 이력의 비밀정보 검사를 수행한 것이 아니다.
5. `.agents/skills/frontend-design/`와 `docs/superpowers/`는 이름에 agent/skills가 있어도 캐시가 아니다. 팀 공통 자산을 디렉터리 이름만 보고 일괄 삭제하지 않는다.
