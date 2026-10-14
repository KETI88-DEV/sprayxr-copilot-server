# Unified Motion/Posture Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 모션뷰어의 전문가 CSV·Zone·도장 경로 재생과 김미진 선임의 자세 비교 기능을 하나의 선택 흐름, 하나의 3D 화면, 하나의 타임라인으로 통합한다.

**Architecture:** `motion_viewer.py`는 통합 HTTP 경계와 HTML 셸을 담당하고, `motion_viewer_pose.py`는 운영 자세 평가와 pose_lab 전문가 기준을 서로 다른 출처로 보존한 읽기 모델을 만든다. 브라우저는 공통 초 단위 playhead로 모든 레이어를 동기화하며, 카메라 조작은 독립 JS 모듈로 분리한다. pose_lab의 CSV 실험 점수와 운영 `pose_recomm` 점수는 혼합하지 않고 각각 출처를 표시한다.

**Tech Stack:** Python 3.11, FastAPI, pytest, 바닐라 JavaScript/Canvas 2D 기반 3D 투영, Docker Compose

**Spec:** `docs/superpowers/specs/2026-10-14-motion-viewer-pose-integration-design.md`

## Global Constraints

- TSB 요청 JSON 계약은 변경하지 않는다.
- `origin/feat/pose-recomm` 최신 버전을 병합하고 pose_lab의 실제 기준 선택 경로를 재사용한다.
- 운영 점수 정본은 `POST /pose_estimation/sequence`; pose_lab 점수는 CSV 전문가 비교로 명시한다.
- 자세 비교와 도장 경로를 탭으로 분리하지 않는다.
- 내부 채점 점·가중치·수식은 UI·Swagger·대외 문서에 노출하지 않는다.
- pose_lab 산포는 `/output/spread_reference_expert_m3.json`만 허용하고 접두어 불일치는 실패로 처리한다.
- 모델 파일은 Git에 넣지 않으며 `alignment_status == verified`인 경우만 표시한다.
- 새 기능과 버그 수정은 테스트가 먼저 실패한 뒤 구현한다.

## Review Focus

- 작업자/모드/기록을 빠르게 바꿀 때 오래된 비동기 응답이 새 선택을 덮어쓰지 않는다 — Task 4 회귀 테스트.
- m1 또는 누락된 산포 파일이 들어오면 pose_lab 결과를 정상처럼 표시하지 않는다 — Task 2 테스트.
- 작업자와 전문가 기록 길이가 다를 때 상대 시간 정렬이 범위를 벗어나지 않는다 — Task 3 테스트.
- 피도면이 원점에서 멀리 있어도 초점·홈·축 뷰가 가시 객체 중심을 유지한다 — Task 5 테스트.
- 1200px 이하에서 평가 정보가 숨지 않고 3D 아래로 이동한다 — Task 4 정적 자산 테스트.

---

### Task 1: 최신 pose-recomm 병합과 실행 기준 고정

**Files:**
- Modify: `deploy/start_pose_lab.sh`
- Modify: `README.md`
- Test: `trainer/pose_lab/tests/test_app.py`
- Test: `trainer/pose_lab/tests/test_static_assets.py`

**Interfaces:**
- Consumes: `origin/feat/pose-recomm`의 `CsvPassSource`, `reference_candidates`, `/api/compare`
- Produces: m3 산포로만 기동 가능한 최신 pose_lab 코드와 검증된 병합 기준

- [ ] **Step 1: 현재 작업 파일을 경로 지정 stash로 보존하고 `origin/feat/pose-recomm` 최신 커밋을 병합한다.**
- [ ] **Step 2: 병합 결과의 산포 기본값과 접두어 검증 테스트를 먼저 실행해 실패 여부를 확인한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/pose_lab/tests/test_app.py trainer/pose_lab/tests/test_static_assets.py -q`
Expected: 병합 충돌 또는 잘못된 m1 기본값이 있으면 관련 테스트 FAIL, 그렇지 않으면 PASS.

- [ ] **Step 3: 모든 시작 경로가 `spread_reference_expert_m3.json`을 사용하고 키 불일치를 거부하도록 최소 수정한다.**
- [ ] **Step 4: pose_lab 전체 테스트를 실행한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/pose_lab -q`
Expected: PASS.

- [ ] **Step 5: 병합 및 운영 기준을 커밋한다.**

```bash
git add deploy/start_pose_lab.sh README.md trainer/pose_lab
git commit -m "merge(pose): 최신 전문가 자세 비교 기능 반영"
```

### Task 2: 실제 전문가 기준 선택 어댑터

**Files:**
- Create: `trainer/demo_api/motion_viewer_reference.py`
- Modify: `trainer/demo_api/motion_viewer_pose.py`
- Test: `trainer/demo_api/tests/test_motion_viewer_reference.py`
- Test: `trainer/demo_api/tests/test_motion_viewer_pose.py`

**Interfaces:**
- Consumes: `CsvPassSource`, `service.reference_candidates(source, query, limit)`, `service.compare(...)`
- Produces: `PoseLabReferenceAdapter.catalog()`, `PoseLabReferenceAdapter.compare(query_ref)`, 출처가 분리된 `expert_reference` 읽기 모델

- [ ] **Step 1: 같은 레벨·동작 키의 전형기 기준이 실제 후보에서 선택되고, m3 산포 미검증 시 unavailable을 반환하는 실패 테스트를 작성한다.**
- [ ] **Step 2: 테스트를 실행해 어댑터 부재로 실패하는지 확인한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_motion_viewer_reference.py -q`
Expected: FAIL because `motion_viewer_reference` is missing.

- [ ] **Step 3: `PoseLabReferenceAdapter`를 구현하고 운영 점수와 CSV 비교 점수를 별도 필드·라벨로 유지한다.**
- [ ] **Step 4: 전문가 골격/머리·양손/교정 표시가 임의 `reference_layer`가 아니라 compare 응답에서만 생성되도록 `build_pose_view`를 수정한다.**
- [ ] **Step 5: 관련 테스트를 실행한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_motion_viewer_reference.py trainer/demo_api/tests/test_motion_viewer_pose.py -q`
Expected: PASS.

- [ ] **Step 6: 커밋한다.**

```bash
git add trainer/demo_api/motion_viewer_reference.py trainer/demo_api/motion_viewer_pose.py trainer/demo_api/tests/test_motion_viewer_reference.py trainer/demo_api/tests/test_motion_viewer_pose.py
git commit -m "feat(viewer): 실제 전문가 자세 기준 연결"
```

### Task 3: 공통 시간축과 Zone/패스 동기화

**Files:**
- Modify: `trainer/demo_api/motion_viewer_pose.py`
- Modify: `trainer/demo_api/motion_viewer.py`
- Test: `trainer/demo_api/tests/test_motion_viewer_pose.py`
- Test: `trainer/demo_api/tests/test_motion_viewer.py`

**Interfaces:**
- Consumes: Task 2의 `expert_reference`, 기존 모션 프레임 `time_seconds`, 시퀀스 Zone ID
- Produces: `timeline.duration_seconds`, 각 레이어의 `time_seconds`, 연속 Zone 구간, 실제 스트로크 수

- [ ] **Step 1: 길이가 다른 기록의 상대 시간 정렬, 복수 Zone 연속 구간 보존, 빈 스트로크를 1로 만들지 않는 실패 테스트를 작성한다.**
- [ ] **Step 2: 테스트가 현재 이중 인덱스/다수결/대체값 때문에 실패하는지 확인한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_motion_viewer.py trainer/demo_api/tests/test_motion_viewer_pose.py -q`
Expected: 새 테스트 FAIL.

- [ ] **Step 3: 서버 응답을 공통 초 단위 타임라인과 연속 Zone 구간으로 정규화한다.**
- [ ] **Step 4: 프런트 재생을 단일 playhead와 한 세트의 재생/속도/슬라이더 제어로 변경한다.**
- [ ] **Step 5: 관련 테스트를 실행한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_motion_viewer.py trainer/demo_api/tests/test_motion_viewer_pose.py -q`
Expected: PASS.

- [ ] **Step 6: 커밋한다.**

```bash
git add trainer/demo_api/motion_viewer.py trainer/demo_api/motion_viewer_pose.py trainer/demo_api/tests/test_motion_viewer.py trainer/demo_api/tests/test_motion_viewer_pose.py
git commit -m "feat(viewer): 경로와 자세 재생 시간축 통합"
```

### Task 4: 단일 캔버스와 통합 선택/평가 UI

**Files:**
- Modify: `trainer/demo_api/motion_viewer.py`
- Modify: `trainer/demo_api/static/motion_viewer_pose.js`
- Modify: `trainer/demo_api/static/motion_viewer_pose.css`
- Test: `trainer/demo_api/tests/test_motion_viewer.py`

**Interfaces:**
- Consumes: Task 2/3의 통합 응답과 타임라인
- Produces: 훈련종류→모드→작업자→기록→시퀀스 선택, 단일 캔버스 레이어, 중복 없는 평가 패널

- [ ] **Step 1: 탭/두 번째 캔버스/두 번째 타임라인이 없고 통합 선택기·레이어 토글·세대 토큰·반응형 패널이 존재하는 실패 테스트를 작성한다.**
- [ ] **Step 2: 정적 자산 테스트가 기존 탭 구조 때문에 실패하는지 확인한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_motion_viewer.py -q`
Expected: 새 테스트 FAIL.

- [ ] **Step 3: 기존 탭 템플릿을 제거하고 단일 캔버스에서 피도면·Zone·경로·작업자·전문가·교정 레이어를 그린다.**
- [ ] **Step 4: 운영 평가와 CSV 전문가 비교의 출처를 명시하고, 핵심 교정 1건만 상시 표시하며 나머지는 상세 영역으로 이동한다.**
- [ ] **Step 5: 비동기 선택 요청에 generation token을 적용하고 1200px 이하에서 평가 패널을 3D 아래로 배치한다.**
- [ ] **Step 6: 테스트를 실행한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_motion_viewer.py -q`
Expected: PASS.

- [ ] **Step 7: 커밋한다.**

```bash
git add trainer/demo_api/motion_viewer.py trainer/demo_api/static/motion_viewer_pose.js trainer/demo_api/static/motion_viewer_pose.css trainer/demo_api/tests/test_motion_viewer.py
git commit -m "feat(viewer): 도장 경로와 자세 비교 단일 화면 통합"
```

### Task 5: Unity식 카메라와 홈/축 기즈모

**Files:**
- Create: `trainer/demo_api/static/motion_viewer_camera.js`
- Modify: `trainer/demo_api/motion_viewer.py`
- Modify: `trainer/demo_api/static/motion_viewer_pose.js`
- Modify: `trainer/demo_api/static/motion_viewer_pose.css`
- Test: `trainer/demo_api/tests/test_motion_viewer.py`

**Interfaces:**
- Consumes: 현재 표시 레이어의 3D 점 배열
- Produces: `MotionViewerCamera.fit(points)`, `orbit`, `pan`, `dolly`, `look`, `fly`, `focus`, `axisView`, `home`

- [ ] **Step 1: 원점에서 떨어진 경계 상자의 중심/거리, pan 시 target 동행, home 복원, X/Y/Z 축 뷰를 검증하는 실패 테스트를 작성한다.**
- [ ] **Step 2: 카메라 모듈 부재로 테스트가 실패하는지 확인한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_motion_viewer.py -q`
Expected: 새 테스트 FAIL.

- [ ] **Step 3: 카메라 모듈을 구현하고 좌클릭 orbit, 휠클릭 pan, 휠 dolly, 우클릭 look+WASD/QE, F focus를 연결한다.**
- [ ] **Step 4: 화면 상단에 X/Y/Z 축과 홈 기즈모를 추가하고 기록 로드시 초기 스냅샷을 갱신한다.**
- [ ] **Step 5: 테스트를 실행한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_motion_viewer.py -q`
Expected: PASS.

- [ ] **Step 6: 커밋한다.**

```bash
git add trainer/demo_api/motion_viewer.py trainer/demo_api/static/motion_viewer_camera.js trainer/demo_api/static/motion_viewer_pose.js trainer/demo_api/static/motion_viewer_pose.css trainer/demo_api/tests/test_motion_viewer.py
git commit -m "feat(viewer): 객체 중심 자유 카메라와 기즈모 추가"
```

### Task 6: 전체 회귀, 배포, 브라우저 검증

**Files:**
- Modify: `deploy/docker-compose.yml` only if pose_lab runtime wiring is missing
- Test: existing Python suites and public browser page

**Interfaces:**
- Consumes: Tasks 1–5의 서버·정적 자산·런타임 설정
- Produces: Spark 실서버의 검증된 `/motion-viewer`

- [ ] **Step 1: pose_lab과 demo API 회귀 테스트를 실행한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/pose_lab trainer/demo_api -q`
Expected: PASS.

- [ ] **Step 2: 전체 Python 테스트를 실행한다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: PASS.

- [ ] **Step 3: 새 이미지를 빌드하고 canary 컨테이너에서 `/api/health`, 모션 카탈로그, 실제 전문가 CSV, 실제 TSB 세션 응답을 검증한다.**
- [ ] **Step 4: 운영 컨테이너를 교체하고 pose_lab을 m3 설정으로 재기동한다.**
- [ ] **Step 5: 실제 브라우저에서 통합 선택, 동기 재생, 레이어, 객체 중심 회전, pan/dolly/fly, 홈/축 기즈모, 좁은 화면 배치를 확인한다.**
- [ ] **Step 6: 최종 변경을 커밋하고 승인된 공유 브랜치에 push한다.**

```bash
git add deploy/docker-compose.yml
git commit -m "deploy(viewer): 통합 모션 자세 뷰어 운영 반영"
git push origin test/pose-recomm
```

