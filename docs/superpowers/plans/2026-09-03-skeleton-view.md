# 스켈레톤 뷰 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 훈련생·전문가·교정후 세 자세를 월드 좌표 3D로 겹쳐 그리고, 시각적 강조를 채점 가중치에서 파생시킨다.

**Architecture:** 서버가 `compare()` 응답에 `view` 블록(월드 좌표 + `weights` + 메시)을 싣고, `trainer/pose_lab/static`의 three.js 렌더러가 `weights`만 읽어 강조를 정한다. 전신은 `pose_fk_csv.forward_kinematics`가 내고 Kabsch 강체 정렬로 실측 3점에 맞춘다.

**Tech Stack:** Python 3(표준 라이브러리만), three.js(CDN, UMD)

**Spec:** `docs/superpowers/specs/2026-09-03-skeleton-view-design.md`

## Global Constraints

- 축 규약은 `trainer/scripts/world_axes.py`에서만 온다. 축 인덱스 리터럴(`[1]`, `[2]`)을 새로 쓰지 않는다.
- 화면 문구에 "사용하지 않음"·"미사용"·"unused" 계열 표현을 쓰지 않는다. 스키마에도 그런 라벨을 두지 않는다.
- 가중치 0인 관절에 수치·편차 마커·측정값 툴팁을 붙이지 않는다.
- 채점(σ·교정·개선율)은 이 경로를 타지 않는다. FK가 틀려도 지표가 오염되면 안 된다.
- 격자점 수는 `_local_tracks`의 리샘플 수와 같아야 한다.
- 성질을 보호한다는 테스트는 그 성질을 **일부러 깨뜨려 실패를 확인한 뒤에만** 인정한다. 보고서에 테스트 이름과 단언문을 인용한다.
- `git add -A` 금지. 명시 경로만 스테이징한다.

---

### Task 1: 전신 스켈레톤을 실측 3점에 정렬한다

**Files:**
- Create: `trainer/pose_lab/skeleton.py`
- Test: `trainer/pose_lab/tests/test_skeleton.py`

**Interfaces:**
- Consumes: `pose_fk_csv.bind_offsets(row)`, `pose_fk_csv.forward_kinematics(row, offsets) -> dict[str, Vec3]`(루트 `Pelvis`가 원점), `measured_pose.MeasuredFrame`(`head`, `right`, `left`, `hit`, `yaw`)
- Produces:
  - `ANCHORS: tuple[tuple[str, str], ...]` — FK 관절명과 `MeasuredFrame` 필드명의 쌍
  - `kabsch(source: list[Vec3], target: list[Vec3]) -> tuple[Matrix, Vec3]` — 회전·평행이동. 스케일 없음(뼈 길이 고정)
  - `place_skeleton(local: Mapping[str, Vec3], frame) -> tuple[dict[str, Vec3], float]` — 월드 관절 위치와 잔차(m)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
# trainer/pose_lab/tests/test_skeleton.py
import math
from pose_lab import skeleton


def _frame(head, right, left):
    class F:
        pass
    f = F()
    f.head, f.right, f.left, f.hit, f.yaw = head, right, left, None, 0.0
    return f


def test_kabsch_recovers_a_known_rotation():
    """알려진 90도 회전을 넣고 그것을 되찾는지 본다."""
    src = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
    # Y축 둘레 90도: (x,y,z) -> (z, y, -x)
    dst = [(0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    rotation, translation = skeleton.kabsch(src, dst)
    for s, d in zip(src, dst):
        moved = skeleton.apply(rotation, translation, s)
        assert all(abs(a - b) < 1e-9 for a, b in zip(moved, d))


def test_place_skeleton_moves_every_joint_not_just_the_anchors():
    """앵커가 아닌 관절도 같은 강체 변환을 받아야 한다."""
    local = {
        "Head": (0.0, 1.6, 0.0),
        "RightHand": (0.3, 1.2, 0.0),
        "LeftHand": (-0.3, 1.2, 0.0),
        "LeftToe": (-0.1, 0.0, 0.1),
    }
    shifted = _frame((0.0, 1.6, 5.0), (0.3, 1.2, 5.0), (-0.3, 1.2, 5.0))
    placed, residual = skeleton.place_skeleton(local, shifted)
    assert residual < 1e-9
    assert all(abs(a - b) < 1e-9 for a, b in zip(placed["LeftToe"], (-0.1, 0.0, 5.1)))


def test_residual_reports_the_real_disagreement():
    """리그가 낸 손이 실측과 어긋나면 잔차가 그 크기를 낸다."""
    local = {"Head": (0.0, 1.6, 0.0), "RightHand": (0.3, 1.2, 0.0), "LeftHand": (-0.3, 1.2, 0.0)}
    # 오른손만 10cm 벌린 목표 — 강체로는 맞출 수 없다
    stretched = _frame((0.0, 1.6, 0.0), (0.4, 1.2, 0.0), (-0.3, 1.2, 0.0))
    _, residual = skeleton.place_skeleton(local, stretched)
    assert 0.01 < residual < 0.1


def test_missing_left_hand_falls_back_to_two_anchors():
    """왼손이 None이어도 머리+오른손으로 배치한다."""
    local = {"Head": (0.0, 1.6, 0.0), "RightHand": (0.3, 1.2, 0.0), "LeftHand": (-0.3, 1.2, 0.0)}
    f = _frame((0.0, 1.6, 2.0), (0.3, 1.2, 2.0), None)
    placed, residual = skeleton.place_skeleton(local, f)
    assert placed is not None
    assert residual < 1e-6
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cd trainer && python -m pytest pose_lab/tests/test_skeleton.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pose_lab.skeleton'`

- [ ] **Step 3: 최소 구현**

`trainer/pose_lab/skeleton.py`를 쓴다. 요구사항:

- `ANCHORS = (("Head", "head"), ("RightHand", "right"), ("LeftHand", "left"))`
- `kabsch(source, target)`: 두 점군의 중심을 빼고 공분산 `H = Σ sᵢ tᵢᵀ`를 만들고 SVD 없이 3x3을 푼다. 표준 라이브러리만 쓰므로 **자코비 고유값 반복으로 `HᵀH`를 대각화해 SVD를 직접 구현**한다(`level_mesh.py`가 같은 일을 이미 하므로 그 헬퍼를 재사용할 수 있으면 재사용한다 — 먼저 `grep -n "def .*svd\|jacobi\|umeyama" trainer/scripts/level_mesh.py`로 확인할 것). 반사(det < 0)는 마지막 특이벡터를 뒤집어 막는다.
- 앵커가 2개면 회전이 한 자유도 덜 정해진다. 그때는 **월드 up(`world_axes.UP`)을 세 번째 가상 대응점으로 넣어** 자세를 고정한다. 이 선택을 독스트링에 적는다.
- 앵커가 1개 이하이면 `(local 그대로 평행이동만, float("nan"))`이 아니라 **`None`을 반환**한다 — 배치할 수 없다는 것이 결과다.
- `place_skeleton`은 잔차로 앵커별 거리의 **RMS**를 낸다. 무엇의 평균인지 독스트링에 적는다.
- `apply(rotation, translation, v) -> Vec3`를 공개한다(테스트가 쓴다).

- [ ] **Step 4: 통과를 확인한다**

Run: `cd trainer && python -m pytest pose_lab/tests/test_skeleton.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: 검증 — 테스트가 실제로 성질을 지키는지 깨뜨려 본다**

세 가지를 각각 일부러 망가뜨리고 어느 테스트가 잡는지 기록한다. 하나라도 안 잡히면 그 테스트를 고친다.

1. `place_skeleton`에서 앵커만 변환하고 나머지 관절은 `local` 그대로 반환 → `test_place_skeleton_moves_every_joint_not_just_the_anchors`가 잡아야 한다
2. 잔차를 항상 `0.0`으로 반환 → `test_residual_reports_the_real_disagreement`가 잡아야 한다
3. `kabsch`에서 회전을 항등행렬로 고정 → `test_kabsch_recovers_a_known_rotation`이 잡아야 한다

보고서에 **각 파괴마다 실패한 테스트 이름과 단언문을 인용**한다.

- [ ] **Step 6: 실측 데이터로 잔차를 잰다 (D0)**

학습모드 CSV 하나(`1_*.csv`, 실전모드 `1-A_*` 아님)를 읽어 분사 프레임 100개에 대해 잔차를 재고 중앙값·p95를 보고서에 적는다. 이 숫자가 설계 판단을 바꾼다.

```bash
cd trainer && python -c "
import sys; sys.path.insert(0, 'scripts')
# ... 실제 경로는 _collect_level_hits 가 쓰는 규칙을 따를 것
"
```

- [ ] **Step 7: 커밋**

```bash
git add trainer/pose_lab/skeleton.py trainer/pose_lab/tests/test_skeleton.py
git commit -m "feat(pose): 전신 FK 스켈레톤을 실측 3점에 강체 정렬"
```

---

### Task 2: `view` 블록과 `weights`를 응답에 싣는다

**Files:**
- Modify: `trainer/pose_lab/service.py`(`compare` 근처, 현재 352행)
- Test: `trainer/pose_lab/tests/test_view_block.py`

**Interfaces:**
- Consumes: Task 1의 `skeleton.place_skeleton`; 기존 `pass_frames`, `_local_tracks`, `axes_block`, `measured_score.issued_axes`
- Produces: `view_block(query_frames, reference_frames, corrections, level) -> dict`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_weights_cover_every_drawn_joint():
    """그리는 관절 중 weights에 없는 것이 있으면 실패한다."""
    block = ...  # view_block(...)
    for layer in block["layers"].values():
        if layer is None:
            continue
        for name in layer["joints"]:
            assert name in block["weights"], f"{name} has no weight"


def test_corrected_layer_equals_issued_axes_shift():
    """교정 층의 이동이 issued_axes 합과 정확히 같다. 별도 계산 경로가 없다."""
    # magnitude_cm/100 * unit 의 합을 독립적으로 계산해 비교


def test_grid_point_count_matches_the_scorer():
    """그림과 채점이 같은 표본을 본다."""
    assert block["grid_n"] == len(list(service._local_tracks(frames).values())[0])


def test_zero_weight_joints_carry_no_numbers():
    """가중치 0인 관절에 수치가 붙지 않는다."""
    for name, w in block["weights"].items():
        if w == 0.0:
            assert name not in block.get("deviations", {})
```

- [ ] **Step 2-4:** 실패 확인 → 구현 → 통과 확인. `WEIGHTS` 상수는 `measured_score`가 실제로 채점하는 점에서 파생시킨다 — 하드코딩한 목록을 두 곳에 두지 않는다.

- [ ] **Step 5: 검증** — `weights`에서 관절 하나를 빼고 첫 테스트가 실패하는지, 교정 이동을 2배로 하고 둘째가 실패하는지 확인하고 인용한다.

- [ ] **Step 6: 커밋**

```bash
git add trainer/pose_lab/service.py trainer/pose_lab/tests/test_view_block.py
git commit -m "feat(pose): compare 응답에 월드 좌표 view 블록과 채점 가중치 추가"
```

---

### Task 3: 벽면 메시를 view에 싣는다

**Files:**
- Modify: `trainer/pose_lab/service.py`
- Test: `trainer/pose_lab/tests/test_view_block.py`(추가)

**Interfaces:**
- Consumes: `level_mesh.load_alignment(level)`, `level_mesh.read_fbx`, `level_mesh.place`
- Produces: `view["mesh"]`

- [ ] **Step 1: 실패하는 테스트**

```python
def test_unverified_level_reports_unavailable_not_missing():
    """검증된 정렬이 없는 레벨은 available: False 이고 null 이 아니다."""
    block = view_block(..., level=10)
    assert block["mesh"] is not None
    assert block["mesh"]["available"] is False
    assert block["mesh"].get("vertices") in (None, [])


def test_level_1_carries_placed_geometry():
    block = view_block(..., level=1)
    assert block["mesh"]["available"] is True
    assert len(block["mesh"]["vertices"]) > 0
```

- [ ] **Step 2-4:** 실패 확인 → 구현 → 통과. `level_alignment.json`에 항목이 없으면 `available: False`, 레벨 자체를 모르면 `mesh: None`.

- [ ] **Step 5: 검증** — `level_alignment.json`을 빈 배열로 바꾸고 두 번째 테스트가 실패하는지 확인한다(원상 복구할 것).

- [ ] **Step 6: 커밋**

```bash
git add trainer/pose_lab/service.py trainer/pose_lab/tests/test_view_block.py
git commit -m "feat(pose): 검증된 정렬이 있는 레벨에만 벽면 지오메트리 제공"
```

---

### Task 4: 렌더러

**Files:**
- Create: `trainer/pose_lab/static/skeleton_view.js`
- Modify: `trainer/pose_lab/static/index.html`, `trainer/pose_lab/static/app.js`, `trainer/pose_lab/static/style.css`

**Interfaces:**
- Consumes: Task 2·3의 `measured.view`
- Produces: `renderSkeletonView(container, view)` — 전역 함수 하나

- [ ] **Step 1: three.js를 CDN UMD로 싣는다**

`index.html`에 `<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>`를 인라인 스크립트보다 **앞에** 둔다. 버전을 고정한다.

- [ ] **Step 2: 강조를 `weights`에서만 파생시킨다**

부위 이름으로 분기하는 코드를 쓰지 않는다. 색·굵기·마커·라벨이 전부 `view.weights[name]`의 함수다.

```js
function emphasis(weight) {
  // weight 0 -> 조용한 중립 형태(선만). weight 1 -> 고유색, 굵게, 마커, 라벨.
  return {
    opacity: 0.25 + 0.75 * weight,
    linewidth: 1 + 2 * weight,
    showMarker: weight > 0,
    showLabel: weight > 0,
  };
}
```

- [ ] **Step 3: 층 토글**

`query`(켬) / `reference`(켬) / `corrected`(끔) / 벽면(있으면 켬). 체크박스 넷.

- [ ] **Step 4: 없는 값은 `—`**

`0`도 falsy이므로 `||` 대신 `??` 또는 `typeof x === "number"`.

- [ ] **Step 5: 화면 문구 점검**

파일 전체를 훑어 "미사용"·"사용하지 않"·"unused"·"not used"가 **한 번도** 안 나오는지 확인한다.

```bash
grep -niE "미사용|사용하지 않|unused|not used" trainer/pose_lab/static/*.js trainer/pose_lab/static/*.html
```
Expected: 출력 없음

- [ ] **Step 6: 브라우저 확인**

서버를 띄우고 실제로 렌더되는지 본다. 스크린샷 대신 콘솔 오류 0건과 캔버스에 그려진 관절 수를 보고한다.

- [ ] **Step 7: 커밋**

```bash
git add trainer/pose_lab/static/skeleton_view.js trainer/pose_lab/static/index.html trainer/pose_lab/static/app.js trainer/pose_lab/static/style.css
git commit -m "feat(pose): 가중치에서 강조를 파생시키는 3D 스켈레톤 렌더러"
```
