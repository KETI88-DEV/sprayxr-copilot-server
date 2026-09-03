# 월드 축 규약 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 월드 축 규약을 유니티 Y-up으로 바로잡고, 규약을 한 모듈에 모아 다시 흩어질 수 없게 하며, 착탄점이 레벨 메시 면 위에 있음을 커밋된 검사로 고정한 뒤, 산포 아티팩트를 다시 굽고 전 수치를 재측정한다.

**Architecture:** 신규 `world_axes`가 규약을 **연산으로** 소유한다(상수만 두면 호출자가 다시 리터럴을 적는다). 네 모듈(`measured_score`·`stroke_frame`·`motion_key`·`measured_pose`)이 그 연산만 쓰도록 **한 번에** 전환한다 — 부분 전환은 기하가 모순이라 중간 상태가 존재할 수 없다. 그 뒤 FBX 리더와 정렬을 추가해 메시 앵커 검사를 세우고, 마지막에 재빌드·재측정한다.

**Tech Stack:** Python 3.11, pytest, 표준 라이브러리만(`struct`·`zlib`으로 FBX; 컨테이너에 mesh 라이브러리가 없고 설치하지 않는다). 컨테이너 `ship-paint-trainer`.

**Spec:** `docs/superpowers/specs/2026-09-02-world-axis-convention-design.md`

## Global Constraints

- 테스트: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`. **`trainer`와 `mcp_servers`를 모두** 넘겨야 한다 — 저장소 루트에서 맨 `pytest -q`는 수집 단계에서 `ModuleNotFoundError: No module named 'simulator_csv'`로 죽는다(기존 문제, 고치지 않는다).
- **시작 베이스라인: 640 passed.**
- 사전등록 상수는 움직이지 않는다: `CORRECTION_SIGMA_THRESHOLD = 2.0`, `MIN_CORRECTION_CM = 1.0`, `SIGMA_FLOOR_M = 0.02`, `MAX_SIGMA = 6.0`, `GATE_RATIO = 0.5`, `DEFAULT_ERROR_M = 0.144`, `N_MIN = 20`, `PAIR_SIGMA_FACTOR = sqrt(2)`, `DEFAULT_WEIGHTS = {"right":1.0,"head":0.4,"left":0.15}`. 축이 바뀌면 이 임계들이 다른 분포를 만나는데 **그것이 재측정으로 알아내려는 것**이다. 결과를 본 뒤 임계를 옮기면 측정이 무의미해진다.
- 유니티 규약: **Y가 위, 왼손 좌표계**. 수평면은 (X, Z).
- `git add -A`를 절대 쓰지 않는다. 작업 트리는 사람과 공유 중이고 `.opencode/`, `docs/Interface.pdf`, `.superpowers/pose-lab-*.md`는 타인의 미추적 파일이다. 명시한 경로만 스테이징한다.
- `.env`를 읽거나 출력하거나 커밋하지 않는다.
- 성질을 보호한다고 이름 붙인 테스트는 **그 성질을 일부러 깨뜨려 실패를 확인한 뒤에만** 인정한다. 이 저장소에서 실패할 수 없는 단언이 지금까지 일곱 건 나왔다.

## 이 계획의 최대 위험 — 먼저 읽을 것

**기존 테스트 픽스처가 옛 규약을 기하로 품고 있다.**

```python
# test_measured_score.py 현재
HORIZONTAL = stroke_frame.stroke_basis(hit=(1.2,4.5,0.8), tangent=(0.0,1.0,0.0), gun=(0.6,4.5,0.8))
VERTICAL   = stroke_frame.stroke_basis(hit=(1.2,4.5,0.8), tangent=(0.0,0.0,1.0), gun=(0.6,4.5,0.9))
```

Z-up에서 `(0,1,0)`은 수평이고 `(0,0,1)`은 수직이다. **Y-up에서는 정확히 반대다.** 두 픽스처의 의미가 서로 뒤바뀐다.

**규칙 1: 픽스처는 이름이 뜻하는 바를 새 규약에서 참이 되도록 다시 짓는다.** `HORIZONTAL`은 Y-up에서 수평인 접선(`(1,0,0)` 또는 `(0,0,1)`)을, `VERTICAL`은 `(0,1,0)`을 쓴다.

**규칙 2: 바뀌는 기대값은 새 규약에서 손으로 유도하고 그 유도를 주석에 적는다. 돌려서 나온 값을 붙여넣지 않는다.** 관측값 붙여넣기는 테스트가 검사를 멈추는 가장 흔한 경로다.

---

### Task 1: `world_axes` — 규약을 연산으로 소유한다

**Files:**
- Create: `trainer/scripts/world_axes.py`
- Test: `trainer/scripts/tests/test_world_axes.py`

**Interfaces:**
- Produces: `UP_INDEX`, `UP`, `HORIZONTAL_INDICES`, `HANDEDNESS`, `MIN_HORIZONTAL`, `vertical_of`, `horizontal_of`, `flatten`, `yaw_of`, `sign_toward`

**배경:** 결함이 생긴 이유는 규약이 네 곳에 **리터럴로** 흩어져 있었기 때문이다. 상수만 모으면 호출자가 다시 `world[2]`를 적는다. 연산을 모아야 그 경로가 막힌다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

world_axes = importlib.import_module("world_axes")


def test_unity_convention_is_y_up_left_handed() -> None:
    """유니티다. Y가 위, 수평면은 (X, Z), 왼손 좌표계."""
    assert world_axes.UP_INDEX == 1
    assert world_axes.UP == (0.0, 1.0, 0.0)
    assert world_axes.HORIZONTAL_INDICES == (0, 2)
    assert world_axes.HANDEDNESS == "left"


def test_vertical_and_horizontal_split_the_vector() -> None:
    assert world_axes.vertical_of((1.0, 2.0, 3.0)) == 2.0
    assert world_axes.horizontal_of((1.0, 2.0, 3.0)) == (1.0, 3.0)


def test_flatten_removes_only_the_vertical_component() -> None:
    assert world_axes.flatten((1.0, 2.0, 3.0)) == (1.0, 0.0, 3.0)


def test_yaw_is_measured_in_the_horizontal_plane() -> None:
    """수평 두 성분의 atan2다. 전방이 +X면 0, +Z면 90도."""
    assert world_axes.yaw_of((1.0, 0.0, 0.0)) == 0.0
    assert math.isclose(world_axes.yaw_of((0.0, 0.0, 1.0)), math.pi / 2)
    # 수직 성분은 요에 영향을 주지 않는다.
    assert math.isclose(world_axes.yaw_of((1.0, 5.0, 0.0)), 0.0)


def test_yaw_is_none_when_there_is_no_horizontal_component() -> None:
    """정확히 위나 아래를 보면 방위가 정의되지 않는다. 0으로 지어내지 않는다."""
    assert world_axes.yaw_of((0.0, 1.0, 0.0)) is None
    assert world_axes.yaw_of((0.0, -1.0, 0.0)) is None


def test_sign_toward_picks_the_direction_facing_the_reference_point() -> None:
    """왼손 좌표계라 외적의 방향 감각이 오른손계와 반대다. 부호는 기준점으로 정한다."""
    assert world_axes.sign_toward((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)
    assert world_axes.sign_toward((1.0, 0.0, 0.0), (-5.0, 0.0, 0.0)) == (-1.0, 0.0, 0.0)
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_world_axes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'world_axes'`

- [ ] **Step 3: 구현한다**

```python
#!/usr/bin/env python3
"""월드 축 규약. **여기가 유일한 출처다.**

캡처 CSV의 좌표는 유니티의 것이다 — Y가 위, 왼손 좌표계. `measured_pose`가 컬럼을
재배열 없이 읽으므로 그 규약이 그대로 파이프라인에 들어온다.

상수만 두지 않고 **연산**을 둔다. 이 모듈이 생긴 이유가 규약이 네 곳에 리터럴로
흩어져 있었기 때문이고(`world[2]`, `atan2(f[1], f[0])`, `(v[0], v[1], 0)`),
상수만 모으면 호출자가 그 리터럴을 다시 적을 수 있다. 연산을 쓰게 만들어야 막힌다.
"""

from __future__ import annotations

import math
from typing import Final

Vec3 = tuple[float, float, float]

UP_INDEX: Final = 1
UP: Final[Vec3] = (0.0, 1.0, 0.0)
HORIZONTAL_INDICES: Final[tuple[int, int]] = (0, 2)
# 유니티는 왼손 좌표계다. 외적의 방향 감각이 오른손계와 반대이므로 새 외적을 쓸 때마다
# 부호를 기준점으로 다시 정해야 한다 — `sign_toward`가 그 일을 한다.
HANDEDNESS: Final = "left"

# 수평 성분이 이보다 짧으면 방위가 정의되지 않는다(정확히 위/아래를 본 경우).
MIN_HORIZONTAL: Final = 1e-9


def vertical_of(v: Vec3) -> float:
    return v[UP_INDEX]


def horizontal_of(v: Vec3) -> tuple[float, float]:
    a, b = HORIZONTAL_INDICES
    return (v[a], v[b])


def flatten(v: Vec3) -> Vec3:
    out = list(v)
    out[UP_INDEX] = 0.0
    return (out[0], out[1], out[2])


def yaw_of(forward: Vec3) -> float | None:
    """수평면 방위. 수평 성분이 없으면 None이다 — 0으로 지어내지 않는다.

    `motion_key`가 전방 성분을 `cos(yaw)·h0 + sin(yaw)·h1`로 쓰므로 여기서 쓰는
    수평 축 쌍과 그쪽이 같아야 한다. 어긋나면 몸 기준 축이 조용히 월드 방위각으로
    돌아간다.
    """
    h0, h1 = horizontal_of(forward)
    if math.hypot(h0, h1) < MIN_HORIZONTAL:
        return None
    return math.atan2(h1, h0)


def sign_toward(axis: Vec3, toward: Vec3) -> Vec3:
    """축의 부호를 기준 벡터 쪽으로 맞춘다. 왼손 좌표계에서 외적 결과를 쓸 때 필요하다."""
    dot = axis[0] * toward[0] + axis[1] * toward[1] + axis[2] * toward[2]
    return axis if dot >= 0.0 else (-axis[0], -axis[1], -axis[2])
```

- [ ] **Step 4: 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_world_axes.py -q`
Expected: `6 passed`

- [ ] **Step 5: 전체 스위트**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: `646 passed` (640 + 6). 아직 아무도 이 모듈을 쓰지 않으므로 기존 테스트는 그대로다.

- [ ] **Step 6: 커밋**

```bash
git add trainer/scripts/world_axes.py trainer/scripts/tests/test_world_axes.py
git commit -m "feat(pose): 월드 축 규약을 연산으로 소유하는 모듈을 만든다

상수만 모으면 호출자가 world[2]를 다시 적는다. 연산을 쓰게 만들어야 막힌다.
유니티 Y-up, 왼손 좌표계. 아직 아무도 쓰지 않는다."
```

---

### Task 2: 네 모듈을 한 번에 전환한다

**Files:**
- Modify: `trainer/scripts/measured_score.py`, `trainer/scripts/stroke_frame.py`, `trainer/scripts/motion_key.py`, `trainer/scripts/measured_pose.py`
- Modify (픽스처·기대값): `trainer/scripts/tests/test_measured_score.py`, `test_measured_score_multiaxis.py`, `test_stroke_frame.py`, `test_motion_key*.py`, `test_measured_pose.py`, `trainer/pose_lab/tests/` 중 영향받는 것

**Interfaces:**
- Consumes: Task 1의 `world_axes` 전부

**부분 전환은 존재할 수 없다.** `measured_score`만 Y-up으로 바꾸고 `stroke_frame`을 두면 기하가 모순이라 어느 테스트도 의미가 없다. 네 모듈을 한 커밋으로 옮긴다.

- [ ] **Step 1: 픽스처를 새 규약으로 다시 짓는다**

이름이 뜻하는 바가 참이 되게 한다. `test_measured_score.py`:

```python
# 수평 스트로크: 접선이 월드 X(유니티에서 수평)라 up의 직교화가 정확히 월드 Y가 된다.
HORIZONTAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(1.0, 0.0, 0.0), gun=(1.2, 4.5, 0.2)
)
# 수직 스트로크: 접선이 월드 Y(유니티에서 수직)라 up의 직교화가 퇴화하고
# 대체축(수평)이 들어온다. 즉 여기서 up은 수평이고 수직은 tangent다.
VERTICAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(1.2, 4.4, 0.2)
)
```

`gun`은 법선 부호를 정하는 데만 쓰이므로 착탄점에서 **수평으로** 떨어진 위치여야 한다(작업자가 벽 앞에 선다). 위 값은 hit에서 −Z 방향으로 0.6 m다.

`test_measured_score_multiaxis.py`의 `HORIZONTAL`도 같은 값으로 맞춘다. `test_stroke_frame.py`의 `tangent=(0,1,0)` 호출들은 각 테스트가 **무엇을 보이려는지** 읽고 그에 맞게 고친다 — 수평 스트로크를 뜻하면 `(1,0,0)`, 수직 스트로크를 뜻하면 `(0,1,0)`을 유지한다.

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: 다수 실패. **이 시점의 실패 목록을 리포트에 그대로 적는다** — 무엇이 왜 깨졌는지가 전환의 증거다.

- [ ] **Step 3: 네 모듈을 전환한다**

`measured_score.py`:

```python
import world_axes

# 규약은 world_axes가 소유한다. 여기서 다시 정의하지 않는다.
WORLD_UP: Final[Vec3] = world_axes.UP
```

`_axis_components`에서:

```python
    world = _world_delta(local, frame)
    plane: Vec3 = world_axes.flatten(world)
    up_cm = world_axes.vertical_of(world) * 100.0
```

`_WorldFrame`의 기본 축도 새 규약에 맞춘다 — `up`은 `world_axes.UP`, `normal`·`tangent`는 수평 두 축.

`stroke_frame.py`: `WORLD_UP = world_axes.UP`, `FALLBACK_TANGENT`를 **수평** 축으로(`(1,0,0)`), 퇴화 시 대체축도 수평으로. 법선 부호 결정을 `world_axes.sign_toward(normal, to_worker)`로 바꾼다.

`motion_key.py`의 `heading_word`·`body_placement`: `delta`를 `world_axes.horizontal_of`로 뽑는다.

`measured_pose.py`의 `facing_yaw`: `world_axes.yaw_of(forward)`를 그대로 반환한다. `MIN_FACING_HORIZONTAL`은 `world_axes.MIN_HORIZONTAL`로 대체한다.

- [ ] **Step 4: 기대값을 손으로 유도해 고친다**

**돌려서 나온 값을 붙여넣지 않는다.** 바뀌는 단언마다 새 규약에서 유도하고 그 유도를 주석에 남긴다. 예: `HORIZONTAL`의 접선이 월드 X이므로 `up`은 월드 Y, `normal`은 `cross(X, Y)`의 부호를 건 쪽으로 맞춘 것 → 월드 ∓Z. 따라서 `to_local`의 x 성분은 벽 거리, z는 높이다.

유도할 수 없는 단언(예: 여러 회전이 겹친 σ 값)은 **그 테스트가 무엇을 보이려 했는지**로 돌아가 기하를 다시 고르는 편이 낫다. 관측값을 박으면 그 테스트는 그때부터 아무것도 검사하지 않는다.

- [ ] **Step 5: 전체 스위트**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: `646 passed`. 테스트 **수**는 변하지 않아야 한다 — 픽스처와 기대값만 바뀌고 테스트가 사라지거나 생기지 않는다. 수가 줄면 테스트를 지운 것이므로 그 이유를 리포트에 적는다.

- [ ] **Step 6: 커밋**

```bash
git add trainer/scripts/measured_score.py trainer/scripts/stroke_frame.py \
        trainer/scripts/motion_key.py trainer/scripts/measured_pose.py \
        trainer/scripts/tests/ trainer/pose_lab/tests/
git commit -m "fix(pose): 월드 축을 유니티 Y-up으로 바로잡는다

CSV는 유니티 좌표(Y가 위, 왼손계)인데 코드가 Z를 수직으로 다뤘다. 측정·사용자
확인·FBX 선언 세 가지가 일치한다. 640개가 전부 통과한 이유는 축 이름이 데이터
안에서 검증될 수 없기 때문이다.

네 모듈을 한 번에 옮긴다 — 부분 전환은 기하가 모순이라 중간 상태가 없다.
픽스처는 이름이 뜻하는 바가 새 규약에서 참이 되도록 다시 지었다."
```

---

### Task 3: 규약이 실제로 지배하는지 고정한다

**Files:**
- Test: `trainer/scripts/tests/test_world_axes.py`
- Modify (필요시): Task 2가 전환한 네 모듈

**Interfaces:**
- Consumes: Task 2가 전환한 네 모듈

**배경:** Task 2가 리터럴을 하나라도 남기면 그 모듈은 규약을 따르지 않는다. 그리고 그건 **눈으로는 안 보인다** — 지금 규약과 리터럴이 우연히 같은 답을 낼 수 있기 때문이다. 규약을 바꿔치기해 전부 따라 움직이는지 본다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_every_module_follows_the_convention_not_a_literal(monkeypatch) -> None:
    """UP_INDEX를 바꾸면 네 모듈의 출력이 **전부** 따라 움직여야 한다.

    하나라도 안 움직이면 그 모듈이 리터럴을 다시 박은 것이다. 그 상태는 지금
    우연히 옳은 답을 낼 수 있어서 눈으로는 안 보인다.
    """
    measured_score = importlib.import_module("measured_score")
    stroke_frame = importlib.import_module("stroke_frame")
    measured_pose = importlib.import_module("measured_pose")
    motion_key = importlib.import_module("motion_key")

    def sample():
        basis = stroke_frame.stroke_basis(
            hit=(1.2, 4.5, 0.8), tangent=(1.0, 0.0, 0.0), gun=(1.2, 4.5, 0.2)
        )
        return {
            "axis": measured_score._axis_components((0.07, 0.05, 0.03), basis),
            "basis": (basis.normal, basis.tangent, basis.up),
            "yaw": measured_pose.facing_yaw((10.0, 20.0, 30.0)),
            "heading": motion_key.heading_word(0.3, (0.4, 0.5, 0.6)),
        }

    before = sample()
    monkeypatch.setattr(world_axes, "UP_INDEX", 2)
    monkeypatch.setattr(world_axes, "UP", (0.0, 0.0, 1.0))
    monkeypatch.setattr(world_axes, "HORIZONTAL_INDICES", (0, 1))
    after = sample()

    stuck = [k for k in before if before[k] == after[k]]
    assert not stuck, f"규약을 따르지 않는 항목: {stuck}"
```

모듈이 임포트 시점에 값을 복사하면(`WORLD_UP = world_axes.UP`) monkeypatch가 안 먹는다. **복사하지 말고 호출 시점에 읽도록 고친다** — 복사는 규약이 다시 흩어지는 첫걸음이다. 이 테스트가 그 복사를 잡아낸다.

- [ ] **Step 2: 실패를 확인하고 모듈을 고친다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_world_axes.py -q`
안 움직이는 항목이 나오면 **그 모듈이 아직 리터럴을 쓰거나 값을 복사한다.** 테스트를 약화시키지 말고 모듈을 고친다.

- [ ] **Step 3: 전체 스위트와 커밋**

Expected: `647 passed` (646 + 1)

```bash
git add trainer/scripts/world_axes.py trainer/scripts/tests/test_world_axes.py \
        trainer/scripts/measured_score.py trainer/scripts/stroke_frame.py \
        trainer/scripts/motion_key.py trainer/scripts/measured_pose.py
git commit -m "test(pose): 축 규약이 실제로 네 모듈을 지배하는지 고정한다

리터럴이 하나라도 남으면 지금은 우연히 옳은 답을 내서 눈으로 안 보인다.
규약을 바꿔치기해 전부 따라 움직이는지 본다."
```

---

### Task 4: FBX 리더와 레벨 정렬

**Files:**
- Create: `trainer/scripts/level_mesh.py`, `trainer/data/level_alignment.json`
- Test: `trainer/scripts/tests/test_level_mesh.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Mesh` — `vertices: list[Vec3]`, `triangles: list[tuple[int,int,int]]`, `uvs: list[tuple[float,float]]`, `declared: dict[str, object]`
  - `@dataclass(frozen=True) class Alignment` — `level: int`, `provenance: str`, `rotation: tuple[float,...]`(9), `translation: Vec3`, `scale: float`, `declared_scale: float`, `fit: dict[str, float]`
  - `def read_fbx(path: Path) -> Mesh`
  - `def closest_distance(mesh: Mesh, point: Vec3) -> float` — 삼각형까지의 정확한 최근접 거리(정점 최근접이 아니다)
  - `def place(mesh: Mesh, alignment: Alignment) -> Mesh`
  - `def load_alignment(level: int) -> Alignment`
  - `def solve_alignment(mesh: Mesh, hits: Sequence[Vec3]) -> Alignment`

**배경:** 바이너리 FBX 7400이다. 표준 라이브러리만 쓴다 — 컨테이너에 mesh 라이브러리가 없고 설치하지 않는다. 27바이트 헤더 뒤 `(endOffset:u32, numProps:u32, propListLen:u32, nameLen:u8, name, props...)` 레코드가 이어지고, `tell() < endOffset - 13`인 동안 재귀한 뒤 `seek(endOffset)`한다. 속성 타입: `Y C I L F D`는 스칼라, `f d l i b`는 `(len:u32, encoding:u32, compressedLen:u32)` 배열로 `encoding`이 0이 아니면 `zlib.decompress`, `S R`은 길이 선행 바이트열. `PolygonVertexIndex`는 다각형 끝 인덱스가 `~i`(비트 반전)로 표시된다.

**변환은 풀어낸 것이지 선언된 것이 아니다.** FBX가 선언한 노드 변환으로는 3.68 m 어긋난다 — 파일에 기록되지 않은 씬 수준 회전이 빠져 있다. `provenance` 필드가 그 사실을 나른다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
LEVEL1_FBX = Path("/data/sources/modelings/learning_mode_lv.1_rev.1.1.fbx")


def test_reader_parses_a_known_level_exactly() -> None:
    """레벨 1은 정점 16개, UV 32쌍이다(스파이크 실측)."""
    mesh = level_mesh.read_fbx(LEVEL1_FBX)
    assert len(mesh.vertices) == 16
    assert len(mesh.uvs) == 32
    assert mesh.triangles


def test_declared_up_axis_agrees_with_the_world_convention() -> None:
    """FBX가 선언한 위 축이 world_axes의 규약과 같다. 세 번째 독립 확인이다."""
    mesh = level_mesh.read_fbx(LEVEL1_FBX)
    assert mesh.declared["UpAxis"] == world_axes.UP_INDEX


def test_closest_distance_is_to_the_triangle_not_the_nearest_vertex() -> None:
    """면 한가운데 바로 위의 점은 어느 정점보다도 훨씬 가깝다.

    정점 최근접으로 구현하면 이 값이 4.3 근처가 나온다(가장 가까운 정점까지의 거리).
    """
    mesh = level_mesh.Mesh(
        vertices=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 10.0)],
        triangles=[(0, 1, 2)], uvs=[], declared={},
    )
    assert level_mesh.closest_distance(mesh, (3.0, 0.5, 3.0)) == pytest.approx(0.5)


def test_alignment_records_its_provenance() -> None:
    """선언된 변환과 풀어낸 변환은 다른 주장이다. 그 구분이 사라지면 안 된다."""
    a = level_mesh.load_alignment(1)
    assert a.provenance in {"solved", "declared"}
    assert a.provenance == "solved"  # 현재는 풀어낸 것뿐이다
    assert a.declared_scale != a.scale  # 선언값과 다르다는 것이 이 필드의 존재 이유다
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_level_mesh.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'level_mesh'`

- [ ] **Step 3: 구현한다**

위 인터페이스대로. `solve_alignment`는 Umeyama 유사변환(회전+평행이동+균일 스케일)이다. **퇴화 특이값에서 불안정하다** — 레벨 1은 정확히 평면(착탄 X가 7만 5천 프레임에서 비트 단위 동일)이라 이 경우가 실제로 걸린다. 합성 데이터(알려진 회전·스케일을 적용한 점군)로 먼저 검증한 뒤 실데이터에 쓴다.

- [ ] **Step 4: 통과 확인 후 `level_alignment.json`을 생성한다**

`solve_alignment`를 레벨 1 실데이터로 돌려 산출물을 만들고 커밋한다. 스키마는 스펙 §D4대로.
기대값(스파이크 실측): 맞춘 스케일 ≈ 0.02568, 선언값 유래 ≈ 0.02540, 적합 median 0.0 mm / p95 0.0 mm / max 46 mm.
**나온 값이 이와 크게 다르면 멈추고 리포트에 적는다** — 스파이크를 재현하지 못한 것이다.

- [ ] **Step 5: 전체 스위트와 커밋**

Expected: `651 passed` (647 + 4)

```bash
git add trainer/scripts/level_mesh.py trainer/scripts/tests/test_level_mesh.py trainer/data/level_alignment.json
git commit -m "feat(pose): 레벨 FBX 리더와 정렬 — 변환은 풀어낸 것이다

선언된 노드 변환으로는 3.68m 어긋난다. 파일에 없는 씬 수준 회전이 빠져 있다.
provenance 필드가 선언된 것과 풀어낸 것을 구분해 나른다."
```

---

### Task 5: 메시 앵커 — 접촉점이 면 위인가

**Files:**
- Test: `trainer/scripts/tests/test_level_mesh.py`

**Interfaces:**
- Consumes: Task 4의 `read_fbx`, `place`, `load_alignment`, UV 조회(B3가 노출), `closest_distance`(기록용)

**배경:** 축 이름은 데이터 안에서 검증할 수 없다 — 그것이 이 결함이 640개 테스트를 통과한 이유다. **메시가 그 예외를 만든다.** 착탄점이 면 위에 놓이면 월드 좌표 해석이 외부 기하와 맞는다는 증거가 된다.

**⚠️ 이 태스크의 초안 설계는 틀렸고 고쳤다. 스펙 §D5를 읽을 것.**

초안은 최근접 삼각형 거리로 판정했다(p95 ≤ 10 mm, 최대 ≤ 100 mm). Task 4 리뷰가 그것이 **아무것도 판별하지 못함**을 보였다: 착탄점 15,897개가 정확히 한 평면에 있어(X의 서로 다른 float가 하나뿐) 중앙값·p95가 **모든 후보에서** ~1e-7 mm다 — 메시의 **틀린 면**에 붙은 적합까지 포함해서. `max_mm`은 같은 적합을 재실행해도 크게 흔들리는 외톨이 순서통계량이라(관측 0.64~35.0 mm) 무엇을 통과시킬지 예측할 수 없다. **판별력이 없는 쪽은 p95이고, 논거는 그쪽으로 서야 한다**(스펙 §D5의 정정 참고).

**판정은 UV 일치로 한다.** 옳은 후보가 R² 0.9999~1.0000, 틀린 면 후보가 0.000001~0.0038로 **여섯 자릿수** 갈린다. 그 사이 어디에 선을 그어도 같은 판정이 나오므로 임계가 튜닝된 값이 아니라 자연스러운 간극이다. 거리에는 그런 간극이 없었다.

거리는 **함께 기록하되 판정에 쓰지 않는다.**

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
@pytest.mark.skipif(not LEVEL1_FBX.exists() or not CSV_ROOT.exists(),
                    reason="실데이터가 없는 환경")
def test_hit_points_agree_with_the_level_mesh_uv() -> None:
    """접촉점의 UV가 메시 면의 UV와 일치한다. 월드 좌표 해석의 외부 앵커다.

    거리로 판정하지 않는다 — 착탄점이 정확히 한 평면에 있어서 거리 백분위는 틀린
    면에 붙은 적합까지 통과시킨다(스펙 §D5). UV는 옳은 후보와 틀린 후보를 여섯
    자릿수로 가른다.

    허용치는 측정을 본 뒤 정했지만, 실측 간극이 0.9999 대 0.0038이라 그 사이
    어디에 선을 그어도 판정이 같다. 회귀를 잡는 용도이므로 사후 선택이 허용된다.
    """
    placed = level_mesh.place(level_mesh.read_fbx(LEVEL1_FBX), level_mesh.load_alignment(1))
    hits = _level1_hits(limit=5000)
    assert len(hits) >= 5000, "표본이 모자라면 이 검사는 아무것도 보증하지 않는다"

    # 옆면 삼각형(전체의 약 5%)은 무관한 UV를 갖는다. 큰 면으로 제한한다 —
    # 안 걷어내면 순수 이상치로 R²를 끌어내린다.
    pairs = level_mesh.uv_pairs_on_dominant_face(placed, hits)
    assert len(pairs) >= 0.9 * len(hits), f"큰 면 비율 {len(pairs) / len(hits):.1%}"

    for axis in (0, 1):
        slope, r2 = level_mesh.uv_regression(pairs, axis)
        # **부호까지** 비교한다. abs(slope)로 재면 180도 뒤집힌 정렬이 그대로 통과한다 —
        # 실제로 시드마다 그 모호성이 나왔고(Rx(180) @ R), R^2도 부호 불변이라 못 잡는다.
        # 기준값은 level_mesh.EXPECTED_UV_SLOPE(-2048.0)이고 그 출처는 실측이다.
        target = level_mesh.EXPECTED_UV_SLOPE
        assert abs(slope - target) <= 0.01 * abs(target), f"axis {axis} 기울기 {slope:.1f}"
        assert r2 >= 0.99, f"axis {axis} R^2 {r2:.6f}"
```

`_level1_hits`는 `/data/sources/professionals` 아래 레벨 1 CSV에서 착탄점을 읽는 헬퍼다. `simulator_csv` / `measured_pose`의 기존 읽기 경로를 재사용하고 새로 쓰지 않는다.

`uv_pairs_on_dominant_face` / `uv_regression`이 `level_mesh`에 없으면 이 태스크에서 추가한다 — B2/B3가 같은 계산을 이미 하고 있으므로 **그 구현을 재사용하고 복제하지 않는다.**

- [ ] **Step 2: 앵커가 진짜인지 실험으로 확인한다 — 두 가지로**

(a) 정렬의 평행이동에 **5 cm**를 더한다. Expected: FAIL.
(b) 정렬을 **틀린 패널 면**의 후보로 바꾼다(Task 4의 B2가 그 후보들을 계산할 수 있다).
    Expected: FAIL. **초안의 거리 검사는 이것을 통과시켰다** — 그래서 이 실험이 핵심이다.

각각 확인 후 되돌리고 통과를 재확인한다. **관측한 실패 출력 둘을 리포트에 적는다.**

- [ ] **Step 3: 전체 스위트와 커밋**

Expected: `652 passed` (651 + 1)

```bash
git add trainer/scripts/tests/test_level_mesh.py
git commit -m "test(pose): 접촉점이 레벨 메시 면 위에 있음을 고정한다

축 이름은 데이터 안에서 검증할 수 없다 — 그것이 이 결함이 640개를 통과한 이유다.
메시가 그 예외를 만든다. 허용치는 실측을 본 뒤 정했고 그 사실을 독스트링에 적었다."
```

---

### Task 6: 재빌드와 재측정 — 전후 비교

**Files:**
- 코드 변경 없음. 산출물과 리포트만.

**배경:** 축이 바뀌면 동작 키가 96.5% 다른 버킷으로 가므로 **산포 아티팩트가 잘못된 버킷 위에 구워져 있다.**

**표집을 바꾸지 않는다:** 200쌍, 시드 `20260828`, 같은 선정 규칙.

- [ ] **Step 1: 산포 아티팩트 재빌드**

```bash
docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/build_spread_reference.py \
  --root /data/sources/professionals --worker 전형기 \
  --out /output/spread_reference_expert_yup.json --log-level INFO
```

층별 커버리지 표를 낸다. 이전 값: uv8 25.0% / uv4 55.7% / uv2 81.4% / uv_off 95.0%.

- [ ] **Step 2: 개선율 재측정**

```bash
docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/improvement_study.py \
  --root /data/sources/professionals \
  --reference /output/spread_reference_expert_yup.json \
  --pairs 200 --seed 20260828 \
  --index-cache /output/improvement_study_index_cache_yup.json \
  --out /output/improvement_study_yup.json --progress --log-level INFO
```

인덱스 캐시는 **새 경로로 새로 만든다** — 옛 캐시는 옛 축의 동작 키로 지어졌다. 옛 캐시를 재사용하면 옛 버킷으로 표집하게 된다.

**포그라운드로 붙들거나 `docker exec -d`로 로그에 쓰고 그 로그를 직접 폴링한다.** 외부 알림에 맡기고 대기 상태로 들어가지 마라 — 이 프로젝트에서 그 대기가 여러 차례 에이전트 턴을 통째로 태웠다.

- [ ] **Step 3: 전후 비교표**

`.superpowers/sdd/2026-09-02-world-axis-convention/task-6-report.md`에 낸다. 표집 규칙을 숫자보다 먼저 적는다.

| 지표 | 수정 전 (Z-up) | 수정 후 (Y-up) |
|---|---|---|
| `split` 중앙 / q3 / ≥0.70 | 0.0 / 0.158 / 3 of 149 | ? |
| `same` 중앙 / ≥0.70 | 0.0 / 0 of 157 | ? |
| 발동률(어떤 지시든) | 43.3% (68/157) | ? |
| 천장(발동분) 중앙 / ≥0.70 가능 | 0.846 / 76.5% | ? |
| 회수율 중앙 (same / split) | 0.294 / 0.329 | ? |
| REQ-19 게이트 | 149 fail / 0 pass | ? |
| 커버리지 uv8/uv4/uv2/uv_off | 25.0 / 55.7 / 81.4 / 95.0% | ? |

**새 값이 무엇이 나오든 그대로 적는다.** 좋아지면 이전 결론을, 나빠지면 이번 수정을 의심하고 싶어지는데 **둘 다 하지 않는다** — 같은 표집으로 잰 두 값을 나란히 적고 판단은 사람에게 넘긴다.

70% 목표 달성 여부를 한 문장으로 명확히 적는다. 완곡하게 쓰지 않는다.

- [ ] **Step 4: 전체 스위트와 커밋**

Expected: `652 passed` (코드 변경 없음)

```bash
git add .superpowers/sdd/2026-09-02-world-axis-convention/task-6-report.md
git commit -m "docs(pose): 축 수정 후 재측정 — 같은 표집으로 전후를 나란히 낸다"
```

---

## 이번 계획에서 하지 않는 것

| 항목 | 이유 |
|---|---|
| 격자점별 기저 평균 | 축과 함께 고치면 숫자가 왜 움직였는지 귀속이 불가능하다 |
| 레벨 2~10 정렬 | 별건. 레벨 10은 아직 신뢰할 수 없다(UV R² 0.23~0.32) |
| 표면 법선 기저 | 위가 끝나야 일반적으로 쓸 수 있다 |
| 각도 교정 재검토 | 법선이 생겼으나 레벨 1뿐 |
| 스켈레톤 뷰 | 올바른 축 확정 후 |

## Self-Review 결과

**1. 스펙 커버리지**

| 스펙 절 | 태스크 |
|---|---|
| §4 D1 규약 모듈 | Task 1 |
| §4 D2 네 모듈 적용 | Task 2 |
| §8 규약 지배 검사 | Task 3 |
| §5 D4 FBX·정렬·provenance | Task 4 |
| §5 D5 메시 앵커와 허용치 | Task 5 |
| §6 재빌드·재측정 | Task 6 |
| §3 점수 비불변 | 기록만(§7 범위 밖) |

**2. Placeholder 스캔:** 통과. Task 4의 구현 본문은 코드 전문 대신 인터페이스·파싱 규칙·기대 수치를 적었다 — 인터페이스와 테스트가 동작을 고정하고, 파싱 규칙은 스파이크에서 이미 검증된 것을 그대로 옮겼다.

**3. 타입 일관성**
- `world_axes`의 이름(T1)을 T2가 소비하고 T3가 monkeypatch한다. T3가 통과하려면 T2가 값을 **복사하지 않아야** 하며, 그 요구를 T3 Step 1에 적었다.
- `Mesh`·`Alignment`(T4)를 T5가 소비한다.
- 테스트 누적: 640 → 646(T1) → 646(T2, 픽스처만) → 647(T3) → 651(T4) → 652(T5) → 652(T6).
- **T2가 테스트 수를 바꾸면 안 된다.** 줄면 테스트를 지운 것이고 이유를 대야 한다.
