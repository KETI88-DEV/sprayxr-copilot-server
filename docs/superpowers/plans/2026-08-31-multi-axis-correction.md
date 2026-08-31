# 다축 교정 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 교정을 오른손 한 축에서 오른손 다축 + 머리 자세 문장으로 넓히되, 개선율이 오직 실제로 발부한 지시만 계상하도록 불변식을 시그니처에 박는다.

**Architecture:** `measured_score`에 `AxisCorrection` 구조체와 이를 내는 선택 함수 셋(`right_axes`, `head_stance`, `issued_axes`)을 두고, `issued_shift(-> Vec3|None)`를 `issued_shifts(-> dict[str, Vec3])`로 바꿔 `apply_measured`가 딕셔너리에 없는 점을 옮길 방법 자체를 없앤다. 서비스는 `stance`와 `axes[]`를 응답에 싣고, 화면이 그것을 그린다. 마지막에 개선율 하네스를 같은 표집으로 다시 돌려 전후를 비교한다.

**Tech Stack:** Python 3.11, pytest, FastAPI, 순수 HTML/CSS/ES 모듈(빌드 단계 없음). 컨테이너 `ship-paint-trainer`.

**Spec:** `docs/superpowers/specs/2026-08-31-multi-axis-correction-design.md`

## Global Constraints

- 테스트는 컨테이너에서 돈다: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`. **`trainer`와 `mcp_servers`를 모두 넘겨야 한다** — 저장소 루트에서 맨 `pytest -q`만 돌리면 수집 단계에서 `ModuleNotFoundError: No module named 'simulator_csv'`로 죽는다(기존 문제이며 이 작업과 무관하고 고치지 않는다).
- **시작 베이스라인: 612 passed.**
- 사전등록 상수는 움직이지 않는다: `CORRECTION_SIGMA_THRESHOLD = 2.0`, `SIGMA_FLOOR_M = 0.02`, `MAX_SIGMA = 6.0`, `MIN_CORRECTION_CM = 1.0`, `GATE_RATIO = 0.5`, `DEFAULT_ERROR_M = 0.144`, `N_MIN = 20`, `PAIR_SIGMA_FACTOR = sqrt(2)`, `DEFAULT_WEIGHTS = {"right": 1.0, "head": 0.4, "left": 0.15}`. 이 중 하나가 틀렸다고 판단되면 고치지 말고 리포트에 적고 멈춘다.
- **새 임계를 만들지 않는다.** 축 포함 규칙은 위 상수만 참조한다. 리터럴 `2.0`이나 `1.0`을 새로 박지 않는다.
- 왼손(`left`)은 절대 지시하지 않는다. 채점 가중치 0.15는 유지한다.
- 머리 `lateral`은 절대 지시하지 않는다(작업자 요가 실측이 아니다 — `service.ASSUMED_YAW`).
- 자세 문구에 `앉아`가 들어가면 안 된다. HMD로는 앉음과 쪼그림을 구별할 수 없다.
- `git add -A`를 절대 쓰지 않는다. 작업 트리는 사람과 공유 중이고 `.opencode/`, `docs/Interface.pdf`, `.superpowers/pose-lab-*.md` 등 타인의 미추적 파일이 있다. 명시한 경로만 스테이징한다.
- `.env`를 읽거나 출력하거나 커밋하지 않는다.
- 성질을 보호한다고 이름 붙인 테스트는 **그 성질을 일부러 깨뜨려 실패를 확인한 뒤** 되돌린다. 확인 결과를 리포트에 적는다.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `trainer/scripts/measured_score.py` | 교정 선택과 불변식. `AxisCorrection`, `right_axes`, `head_stance`, `stance_text`, `issued_axes`, `issued_shifts`, `apply_measured` |
| `trainer/scripts/tests/test_measured_score_multiaxis.py` | (신규) 다축 선택·불변식·단조성 |
| `trainer/scripts/tests/test_measured_score.py` | (수정) 머리가 교정 대상이 되며 깨지는 기존 테스트 2건을 좁힌다 |
| `trainer/pose_lab/service.py` | 응답에 `stance`·`axes[]` |
| `trainer/pose_lab/tests/test_service_multiaxis.py` | (신규) 응답 계약 |
| `trainer/pose_lab/static/{index.html,app.js,style.css}` | 자세 배너, 축 표, split 출처 |
| `trainer/scripts/improvement_study.py` | 천장 정의 갱신, 재측정 |

---

### Task 1: `AxisCorrection`과 오른손 다축 선택

**Files:**
- Modify: `trainer/scripts/measured_score.py`
- Test: `trainer/scripts/tests/test_measured_score_multiaxis.py` (신규)

**Interfaces:**
- Consumes: 기존 `PointDelta(point, sigma, local_delta, samples, axis_sigmas)`, `_axis_components(local, frame) -> dict[str, tuple[float, Vec3 | None, str]]`, `_correction_candidates(local, frame) -> list[tuple[str, float, str, Vec3]]`, `_largest_correction(local, frame, axis_sigmas)`, `CORRECTION_SIGMA_THRESHOLD`, `MIN_CORRECTION_CM`, `AXIS_ORDER`
- Produces:
  - `@dataclass(frozen=True, slots=True) class AxisCorrection` — 필드 `point: str`, `axis: str`, `magnitude_cm: float`, `direction: str`, `sigma: float`, `unit: Vec3`
  - `def right_axes(delta: PointDelta, frame: AxisFrame = WORLD_FRAME) -> list[AxisCorrection]`

**배경(구현자가 알아야 할 것):** `PointDelta.axis_sigmas`는 `score_track`이 채운다. 그러나 구조체를 직접 만드는 옛 호출자(테스트 포함)는 비워 둔다. 축별 σ가 없으면 축별 게이트를 걸 수 없으므로, **비어 있으면 예전처럼 문장 축 하나만** 낸다. `_largest_correction`이 이미 같은 패턴(`if axis_sigmas:`)을 쓴다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`trainer/scripts/tests/test_measured_score_multiaxis.py`:

```python
#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_score = importlib.import_module("measured_score")
stroke_frame = importlib.import_module("stroke_frame")

# 수평 스트로크: 접선이 월드 Y라 up이 정확히 월드 Z이고 normal이 월드 X다.
HORIZONTAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.8)
)
T = measured_score.CORRECTION_SIGMA_THRESHOLD


def _local(basis: object, world: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        sum(world[i] * getattr(basis, name)[i] for i in range(3))
        for name in ("normal", "tangent", "up")
    )


def _right(world: tuple[float, float, float], axis_sigmas: dict[str, float] | None) -> object:
    return measured_score.PointDelta(
        point="right",
        sigma=3.0,
        local_delta=_local(HORIZONTAL, world),
        samples=30,
        axis_sigmas=axis_sigmas or {},
    )


def test_every_axis_over_the_threshold_is_issued() -> None:
    """축별 sigma가 임계를 넘는 축은 전부 나간다. 하나만 내면 개선 여지를 버린다."""
    delta = _right(
        (0.08, 0.05, -0.06),  # wall +8cm, lateral +5cm, vertical -6cm
        {"wall": 3.5, "vertical": 2.6, "lateral": 2.2},
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert {c.axis for c in issued} == {"wall", "vertical", "lateral"}
    assert all(c.point == "right" for c in issued)


def test_axes_below_the_threshold_are_not_issued() -> None:
    """sigma가 임계 아래인 축은 이상하지 않다. 지시하면 지어낸 교정이다."""
    delta = _right(
        (0.08, 0.05, -0.06),
        {"wall": 3.5, "vertical": 1.1, "lateral": 0.4},
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert {c.axis for c in issued} == {"wall"}


def test_axes_below_the_minimum_centimetre_are_not_issued() -> None:
    """sigma가 커도 5mm를 옮기라는 지시는 사람이 따를 수 없다."""
    delta = _right(
        (0.08, 0.005, 0.0),  # lateral 0.5cm — MIN_CORRECTION_CM(1.0) 미만
        {"wall": 3.5, "vertical": 3.0, "lateral": 3.0},
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert "lateral" not in {c.axis for c in issued}
    assert "wall" in {c.axis for c in issued}


def test_the_sentence_axis_is_issued_even_when_its_own_sigma_is_low() -> None:
    """단조 보장. 결합 sigma가 임계를 넘는데 개별 축이 아무도 못 넘는 경우가 있다.

    이 보장이 없으면 오늘 나가던 교정이 새 코드에서 사라진다.
    """
    delta = _right(
        (0.08, 0.0, 0.0),
        {"wall": 1.4, "vertical": 1.3, "lateral": 1.2},  # 아무도 2.0 미만
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert [c.axis for c in issued] == ["wall"]


def test_no_axis_sigmas_falls_back_to_the_single_sentence_axis() -> None:
    """축별 sigma 없이 만든 구조체는 예전 동작 그대로다(옛 호출자 보호)."""
    delta = _right((0.08, 0.05, -0.06), None)
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert len(issued) == 1


def test_below_the_point_gate_nothing_is_issued() -> None:
    """점 게이트는 그대로다. 오른손이 이상하지 않으면 아무 말도 하지 않는다."""
    delta = measured_score.PointDelta(
        point="right",
        sigma=T,  # 초과가 아니라 같음 — 열리지 않아야 한다
        local_delta=_local(HORIZONTAL, (0.08, 0.05, -0.06)),
        samples=30,
        axis_sigmas={"wall": 3.5, "vertical": 2.6, "lateral": 2.2},
    )
    assert measured_score.right_axes(delta, HORIZONTAL) == []


def test_magnitude_and_unit_agree_so_the_sentence_cannot_drift() -> None:
    """cm와 단위벡터가 같은 곳에서 나온다. 어긋나면 문장과 시뮬레이션이 갈라진다."""
    delta = _right((0.08, 0.0, 0.0), {"wall": 3.5, "vertical": 0.1, "lateral": 0.1})
    (wall,) = measured_score.right_axes(delta, HORIZONTAL)
    assert wall.axis == "wall"
    # HORIZONTAL의 normal은 월드 X다.
    assert abs(wall.unit[0]) > 0.99
    assert abs(wall.magnitude_cm - 8.0) < 0.5
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_measured_score_multiaxis.py -q`
Expected: FAIL — `AttributeError: module 'measured_score' has no attribute 'right_axes'`

- [ ] **Step 3: 구현한다**

`measured_score.py`의 `RightHandFinding` 정의 아래에 구조체를 더한다:

```python
@dataclass(frozen=True, slots=True)
class AxisCorrection:
    """발부한 축 교정 하나. **이 목록에 있는 것만 시뮬레이션이 옮긴다.**

    `magnitude_cm`과 `unit`이 같은 곳(`_axis_components`)에서 나오므로 문장이 말한
    크기와 시뮬레이션이 옮기는 양이 갈라질 수 없다. `sigma`는 그 축 자신의 값이지
    점의 결합 sigma가 아니다 — 읽는 쪽이 어느 축이 얼마나 이상한지 봐야 한다.
    """

    point: str  # "right" | "head". "left"는 절대 들어가지 않는다.
    axis: str  # "wall" | "vertical" | "lateral"
    magnitude_cm: float  # 부호 있음
    direction: str  # 동사/방향어. 판정할 수 없으면 빈 문자열
    sigma: float  # 그 축의 sigma
    unit: Vec3  # 월드 단위벡터
```

`_largest_correction` 아래에 선택 함수를 더한다:

```python
def right_axes(
    delta: PointDelta, frame: AxisFrame = WORLD_FRAME
) -> list[AxisCorrection]:
    """오른손에서 발부할 축 전부. 점 게이트는 그대로, 축은 자기 sigma로 판정한다.

    문장 축(sigma가 가장 큰 축)은 자기 축 sigma와 무관하게 **항상** 포함한다. 이것이
    단조 보장이다 — 결합 sigma는 축별 sigma의 제곱합의 제곱근이라 "합쳐서는 임계를
    넘지만 개별 축은 아무도 못 넘는" 경우가 실제로 있고, 그때 축별 게이트만 걸면
    오늘 나가던 교정이 조용히 사라진다.

    `axis_sigmas`가 비어 있으면(구조체를 직접 만든 옛 호출자) 축별 판정을 할 수
    없으므로 예전처럼 문장 축 하나만 낸다.
    """
    if delta.sigma <= CORRECTION_SIGMA_THRESHOLD:
        return []
    chosen = _largest_correction(delta.local_delta, frame, delta.axis_sigmas)
    if chosen is None:
        return []
    axis_sigmas = delta.axis_sigmas
    picked: dict[str, tuple[str, float, str, Vec3]] = {chosen[0]: chosen}
    if axis_sigmas:
        for axis, magnitude_cm, direction, unit in _correction_candidates(
            delta.local_delta, frame
        ):
            if axis_sigmas.get(axis, 0.0) > CORRECTION_SIGMA_THRESHOLD:
                picked[axis] = (axis, magnitude_cm, direction, unit)
    return [
        AxisCorrection(
            point=delta.point,
            axis=axis,
            magnitude_cm=magnitude_cm,
            direction=direction,
            sigma=float(axis_sigmas.get(axis, delta.sigma)),
            unit=unit,
        )
        for axis, magnitude_cm, direction, unit in (
            picked[name] for name in AXIS_ORDER if name in picked
        )
    ]
```

`_correction_candidates`가 이미 `MIN_CORRECTION_CM` 미만과 후보 불가 축을 걸러내므로 별도 필터가 필요 없다.

- [ ] **Step 4: 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_measured_score_multiaxis.py -q`
Expected: `7 passed`

- [ ] **Step 5: 단조 보장이 진짜인지 실험으로 확인한다**

`right_axes`에서 `picked: dict[...] = {chosen[0]: chosen}`을 임시로 `picked = {}`로 바꾸고 위 테스트를 돌린다.
Expected: `test_the_sentence_axis_is_issued_even_when_its_own_sigma_is_low` FAIL.
확인 후 되돌리고 다시 통과를 확인한다. 관측한 실패 출력을 리포트에 적는다.

- [ ] **Step 6: 전체 스위트**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: `619 passed` (612 + 7)

- [ ] **Step 7: 커밋**

```bash
git add trainer/scripts/measured_score.py trainer/scripts/tests/test_measured_score_multiaxis.py
git commit -m "feat(pose): 오른손 교정을 축별 sigma로 다축화한다

임계를 넘는 축은 전부 발부한다. 문장 축은 자기 축 sigma와 무관하게 항상 포함해
오늘보다 적게 내보내는 경우를 막는다. 새 임계는 만들지 않았다."
```

---

### Task 2: 머리를 자세 문장으로 번역

**Files:**
- Modify: `trainer/scripts/measured_score.py`
- Test: `trainer/scripts/tests/test_measured_score_multiaxis.py`

**Interfaces:**
- Consumes: Task 1의 `AxisCorrection`, `right_axes`. 기존 `_correction_candidates`, `CORRECTION_SIGMA_THRESHOLD`
- Produces:
  - `STANCE_AXES: Final[tuple[str, ...]]` — `("wall", "vertical")`
  - `def head_stance(delta: PointDelta, frame: AxisFrame = WORLD_FRAME) -> AxisCorrection | None`
  - `def stance_text(correction: AxisCorrection) -> str`

**배경:** 머리에는 별도의 점 게이트를 두지 않는다. 결합 sigma는 축별 sigma의 제곱합의 제곱근이므로 어느 축이 임계를 넘으면 결합 sigma도 반드시 넘는다 — 점 게이트를 더해도 판정이 바뀌지 않는다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

같은 파일에 이어 붙인다:

```python
def _head(world: tuple[float, float, float], axis_sigmas: dict[str, float]) -> object:
    return measured_score.PointDelta(
        point="head",
        sigma=3.0,
        local_delta=_local(HORIZONTAL, world),
        samples=30,
        axis_sigmas=axis_sigmas,
    )


def test_head_height_becomes_a_stance_sentence() -> None:
    delta = _head((0.0, 0.0, -0.12), {"wall": 0.2, "vertical": 3.1, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None
    assert stance.point == "head" and stance.axis == "vertical"
    text = measured_score.stance_text(stance)
    assert "낮춰" in text


def test_head_wall_distance_becomes_a_step_back_sentence() -> None:
    delta = _head((0.18, 0.0, 0.0), {"wall": 3.4, "vertical": 0.2, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None and stance.axis == "wall"
    assert "걸음" in measured_score.stance_text(stance)


def test_only_one_stance_sentence_even_when_both_axes_fire() -> None:
    """자세 문장도 하나다. 두 가지를 동시에 시키면 아무것도 안 한다."""
    delta = _head((0.18, 0.0, -0.12), {"wall": 3.4, "vertical": 2.9, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None and stance.axis == "wall"  # sigma가 큰 쪽


def test_head_lateral_is_never_issued() -> None:
    """작업자 요가 실측이 아니라 좌우를 판정할 기준이 없다(service.ASSUMED_YAW)."""
    delta = _head((0.0, 0.20, 0.0), {"wall": 0.1, "vertical": 0.1, "lateral": 5.0})
    assert measured_score.head_stance(delta, HORIZONTAL) is None


def test_head_below_the_threshold_says_nothing() -> None:
    delta = _head((0.0, 0.0, -0.12), {"wall": 0.2, "vertical": 1.9, "lateral": 0.1})
    assert measured_score.head_stance(delta, HORIZONTAL) is None


def test_stance_text_never_claims_a_posture_we_cannot_observe() -> None:
    """HMD로는 앉음과 쪼그림과 무릎 굽힘을 구별할 수 없다. 그러니 '앉아'라고 쓰지 않는다."""
    for world, sigmas in (
        ((0.0, 0.0, -0.12), {"vertical": 3.1}),
        ((0.0, 0.0, 0.12), {"vertical": 3.1}),
        ((0.18, 0.0, 0.0), {"wall": 3.4}),
        ((-0.18, 0.0, 0.0), {"wall": 3.4}),
    ):
        stance = measured_score.head_stance(_head(world, sigmas), HORIZONTAL)
        assert stance is not None
        assert "앉아" not in measured_score.stance_text(stance)
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_measured_score_multiaxis.py -q`
Expected: FAIL — `AttributeError: module 'measured_score' has no attribute 'head_stance'`

- [ ] **Step 3: 구현한다**

```python
# 자세 문구. 머리 위치를 cm로 지시하지 않는 이유는 그것이 자세의 **결과**이지
# 직접 조작하는 대상이 아니기 때문이다 — "머리를 3cm 내리세요"는 따를 수 없는 지시다.
# 문구는 정성적이고 정확한 cm는 `AxisCorrection.magnitude_cm`이 들고 있다.
STANCE_PHRASES: Final[dict[tuple[str, bool], str]] = {
    ("vertical", False): "자세를 한 단계 낮춰 작업하십시오.",
    ("vertical", True): "자세를 한 단계 높여 작업하십시오.",
    ("wall", True): "벽에서 한 걸음 더 떨어져 서십시오.",
    ("wall", False): "벽에 한 걸음 더 붙어 서십시오.",
}
# 머리에서 자세로 번역할 수 있는 축. lateral은 없다 — 작업자 요가 실측이 아니라
# 좌우를 판정할 기준이 없다(`service.ASSUMED_YAW`).
STANCE_AXES: Final[tuple[str, ...]] = ("wall", "vertical")


def head_stance(
    delta: PointDelta, frame: AxisFrame = WORLD_FRAME
) -> AxisCorrection | None:
    """머리의 이탈을 자세 지시 하나로 번역한다. 없으면 None.

    점 게이트를 따로 두지 않는다. 결합 sigma는 축별 sigma의 제곱합의 제곱근이므로
    어느 축이 임계를 넘으면 결합 sigma도 반드시 넘는다 — 더해도 판정이 안 바뀐다.
    """
    axis_sigmas = delta.axis_sigmas
    if not axis_sigmas:
        return None
    best: tuple[str, float, str, Vec3] | None = None
    best_sigma = CORRECTION_SIGMA_THRESHOLD
    for axis, magnitude_cm, direction, unit in _correction_candidates(
        delta.local_delta, frame
    ):
        if axis not in STANCE_AXES:
            continue
        sigma = float(axis_sigmas.get(axis, 0.0))
        if sigma > best_sigma:
            best, best_sigma = (axis, magnitude_cm, direction, unit), sigma
    if best is None:
        return None
    axis, magnitude_cm, direction, unit = best
    return AxisCorrection(
        point="head",
        axis=axis,
        magnitude_cm=magnitude_cm,
        direction=direction,
        sigma=best_sigma,
        unit=unit,
    )


def stance_text(correction: AxisCorrection) -> str:
    """자세 문장 하나. 구조체에서 지으므로 문장과 부호가 어긋날 수 없다."""
    return STANCE_PHRASES[(correction.axis, correction.magnitude_cm > 0)]
```

- [ ] **Step 4: 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_measured_score_multiaxis.py -q`
Expected: `13 passed`

- [ ] **Step 5: 전체 스위트와 커밋**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: `625 passed` (619 + 6)

```bash
git add trainer/scripts/measured_score.py trainer/scripts/tests/test_measured_score_multiaxis.py
git commit -m "feat(pose): 머리 이탈을 자세 문장 하나로 번역한다

cm로 지시하지 않는다 — 머리 높이는 자세의 결과이지 조작 대상이 아니다.
HMD로 구별할 수 없는 자세(앉음/쪼그림)를 주장하지 않는다. lateral은 요를
실측하지 않아 좌우를 판정할 수 없으므로 지시하지 않는다."
```

---

### Task 3: 불변식 — 발부한 것만 옮긴다

**Files:**
- Modify: `trainer/scripts/measured_score.py`
- Modify: `trainer/scripts/tests/test_measured_score.py:296-315`, `:371-382`
- Test: `trainer/scripts/tests/test_measured_score_multiaxis.py`

**Interfaces:**
- Consumes: Task 1의 `right_axes`, Task 2의 `head_stance`·`STANCE_AXES`
- Produces:
  - `def issued_axes(deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME) -> list[AxisCorrection]` — **sigma 내림차순**
  - `def issued_shifts(deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME) -> dict[str, Vec3]`
  - `apply_measured(query, deltas, frame=WORLD_FRAME, fraction=1.0)` — 시그니처 불변, 내부만 교체
  - `issued_shift`는 **삭제한다**(호출자는 `measured_score.py` 내부뿐이다)

**주의 — 기존 테스트 2건이 반드시 깨진다.** 머리가 교정 대상이 되기 때문이다. 아래 Step 5에서 정확한 대체 코드를 준다. **단언을 약화시키지 말고 좁혀라** — "머리도 안 움직인다"에서 "왼손은 절대 안 움직이고, 머리는 발부된 축으로만 움직인다"로.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`test_measured_score_multiaxis.py`에 이어 붙인다:

```python
def test_issued_axes_are_sorted_by_sigma_descending() -> None:
    """읽는 사람이 위에서부터 심각한 순으로 본다."""
    right = _right((0.08, 0.05, -0.06), {"wall": 2.2, "vertical": 3.9, "lateral": 2.6})
    head = _head((0.18, 0.0, 0.0), {"wall": 3.4, "vertical": 0.2, "lateral": 0.1})
    issued = measured_score.issued_axes([right, head], HORIZONTAL)
    sigmas = [c.sigma for c in issued]
    assert sigmas == sorted(sigmas, reverse=True)


def test_left_hand_is_never_issued() -> None:
    """왼손은 채점에만 남는다. 작업 결과와의 인과가 약하다."""
    left = measured_score.PointDelta(
        point="left",
        sigma=5.0,
        local_delta=_local(HORIZONTAL, (0.30, 0.30, 0.30)),
        samples=30,
        axis_sigmas={"wall": 5.0, "vertical": 5.0, "lateral": 5.0},
    )
    assert measured_score.issued_axes([left], HORIZONTAL) == []
    assert measured_score.issued_shifts([left], HORIZONTAL) == {}


def test_shifts_contain_only_points_that_were_issued() -> None:
    """지시하지 않은 점은 키가 없다. 없으면 apply_measured가 옮길 방법이 없다."""
    right = _right((0.08, 0.0, 0.0), {"wall": 3.5, "vertical": 0.1, "lateral": 0.1})
    quiet_head = _head((0.0, 0.0, -0.01), {"wall": 0.1, "vertical": 0.1, "lateral": 0.1})
    shifts = measured_score.issued_shifts([right, quiet_head], HORIZONTAL)
    assert set(shifts) == {"right"}


def test_apply_measured_moves_exactly_the_issued_axes_and_nothing_else() -> None:
    """불변식. 발부한 축의 합만큼만 옮긴다."""
    right = _right((0.08, 0.0, -0.06), {"wall": 3.5, "vertical": 2.6, "lateral": 0.1})
    query = {
        "right": [(0.0, 0.0, 0.0)] * 3,
        "head": [(0.0, 0.0, 0.0)] * 3,
        "left": [(0.0, 0.0, 0.0)] * 3,
    }
    fixed = measured_score.apply_measured(query, [right], HORIZONTAL)
    assert fixed["head"] == query["head"]
    assert fixed["left"] == query["left"]
    expected = measured_score.issued_shifts([right], HORIZONTAL)["right"]
    moved = tuple(fixed["right"][0][i] - query["right"][0][i] for i in range(3))
    for got, want in zip(moved, expected):
        assert abs(got - want) < 1e-9


def test_head_moves_only_when_a_stance_sentence_was_issued() -> None:
    """자세 문장이 나가면 머리도 옮긴다 — 말한 만큼만."""
    head = _head((0.18, 0.0, 0.0), {"wall": 3.4, "vertical": 0.2, "lateral": 0.1})
    query = {name: [(0.0, 0.0, 0.0)] * 3 for name in ("right", "head", "left")}
    fixed = measured_score.apply_measured(query, [head], HORIZONTAL)
    assert fixed["head"] != query["head"]
    assert fixed["left"] == query["left"]
    assert fixed["right"] == query["right"]
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_measured_score_multiaxis.py -q`
Expected: FAIL — `AttributeError: module 'measured_score' has no attribute 'issued_axes'`

- [ ] **Step 3: 구현한다**

`issued_shift`를 지우고 그 자리에 넣는다:

```python
def issued_axes(
    deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME
) -> list[AxisCorrection]:
    """실제로 발부하는 축 전부. sigma 내림차순.

    오른손은 다축, 머리는 자세 문장 하나, 왼손은 없다. 이 목록이 작업자에게 가는
    내용과 정확히 같아야 개선율이 정직하다 — `apply_measured`가 이것만 옮긴다.
    """
    issued: list[AxisCorrection] = []
    for delta in deltas:
        if delta.point == "right":
            issued.extend(right_axes(delta, frame))
        elif delta.point == "head":
            stance = head_stance(delta, frame)
            if stance is not None:
                issued.append(stance)
    return sorted(issued, key=lambda c: c.sigma, reverse=True)


def issued_shifts(
    deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME
) -> dict[str, Vec3]:
    """점 -> 스트로크 프레임 이동벡터. **발부한 점만 키를 갖는다.**

    딕셔너리를 내는 것이 요점이다. 키가 없는 점은 `apply_measured`가 옮길 방법
    자체가 없으므로, 지시하지 않은 이동이 개선율에 섞이는 경로가 타입에서 막힌다.
    한 점의 여러 축은 월드에서 더한 뒤 기저로 되돌린다.
    """
    world: dict[str, list[float]] = {}
    for correction in issued_axes(deltas, frame):
        metres = correction.magnitude_cm / 100.0
        acc = world.setdefault(correction.point, [0.0, 0.0, 0.0])
        for i in range(3):
            acc[i] += correction.unit[i] * metres
    # 기저가 정규직교라 각 축과의 내적이 곧 그 축의 성분이다.
    return {
        point: (
            _dot((v[0], v[1], v[2]), frame.normal),
            _dot((v[0], v[1], v[2]), frame.tangent),
            _dot((v[0], v[1], v[2]), frame.up),
        )
        for point, v in world.items()
    }
```

`apply_measured`의 본문을 교체한다(시그니처는 그대로):

```python
    shifts = issued_shifts(deltas, frame)
    output: dict[str, list[Vec3 | None]] = {}
    for name, points in query.items():
        shift = shifts.get(name)
        if shift is None:
            output[name] = list(points)
            continue
        output[name] = [
            None
            if point is None
            else (
                point[0] + shift[0] * fraction,
                point[1] + shift[1] * fraction,
                point[2] + shift[2] * fraction,
            )
            for point in points
        ]
    return output
```

독스트링의 "오른손 하나를, 그 성분 하나만 옮긴다"를 다음으로 바꾼다:

```
    `issued_axes`가 낸 축만, 그만큼만 옮긴다. 왼손은 어떤 경우에도 제자리다 —
    지시하지 않은 이동을 개선율에 계상하면 작업자에게 말한 것과 보고하는 숫자가
    달라진다. 이 함수가 `issued_shifts`의 딕셔너리를 쓰는 이유가 그것이다:
    키가 없는 점은 옮길 방법 자체가 없다.
```

- [ ] **Step 4: 신규 테스트 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_measured_score_multiaxis.py -q`
Expected: `18 passed`

- [ ] **Step 5: 깨진 기존 테스트 2건을 좁힌다**

`trainer/scripts/tests/test_measured_score.py`에서 `test_apply_measured_leaves_head_and_left_where_they_are`(296행부터)를 통째로 아래로 교체한다:

```python
def test_apply_measured_never_moves_the_left_hand() -> None:
    """왼손은 어떤 경우에도 제자리다. 지시하지 않는 점이기 때문이다.

    fa9c4b0이 문장을 좁혔는데 시뮬레이션은 임계를 넘은 세 점을 모두 옮기고 있었다.
    그러면 작업자에게 말한 것과 개선율이 재는 것이 달라진다. 다축 교정 이후에도
    왼손만은 계속 지시 대상이 아니므로 이 단언은 그대로 성립해야 한다.
    """
    query, reference = _three_point_case()
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    sigmas = {delta.point: delta.sigma for delta in before.deltas}
    assert min(sigmas.values()) > measured_score.CORRECTION_SIGMA_THRESHOLD

    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    assert fixed["left"] == query["left"]
    assert fixed["right"] != query["right"]


def test_apply_measured_moves_the_head_only_along_an_issued_stance_axis() -> None:
    """머리는 자세 문장이 나갈 때만, 그 축으로만 움직인다."""
    query, reference = _three_point_case()
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    head = next(d for d in before.deltas if d.point == "head")
    stance = measured_score.head_stance(head, HORIZONTAL)
    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    if stance is None:
        assert fixed["head"] == query["head"]
    else:
        assert stance.axis in measured_score.STANCE_AXES
        expected = measured_score.issued_shifts(before.deltas, HORIZONTAL)["head"]
        moved = tuple(fixed["head"][0][i] - query["head"][0][i] for i in range(3))
        for got, want in zip(moved, expected):
            assert abs(got - want) < 1e-9
```

`test_apply_measured_does_nothing_when_no_sentence_is_issued`(371행부터)는 질의의 머리를 옮겨 놓기 때문에 이제 자세 문장이 나갈 수 있다. 이름과 본문을 아래로 교체한다:

```python
def test_apply_measured_does_nothing_when_nothing_is_issued() -> None:
    """아무 지시도 없으면 아무것도 옮기지 않는다. 없는 지시를 개선율에 계상하지 않는다."""
    reference = {name: [(0.30, 0.00, 0.10)] * 5 for name in ("right", "head", "left")}
    query = {name: list(points) for name, points in reference.items()}
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    assert measured_score.issued_axes(before.deltas, HORIZONTAL) == []
    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    assert fixed == {name: list(points) for name, points in query.items()}
```

- [ ] **Step 6: 불변식이 진짜인지 실험으로 확인한다**

`apply_measured`에서 `shift = shifts.get(name)`을 임시로 `shift = shifts.get(name) or shifts.get("right")`로 바꾼다(지시하지 않은 점까지 옮기는 버그).
Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_measured_score_multiaxis.py trainer/scripts/tests/test_measured_score.py -q`
Expected: `test_apply_measured_never_moves_the_left_hand`와 `test_shifts_contain_only_points_that_were_issued` 중 최소 하나가 FAIL.
확인 후 되돌리고 통과를 재확인한다. 관측한 실패 출력을 리포트에 적는다.

- [ ] **Step 7: 전체 스위트와 커밋**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: `631 passed` (625 + 5 신규 + 1 분할 증가)

실측이 다르면 그 수를 쓰고 차이를 리포트에 설명한다.

```bash
git add trainer/scripts/measured_score.py trainer/scripts/tests/test_measured_score.py trainer/scripts/tests/test_measured_score_multiaxis.py
git commit -m "feat(pose): issued_shifts가 dict를 내어 지시하지 않은 점을 타입에서 막는다

apply_measured가 딕셔너리에 키가 없는 점을 옮길 방법 자체가 없다. 개선율은
작업자에게 실제로 발부한 것만 계상한다. 왼손은 어떤 경우에도 제자리다."
```

---

### Task 4: 서비스 응답에 `stance`와 `axes[]`

**Files:**
- Modify: `trainer/pose_lab/service.py:520-535`
- Test: `trainer/pose_lab/tests/test_service_multiaxis.py` (신규)

**Interfaces:**
- Consumes: Task 3의 `issued_axes`, Task 2의 `head_stance`·`stance_text`
- Produces:
  - `def axes_block(deltas, frame) -> list[dict[str, object]]`
  - `def stance_block(deltas, frame) -> dict[str, object] | None`
  - 응답 `measured` 블록에 `stance`와 `axes`

**배경:** 기존 `finding`·`corrections` 필드는 그대로 둔다(하위 호환). `compare()`에는 정상 경로 외에 오류/키불일치 경로가 있고 그쪽에도 `"corrections": []`, `"finding": None`이 있다 — 거기에도 새 키를 넣어야 화면이 키 없는 응답을 만나지 않는다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""다축 교정이 응답까지 배선됐는지. 선택 규칙 자체는 test_measured_score_multiaxis가 본다."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_score = importlib.import_module("measured_score")

from trainer.pose_lab import service

FRAME = measured_score.WORLD_FRAME


def _deltas() -> list[object]:
    return [
        measured_score.PointDelta(
            point="right",
            sigma=4.0,
            local_delta=(0.08, 0.0, -0.06),
            samples=30,
            axis_sigmas={"wall": 3.5, "vertical": 2.6, "lateral": 0.1},
        ),
        measured_score.PointDelta(
            point="head",
            sigma=3.4,
            local_delta=(0.18, 0.0, 0.0),
            samples=30,
            axis_sigmas={"wall": 3.4, "vertical": 0.2, "lateral": 0.1},
        ),
    ]


def test_axes_block_lists_every_issued_axis_with_its_own_sigma() -> None:
    block = service.axes_block(_deltas(), FRAME)
    assert block, "발부된 축이 있어야 한다"
    sigmas = [entry["sigma"] for entry in block]
    assert sigmas == sorted(sigmas, reverse=True)
    for entry in block:
        assert set(entry) == {"point", "axis", "magnitude_cm", "direction", "sigma"}
        assert entry["point"] in {"right", "head"}


def test_axes_block_never_lists_the_left_hand() -> None:
    left = measured_score.PointDelta(
        point="left",
        sigma=5.0,
        local_delta=(0.30, 0.30, 0.30),
        samples=30,
        axis_sigmas={"wall": 5.0, "vertical": 5.0, "lateral": 5.0},
    )
    assert service.axes_block([left], FRAME) == []


def test_stance_block_is_none_when_the_head_is_quiet() -> None:
    quiet = [
        measured_score.PointDelta(
            point="head",
            sigma=0.5,
            local_delta=(0.01, 0.0, 0.0),
            samples=30,
            axis_sigmas={"wall": 0.2, "vertical": 0.1, "lateral": 0.1},
        )
    ]
    assert service.stance_block(quiet, FRAME) is None


def test_stance_block_carries_the_sentence_and_the_exact_centimetres() -> None:
    """문장은 정성적이고 숫자는 정량적이다. 시뮬레이션은 숫자를 쓴다."""
    block = service.stance_block(_deltas(), FRAME)
    assert block is not None
    assert set(block) == {"text", "axis", "magnitude_cm", "sigma"}
    assert isinstance(block["text"], str) and block["text"]
    assert "앉아" not in block["text"]
    assert abs(block["magnitude_cm"]) >= measured_score.MIN_CORRECTION_CM
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/pose_lab/tests/test_service_multiaxis.py -q`
Expected: FAIL — `AttributeError: module 'trainer.pose_lab.service' has no attribute 'axes_block'`

- [ ] **Step 3: 구현한다**

`service.py`에 두 함수를 더한다:

```python
def axes_block(
    deltas: Sequence[measured_score.PointDelta], frame: measured_score.AxisFrame
) -> list[dict[str, object]]:
    """발부한 축 전부를 sigma 내림차순으로. 작업자에게 가는 숫자 패널의 원본이다.

    이 목록이 곧 `apply_measured`가 옮기는 것이므로, 화면에 보이는 숫자와 개선율이
    재는 이동이 같다.
    """
    return [
        {
            "point": correction.point,
            "axis": correction.axis,
            "magnitude_cm": correction.magnitude_cm,
            "direction": correction.direction,
            "sigma": correction.sigma,
        }
        for correction in measured_score.issued_axes(deltas, frame)
    ]


def stance_block(
    deltas: Sequence[measured_score.PointDelta], frame: measured_score.AxisFrame
) -> dict[str, object] | None:
    """자세 문장 하나. 머리가 조용하면 None이다(0이 아니다)."""
    head = next((delta for delta in deltas if delta.point == "head"), None)
    if head is None:
        return None
    stance = measured_score.head_stance(head, frame)
    if stance is None:
        return None
    return {
        "text": measured_score.stance_text(stance),
        "axis": stance.axis,
        "magnitude_cm": stance.magnitude_cm,
        "sigma": stance.sigma,
    }
```

`compare()`의 `measured` 딕셔너리에서 `"corrections"` 줄 바로 아래에 더한다:

```python
            "stance": stance_block(before.deltas, basis),
            "axes": axes_block(before.deltas, basis),
```

`"corrections": []`와 `"finding": None`이 있는 오류/키불일치 경로에도 더한다:

```python
            "stance": None,
            "axes": [],
```

- [ ] **Step 4: 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/pose_lab/tests/test_service_multiaxis.py -q`
Expected: `4 passed`

- [ ] **Step 5: 전체 스위트와 커밋**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: `635 passed` (631 + 4)

```bash
git add trainer/pose_lab/service.py trainer/pose_lab/tests/test_service_multiaxis.py
git commit -m "feat(pose): 응답에 자세 문장과 발부한 축 목록을 싣는다

axes[]가 곧 apply_measured가 옮기는 것이라 화면의 숫자와 개선율이 재는 이동이
같다. finding과 corrections는 하위 호환을 위해 그대로 둔다."
```

---

### Task 5: 화면 — 자세 배너, 축 표, split 출처

**Files:**
- Modify: `trainer/pose_lab/static/index.html`, `app.js`, `style.css`

**Interfaces:**
- Consumes: Task 4의 `measured.stance`(객체 또는 `null`), `measured.axes[]`. 기존 `measured.split_reason`, `measured.split_n_a`, `measured.split_n_b`

**배경:** 파이썬을 바꾸지 않는다. 필요한 필드가 응답에 없으면 고치지 말고 리포트에 적는다. 브라우저를 열 수 없으므로 `curl`로 필드 단위 대조하고, **육안 확인을 못 했다고 리포트에 그대로 적는다.** `app.js`에 이미 `num()`(값이 숫자일 때만 포맷하고 아니면 `—`)과 `AXIS_LABELS`가 있다. 없으면 같은 규칙으로 만든다.

- [ ] **Step 1: 자세 배너와 축 표를 붙인다**

`index.html`의 실측 축 패널 안, 기존 `.corrections` 목록 위에 넣는다:

```html
<p class="stance" hidden></p>
<table class="issued-axes">
  <thead><tr><th>점</th><th>축</th><th>편차</th><th>σ</th></tr></thead>
  <tbody></tbody>
</table>
```

`app.js`에서 렌더한다. **`||`를 쓰지 마라** — 자바스크립트에서 `0`도 falsy라 진짜 0이 `—`로 그려지는 역방향 거짓말이 된다:

```js
const POINT_LABELS = { right: "오른손", head: "머리", left: "왼손" };

const stanceEl = root.querySelector(".stance");
if (stanceEl) {
  const stance = payload.stance ?? null;
  stanceEl.hidden = stance === null;
  stanceEl.textContent = stance ? stance.text : "";
}

const axesBody = root.querySelector(".issued-axes tbody");
if (axesBody) {
  axesBody.replaceChildren();
  const axes = payload.axes ?? [];
  if (axes.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.textContent = "발부된 교정 없음";
    row.append(cell);
    axesBody.append(row);
  }
  for (const entry of axes) {
    const row = document.createElement("tr");
    for (const text of [
      POINT_LABELS[entry.point] ?? entry.point,
      AXIS_LABELS[entry.axis] ?? entry.axis,
      typeof entry.magnitude_cm === "number"
        ? `${entry.magnitude_cm >= 0 ? "+" : ""}${entry.magnitude_cm.toFixed(1)}cm`
        : "—",
      typeof entry.sigma === "number" ? `${entry.sigma.toFixed(2)}σ` : "—",
    ]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    axesBody.append(row);
  }
}
```

`style.css`에 최소한만 더한다:

```css
.stance { font-weight: 600; margin: 0 0 .5rem; }
.issued-axes { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.issued-axes th, .issued-axes td { text-align: right; padding: .15rem .4rem; }
.issued-axes th:first-child, .issued-axes td:first-child { text-align: left; }
```

- [ ] **Step 2: split 출처를 그린다 (파킹 항목)**

지금은 `개선율(분리) —`가 사유 없이 뜨고, 그 위의 `레퍼런스 날짜`(전체 레퍼런스 기준)가 split의 출처처럼 읽힌다. `index.html`의 개선율(분리) 행 옆에 `<span class="split-note"></span>`를 두고 `app.js`에서 채운다:

```js
const SPLIT_REASONS = {
  insufficient_dates: "날짜가 둘로 갈리지 않음",
  insufficient_samples: "반쪽 표본이 2개 미만",
  no_reference_distribution: "실측 레퍼런스 없음",
};

const splitNote = root.querySelector(".split-note");
if (splitNote) {
  const reason = payload.split_reason ?? null;
  const a = payload.split_n_a;
  const b = payload.split_n_b;
  splitNote.textContent =
    reason !== null
      ? `산출 안 됨: ${SPLIT_REASONS[reason] ?? reason}`
      : `A n=${typeof a === "number" ? a : "—"} / B n=${typeof b === "number" ? b : "—"}`;
}
```

- [ ] **Step 3: 컨테이너에서 띄워 필드 대조**

```bash
docker exec -d -w /workspace \
  -e POSE_LAB_DATA_ROOT=/data/sources/professionals \
  -e POSE_LAB_SPREAD_REFERENCE=/output/spread_reference_expert.json \
  ship-paint-trainer python3 -m uvicorn trainer.pose_lab.app:create_app --factory --host 0.0.0.0 --port 8131
```

`curl`로 페이지와 `/api/compare` 응답을 받아, **JS가 읽는 키가 실제로 오는지 필드 단위로 대조한다.** `stance`가 `null`인 경우와 객체인 경우를 모두 확인한다. 육안 확인은 못 했다고 리포트에 적는다.

- [ ] **Step 4: 전체 스위트와 커밋**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: `635 passed` (파이썬 무변경)

```bash
git add trainer/pose_lab/static
git commit -m "feat(pose): 자세 문장과 발부한 축을 화면에 그리고 split 출처를 밝힌다

없는 값은 0이 아니라 대시다. 자바스크립트에서 0도 falsy이므로 ||가 아니라
typeof 판정을 쓴다 — 진짜 0을 대시로 그리는 역방향 거짓말도 막는다.
개선율(분리)이 산출되지 않은 이유와 반쪽 표본수를 옆에 적는다."
```

---

### Task 6: 개선율 재측정 — 70% 판정 갱신

**Files:**
- Modify: `trainer/scripts/improvement_study.py`
- Test: `trainer/scripts/tests/test_improvement_study.py`

**Interfaces:**
- Consumes: Task 2의 `measured_score.STANCE_AXES`, Task 3의 `issued_axes`
- Produces: `CORRECTABLE`, `def ceiling_residual_sigma(axis_sigmas, weights) -> dict[str, float]`

**배경 — 이 태스크의 산출물은 기능이 아니라 숫자다.**

**비교 기준(이미 고정됨, 바꾸지 않는다):** 200쌍, 시드 `20260828`, 같은 선정 규칙.
**직전 값:** `split` 중앙 0.0, ≥70% **0/149**. `same` 중앙 0.0, ≥70% 0/157. 발동률 39.5%. 게이트 149 fail / 0 pass. 옛 천장(발동분) 중앙 0.706, 51.6%가 ≥0.70.

**새 값이 무엇이 나오든 그대로 적는다. 결과를 본 뒤 유리한 절단·부분집합을 고르지 않는다.**

- [ ] **Step 1: 천장 정의를 고치는 테스트를 쓴다**

현재 `_ceiling_gap_closure`는 "오른손 σ만 완전 소거, 머리·왼손 불변"이다. 머리가 교정 대상이 되었으므로 만질 수 없는 몫은 **왼손 전체와 머리의 `lateral`**이다.

`trainer/scripts/tests/test_improvement_study.py`에 더한다(파일 상단에 `import math`와 `import pytest`가 이미 있다):

```python
def test_ceiling_leaves_the_left_hand_and_the_head_lateral_untouched() -> None:
    """천장은 '지시할 수 있는 것을 전부 완벽히 따랐을 때'다.

    왼손은 지시 대상이 아니고 머리 lateral도 아니다. 그 둘을 0으로 치면 천장이
    실제보다 높게 나오고, 미달 폭이 실제보다 작아 보인다.
    """
    sigmas = {
        "right": {"wall": 3.0, "vertical": 3.0, "lateral": 3.0},
        "head": {"wall": 2.5, "vertical": 2.5, "lateral": 2.5},
        "left": {"wall": 4.0, "vertical": 4.0, "lateral": 4.0},
    }
    weights = {"right": 1.0, "head": 0.4, "left": 0.15}
    residual = improvement_study.ceiling_residual_sigma(sigmas, weights)
    assert residual["left"] == pytest.approx(4.0 * math.sqrt(3))
    assert residual["head"] == pytest.approx(2.5)
    assert residual["right"] == pytest.approx(0.0)


def test_correctable_head_axes_match_the_scorer() -> None:
    """천장이 쓰는 '지시 가능한 축'과 실제 지시 규칙이 갈라지면 천장이 틀린다."""
    assert improvement_study.CORRECTABLE["head"] == measured_score.STANCE_AXES
```

두 번째 테스트를 위해 파일 상단에 `measured_score = importlib.import_module("measured_score")`를 더한다(이미 있으면 그대로 둔다).

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_improvement_study.py -q`
Expected: FAIL — `AttributeError: module 'improvement_study' has no attribute 'ceiling_residual_sigma'`

- [ ] **Step 3: 구현한다**

```python
# 지시할 수 있는 축. 이것 밖은 완벽한 교정으로도 남는다.
# head는 measured_score.STANCE_AXES와 같아야 한다 — 갈라지면 천장이 틀린다.
CORRECTABLE: Final[dict[str, tuple[str, ...]]] = {
    "right": measured_score.AXIS_ORDER,
    "head": measured_score.STANCE_AXES,
    "left": (),  # 지시 대상이 아니다
}


def ceiling_residual_sigma(
    axis_sigmas: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> dict[str, float]:
    """지시 가능한 축을 전부 완벽히 따랐을 때 각 점에 남는 sigma.

    `CORRECTABLE`에 없는 축은 그대로 남는다. 남는 것을 0으로 치면 천장이 실제보다
    높게 나오고 미달 폭이 작아 보인다.
    """
    residual: dict[str, float] = {}
    for point, axes in axis_sigmas.items():
        correctable = CORRECTABLE.get(point, ())
        total = sum(
            float(value) ** 2 for axis, value in axes.items() if axis not in correctable
        )
        residual[point] = math.sqrt(total)
    return residual
```

`_ceiling_gap_closure`가 이 함수를 쓰도록 고치고, 독스트링에서 "오른손 σ만 소거"를 갱신한다. `1.0/1.55 = 64.5%` 류의 문장이 남아 있으면 지운다 — **그 산술은 이미 반박됐다**(가중 σ의 몫은 gap closure의 몫이 아니다).

- [ ] **Step 4: 통과 확인 후 실데이터로 재측정**

```bash
docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/improvement_study.py \
  --root /data/sources/professionals \
  --reference /output/spread_reference_expert.json \
  --pairs 200 --seed 20260828 \
  --index-cache /output/improvement_study_index_cache.json \
  --out /output/improvement_study_multiaxis.json --progress --log-level INFO
```

**포그라운드에서 붙들고 기다리거나, `docker exec -d`로 로그에 쓰고 그 로그를 직접 폴링한다.** 외부 알림에 작업을 맡기고 대기 상태로 들어가지 마라. 인덱스 캐시가 있어 4분짜리 재파싱은 건너뛴다.

- [ ] **Step 5: 리포트에 전후 비교를 낸다**

**표집 규칙을 숫자보다 먼저 적는다.**

| 지표 | 이전(단축) | 이후(다축) |
|---|---|---|
| `split` 중앙 / q1 / q3 / ≥0.70 | 0.0 / 0.0 / 0.066 / **0/149** | ? |
| `same` 중앙 / ≥0.70 | 0.0 / 0/157 | ? |
| 교정 발동률 | 39.5% (62/157) | ? |
| 천장(발동분) 중앙 / ≥0.70 가능 | 0.706 / 51.6% | ? |
| REQ-19 게이트 | 149 fail / 0 pass | ? |

추가로 낸다: 입도 층별 분해, `split_reason` 분포, `split_n_a`/`split_n_b` 두께 층화, 그리고 **머리만 발동 / 오른손만 발동 / 둘 다 발동**으로 나눈 분포(스펙 §10의 열린 위험 — 두 지시가 같은 자세 문제의 두 증상이면 중복될 수 있다).

**70% 목표 달성 여부를 한 문장으로 명확히 적는다. 완곡하게 쓰지 않는다.**

- [ ] **Step 6: 전체 스위트와 커밋**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: `637 passed` (635 + 2)

```bash
git add trainer/scripts/improvement_study.py trainer/scripts/tests/test_improvement_study.py
git commit -m "feat(pose): 다축 교정 이후의 개선율을 같은 표집으로 재측정한다

천장 정의를 고쳤다 — 지시할 수 있는 축만 소거하므로 왼손 전체와 머리 lateral이
남는다. 남는 것을 0으로 치면 천장이 높게 나오고 미달 폭이 작아 보인다.
표집·시드는 이전과 동일하며 결과를 보기 전에 고정돼 있었다."
```

---

## 이번 계획에서 하지 않는 것

| 항목 | 이유 |
|---|---|
| 왼손 지시 | 작업 결과와의 인과가 약함. 채점 가중치 0.15는 유지 |
| 허리 | 데이터에 없음(스펙 §3). 확보되면 **각도**로 재검토 |
| 각도 교정 | REQ-19 게이트 149/149 fail |
| 스켈레톤 뷰 | 별건 |
| 임계 조정 | 사전등록 상수 불변 |

## Self-Review 결과

**1. 스펙 커버리지**

| 스펙 절 | 태스크 |
|---|---|
| §4 D2 축 포함 규칙, 단조 보장 | Task 1 |
| §4 D3 머리 → 자세 번역 | Task 2 |
| §4 D1 불변식, `issued_shifts` | Task 3 |
| §7 응답 변경 | Task 4 |
| §8 화면 + 파킹 항목 | Task 5 |
| §6 재측정, 천장 갱신 | Task 6 |
| §9 테스트 규율 | 전 태스크(Task 1 Step 5, Task 3 Step 6에 파괴 실험) |
| §5 상수 불변 | Global Constraints |
| §3 허리 조사 기록 | 스펙에만 — 구현 없음이 옳다 |

**2. Placeholder 스캔:** 통과. 모든 코드 단계에 실제 코드가 있다. Task 5의 `curl` 대조는 명령과 확인 대상을 명시했다.

**3. 타입 일관성**
- `AxisCorrection`(T1) → `right_axes`(T1) → `head_stance`(T2) → `issued_axes`(T3) → `axes_block`(T4)가 같은 필드 이름을 쓴다.
- `issued_shifts`가 내는 `dict[str, Vec3]`를 `apply_measured`(T3)만 소비한다.
- `STANCE_AXES`(T2)를 T3의 대체 테스트와 T6의 `CORRECTABLE`이 참조한다. 갈라지면 천장이 틀리므로 T6 Step 1에 `CORRECTABLE["head"] == measured_score.STANCE_AXES`를 고정하는 테스트를 넣었다.
- 테스트 누적: 612 → 619(T1) → 625(T2) → 631(T3) → 635(T4) → 635(T5, 파이썬 무변경) → 637(T6).
