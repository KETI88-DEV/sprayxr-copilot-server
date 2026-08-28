# 숙련자 실측 산포 레퍼런스 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 임시 상수 0.05 m를 전형기 실측 산포로 대체해 σ·REQ-19 게이트·개선율을 의미 있게 만들고, 개선율 70% 목표에 대한 답을 숫자로 낸다.

**Architecture:** 전형기 학습모드 패스에서 버킷별·점별·축별 평균과 산포를 뽑아 JSON 아티팩트로 굽는다. 날짜별 부분합을 함께 실어 질의 날짜 제외와 A/B 분리를 전수 재빌드 없이 한다. 입도 네 층(uv8/uv4/uv2/uv_off)을 한 아티팩트에 담고, 질의 시 `n ≥ 20`을 처음 만족하는 층을 쓴다.

**Tech Stack:** Python 3.11 (컨테이너 `ship-paint-trainer`), 표준 라이브러리만 — numpy·scipy 금지. 화면은 기존 plain HTML/CSS/ES module.

**Spec:** [`docs/superpowers/specs/2026-08-28-expert-spread-reference-design.md`](../specs/2026-08-28-expert-spread-reference-design.md)

## Global Constraints

- **모든 Python은 컨테이너에서, 저장소 루트에서** 실행한다: `docker exec -w /workspace ship-paint-trainer python3 -m pytest ...`. heredoc은 `docker exec -i`.
- **새 Python·JavaScript 의존성 금지.** numpy·scipy·npm·CDN·번들러 전부 안 된다.
- 테스트 import 규약: `trainer/scripts/tests/`는 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 후 `importlib.import_module("<name>")`. `trainer/pose_lab/tests/`는 `from trainer.pose_lab import ...`.
- `trainer/scripts/pass_segmentation.py`와 `simulator_csv.py`를 수정하지 않는다. `pose_similarity.py`에서 커밋 `790c0d8` 이전에 있던 함수를 바꾸지 않는다.
- **교정문에 각도를 넣지 않는다.** 문자 `도`가 어떤 교정 문장에도 나오면 안 된다. 각도 기준은 2026-08-28 측정으로 기각됐다(스펙 §2.3).
- `joint` 축은 리터럴만 담는다(`status: "not_wired"`, `score: null`, `improvement: null`, `corrections: []`). 실측 축 값을 빌려오지 않는다.
- **없는 값을 0으로 표시하지 않는다.** 화면은 `—`를 낸다. 값이 없으면 `null`을 그대로 흘린다.
- 산포 축 이름은 **`wall` / `vertical` / `lateral`** 셋이다. `_correction_candidates`가 내는 축과 같아야 σ로 축을 고를 수 있다.
- 데이터 루트 `/data/sources/professionals`. 레벨 `1`~`10`만. 파일명이 `1-A`/`1-B`/`1-C`로 시작하면 실전모드라 범위 밖이다.
- Conventional Commits, 한국어 요약, 명시 경로만 stage, `git add -A` 금지. `.opencode/`와 `docs/Interface.pdf`는 남의 것이니 절대 stage하지 않는다.
- **검증 기준선**: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q` 가 **548개 통과**. 줄면 안 된다.

---

## 재사용하는 기존 코드 (정확한 시그니처)

```python
# trainer/scripts/measured_score.py
DEFAULT_WEIGHTS: dict[str, float] = {"right": 1.0, "head": 0.4, "left": 0.15}
POINT_ORDER: tuple[str, ...] = ("right", "head", "left")
SIGMA_FLOOR_M = 0.02 ; CORRECTION_SIGMA_THRESHOLD = 2.0 ; MAX_SIGMA = 6.0
WORLD_FRAME: AxisFrame            # normal/tangent/up 세 벡터를 가진 Protocol
Track = list[Vec3 | None]

@dataclass(frozen=True, slots=True)
class PointDelta:
    point: str ; sigma: float ; local_delta: Vec3 ; samples: int = 0

@dataclass(frozen=True, slots=True)
class MeasuredResult:
    score: float ; deltas: list[PointDelta] ; samples: int

@dataclass(frozen=True, slots=True)
class RightHandFinding:
    axis: str          # "wall" | "vertical" | "lateral"
    magnitude_cm: float ; direction: str ; sigma: float ; samples: int

def score_track(query, reference, spreads: Mapping[str, float], weights) -> MeasuredResult
def _correction_candidates(local: Vec3, frame: AxisFrame) -> list[tuple[str, float, str, Vec3]]
    # (axis, magnitude_cm, direction, world_unit_vector)
def _largest_correction(local, frame) -> tuple[str, float, str, Vec3] | None
def right_hand_finding(deltas, frame=WORLD_FRAME) -> RightHandFinding | None
def issued_shift(deltas, frame=WORLD_FRAME) -> Vec3 | None
def corrections_text(deltas, frame=WORLD_FRAME) -> list[str]
def apply_measured(query, deltas, frame=WORLD_FRAME, fraction=1.0) -> dict[str, Track]

# trainer/pose_lab/service.py
GATE_RATIO = 0.5 ; DEFAULT_ERROR_M = 0.144
PROVISIONAL_SPREADS: dict[str, float] = {"right": 0.05, "head": 0.05, "left": 0.05}
def gate_verdict(spread: float, error: float) -> str | None
def improvement(before, after_split: float | None, after_same: float) -> dict[str, float | None]
def pass_frames(raw: RawPass, samples=20) -> list[Any]
def key_inputs(frames) -> KeyInputs | None
def key_from(inputs: KeyInputs, params) -> str
def representative_basis(frames) -> Any        # AxisFrame
def _local_tracks(frames) -> dict[str, Track]
def compare(source, query, reference, params=None, samples=20, spreads=None, weights=None) -> dict

# trainer/pose_lab/csv_source.py
@dataclass class SessionFile: worker, date, level, session, path
@dataclass class PassIndexEntry: pass_index, start, end
class CsvPassSource:
    def __init__(self, root: Path, samples: int = 20, cache_size: int = 2)
    def sessions(self) -> list[SessionFile]
    def workers(self) -> list[str]
    def levels(self, worker: str) -> list[str]
    def index_of(self, item: SessionFile) -> list[PassIndexEntry]
    def load(self, ref: PassRef) -> RawPass
    def key_of(self, entry: PassIndexEntry, params) -> str

# trainer/scripts/motion_key.py
@dataclass(frozen=True, slots=True) class KeyParams:
    uv_grid: int = 8 ; lateral_deg: float = 30.0 ; reach_m: float = 0.8 ; uv_size: float = 2048.0
```

## 파일 구조

| 파일 | 책임 |
|---|---|
| `trainer/scripts/measured_score.py` (수정) | 축별 산포로 채점, σ로 교정 축 선택 |
| `trainer/scripts/spread_stats.py` (신규) | 부분합 누적과 평균·산포 재구성. 순수 계산 |
| `trainer/scripts/build_spread_reference.py` (신규) | CLI. 전형기 패스 → 아티팩트 JSON |
| `trainer/pose_lab/spread_reference.py` (신규) | 아티팩트 읽기, 날짜 제외, 입도 폴백 |
| `trainer/pose_lab/service.py` (수정) | 실측 산포 배선, 실제 게이트, split 산출 |
| `trainer/pose_lab/static/` (수정) | 입도·표본수·축별 σ 표시 |
| `trainer/scripts/improvement_study.py` (신규) | 스펙 §10 표 산출 |

---

## Task 1: 축별 산포로 채점하고 σ로 교정 축을 고른다

지금 산포가 점당 스칼라 하나라, 교정 축을 raw 크기로 고른다. 벽 거리가 3 cm로 일정하고 진행 방향이 20 cm로 느슨한 사람에게 "진행 방향 17 cm"는 정상인데 그걸 지적한다.

**Files:**
- Modify: `trainer/scripts/measured_score.py`, `trainer/pose_lab/service.py`
- Modify: `docs/superpowers/specs/2026-08-28-expert-spread-reference-design.md` (§4 축 이름 정정)
- Test: `trainer/scripts/tests/test_measured_score_axes.py`

**Interfaces:**
- Consumes: `_correction_candidates`, `WORLD_FRAME`, `PointDelta`, `MeasuredResult`
- Produces:
  - `AXIS_ORDER: tuple[str, ...] = ("wall", "vertical", "lateral")`
  - `AxisSpreads = Mapping[str, Mapping[str, float]]` — 점 → 축 → 산포(m)
  - `def uniform_spreads(value: float) -> dict[str, dict[str, float]]`
  - `PointDelta`에 필드 추가: `axis_sigmas: dict[str, float] = field(default_factory=dict)`
  - `score_track(query, reference, spreads: AxisSpreads, weights, frame: AxisFrame = WORLD_FRAME) -> MeasuredResult`
  - `_largest_correction(local, frame, axis_sigmas: Mapping[str, float] | None = None)`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`trainer/scripts/tests/test_measured_score_axes.py`:

```python
#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_score = importlib.import_module("measured_score")

WORLD = measured_score.WORLD_FRAME


def _track(point: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    return [point] * 5


def _tracks(right, head=(0.3, 0.0, 0.1), left=(0.3, 0.0, 0.1)):
    return {"right": _track(right), "head": _track(head), "left": _track(left)}


def test_uniform_spreads_reduces_to_the_old_scalar_behaviour() -> None:
    """축별 산포가 모두 같으면 이전 스칼라 채점과 같은 sigma가 나와야 한다."""
    spreads = measured_score.uniform_spreads(0.05)
    assert set(spreads) == {"right", "head", "left"}
    assert set(spreads["right"]) == set(measured_score.AXIS_ORDER)
    result = measured_score.score_track(
        _tracks((0.4, 0.0, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    right = next(d for d in result.deltas if d.point == "right")
    assert math.isclose(right.sigma, 2.0, abs_tol=1e-6)


def test_axis_sigmas_are_reported_per_axis() -> None:
    spreads = {name: {"wall": 0.02, "vertical": 0.20, "lateral": 0.20}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.34, 0.0, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    right = next(d for d in result.deltas if d.point == "right")
    assert set(right.axis_sigmas) == set(measured_score.AXIS_ORDER)
    assert math.isclose(right.axis_sigmas["wall"], 2.0, abs_tol=1e-6)
    assert right.axis_sigmas["vertical"] < 0.5


def test_tight_axis_makes_a_small_deviation_significant() -> None:
    """벽 산포가 좁으면 작은 벽 편차가 큰 진행 방향 편차보다 이상하다."""
    spreads = {name: {"wall": 0.02, "vertical": 0.20, "lateral": 0.30}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.35, 0.17, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    finding = measured_score.right_hand_finding(result.deltas, WORLD)
    assert finding is not None
    assert finding.axis == "wall", f"raw 크기가 아니라 sigma로 골라야 한다: {finding}"


def test_loose_axis_is_not_flagged_even_when_raw_magnitude_is_large() -> None:
    """진행 방향 산포가 넓으면 17cm는 정상이다."""
    spreads = {name: {"wall": 0.30, "vertical": 0.30, "lateral": 0.30}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.3, 0.17, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    assert measured_score.right_hand_finding(result.deltas, WORLD) is None


def test_spread_floor_still_applies_per_axis() -> None:
    spreads = {name: {"wall": 0.0, "vertical": 0.0, "lateral": 0.0}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.31, 0.0, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    right = next(d for d in result.deltas if d.point == "right")
    assert math.isclose(right.axis_sigmas["wall"], 0.5, abs_tol=1e-6)


def test_sigma_is_capped() -> None:
    result = measured_score.score_track(
        _tracks((1.3, 0.0, 0.1)), _tracks((0.3, 0.0, 0.1)),
        measured_score.uniform_spreads(0.01), measured_score.DEFAULT_WEIGHTS,
    )
    right = next(d for d in result.deltas if d.point == "right")
    assert right.sigma <= measured_score.MAX_SIGMA


def test_no_correction_sentence_contains_an_angle() -> None:
    spreads = {name: {"wall": 0.02, "vertical": 0.02, "lateral": 0.02}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.4, 0.1, 0.3)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    for line in measured_score.corrections_text(result.deltas, WORLD):
        assert "도" not in line


def test_head_and_left_still_produce_no_sentence() -> None:
    spreads = {name: {"wall": 0.02, "vertical": 0.02, "lateral": 0.02}
               for name in measured_score.POINT_ORDER}
    query = _tracks((0.3, 0.0, 0.1), head=(0.9, 0.0, 0.1), left=(0.9, 0.0, 0.1))
    result = measured_score.score_track(
        query, _tracks((0.3, 0.0, 0.1)), spreads, measured_score.DEFAULT_WEIGHTS
    )
    assert measured_score.corrections_text(result.deltas, WORLD) == []
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_measured_score_axes.py -q`
Expected: FAIL — `AttributeError: module 'measured_score' has no attribute 'uniform_spreads'`

- [ ] **Step 3: 구현한다**

`trainer/scripts/measured_score.py`:

```python
AXIS_ORDER: Final[tuple[str, ...]] = ("wall", "vertical", "lateral")
AxisSpreads = Mapping[str, Mapping[str, float]]


def uniform_spreads(value: float) -> dict[str, dict[str, float]]:
    """스칼라 하나를 점·축 전부에 편다. 실측 산포가 없을 때의 잠정값이다."""
    return {point: {axis: value for axis in AXIS_ORDER} for point in POINT_ORDER}
```

`PointDelta`에 필드를 더한다 (`from dataclasses import dataclass, field`):

```python
    axis_sigmas: dict[str, float] = field(default_factory=dict)
```

`score_track`의 σ 계산을 축별 대각 거리로 바꾼다. 시그니처에 `frame: AxisFrame = WORLD_FRAME`을 더한다:

```python
        axis_spread = spreads.get(name) or {}
        axis_sigmas: dict[str, float] = {}
        total = 0.0
        for axis, magnitude_cm, _direction, _unit in _correction_candidates(mean_delta, frame):
            floor = max(float(axis_spread.get(axis, SIGMA_FLOOR_M)), SIGMA_FLOOR_M)
            value = min(MAX_SIGMA, abs(magnitude_cm) / 100.0 / floor)
            axis_sigmas[axis] = value
            total += value * value
        combined = min(MAX_SIGMA, math.sqrt(total))
```

`PointDelta(..., sigma=combined, axis_sigmas=axis_sigmas)`로 만든다.

`_largest_correction`이 σ로 고르게 한다:

```python
def _largest_correction(
    local: Vec3, frame: AxisFrame, axis_sigmas: Mapping[str, float] | None = None
) -> tuple[str, float, str, Vec3] | None:
    """가장 이상한 후보 하나. 가장 큰 것과 가장 이상한 것은 다르다.

    축별 산포가 있으면 sigma로, 없으면 예전처럼 절댓값으로 고른다.
    """
    candidates = _correction_candidates(local, frame)
    if not candidates:
        return None
    if axis_sigmas:
        return max(candidates, key=lambda c: axis_sigmas.get(c[0], 0.0))
    return max(candidates, key=lambda candidate: abs(candidate[1]))
```

`right_hand_finding`과 `issued_shift`가 `right.axis_sigmas`를 넘긴다:

```python
    candidate = _largest_correction(right.local_delta, frame, right.axis_sigmas)
```

`trainer/pose_lab/service.py`의 `PROVISIONAL_SPREADS`를 `measured_score.uniform_spreads(0.05)`로 바꾼다.

**스펙도 함께 고친다.** `docs/superpowers/specs/2026-08-28-expert-spread-reference-design.md` §4의 아티팩트 구조에서 축 이름 `"normal"|"tangent"|"up"`을 `"wall"|"vertical"|"lateral"`로 바꾸고, 왜 교정 축과 같아야 하는지 한 줄 적는다.

- [ ] **Step 4: 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_measured_score_axes.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: 기존 호출부 회귀를 확인한다**

기존 테스트 중 점당 스칼라 산포를 넘기는 것들이 있다. `uniform_spreads`로 고친다. **어떤 파일의 어떤 테스트를 왜 고쳤는지 보고서에 적는다.**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: 556 passed (548 + 8)

- [ ] **Step 6: 커밋한다**

```bash
git add trainer/scripts/measured_score.py trainer/scripts/tests/test_measured_score_axes.py trainer/pose_lab/service.py docs/superpowers/specs/2026-08-28-expert-spread-reference-design.md
git commit -m "feat(pose): 축별 산포로 채점하고 sigma로 교정 축을 고른다

점당 스칼라 하나로는 큰 것과 이상한 것을 구분하지 못한다. 벽 거리가 일정하고 진행
방향이 느슨한 사람에게 진행 방향 17cm는 정상인데 raw 크기로 고르면 그걸 지적한다.

축별 산포로 재고 sigma가 최대인 축을 고른다. 축별 산포가 모두 같으면 이전과 같은
sigma가 나오므로 하위호환이 유지된다. 축 이름을 교정 후보와 맞추려 스펙도 고쳤다."
```

---

## Task 2: 부분합 누적과 재구성

날짜를 빼거나 나눠도 평균·산포를 다시 계산할 수 있어야 한다. 전수 재빌드 없이 하려면 합과 제곱합을 날짜별로 들고 있어야 한다.

**Files:**
- Create: `trainer/scripts/spread_stats.py`
- Test: `trainer/scripts/tests/test_spread_stats.py`

**Interfaces:**
- Consumes: 없음 (순수 계산)
- Produces:
  - `@dataclass(frozen=True, slots=True) class AxisAccum` — `n: int = 0`, `total: float = 0.0`, `total_sq: float = 0.0`
  - `def accumulate(values: Iterable[float]) -> AxisAccum`
  - `def merge(parts: Iterable[AxisAccum]) -> AxisAccum`
  - `def summarize(accum: AxisAccum) -> tuple[float, float]`
  - `def to_json(accum: AxisAccum) -> dict[str, float]`
  - `def from_json(raw: Mapping[str, object]) -> AxisAccum`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`trainer/scripts/tests/test_spread_stats.py`:

```python
#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

spread_stats = importlib.import_module("spread_stats")


def test_summarize_matches_direct_statistics() -> None:
    values = [0.10, 0.12, 0.09, 0.15, 0.11]
    mean, spread = spread_stats.summarize(spread_stats.accumulate(values))
    assert math.isclose(mean, st.mean(values), abs_tol=1e-12)
    assert math.isclose(spread, st.pstdev(values), abs_tol=1e-9)


def test_merged_parts_equal_one_pass_over_everything() -> None:
    """부분합을 합친 결과가 전수 계산과 같아야 날짜 제외가 성립한다."""
    left = [0.10, 0.12, 0.09]
    right = [0.15, 0.11, 0.13, 0.14]
    merged = spread_stats.merge(
        [spread_stats.accumulate(left), spread_stats.accumulate(right)]
    )
    direct = spread_stats.accumulate(left + right)
    assert merged.n == direct.n
    assert math.isclose(merged.total, direct.total, abs_tol=1e-12)
    assert math.isclose(merged.total_sq, direct.total_sq, abs_tol=1e-12)
    for a, b in zip(spread_stats.summarize(merged), spread_stats.summarize(direct)):
        assert math.isclose(a, b, abs_tol=1e-12)


def test_excluding_a_part_equals_building_without_it() -> None:
    a = spread_stats.accumulate([0.10, 0.12])
    b = spread_stats.accumulate([0.15, 0.11, 0.13])
    c = spread_stats.accumulate([0.20, 0.21])
    assert b.n == 3  # b는 제외 대상이며 결과에 들어가면 안 된다
    without_b = spread_stats.merge([a, c])
    direct = spread_stats.accumulate([0.10, 0.12, 0.20, 0.21])
    assert without_b.n == direct.n
    for x, y in zip(spread_stats.summarize(without_b), spread_stats.summarize(direct)):
        assert math.isclose(x, y, abs_tol=1e-9)


def test_single_sample_has_no_spread() -> None:
    mean, spread = spread_stats.summarize(spread_stats.accumulate([0.42]))
    assert math.isclose(mean, 0.42, abs_tol=1e-12)
    assert spread == 0.0


def test_empty_accumulator_is_zero() -> None:
    accum = spread_stats.merge([])
    assert accum.n == 0
    assert spread_stats.summarize(accum) == (0.0, 0.0)


def test_json_round_trip() -> None:
    accum = spread_stats.accumulate([0.10, 0.12, 0.09])
    assert spread_stats.from_json(spread_stats.to_json(accum)) == accum


def test_negative_variance_from_rounding_is_clamped() -> None:
    """제곱합 방식은 부동소수 오차로 음수 분산이 나올 수 있다. 0으로 막는다."""
    accum = spread_stats.AxisAccum(n=3, total=3.0, total_sq=2.9999999999)
    _mean, spread = spread_stats.summarize(accum)
    assert spread >= 0.0
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_spread_stats.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'spread_stats'`

- [ ] **Step 3: 구현한다**

`trainer/scripts/spread_stats.py`:

```python
#!/usr/bin/env python3
"""날짜별 부분합을 들고 임의의 부분집합에 대해 평균·산포를 재구성한다.

질의가 전형기 자신의 패스면 그 날짜를 레퍼런스에서 빼야 한다. 자기가 자기 평균에
가까운 것은 당연하기 때문이다. 개선율의 분리 채점도 날짜를 두 그룹으로 갈라야 한다.

전수 재빌드 없이 하려면 합과 제곱합을 날짜별로 들고 있으면 된다. 둘 다 가산적이라
부분집합의 통계가 덧셈만으로 나온다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class AxisAccum:
    """한 축의 부분합. n=0이면 빈 누적이다."""

    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0


def accumulate(values: Iterable[float]) -> AxisAccum:
    count = 0
    total = 0.0
    total_sq = 0.0
    for value in values:
        count += 1
        total += value
        total_sq += value * value
    return AxisAccum(n=count, total=total, total_sq=total_sq)


def merge(parts: Iterable[AxisAccum]) -> AxisAccum:
    count = 0
    total = 0.0
    total_sq = 0.0
    for part in parts:
        count += part.n
        total += part.total
        total_sq += part.total_sq
    return AxisAccum(n=count, total=total, total_sq=total_sq)


def summarize(accum: AxisAccum) -> tuple[float, float]:
    """(평균, 모표준편차). 표본이 2개 미만이면 산포는 0이다."""
    if accum.n <= 0:
        return (0.0, 0.0)
    mean = accum.total / accum.n
    if accum.n < 2:
        return (mean, 0.0)
    variance = accum.total_sq / accum.n - mean * mean
    # 제곱합 방식은 부동소수 오차로 음수가 나올 수 있다.
    return (mean, math.sqrt(variance) if variance > 0.0 else 0.0)


def to_json(accum: AxisAccum) -> dict[str, float]:
    return {"n": accum.n, "total": accum.total, "total_sq": accum.total_sq}


def from_json(raw: Mapping[str, object]) -> AxisAccum:
    return AxisAccum(
        n=int(raw.get("n", 0)),  # type: ignore[arg-type]
        total=float(raw.get("total", 0.0)),  # type: ignore[arg-type]
        total_sq=float(raw.get("total_sq", 0.0)),  # type: ignore[arg-type]
    )
```

- [ ] **Step 4: 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_spread_stats.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: 커밋한다**

```bash
git add trainer/scripts/spread_stats.py trainer/scripts/tests/test_spread_stats.py
git commit -m "feat(pose): 날짜별 부분합으로 평균·산포를 재구성하는 계산 추가

합과 제곱합이 가산적이라 임의의 날짜 부분집합 통계가 전수 재빌드 없이 나온다.
질의 날짜 제외와 개선율의 A/B 분리가 이것 위에 선다."
```

---

## Task 3: 레퍼런스 아티팩트 빌더

**Files:**
- Create: `trainer/scripts/build_spread_reference.py`
- Test: `trainer/scripts/tests/test_build_spread_reference.py`

**Interfaces:**
- Consumes: `spread_stats`, `measured_score.AXIS_ORDER`/`POINT_ORDER`/`_correction_candidates`, `csv_source.CsvPassSource`, `service.pass_frames`/`key_inputs`/`key_from`/`representative_basis`/`_local_tracks`, `motion_key.KeyParams`
- Produces:
  - `GRANULARITIES: tuple[tuple[str, int], ...] = (("uv8", 8), ("uv4", 4), ("uv2", 2), ("uv_off", 0))`
  - `N_MIN: int = 20`
  - `def key_without_uv(key: str) -> str`
  - `def pass_axis_values(tracks, frame) -> dict[str, dict[str, float]]`
  - `def build_payload(worker, levels, sessions, buckets) -> dict[str, object]`
  - `def main() -> int` — CLI `--root`, `--worker`, `--out`, `--levels`, `--log-level`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`trainer/scripts/tests/test_build_spread_reference.py`:

```python
#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

builder = importlib.import_module("build_spread_reference")
spread_stats = importlib.import_module("spread_stats")
measured_score = importlib.import_module("measured_score")

KEY = "m1:horizontal:right:1x2:center-near-high"


def _buckets(by_date):
    return {"uv8": {KEY: by_date}, "uv4": {}, "uv2": {}, "uv_off": {}}


def test_granularity_ladder_is_fine_to_coarse_ordered() -> None:
    assert [name for name, _grid in builder.GRANULARITIES] == ["uv8", "uv4", "uv2", "uv_off"]


def test_n_min_is_inherited_not_reinvented() -> None:
    """이전 연구에서 사전등록한 값이다. 이번 수치를 보고 정하지 않는다."""
    assert builder.N_MIN == 20


def test_artifact_has_every_layer_and_carries_dates() -> None:
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=3,
        buckets=_buckets({"20250901": {"right": {"wall": spread_stats.accumulate([0.10, 0.12])}}}),
    )
    assert payload["worker"] == "전형기"
    assert payload["n_min"] == 20
    assert set(payload["granularities"]) == {"uv8", "uv4", "uv2", "uv_off"}
    bucket = payload["granularities"]["uv8"][KEY]
    assert bucket["dates"] == ["20250901"]
    assert bucket["n"] == 2
    assert "by_date" in bucket
    assert bucket["points"]["right"]["wall"]["spread"] >= 0.0


def test_payload_round_trips_through_json() -> None:
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=1,
        buckets=_buckets({"20250901": {"right": {"wall": spread_stats.accumulate([0.1, 0.2])}}}),
    )
    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["granularities"]["uv8"][KEY]["n"] == 2


def test_two_dates_merge_into_one_bucket() -> None:
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=2,
        buckets=_buckets({
            "20250901": {"right": {"wall": spread_stats.accumulate([0.10, 0.12])}},
            "20250904": {"right": {"wall": spread_stats.accumulate([0.14])}},
        }),
    )
    bucket = payload["granularities"]["uv8"][KEY]
    assert bucket["dates"] == ["20250901", "20250904"]
    assert bucket["n"] == 3


def test_pass_axis_values_uses_the_correction_axes() -> None:
    """산포 축은 교정 축과 같아야 sigma로 축을 고를 수 있다."""
    class _Frame:
        normal = (1.0, 0.0, 0.0)
        tangent = (0.0, 1.0, 0.0)
        up = (0.0, 0.0, 1.0)

    tracks = {"right": [(0.3, 0.1, 0.2)] * 4, "head": [(0.5, 0.0, 0.0)] * 4, "left": []}
    values = builder.pass_axis_values(tracks, _Frame())
    assert set(values["right"]) == set(measured_score.AXIS_ORDER)
    assert "left" not in values


def test_key_without_uv_replaces_the_cell_component() -> None:
    assert builder.key_without_uv(KEY) == "m1:horizontal:right:na:center-near-high"


def test_thin_bucket_is_kept_in_the_artifact() -> None:
    """n<20이어도 아티팩트에는 남긴다. 폴백 판정은 읽는 쪽이 한다."""
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=1,
        buckets=_buckets({"20250901": {"right": {"wall": spread_stats.accumulate([0.1])}}}),
    )
    assert payload["granularities"]["uv8"][KEY]["n"] == 1
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_build_spread_reference.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_spread_reference'`

- [ ] **Step 3: 구현한다**

`trainer/scripts/build_spread_reference.py`:

```python
#!/usr/bin/env python3
"""전형기 패스에서 버킷별·점별·축별 산포를 뽑아 아티팩트로 굽는다.

산포를 전형기 한 사람의 변동으로 재는 이유는 6인으로 재면 작업자 간 차이가 잣대에
들어가기 때문이다. 그것이 바로 교정하려는 대상이다.

축은 wall/vertical/lateral이다. 교정 후보를 고르는 축과 같아야 sigma로 축을 고를 수 있다.

날짜별 부분합을 함께 실어 질의 날짜 제외와 개선율의 A/B 분리를 전수 재빌드 없이 한다.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import measured_score
import spread_stats

logger = logging.getLogger(__name__)

GRANULARITIES: Final[tuple[tuple[str, int], ...]] = (
    ("uv8", 8), ("uv4", 4), ("uv2", 2), ("uv_off", 0),
)
# 이전 연구에서 사전등록한 값. 이번 수치를 보고 정하지 않는다.
N_MIN: Final = 20


def key_without_uv(key: str) -> str:
    """UV 셀 성분을 na로 바꾼다. motion_key가 uv=None일 때 내는 형태와 같다."""
    parts = key.split(":")
    if len(parts) < 5:
        return key
    parts[3] = "na"
    return ":".join(parts)


def pass_axis_values(
    tracks: Mapping[str, Any], frame: Any
) -> dict[str, dict[str, float]]:
    """한 패스의 점별·축별 값(m). 교정 후보와 같은 축 분해를 쓴다."""
    output: dict[str, dict[str, float]] = {}
    for point in measured_score.POINT_ORDER:
        points = [p for p in (tracks.get(point) or []) if p is not None]
        if not points:
            continue
        mean = (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
            sum(p[2] for p in points) / len(points),
        )
        axes = {
            axis: magnitude_cm / 100.0
            for axis, magnitude_cm, _direction, _unit
            in measured_score._correction_candidates(mean, frame)
        }
        if axes:
            output[point] = axes
    return output


def build_payload(
    worker: str,
    levels: Sequence[str],
    sessions: int,
    buckets: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Mapping[str, spread_stats.AxisAccum]]]]],
) -> dict[str, object]:
    """buckets: layer -> key -> date -> point -> axis -> AxisAccum"""
    granularities: dict[str, object] = {}
    for layer, _grid in GRANULARITIES:
        layer_out: dict[str, object] = {}
        for key, by_date in (buckets.get(layer) or {}).items():
            dates = sorted(by_date)
            merged: dict[str, dict[str, spread_stats.AxisAccum]] = {}
            for date in dates:
                for point, axes in by_date[date].items():
                    slot = merged.setdefault(point, {})
                    for axis, accum in axes.items():
                        slot[axis] = spread_stats.merge(
                            [slot.get(axis, spread_stats.AxisAccum()), accum]
                        )
            points_out: dict[str, dict[str, dict[str, float]]] = {}
            for point, axes in merged.items():
                points_out[point] = {}
                for axis, accum in axes.items():
                    mean, spread = spread_stats.summarize(accum)
                    points_out[point][axis] = {"mean": mean, "spread": spread}
            counts = [accum.n for axes in merged.values() for accum in axes.values()]
            layer_out[key] = {
                "n": max(counts) if counts else 0,
                "dates": dates,
                "by_date": {
                    date: {
                        point: {axis: spread_stats.to_json(accum) for axis, accum in axes.items()}
                        for point, axes in by_date[date].items()
                    }
                    for date in dates
                },
                "points": points_out,
            }
        granularities[layer] = layer_out

    return {
        "version": 1,
        "worker": worker,
        "levels": list(levels),
        "sessions": sessions,
        "n_min": N_MIN,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "granularities": granularities,
    }
```

`main()`은 `CsvPassSource(root)`로 `worker`의 세션을 훑는다. 세션마다 `index_of`로 패스를 얻고, 각 패스에 대해 `service.pass_frames` → `service.key_inputs` → `service.representative_basis` → `service._local_tracks` → `pass_axis_values`를 돌린다. 키는 `uv8/uv4/uv2`가 `service.key_from(inputs, motion_key.KeyParams(uv_grid=N))`이고 `uv_off`는 `key_without_uv(uv8_key)`다. 축 값 하나를 `spread_stats.accumulate([value])`로 만들어 `buckets[layer][key][date][point][axis]`에 `merge`로 누적한다. 마지막에 `build_payload`를 호출해 `--out`에 쓴다. 세션마다 `logger.info`로 진행을 알린다.

- [ ] **Step 4: 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_build_spread_reference.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: 실데이터로 굽고 커버리지를 보고한다**

```bash
docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/build_spread_reference.py \
  --root /data/sources/professionals --worker 전형기 \
  --out /output/spread_reference_expert.json --log-level INFO
```

**보고서에 반드시 적을 것**: 층별로 (키 개수, `n ≥ 20`인 키 개수, 중앙 n, 최대 n), 그리고 전형기 패스 총수. 스펙 §11이 "전형기 단독 커버리지 미측정"이라 적어 뒀고 이 단계가 그것을 처음 잰다. **커버리지가 낮으면 폴백이 대부분의 질의를 굵은 층으로 보낸다는 뜻이며, 그것은 실패가 아니라 측정 결과다.**

- [ ] **Step 6: 커밋한다**

```bash
git add trainer/scripts/build_spread_reference.py trainer/scripts/tests/test_build_spread_reference.py
git commit -m "feat(pose): 전형기 실측 산포 레퍼런스 빌더 추가

축은 wall/vertical/lateral로 교정 후보와 맞춘다. 그래야 sigma로 축을 고를 수 있다.
날짜별 부분합을 함께 실어 질의 날짜 제외와 A/B 분리를 전수 재빌드 없이 한다.
n<20인 버킷도 아티팩트에는 남긴다. 폴백 판정은 읽는 쪽이 한다."
```

---

## Task 4: 레퍼런스 읽기, 날짜 제외, 입도 폴백

**Files:**
- Create: `trainer/pose_lab/spread_reference.py`
- Test: `trainer/pose_lab/tests/test_spread_reference.py`

**Interfaces:**
- Consumes: `spread_stats`, `build_spread_reference.GRANULARITIES`/`N_MIN`
- Produces:
  - `@dataclass(frozen=True, slots=True) class Resolved` — `granularity: str`, `key: str`, `n: int`, `dates: list[str]`, `spreads: dict[str, dict[str, float]]`, `means: dict[str, dict[str, float]]`
  - `class SpreadReference:`
    - `def __init__(self, payload: Mapping[str, object])`
    - `@classmethod def load(cls, path: Path) -> "SpreadReference"`
    - `@property def worker(self) -> str`
    - `def resolve(self, keys: Mapping[str, str], exclude_dates: Sequence[str] = ()) -> Resolved | None`
    - `def split_dates(self, granularity: str, key: str) -> tuple[list[str], list[str]] | None`
    - `def spreads_for(self, granularity: str, key: str, dates: Sequence[str]) -> dict[str, dict[str, float]] | None`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`trainer/pose_lab/tests/test_spread_reference.py`:

```python
#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

spread_stats = importlib.import_module("spread_stats")

from trainer.pose_lab.spread_reference import SpreadReference

KEY8 = "m1:horizontal:right:1x2:center-near-high"
KEY_OFF = "m1:horizontal:right:na:center-near-high"


def _bucket(by_date):
    dates = sorted(by_date)
    merged: dict[str, list[float]] = {}
    for date in dates:
        for axis, values in by_date[date].items():
            merged.setdefault(axis, []).extend(values)
    return {
        "n": max(len(values) for values in merged.values()),
        "dates": dates,
        "by_date": {
            date: {
                "right": {
                    axis: spread_stats.to_json(spread_stats.accumulate(values))
                    for axis, values in axes.items()
                }
            }
            for date, axes in by_date.items()
        },
        "points": {
            "right": {
                axis: {
                    "mean": sum(values) / len(values),
                    "spread": spread_stats.summarize(spread_stats.accumulate(values))[1],
                }
                for axis, values in merged.items()
            }
        },
    }


def _payload(uv8=None, uv_off=None):
    return {
        "version": 1, "worker": "전형기", "levels": ["10"], "sessions": 5, "n_min": 20,
        "granularities": {
            "uv8": {KEY8: uv8} if uv8 else {},
            "uv4": {}, "uv2": {},
            "uv_off": {KEY_OFF: uv_off} if uv_off else {},
        },
    }


def _thick(date_count=3, per_date=10, base=0.10):
    return {
        f"2025090{index + 1}": {"wall": [base + 0.01 * step for step in range(per_date)]}
        for index in range(date_count)
    }


KEYS = {"uv8": KEY8, "uv_off": KEY_OFF}


def test_resolve_uses_the_finest_layer_that_meets_n_min() -> None:
    resolved = SpreadReference(_payload(uv8=_bucket(_thick()))).resolve(KEYS)
    assert resolved is not None
    assert resolved.granularity == "uv8"
    assert resolved.n >= 20


def test_resolve_falls_back_when_the_fine_layer_is_thin() -> None:
    thin = _bucket({"20250901": {"wall": [0.10, 0.11, 0.12]}})
    resolved = SpreadReference(_payload(uv8=thin, uv_off=_bucket(_thick()))).resolve(KEYS)
    assert resolved is not None
    assert resolved.granularity == "uv_off", "얇으면 굵은 층으로 내려가야 한다"


def test_resolve_returns_none_when_every_layer_is_thin() -> None:
    thin = _bucket({"20250901": {"wall": [0.10, 0.11]}})
    assert SpreadReference(_payload(uv8=thin)).resolve(KEYS) is None


def test_excluding_a_date_changes_the_statistics() -> None:
    """질의가 전형기 자신이면 그 날짜를 빼야 한다. 자기가 자기 평균에 가까운 건 당연하다."""
    by_date = _thick()
    by_date["20250903"] = {"wall": [0.90 + 0.01 * step for step in range(10)]}
    reference = SpreadReference(_payload(uv8=_bucket(by_date)))
    full = reference.resolve(KEYS)
    without = reference.resolve(KEYS, exclude_dates=["20250903"])
    assert full is not None and without is not None
    assert "20250903" not in without.dates
    assert not math.isclose(
        full.means["right"]["wall"], without.means["right"]["wall"], abs_tol=1e-9
    )


def test_excluded_reconstruction_matches_a_build_without_that_date() -> None:
    by_date = _thick()
    without = SpreadReference(_payload(uv8=_bucket(by_date))).resolve(
        KEYS, exclude_dates=["20250901"]
    )
    trimmed = {date: axes for date, axes in by_date.items() if date != "20250901"}
    baseline = SpreadReference(_payload(uv8=_bucket(trimmed))).resolve(KEYS)
    assert without is not None and baseline is not None
    assert math.isclose(
        without.spreads["right"]["wall"], baseline.spreads["right"]["wall"], abs_tol=1e-9
    )


def test_excluding_every_date_yields_none() -> None:
    by_date = _thick()
    reference = SpreadReference(_payload(uv8=_bucket(by_date)))
    assert reference.resolve(KEYS, exclude_dates=list(by_date)) is None


def test_split_dates_partitions_without_overlap() -> None:
    split = SpreadReference(_payload(uv8=_bucket(_thick(date_count=4)))).split_dates("uv8", KEY8)
    assert split is not None
    first, second = split
    assert first and second
    assert not set(first) & set(second)


def test_split_dates_is_none_with_a_single_date() -> None:
    reference = SpreadReference(_payload(uv8=_bucket({"20250901": {"wall": [0.1] * 25}})))
    assert reference.split_dates("uv8", KEY8) is None


def test_worker_is_exposed_so_callers_know_whose_dates_to_exclude() -> None:
    assert SpreadReference(_payload(uv8=_bucket(_thick()))).worker == "전형기"
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/pose_lab/tests/test_spread_reference.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'trainer.pose_lab.spread_reference'`

- [ ] **Step 3: 구현한다**

`resolve`는 `GRANULARITIES` 순서대로 층을 훑는다. 각 층에서 `keys`에 그 층의 키가 있고 아티팩트에 그 버킷이 있으면, `by_date`에서 `exclude_dates`를 뺀 부분합을 점·축별로 `spread_stats.merge`하고 `summarize`한다. 그 결과의 최대 `n`이 `N_MIN` 이상이면 `Resolved`를 만들어 반환한다. 어느 층도 만족하지 않으면 `None`이다.

`split_dates`는 버킷의 `dates`를 정렬해 인덱스 홀짝으로 두 그룹에 나눈다. 어느 한쪽이 비면 `None`이다.

`spreads_for`는 주어진 날짜 집합만으로 산포를 만든다 — 개선율의 A/B 분리가 쓴다.

- [ ] **Step 4: 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/pose_lab/tests/test_spread_reference.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: 커밋한다**

```bash
git add trainer/pose_lab/spread_reference.py trainer/pose_lab/tests/test_spread_reference.py
git commit -m "feat(pose): 산포 레퍼런스 읽기와 날짜 제외·입도 폴백 추가

n>=20을 처음 만족하는 층을 쓰고 전부 미달이면 없다고 답한다. 날짜 제외는 부분합
재구성으로 하며, 그 결과가 그 날짜 없이 전수 빌드한 것과 같음을 테스트가 고정한다."
```

---

## Task 5: 서비스 배선 — 실측 산포, 실제 게이트, split 산출

**Files:**
- Modify: `trainer/pose_lab/service.py`, `trainer/pose_lab/app.py`
- Test: `trainer/pose_lab/tests/test_service_measured.py`

**Interfaces:**
- Consumes: Task 1~4 전부. `FakeSource`는 기존 `trainer/pose_lab/tests/test_app.py`에서 import해 재사용한다(복제 금지).
- Produces:
  - `compare(..., reference_spreads: SpreadReference | None = None)`
  - 응답 `measured`에 추가: `normalization`, `granularity`, `reference_n`, `reference_dates`, `split_reason`
  - `create_app`이 `POSE_LAB_SPREAD_REFERENCE` 환경변수를 읽는다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_measured_spreads_flip_normalization_and_enable_the_gate() -> None:
    body = _compare_with_reference()
    assert body["measured"]["normalization"] == "measured"
    assert body["measured"]["gate"] in {"pass", "fail"}
    assert body["measured"].get("gate_reason") is None


def test_response_says_which_layer_and_how_many_samples() -> None:
    body = _compare_with_reference()
    assert body["measured"]["granularity"] in {"uv8", "uv4", "uv2", "uv_off"}
    assert isinstance(body["measured"]["reference_n"], int)
    assert body["measured"]["reference_n"] >= 20
    assert body["measured"]["reference_dates"]


def test_split_is_a_real_number_or_an_explained_null() -> None:
    body = _compare_with_reference()
    split = body["measured"]["improvement"]["split"]
    if split is None:
        assert body["measured"]["split_reason"] == "insufficient_dates"
    else:
        assert isinstance(split, float)


def test_split_never_exceeds_same() -> None:
    """교정에 쓴 분포로 다시 채점하면 향상이 정의상 보장된다. split이 그보다 크면 버그다."""
    body = _compare_with_reference()
    improvement = body["measured"]["improvement"]
    if improvement["split"] is not None:
        assert improvement["split"] <= improvement["same"] + 1e-9


def test_insufficient_dates_are_reported_not_invented() -> None:
    body = _compare_with_single_date_reference()
    assert body["measured"]["improvement"]["split"] is None
    assert body["measured"]["split_reason"] == "insufficient_dates"


def test_query_own_date_is_excluded_from_the_reference() -> None:
    body = _compare_expert_against_own_reference()
    assert _QUERY_DATE not in body["measured"]["reference_dates"]


def test_without_a_reference_it_stays_provisional() -> None:
    body = _compare_without_reference()
    assert body["measured"]["normalization"] == "provisional"
    assert body["measured"]["gate"] is None
    assert body["measured"]["gate_reason"] == "no_reference_distribution"


def test_joint_axis_still_borrows_nothing() -> None:
    body = _compare_with_reference()
    assert body["joint"]["status"] == "not_wired"
    assert body["joint"]["score"] is None
    assert body["joint"]["improvement"] is None
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/pose_lab/tests/test_service_measured.py -q`
Expected: FAIL

- [ ] **Step 3: 구현한다**

`compare`가 `reference_spreads`를 받으면:

1. 질의 패스의 네 층 키를 만든다 — `uv8/uv4/uv2`는 `key_from(inputs, KeyParams(uv_grid=N))`, `uv_off`는 `build_spread_reference.key_without_uv(uv8_key)`.
2. `query.worker == reference_spreads.worker`면 `exclude_dates=[query.date]`를 넘긴다.
3. `resolve` 결과가 있으면 그 `spreads`를 `score_track`에 넘기고, `normalization="measured"`, `gate=gate_verdict(최대 산포, DEFAULT_ERROR_M)`, `granularity`/`reference_n`/`reference_dates`를 응답에 싣는다. 없으면 현행 잠정 처리를 유지한다.
4. `split_dates(granularity, key)`로 A/B를 갈라 `spreads_for(...A)`로 교정하고 `spreads_for(...B)`로 채점해 `improvement.split`을 낸다. 분할 불가면 `split=None`, `split_reason="insufficient_dates"`.

`app.py`의 `create_app`은 `POSE_LAB_SPREAD_REFERENCE` 경로가 있으면 `SpreadReference.load`로 읽어 `compare`에 넘긴다. 없으면 현행과 동일하다.

- [ ] **Step 4: 통과를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/pose_lab -q`
Expected: PASS

- [ ] **Step 5: 전체 회귀**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: 586 passed (548 + 8 + 7 + 8 + 9 + 8·이 태스크). **자기 시점의 실측값을 쓰고 차이를 보고서에 설명한다.**

- [ ] **Step 6: 커밋한다**

```bash
git add trainer/pose_lab/service.py trainer/pose_lab/app.py trainer/pose_lab/tests/test_service_measured.py
git commit -m "feat(pose): 실측 산포를 배선해 게이트와 분리 개선율을 실제로 낸다

산포가 실측이면 normalization을 measured로 올리고 REQ-19 게이트를 실제로 판정한다.
어느 입도에서 표본 몇 개로 나온 값인지 응답에 싣는다.

질의가 전형기 자신이면 그 날짜를 레퍼런스에서 뺀다. 날짜를 A/B로 갈라 교정과 채점을
분리해 improvement.split을 처음으로 실제 값으로 낸다. 날짜가 모자라면 만들어내지 않고
insufficient_dates로 비운다."
```

---

## Task 6: 화면에 근거를 드러낸다

**Files:**
- Modify: `trainer/pose_lab/static/index.html`, `app.js`, `style.css`, `README.md`

- [ ] **Step 1: 화면을 고친다**

실측 축 패널에 더한다. 값이 없으면 `—`이고 `0`이 아니다.

- `정규화`: `측정된 분포` / `임시`
- `입도 · 표본`: `uv2 · n=36` 형태
- `레퍼런스 날짜`: 개수와 목록
- 점별 막대 옆에 **축별 σ**(wall / vertical / lateral)
- `REQ-19 게이트`: `pass` / `fail` / `—`, `—`일 때 사유 병기

`README.md` §2.5에 `POSE_LAB_SPREAD_REFERENCE` 환경변수와 아티팩트 굽는 명령을 적는다.

- [ ] **Step 2: 컨테이너에서 띄워 확인한다**

```bash
docker exec -d -w /workspace \
  -e POSE_LAB_DATA_ROOT=/data/sources/professionals \
  -e POSE_LAB_SPREAD_REFERENCE=/output/spread_reference_expert.json \
  ship-paint-trainer python3 -m uvicorn trainer.pose_lab.app:create_app --factory --host 0.0.0.0 --port 8130
```

브라우저를 못 여니 `curl`로 페이지와 `/api/compare` 응답을 받아 **화면이 읽는 키가 실제로 오는지 필드 단위로 대조**한다. **육안 확인은 못 했다고 보고서에 그대로 적는다.**

- [ ] **Step 3: 커밋한다**

```bash
git add trainer/pose_lab/static README.md
git commit -m "feat(pose): 입도·표본수·축별 sigma를 화면에 드러낸다

같은 화면의 두 숫자가 서로 다른 입도에서 나올 수 있고 그 사실이 보이지 않으면 비교가
거짓이 된다. 값이 없으면 0이 아니라 대시로 낸다."
```

---

## Task 7: 70% 질문에 답하는 표

**Files:**
- Create: `trainer/scripts/improvement_study.py`
- Test: `trainer/scripts/tests/test_improvement_study.py`

이 태스크의 산출물은 기능이 아니라 **숫자와 결론**이다.

- [ ] **Step 1: 집계 함수의 테스트를 쓴다**

```python
#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

improvement_study = importlib.import_module("improvement_study")


def test_summary_reports_median_quartiles_and_threshold_share() -> None:
    summary = improvement_study.summarize([0.1, 0.2, 0.5, 0.75, 0.8, 0.9], threshold=0.70)
    assert summary["n"] == 6
    assert math.isclose(summary["median"], 0.625, abs_tol=1e-9)
    assert math.isclose(summary["at_or_above"], 0.5, abs_tol=1e-9)


def test_none_values_are_excluded_not_zeroed() -> None:
    """계산되지 않은 split을 0으로 넣으면 분포가 근거 없이 내려간다."""
    summary = improvement_study.summarize([0.8, None, 0.9], threshold=0.70)
    assert summary["n"] == 2
    assert math.isclose(summary["at_or_above"], 1.0, abs_tol=1e-9)


def test_empty_input_reports_none_not_zero() -> None:
    summary = improvement_study.summarize([], threshold=0.70)
    assert summary["n"] == 0
    assert summary["median"] is None
    assert summary["at_or_above"] is None


def test_quartiles_are_reported() -> None:
    summary = improvement_study.summarize([0.1, 0.2, 0.3, 0.4], threshold=0.70)
    assert summary["q1"] is not None
    assert summary["q3"] is not None
    assert summary["q1"] <= summary["median"] <= summary["q3"]
```

- [ ] **Step 2: 실패를 확인한다**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/scripts/tests/test_improvement_study.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'improvement_study'`

- [ ] **Step 3: 구현하고 실데이터로 돌린다**

CLI가 비교 쌍을 표집해 `service.compare`를 돌리고 스펙 §10의 표를 낸다. **표집 방식(무작위인지 전수인지, 시드, 쌍 수)을 코드에 먼저 고정하고 그 다음에 결과를 본다.**

```bash
docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/improvement_study.py \
  --root /data/sources/professionals \
  --reference /output/spread_reference_expert.json \
  --pairs 200 --seed 20260828 --out /output/improvement_study.json
```

보고서에 스펙 §10이 정한 항목을 **전부** 싣는다:
- 쌍 수와 선정 방식
- `same`의 중앙값·사분위·70% 이상 비율
- `split`의 같은 값들
- 입도 층별 분해 — 굵은 층에서 나온 값이 얼마나 되는지
- 위치에 대한 REQ-19 게이트 통과 버킷 비율

**결과가 나온 뒤 유리한 절단이나 부분집합을 고르지 않는다. 목표 미달이면 미달이라고 적는다.**

- [ ] **Step 4: 전체 게이트**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: 590 passed (Task 5 시점 + 4)

- [ ] **Step 5: 커밋한다**

```bash
git add trainer/scripts/improvement_study.py trainer/scripts/tests/test_improvement_study.py
git commit -m "feat(pose): 개선율 분포로 70% 목표를 판정하는 하네스 추가

납품 문구대로 재면(동일 채점) 몇 %, 정직하게 재면(분리 채점) 몇 %인지를 같은 표에
낸다. 계산되지 않은 split을 0으로 넣지 않는다. 표의 형식은 숫자를 보기 전에 고정했다."
```

---

## 이번 계획에서 하지 않는 것

| 항목 | 이유 |
|---|---|
| 각도 기준 | 스펙 §2.3에서 기각. 표면 법선을 벤더가 줘야 재검정 가능 |
| 표시용 스켈레톤 / React | 검증된 FK 없음. 별건 |
| 6인 풀링 산포 | D1이 전형기 1인으로 정했다 |
| 발주처 대응 | Task 7의 표를 낼 뿐, 미달 처리는 사용자 판단 |

---

## Self-Review 결과

**1. 스펙 커버리지**

| 스펙 절 | 태스크 |
|---|---|
| §3 D2 축별 산포, §7 교정 선택 | Task 1 |
| §4 부분합 | Task 2 |
| §4 아티팩트, §11 전형기 커버리지 측정 | Task 3 |
| §5 자기충족 차단, §6 폴백, §3 D3·D5 | Task 4 |
| §3 D4, §8 응답 변경 | Task 5 |
| §6 대가의 표시 | Task 6 |
| §10 70% 표 | Task 7 |

**2. Placeholder 스캔:** 통과. Task 3 `main()`, Task 4 `resolve`/`split_dates`, Task 5·6은 코드 전문 대신 알고리즘을 문장으로 적었다. 네 곳 모두 인터페이스와 테스트가 동작을 완전히 고정하므로 구현이 유일하게 결정된다.

**3. 타입 일관성**
- 산포 축 이름은 어디서나 `wall`/`vertical`/`lateral`이고 `measured_score.AXIS_ORDER`가 단일 출처다. **스펙 §4는 `normal/tangent/up`이라 적혀 있어 어긋나므로 Task 1 Step 3에서 스펙도 함께 고친다.**
- `AxisAccum`(T2) → `build_payload`(T3) → `SpreadReference`(T4)가 `to_json`/`from_json`으로만 오간다.
- `Resolved.spreads`(T4)와 `score_track(spreads=...)`(T1)가 같은 `점 → 축 → float` 모양이다.
- `N_MIN`은 `build_spread_reference`가 단일 출처이고 T4가 import한다.
- `GRANULARITIES`도 같은 단일 출처를 T4·T5가 쓴다.
- 테스트 누적: 548 → 556(T1) → 563(T2) → 571(T3) → 580(T4) → T5 → +4(T7). T5의 신규 테스트 수는 구현하며 정해지므로 실행자가 자기 시점 실측값을 쓰고 차이를 설명한다.
