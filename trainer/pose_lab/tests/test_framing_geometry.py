#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""'상체 확대'가 '자세 맞춤'보다 **실제로 크게** 보이는가(D6).

`test_static_assets.torso_preset_faults`는 `skeleton_view.js`가 어떤 규칙을 쓰는지(소스 문자열)만
본다. 그것만으로는 e69006d가 바로잡은 실패 — AABB 8꼭짓점에 맞추는 바람에 **확대 프리셋이 기본보다
작아진 것**(L3 0.4864 < 기본 0.5827) — 이 되돌아와도 잡히지 않는다. 여기서는 그 기하를 직접 센다.

브라우저가 없으니(컨테이너에 JS 엔진이 없다) `frameTo`의 순수 기하 — AABB 중심을 보고, 목표 비율이
되는 거리를 이분법으로 찾고, 그 거리에서 점을 투영하는 것 — 를 파이썬으로 옮겨 잰다. 옮긴 값이 아니라
**`skeleton.view.js`가 실제로 쓰는 값**을 쓰도록 목표 비율(`FRAMING_TARGET`)과 "어느 모드가 실제 점에
맞추는가"는 소스에서 읽어 온다. 그래서 목표를 바꾸거나 torso를 꼭짓점 맞춤으로 되돌리면 여기가 깨진다.

이 파일이 지키지 **못하는** 것: 파이썬 이식본과 JS 구현이 갈라지는 것. 그 부분은 소스 규칙 검사
(`test_static_assets`)와 헤드리스 브라우저 실측이 맡는다.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "static"


def camera_constants(js: str) -> tuple[float, float]:
    """`skeleton_view.js`가 실제로 쓰는 카메라 상수(fov, near).

    숫자를 여기 박아 두면 안 된다 — 예전에는 `FOV_DEG = 45.0  # skeleton_view.js의 …`,
    `NEAR = 0.01`이라고 적혀 있었는데 소스는 `PerspectiveCamera(50, 1, 0.02, 200)`였다.
    출처와 다른 라벨이 붙은 숫자이고, 이 파일의 모든 투영이 그 값을 썼다.
    """
    found = re.search(r"new THREE\.PerspectiveCamera\(\s*([\d.]+)\s*,\s*[\d.]+\s*,\s*([\d.]+)", js)
    assert found is not None, "PerspectiveCamera 생성자를 찾지 못했다"
    return float(found.group(1)), float(found.group(2))


def min_torso_fraction(js: str) -> float:
    """기본 프리셋이 지키는 상체 합격선(F). 소스가 하나의 출처다."""
    found = re.search(r"const MIN_TORSO_FRACTION = ([\d.]+);", js)
    assert found is not None, "MIN_TORSO_FRACTION을 찾지 못했다"
    return float(found.group(1))


FOV_DEG, NEAR = camera_constants((STATIC / "skeleton_view.js").read_text(encoding="utf-8"))


def framing_targets(js: str) -> dict[str, float]:
    body = re.search(r"const FRAMING_TARGET = \{([^}]*)\}", js)
    assert body is not None, "FRAMING_TARGET을 찾지 못했다"
    return {k: float(v) for k, v in re.findall(r"(\w+):\s*([\d.]+)", body.group(1))}


def real_point_modes(js: str) -> set[str]:
    """실제 점에 맞추는 모드(나머지는 AABB 8꼭짓점). 소스의 삼항식에서 읽는다."""
    line = re.search(r"\n\s*mode === ([^?]*)\? points\.map", js)
    assert line is not None, "맞춤 대상 선택 식을 찾지 못했다"
    return set(re.findall(r'"(\w+)"', line.group(1)))


def _bounds(points):
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return lo, hi


def _basis(offset_dir):
    forward = tuple(-c for c in offset_dir)
    right = _cross(forward, (0.0, 1.0, 0.0))
    if _dot(right, right) < 1e-12:
        right = (1.0, 0.0, 0.0)
    right = _unit(right)
    up = _unit(_cross(right, forward))
    return forward, right, up


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _unit(v):
    n = math.sqrt(_dot(v, v))
    return tuple(c / n for c in v)


def project_ndc(point, target, offset_dir, distance, aspect):
    """`projectedRatio`와 같은 기저·같은 원근으로 NDC 좌표를 낸다. 뒤로 넘어가면 None."""
    forward, right, up = _basis(offset_dir)
    position = tuple(target[i] + offset_dir[i] * distance for i in range(3))
    v = tuple(point[i] - position[i] for i in range(3))
    z = _dot(v, forward)
    if z <= NEAR:
        return None
    tan_v = math.tan(math.radians(FOV_DEG) / 2)
    return (_dot(v, right) / (z * tan_v * aspect), _dot(v, up) / (z * tan_v))


def projected_extent(points, target, offset_dir, distance, aspect):
    """맞춤 대상의 화면(NDC) 범위. 뒤로 넘어간 점이 하나라도 있으면 None."""
    xs, ys = [], []
    for p in points:
        ndc = project_ndc(p, target, offset_dir, distance, aspect)
        if ndc is None:
            return None
        xs.append(ndc[0])
        ys.append(ndc[1])
    return min(xs), max(xs), min(ys), max(ys)


def projected_ratio(points, target, offset_dir, distance, aspect):
    box = projected_extent(points, target, offset_dir, distance, aspect)
    if box is None:
        return math.inf
    xmin, xmax, ymin, ymax = box
    return max((xmax - xmin) / 2, (ymax - ymin) / 2)


def projected_reach(points, target, offset_dir, distance, aspect):
    """맞춤 대상이 화면 중심에서 가장 멀리 나간 정도(NDC). 1.0이 캔버스 가장자리다."""
    box = projected_extent(points, target, offset_dir, distance, aspect)
    if box is None:
        return math.inf
    xmin, xmax, ymin, ymax = box
    return max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))


def reach_modes(js: str) -> set[str]:
    """범위 크기가 아니라 **가장 멀리 나간 점**에 맞추는 모드. 소스의 조건식에서 읽는다."""
    line = re.search(r"\n\s*const measure = mode === ([^?]*)\? projectedReach", js)
    assert line is not None, "맞춤 척도 선택 식을 찾지 못했다"
    return set(re.findall(r'"(\w+)"', line.group(1)))


def solve_framing(points, mode, targets, real_modes, theta, phi, aspect, reach=()):
    """`frameTo`의 이분법. 실제 점 모드가 아니면 AABB 8꼭짓점에 맞추고, `reach` 모드는 범위 크기
    대신 **가장 멀리 나간 점**을 목표 아래로 누른다."""
    lo, hi = _bounds(points)
    target = tuple((lo[i] + hi[i]) / 2 for i in range(3))
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    sin_phi = math.sin(phi)
    offset_dir = _unit((sin_phi * math.sin(theta), math.cos(phi), sin_phi * math.cos(theta)))
    fit = list(points) if mode in real_modes else corners
    goal = targets.get(mode, 0.75)
    measure = projected_reach if mode in reach else projected_ratio
    low, high = 0.01, 100.0
    for _ in range(60):
        mid = (low + high) / 2
        if measure(fit, target, offset_dir, mid, aspect) > goal:
            low = mid
        else:
            high = mid
    return target, offset_dir, high


def pull_in_to_torso_floor(target, offset_dir, distance, aspect, head, pelvis, upper, minimum):
    """**옛** 규칙(`pullInToTorsoFloor`)의 이식본: 상체가 합격선에 못 미치면 그 선까지 당긴다.

    화면에서는 지워진 규칙이고, 여기에만 남아 **반례**로 쓰인다 — 당기면 합격선은 넘지만 그려지는
    것이 잘린다는 것을 같은 장면에서 숫자로 보이는 자리다(`test_meeting_the_pass_line_would_cut_…`).
    보는 곳은 상체 상자의 중심이고(전신 중심을 보면 같은 배율에서 머리가 먼저 잘린다), 거리는 이
    비율에 단조라 같은 이분법을 쓴다.
    """
    if torso_span(head, pelvis, target, offset_dir, distance, aspect) >= minimum:
        return target, distance
    lo, hi = _bounds(upper)
    centre = tuple((lo[i] + hi[i]) / 2 for i in range(3))
    low, high = 0.01, distance
    for _ in range(60):
        mid = (low + high) / 2
        if torso_span(head, pelvis, centre, offset_dir, mid, aspect) >= minimum:
            low = mid
        else:
            high = mid
    return centre, low


def torso_span(head, pelvis, target, offset_dir, distance, aspect):
    """머리 실측 마커 ↔ 골반 관절의 화면 세로 비율(캔버스 높이 대비) — 화면의 `torsoScreenSpan`과 같은 정의."""
    a = project_ndc(head, target, offset_dir, distance, aspect)
    b = project_ndc(pelvis, target, offset_dir, distance, aspect)
    assert a is not None and b is not None
    return abs(a[1] - b[1]) / 2


# 합성 장면: 상체는 앞뒤로 두께가 있고(팔·마커가 몸 앞뒤로 벌어진다), 교정 화살표가 상체 밖으로
# 조금 더 뻗는다 — 실제 패스의 모양이다. 이 장면에서 '자세 맞춤'(당기기 전)의 상체 비율은
# 1875×1000 화면비에서 0.495 · 840×1050에서 0.422다.
#
# **두께(DEPTH)를 0.5에서 0.7로 키웠다.** 카메라 상수를 소스에서 읽도록 고친 뒤(fov 45→실제 50)
# 두께 0.5짜리 장면에서는 "꼭짓점 맞춤이 기본보다 작아진다"는 성질이 1875 화면비에서 뒤집혔다
# (꼭짓점 0.5277 > 기본 0.5031). 이 검사가 재현하려는 결함(e69006d)은 **몸 앞뒤 두께가 만든
# AABB 모서리**가 원인이므로, 그 두께를 실제 패스 쪽으로 키워 성질을 되살린다 — 문턱을 낮춘 것이
# 아니라 장면을 맞춘 것이다(0.7에서 꼭짓점 0.4739 < 기본 0.4952, 840에서 0.3073 < 0.4221).
HEAD = (0.0, 1.70, 0.0)
PELVIS = (0.0, 1.00, 0.0)
DEPTH = 0.7
UPPER_BODY = [
    HEAD,
    PELVIS,
    (0.0, 1.45, 0.0),
    (0.35, 1.35, DEPTH),
    (-0.35, 1.30, -DEPTH),
    (0.45, 1.55, DEPTH * 0.7),
]
CORRECTIONS = [(0.80, 1.80, DEPTH), (-0.80, 0.85, -DEPTH)]
POSE_POINTS = UPPER_BODY + CORRECTIONS
ASPECTS = (1875 / 1000, 840 / 1050)
ANGLES = (0.9, math.pi / 2.5)


def _span(points, mode, targets, real_modes, aspect, reach=()):
    target, offset_dir, distance = solve_framing(
        points, mode, targets, real_modes, *ANGLES, aspect, reach=reach
    )
    return torso_span(HEAD, PELVIS, target, offset_dir, distance, aspect)


# 벽면 스트로크는 몸에서 멀고 **한쪽으로 치우쳐** 있다(벽 위 한 줄). 기본 프리셋이 그것을 맞춤
# 대상에 넣은 뒤(S2) 이 치우침이 드러났다 — 이분법은 범위의 크기만 보고 카메라는 AABB 중심을
# 보므로, 크기가 목표 안이어도 한쪽이 화면 밖으로 밀린다. 실측(김경호 L6 1000394_1 #0,
# 1875×1000): 스트로크 정점 77개 중 3개가 캔버스 위로 6px 넘어갔다.
WALL_STROKE = [(1.30, 2.05, 0.9), (1.35, 2.10, 0.9), (1.40, 2.15, 0.9)]
POSE_WITH_STROKE = POSE_POINTS + WALL_STROKE


def _outside_count(points, mode, targets, real_modes, aspect, reach=()):
    """그 프리셋 카메라에서 맞춤 대상 중 캔버스(|ndc| ≤ 1) 밖에 있는 점의 수."""
    target, offset_dir, distance = solve_framing(
        points, mode, targets, real_modes, *ANGLES, aspect, reach=reach
    )
    out = 0
    for p in points:
        ndc = project_ndc(p, target, offset_dir, distance, aspect)
        if ndc is None or abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0:
            out += 1
    return out


@pytest.mark.parametrize("aspect", ASPECTS)
def test_torso_preset_really_is_bigger_than_the_default(aspect: float) -> None:
    """'상체 확대'의 상체 비율이 '자세 맞춤'보다 크다 — 확대 프리셋이 확대여야 한다."""
    js = (STATIC / "skeleton_view.js").read_text(encoding="utf-8")
    targets, real_modes = framing_targets(js), real_point_modes(js)
    assert {"pose", "torso"} <= set(targets), targets
    reach = reach_modes(js)
    pose = _span(POSE_POINTS, "pose", targets, real_modes, aspect, reach=reach)
    torso = _span(UPPER_BODY, "torso", targets, real_modes, aspect, reach=reach)
    assert torso > pose, (torso, pose)


@pytest.mark.parametrize("aspect", ASPECTS)
def test_fitting_the_torso_to_aabb_corners_makes_it_smaller_than_the_default(aspect: float) -> None:
    """e69006d가 고친 실패를 그대로 재현한다: torso가 꼭짓점에 맞으면 기본보다 **작아진다**.

    이 단언이 곧 위 검사가 공허하지 않다는 증거다 — 규칙 하나만 되돌리면 성질이 뒤집힌다.
    """
    js = (STATIC / "skeleton_view.js").read_text(encoding="utf-8")
    targets, real_modes, reach = framing_targets(js), real_point_modes(js), reach_modes(js)
    pose = _span(POSE_POINTS, "pose", targets, real_modes, aspect, reach=reach)
    torso = _span(UPPER_BODY, "torso", targets, real_modes, aspect, reach=reach)
    boxed = _span(UPPER_BODY, "torso", targets, real_modes - {"torso"}, aspect, reach=reach)
    assert boxed < torso, (boxed, torso)
    assert boxed < pose, (boxed, pose)


def test_source_reading_sees_the_values_the_screen_uses() -> None:
    """소스에서 읽은 목표·규칙이 실제 값이다(여기가 비면 위 두 검사가 기본값으로 조용히 통과한다)."""
    js = (STATIC / "skeleton_view.js").read_text(encoding="utf-8")
    assert framing_targets(js) == {"pose": 0.86, "tracks": 0.75, "wall": 0.9, "torso": 0.98}
    assert real_point_modes(js) == {"pose", "torso"}
    # 카메라 상수도 **소스에서** 읽는다(옛 주석의 45.0/0.01이 아니라 실제 값).
    assert camera_constants(js) == (50.0, 0.02)
    assert min_torso_fraction(js) == 0.4


@pytest.mark.parametrize("aspect", ASPECTS)
def test_the_default_preset_keeps_every_fit_target_on_the_canvas(aspect: float) -> None:
    """기본 뷰('자세 맞춤')는 맞춤 대상을 **하나도** 화면 밖으로 내보내지 않는다.

    사용자 불만 1번이 "기준 스트로크(벽면 착탄점)이 보이지 않음"이었다(S2). 스트로크를 맞춤 대상에
    넣기만 하면 되는 줄 알았는데, 스트로크는 벽 위 한 줄이라 점 구름이 한쪽으로 치우쳐 크기가
    목표 안이어도 밖으로 밀린다 — 그래서 기본 뷰는 범위 크기가 아니라 **가장 멀리 나간 점**에
    맞춘다. 자리표시 다리의 발도 같은 규칙으로 화면 안에 남는다(B).
    """
    js = (STATIC / "skeleton_view.js").read_text(encoding="utf-8")
    targets, real_modes, reach = framing_targets(js), real_point_modes(js), reach_modes(js)
    assert "pose" in reach, reach
    outside = _outside_count(POSE_WITH_STROKE, "pose", targets, real_modes, aspect, reach=reach)
    assert outside == 0, outside


def _default_and_pulled(aspect: float) -> tuple[float, int, float, int]:
    """같은 장면에서 (지금 규칙의 기본 뷰, **옛 규칙대로** 합격선까지 당긴 뷰)의 상체 비율과 자른 점 수."""
    js = (STATIC / "skeleton_view.js").read_text(encoding="utf-8")
    targets, real_modes, reach = framing_targets(js), real_point_modes(js), reach_modes(js)
    minimum = min_torso_fraction(js)
    target, offset_dir, distance = solve_framing(
        POSE_WITH_STROKE, "pose", targets, real_modes, *ANGLES, aspect, reach=reach
    )

    def cut_at(centre, radius) -> int:
        out = 0
        for p in POSE_WITH_STROKE:
            ndc = project_ndc(p, centre, offset_dir, radius, aspect)
            if ndc is None or abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0:
                out += 1
        return out

    default = torso_span(HEAD, PELVIS, target, offset_dir, distance, aspect)
    pulled_target, pulled_distance = pull_in_to_torso_floor(
        target, offset_dir, distance, aspect, HEAD, PELVIS, UPPER_BODY, minimum
    )
    pulled = torso_span(HEAD, PELVIS, pulled_target, offset_dir, pulled_distance, aspect)
    return default, cut_at(target, distance), pulled, cut_at(pulled_target, pulled_distance)


@pytest.mark.parametrize("aspect", ASPECTS)
def test_the_default_preset_cuts_nothing_and_falls_short_of_the_pass_line(aspect: float) -> None:
    """기본 뷰는 **아무것도 자르지 않고**, 그 대가로 상체 합격선(F)에 못 미친다.

    두 성질을 한 검사에 묶는다 — 이 화면의 사실이 바로 "둘 다는 안 된다"이기 때문이다. 못 미치는
    것을 화면이 숫자로 말하는지는 `test_static_assets.torso_floor_faults`가 따로 고정한다.
    """
    js = (STATIC / "skeleton_view.js").read_text(encoding="utf-8")
    minimum = min_torso_fraction(js)
    default, default_cut, _pulled, _pulled_cut = _default_and_pulled(aspect)
    assert default_cut == 0, default_cut
    assert default < minimum, (default, minimum)


@pytest.mark.parametrize("aspect", ASPECTS)
def test_meeting_the_pass_line_would_cut_what_the_screen_draws(aspect: float) -> None:
    """옛 규칙(합격선까지 당기기)을 그대로 돌려 보면 **실제로 잘린다** — 위 검사가 공허하지 않다.

    이 한 줄이 "합격선을 포기한 것"과 "합격선을 지킬 수 있었는데 안 한 것"을 가른다: 당기면 합격선은
    넘지만(`pulled >= minimum`) 그림이 잘린다(`pulled_cut > 0`). 실화면에서 잘리는 것은 자리표시
    다리와 벽면 스트로크였다(검증 실측 김경호 L1: 다리 관절 16개 중 14개, 스트로크 끝점 3개).
    """
    js = (STATIC / "skeleton_view.js").read_text(encoding="utf-8")
    minimum = min_torso_fraction(js)
    _default, _default_cut, pulled, pulled_cut = _default_and_pulled(aspect)
    assert pulled >= minimum - 1e-9, (pulled, minimum)
    assert pulled_cut > 0, "이 장면에서는 당겨도 아무것도 안 잘린다 — 검사 장면을 고쳐야 한다"


@pytest.mark.parametrize("aspect", ASPECTS)
def test_without_recentring_the_asymmetric_stroke_is_pushed_off_screen(aspect: float) -> None:
    """범위 크기 맞춤(옛 규칙)으로 되돌리면 같은 장면에서 점이 화면 밖으로 나간다 —
    위 검사가 공허하지 않다는 증거다."""
    js = (STATIC / "skeleton_view.js").read_text(encoding="utf-8")
    targets, real_modes = framing_targets(js), real_point_modes(js)
    outside = _outside_count(POSE_WITH_STROKE, "pose", targets, real_modes, aspect, reach=())
    assert outside > 0, "이 합성 장면이 치우쳐 있지 않다 — 검사 장면을 고쳐야 한다"


def test_the_reach_rule_leaves_the_torso_preset_alone() -> None:
    """'상체 확대'는 **일부러** 맞춤 대상 밖을 잘라 내고 그 이름을 적는다 — 이 규칙 대상이 아니다.

    기본 뷰(pose)만 "가장 멀리 나간 점"을 눌러 하나도 자르지 않는다.
    """
    js = (STATIC / "skeleton_view.js").read_text(encoding="utf-8")
    assert reach_modes(js) == {"pose"}
