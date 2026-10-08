"""교정 시연 — "잘못된 자세 → 교정 안내 → 교정된 결과"를 **사람 몸**으로 한 흐름에 보이는 화면의 자료.

KPI 시험(`docs/year2/`, `docs/year3/`)의 시연 화면(`static/demo.html`)이 쓴다. 숫자는 KPI 측정과 **같은
계산**에서 나온다: `service.compare(correction_policies=...)`를 한 번 부르고, 그 안에서 정책이 받은
궤적·산포·기저와 고른 교정을 그대로 받는다. 화면용으로 다시 채점하지 않는다.

교정 방식(`MODES`)
- `issued`(화면 기본): 교정 알고리즘이 발부한 안내 전부(`issued_axes`, 관절 값 ≤ 4). KPI 정확도의 정의다.
- `delivered`(숨김): 작업자에게 한 번에 전달되는 안내 1건 — 메인 화면의 "예상 결과"와 같다.
- 3차년도 방식은 `trainer/year3/demo_mode.py`가 **숨김**으로 등록한다. 숨김 방식은 화면 목록에 나오지 않고
  주소(`demo.html?mode=…`)로만 열린다.

몸: 8130 3D 뷰와 같은 골격(`view.layers.*.joints`; 다리 추정이 그럴듯하지 않으면 서버가 세운 자리표시 다리)과
살 두께(`view.flesh`)를 쓴다. 교정된 몸은 `corrected_body`가 만든다.
- 손 안내: 손이 **원래 손 + 안내 이동**(채점과 같은 이동)으로 가고, 어깨는 몸을 따라가며, 팔꿈치는 위팔·아래팔
  길이를 지켜 다시 푼다.
- 머리(자세) 안내: 수평 성분은 몸 전체가 옮겨 서는 것, 수직 성분은 발을 디딘 채 골반 위가 오르내리는 것으로
  그린다(무릎은 다리 길이를 지켜 다시 푼다).
- 안내가 없는 손은 몸을 따라간다.
- 어떤 마디도 늘리지 않는다. 손이 팔 길이로 닿지 않으면 쇄골 길이를 지킨 채 어깨 관절을 손 쪽으로 돌리고(최대 35°),
  그래도 모자라면 손이 못 미친 채로 두고 그 거리를 `reach_short_cm`·`reach_note`로 낸다.
- **발은 뜨지 않는다.** 자세 높이기·낮추기는 두 다리가 다 펴지거나 다 접히는 데서 멈추고(몸이 공중에 뜰 때까지 허리를
  올리지 않는다, 사용자 2026-10-01), 못 미친 높이를 같은 방식으로 적는다.
목표 자세의 몸은 작업자 몸에서 채점하는 세 점의 평균 이탈을 **전부** 메운 몸이다(`full_shift_corrections`) —
채점이 보는 목표와 같은 자리다. 그래서 교정이 메우지 않은 값만큼 교정된 몸과 목표 몸 사이가 남는다.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
measured_score = importlib.import_module("measured_score")
world_axes = importlib.import_module("world_axes")

from trainer.pose_lab import instructions, service  # noqa: E402

# 교정 0%→100%를 몇 단계로 나눠 보일지. 단계마다 실제로 채점하고 몸도 그 비율로 만든다(보간한 숫자가 아니다).
SCORE_STEPS: Final = 20
POINT_LABELS: Final = {"right": "오른손(건)", "head": "머리(HMD)", "left": "왼손"}
AXIS_LABELS: Final = {"wall": "벽 쪽", "vertical": "위아래", "lateral": "진행 방향"}
# 안내가 가리키는 몸의 관절(화살표의 뿌리).
ANCHOR_JOINT: Final = {"right": "RightHand", "left": "LeftHand", "head": "Head"}
ARM_CHAINS: Final = {"right": ("RightUpperArm", "RightForeArm", "RightHand"),
                     "left": ("LeftUpperArm", "LeftForeArm", "LeftHand")}
# 팔이 닿지 않을 때 어깨 관절(UpperArm)을 쇄골 뿌리(Shoulder) 둘레로 손 쪽으로 돌린다 — 어깨를 내미는 동작.
# 쇄골 길이는 그대로다. 사람 어깨가 움직이는 범위 안에서만 돌린다.
CLAVICLE_ROOT: Final = {"right": "RightShoulder", "left": "LeftShoulder"}
MAX_SHOULDER_SWING_DEG: Final = 35.0
LEG_CHAINS: Final = (("LeftThigh", "LeftCalf", "LeftFoot"), ("RightThigh", "RightCalf", "RightFoot"))
# 자세를 높이거나 낮출 때 바닥에 남는 관절(발목 아래). 무릎은 다시 푼다.
PLANTED: Final = frozenset({"LeftFoot", "LeftToe", "RightFoot", "RightToe"})
KNEES: Final = frozenset({"LeftCalf", "RightCalf"})
LEG_JOINTS: Final = frozenset({"LeftThigh", "LeftCalf", "LeftFoot", "LeftToe",
                               "RightThigh", "RightCalf", "RightFoot", "RightToe"})

Vec = tuple[float, float, float]
Policy = Callable[..., Sequence[Any]]


def issued_policy(query, reference, spreads, weights, frame, deltas):  # noqa: ANN001, ANN201
    """교정 알고리즘이 발부한 안내 전부(관절 값 ≤ 4). KPI 정확도의 정의."""
    return measured_score.issued_axes(deltas, frame)


def delivered_policy(query, reference, spreads, weights, frame, deltas):  # noqa: ANN001, ANN201
    """작업자에게 한 번에 전달되는 안내(σ 상위 `DELIVERED_CUES`건)."""
    return measured_score.issued_axes(deltas, frame)[: measured_score.DELIVERED_CUES]


# 이름 → (설명, 정책, 숨김). 숨김 방식은 화면 목록에 나오지 않고 주소로만 열린다.
MODES: dict[str, tuple[str, Policy, bool]] = {
    "issued": ("교정 알고리즘 안내", issued_policy, False),
    "delivered": ("작업자에게 한 번에 전달되는 안내 1건", delivered_policy, True),
}
DEFAULT_MODE: Final = "issued"


def register_mode(name: str, label: str, policy: Policy, *, hidden: bool = True) -> None:
    MODES[name] = (label, policy, hidden)


def visible_modes() -> list[dict[str, str]]:
    return [{"mode": name, "label": label} for name, (label, _policy, hidden) in MODES.items() if not hidden]


# --- 벡터 ---------------------------------------------------------------------


def _add(a: Sequence[float], b: Sequence[float], k: float = 1.0) -> Vec:
    return (a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2])


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _scale(a: Sequence[float], k: float) -> Vec:
    return (a[0] * k, a[1] * k, a[2] * k)


def solve_two_bone(root: Vec, pole: Vec, target: Vec, upper: float, lower: float) -> tuple[Vec, Vec]:
    """뿌리를 고정한 2마디 사슬(어깨-팔꿈치-손목, 엉덩이-무릎-발목)의 끝을 `target`에 맞춘다.

    마디 길이(`upper`, `lower`)는 지킨다. 굽는 쪽은 `pole`(원래 가운데 관절이 있던 쪽)이다. 닿지 못하는 거리면
    곧게 펴서 목표 쪽으로 최대한 뻗는다. (가운데 관절, 끝)을 돌려준다.
    """
    reach = _sub(target, root)
    distance = _norm(reach)
    if distance < 1e-9 or upper < 1e-9 or lower < 1e-9:
        return pole, target
    direction = _scale(reach, 1.0 / distance)
    distance = min(max(distance, abs(upper - lower) + 1e-6), upper + lower - 1e-6)
    along = (upper * upper - lower * lower + distance * distance) / (2.0 * distance)
    height = math.sqrt(max(upper * upper - along * along, 0.0))
    bend = _sub(pole, root)
    side = _sub(bend, _scale(direction, _dot(bend, direction)))
    if _norm(side) < 1e-9:
        helper = (0.0, 1.0, 0.0) if abs(direction[1]) < 0.9 else (1.0, 0.0, 0.0)
        side = _sub(helper, _scale(direction, _dot(helper, direction)))
    side = _scale(side, 1.0 / _norm(side))
    return _add(_add(root, direction, along), side, height), _add(root, direction, distance)


def point_shifts(corrections: Sequence[Any], fraction: float = 1.0) -> dict[str, Vec]:
    """점별 월드 이동(m). `measured_score.corrections_shifts`와 같은 합이되 기저로 되돌리지 않는다."""
    shifts: dict[str, Vec] = {}
    for correction in corrections:
        move = _scale(correction.unit, correction.magnitude_cm / 100.0 * fraction)
        shifts[correction.point] = _add(shifts.get(correction.point, (0.0, 0.0, 0.0)), move)
    return shifts


UNCUED_HANDS: Final = ("follow", "stay")


def corrected_body(
    joints: Mapping[str, Sequence[float]], corrections: Sequence[Any], fraction: float = 1.0, uncued_hands: str = "follow"
) -> dict[str, Vec]:
    """교정을 `fraction`만큼 따랐을 때의 몸(월드 좌표, 유니티 Y-up). 원래 몸은 바꾸지 않는다.

    `uncued_hands`: 안내가 없는 손을 어떻게 둘지. `follow`(기본, 시연 화면)는 몸을 따라간다. `stay`는 벽 기준 원래
    자리를 지키려 하고(건을 같은 곳에 겨눈 채 옮겨 서기), 팔이 닿지 않으면 닿는 데까지만 간다 — 몸 제약 측정의
    민감도 분석용이다.
    """
    if uncued_hands not in UNCUED_HANDS:
        raise ValueError(f"uncued_hands must be one of {UNCUED_HANDS}, got {uncued_hands!r}")
    original: dict[str, Vec] = {name: (float(p[0]), float(p[1]), float(p[2])) for name, p in joints.items()}
    shifts = point_shifts(corrections, fraction)
    up = tuple(float(v) for v in world_axes.UP)
    head = shifts.get("head", (0.0, 0.0, 0.0))
    vertical = _scale(up, _dot(head, up))
    horizontal = _sub(head, vertical)

    # 1) 옮겨 서기 — 몸 전체가 수평으로 간다.
    body = {name: _add(p, horizontal) for name, p in original.items()}
    # 2) 자세 높이기·낮추기 — 발은 디딘 채 골반 위가 오르내리고 무릎을 다시 푼다. **발은 절대 뜨지 않는다**:
    #    두 다리가 다 펴지거나(올릴 때) 다 접히는(낮출 때) 데서 멈춘다. 못 미친 높이는 `point_shortfall_cm`이 낸다.
    vertical = _scale(vertical, _feasible_vertical_fraction(body, original, vertical))
    if _norm(vertical) > 0.0:
        for name in body:
            if name not in PLANTED and name not in KNEES:
                body[name] = _add(body[name], vertical)
        for hip, knee, ankle in LEG_CHAINS:
            if hip in body and knee in body and ankle in body:
                body[knee], _ = solve_two_bone(
                    body[hip], _add(original[knee], horizontal), body[ankle],
                    _norm(_sub(original[knee], original[hip])), _norm(_sub(original[ankle], original[knee])),
                )
    # 3) 손 — **안내가 있는 손만** 안내가 말한 곳(원래 손 + 안내 이동)으로 보낸다. 안내가 없는 손은 1·2에서 몸을
    #    따라 옮겨진 그대로다(원래 자리로 되돌리면 팔이 닿지 않아 몸이 늘어난다). 팔 길이로 닿지 않으면 쇄골 길이를
    #    지킨 채 어깨 관절을 손 쪽으로 돌리고, 그래도 모자라면 손을 그만큼 못 미친 채 둔다 — 어떤 마디도 늘리지 않는다.
    for point, (shoulder, elbow, hand) in ARM_CHAINS.items():
        if not all(name in body for name in (shoulder, elbow, hand)):
            continue
        if point not in shifts and uncued_hands == "follow":
            continue
        target = _add(original[hand], shifts.get(point, (0.0, 0.0, 0.0)))
        upper = _norm(_sub(original[elbow], original[shoulder]))
        lower = _norm(_sub(original[hand], original[elbow]))
        moved_shoulder = body[shoulder]
        root = CLAVICLE_ROOT[point]
        if _norm(_sub(target, body[shoulder])) > upper + lower and root in body:
            body[shoulder] = _swing_toward(body[root], body[shoulder], target, MAX_SHOULDER_SWING_DEG)
        body[elbow], body[hand] = solve_two_bone(
            body[shoulder], _add(original[elbow], _sub(moved_shoulder, original[shoulder])), target, upper, lower,
        )
    return body


# 다리를 끝까지 펴거나 접지 않는다 — 무릎이 완전히 펴진 채 서는 것도, 완전히 포개지는 것도 사람 자세가 아니다.
LEG_REACH_MAX: Final = 0.995
LEG_FOLD_MIN: Final = 0.35


def _legs_feasible(body: Mapping[str, Vec], original: Mapping[str, Vec], raise_by: Vec) -> bool:
    for hip, knee, ankle in LEG_CHAINS:
        if hip not in body or knee not in body or ankle not in body:
            continue
        thigh = _norm(_sub(original[knee], original[hip]))
        shin = _norm(_sub(original[ankle], original[knee]))
        span = _norm(_sub(_add(body[hip], raise_by), body[ankle]))
        if span > (thigh + shin) * LEG_REACH_MAX or span < max(abs(thigh - shin), (thigh + shin) * LEG_FOLD_MIN):
            return False
    return True


def _feasible_vertical_fraction(body: Mapping[str, Vec], original: Mapping[str, Vec], vertical: Vec) -> float:
    """발을 디딘 채 골반을 `vertical`만큼 옮길 수 있는 비율(0–1). 다 되면 1, 몸이 떠야 하면 그 전에서 멈춘다."""
    if _norm(vertical) == 0.0 or _legs_feasible(body, original, vertical):
        return 1.0
    if not _legs_feasible(body, original, (0.0, 0.0, 0.0)):
        # 원래 자세부터 다리 범위 밖(추정 다리가 어긋남) — 위아래로 옮기지 않는다.
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = (low + high) / 2.0
        if _legs_feasible(body, original, _scale(vertical, middle)):
            low = middle
        else:
            high = middle
    return low


def _swing_toward(root: Vec, joint: Vec, target: Vec, max_deg: float) -> Vec:
    """`joint`를 `root` 둘레로(거리 그대로) `target` 쪽으로 최대 `max_deg`만큼 돌린다."""
    arm = _sub(joint, root)
    length = _norm(arm)
    want = _sub(target, root)
    if length < 1e-9 or _norm(want) < 1e-9:
        return joint
    a = _scale(arm, 1.0 / length)
    b = _scale(want, 1.0 / _norm(want))
    angle = math.acos(max(-1.0, min(1.0, _dot(a, b))))
    if angle < 1e-9:
        return joint
    turn = min(angle, math.radians(max_deg))
    # a에서 b 쪽으로 평면 안에서 `turn`만큼 돈 방향.
    side = _sub(b, _scale(a, _dot(a, b)))
    side = _scale(side, 1.0 / _norm(side))
    direction = _add(_scale(a, math.cos(turn)), side, math.sin(turn))
    return _add(root, direction, length)


def point_shortfall_cm(joints: Mapping[str, Sequence[float]], body: Mapping[str, Vec], corrections: Sequence[Any], point: str) -> float:
    """안내한 위치에 몸의 그 부위(손·머리)가 못 미친 거리(cm). 닿았으면 0."""
    joint = ANCHOR_JOINT.get(point)
    if joint is None or joint not in joints or joint not in body:
        return 0.0
    target = _add(joints[joint], point_shifts(corrections).get(point, (0.0, 0.0, 0.0)))
    return _norm(_sub(body[joint], target)) * 100.0


SHORTFALL_NOTE: Final = {"right": "팔 길이로 {cm}cm 못 미칩니다", "left": "팔 길이로 {cm}cm 못 미칩니다",
                         "head": "발을 디딘 채로는 {cm}cm 못 미칩니다"}


def full_shift_corrections(deltas: Sequence[Any], frame: Any) -> list[Any]:
    """세 점의 평균 이탈을 **전부** 메우는 이동(점마다 하나). 채점이 보는 목표 위치다.

    점의 세 축 교정(벽 쪽·위아래·진행)을 합친 것과 같다 — 세 축이 직교하고 합이 월드 이탈이다.
    """
    out = []
    for delta in deltas:
        world = measured_score._world_delta(delta.local_delta, frame)
        length = _norm(world)
        if length <= 0.0:
            continue
        out.append(measured_score.AxisCorrection(
            point=delta.point, axis="vertical", magnitude_cm=length * 100.0, direction="",
            sigma=float(delta.sigma), unit=_scale(world, 1.0 / length),
        ))
    return out


# --- 몸 자료 ---------------------------------------------------------------------


def _skeleton(layer: Mapping[str, Any] | None, fit: Mapping[str, Any] | None) -> dict[str, Vec] | None:
    """화면 3D 뷰와 같은 몸: 다리 추정이 그럴듯하지 않으면 서버가 세운 자리표시 다리로 바꾼다."""
    joints = (layer or {}).get("joints")
    if not isinstance(joints, Mapping) or not joints:
        return None
    body = {name: (float(p[0]), float(p[1]), float(p[2])) for name, p in joints.items()}
    legs = (fit or {}).get("legs") if isinstance(fit, Mapping) else None
    if isinstance(legs, Mapping) and not legs.get("plausible") and isinstance(legs.get("placeholder_joints"), Mapping):
        for name, p in legs["placeholder_joints"].items():
            body[name] = (float(p[0]), float(p[1]), float(p[2]))
    return body


def _basis(source: Any, ref: Any) -> Any:
    frames, _inputs = service.pass_key_inputs(source.load(ref))
    return service.representative_basis(frames)


def _round(body: Mapping[str, Vec]) -> dict[str, list[float]]:
    return {name: [round(v, 4) for v in p] for name, p in body.items()}


def cue_text(correction: Any) -> str:
    try:
        text, _mode = instructions.axis_text(correction, None)
        return text
    except ValueError:
        return _generic_text(correction)


def _generic_text(correction: Any) -> str:
    """제품 문장표에 없는 점(왼손)용. 오른손 문장과 같은 어휘를 쓴다."""
    label = POINT_LABELS.get(correction.point, correction.point)
    amount = f"{abs(correction.magnitude_cm):.1f}"
    direction = correction.direction
    if correction.axis == "wall":
        body = f"벽에서 {amount}cm 띄우세요." if direction == "띄우" else f"벽 쪽으로 {amount}cm 붙이세요."
    elif correction.axis == "vertical":
        body = f"{amount}cm {'올리' if direction == '올리' else '내리'}세요."
    elif direction:
        body = f"진행 방향 {direction} {amount}cm 옮기세요."
    else:
        body = f"벽을 따라 {amount}cm 옮기세요."
    return f"{label}을 {body}"


def _percent(value: float | None) -> str:
    return instructions.DASH if value is None else f"{value * 100:.1f}%"


def _ref_json(ref: Any) -> dict[str, Any]:
    return asdict(ref) if is_dataclass(ref) else dict(ref._asdict())


def build_demo(source: Any, query: Any, reference: Any, reference_spreads: Any, mode: str = DEFAULT_MODE) -> dict[str, Any]:
    if mode not in MODES:
        return {"status": "unavailable", "reason": "unknown_mode"}
    label, policy, hidden = MODES[mode]
    captured: dict[str, Any] = {}

    def capture(query_track, reference_track, spreads, weights, frame, deltas):  # noqa: ANN001, ANN202
        chosen = list(policy(query_track, reference_track, spreads, weights, frame, deltas))
        # 첫 호출이 same 쪽(전체 산포)이다. 화면은 목표 자세 하나에 대고 보이므로 이쪽을 그린다.
        captured.setdefault("same", (query_track, reference_track, spreads, weights, frame, chosen))
        return chosen

    body = service.compare(
        source, query, reference, reference_spreads=reference_spreads,
        include_view=True, correction_policies={"demo": capture},
    )
    if body.get("status") != "ok":
        return {"status": "unavailable", "reason": body.get("reason")}
    view = body.get("view") or {}
    layers = view.get("layers") or {}
    worker = _skeleton(layers.get("query"), view.get("fit"))
    if worker is None:
        return {"status": "unavailable", "reason": "no_skeleton"}
    query_basis = _basis(source, query)

    result = body["measured"]["policies"]["demo"]
    query_track, reference_track, spreads, weights, frame, chosen = captured["same"]
    # 목표 몸 = 작업자 몸에서 세 점의 평균 이탈을 전부 메운 몸. 숙련자의 몸을 다른 자리에서 옮겨 오면
    # 한 순간의 골격과 패스 평균이 어긋나 채점이 보는 목표와 다른 곳에 선다.
    deltas = measured_score.score_track(query_track, reference_track, spreads, weights, frame).deltas
    target = corrected_body(worker, full_shift_corrections(deltas, frame))
    scores = [
        measured_score.score_track(
            measured_score.apply_corrections(query_track, chosen, frame, fraction=k / SCORE_STEPS),
            reference_track, spreads, weights, frame,
        ).score
        for k in range(SCORE_STEPS + 1)
    ]
    steps = [corrected_body(worker, chosen, k / SCORE_STEPS) for k in range(SCORE_STEPS + 1)]

    cues = []
    for n, correction in enumerate(chosen, start=1):
        anchor = ANCHOR_JOINT[correction.point]
        before = corrected_body(worker, chosen[: n - 1])
        after = corrected_body(worker, chosen[:n])
        # 화살표는 **작업자의 원래 몸** 위 그 부위에서, 그 부위의 앞선 안내 뒤에 이어 그린다 — 안내는 원래 위치를 기준으로
        # 말하기 때문이다(앞의 "옮겨 서세요"로 몸을 따라 움직인 손에서 그리면 엉뚱한 곳을 가리킨다). 길이는 문장의 cm다.
        earlier = [c for c in chosen[: n - 1] if c.point == correction.point]
        start = _add(worker[anchor], point_shifts(earlier).get(correction.point, (0.0, 0.0, 0.0)))
        end = _add(start, _scale(correction.unit, correction.magnitude_cm / 100.0))
        # 몸이 안내한 자리에 못 미친 거리. 같은 부위의 앞선 안내에서 이미 생긴 부족분은 빼서 한 번만 센다.
        short = point_shortfall_cm(worker, after, chosen[:n], correction.point)
        if earlier:
            short -= point_shortfall_cm(worker, before, chosen[: n - 1], correction.point)
        short = max(0.0, short)
        cues.append({
            "n": n,
            "point": correction.point,
            "axis": correction.axis,
            "point_label": POINT_LABELS.get(correction.point, correction.point),
            "axis_label": AXIS_LABELS[correction.axis],
            "text": cue_text(correction),
            "cm": abs(correction.magnitude_cm),
            "cm_display": f"{abs(correction.magnitude_cm):.1f}cm",
            "anchor": anchor,
            "from": [round(v, 4) for v in start],
            "to": [round(v, 4) for v in end],
            "reach_short_cm": round(short, 2),
            "reach_note": SHORTFALL_NOTE[correction.point].format(cm=f"{short:.1f}") if short >= 1.0 else None,
        })

    improvement = result["improvement"]
    legs = (view.get("fit") or {}).get("legs") or {}
    return {
        "status": "ok",
        "mode": mode,
        "label": label,
        "hidden": hidden,
        "query": _ref_json(query),
        "reference": _ref_json(reference),
        "values": len(chosen),
        "cues": cues,
        "bones": (layers.get("query") or {}).get("bones") or [],
        # 작업자 몸이 해부학적으로 맞지 않으면(`service.posture_check`) 시연도 "추정 자세"로 알린다 — 목표 몸도 이 몸에서 나온다.
        "posture": (view.get("fit") or {}).get("posture"),
        "posture_note": (view.get("fit") or {}).get("posture_note"),
        "flesh": view.get("flesh"),
        "bodies": {"worker": _round(worker), "target": _round(target), "steps": [_round(s) for s in steps]},
        "context": {
            "hit": [round(v, 4) for v in query_basis.origin],
            "wall_normal": [round(v, 4) for v in query_basis.normal],
            "stroke_tangent": [round(v, 4) for v in query_basis.tangent],
            "floor": legs.get("floor_vertical"),
        },
        "score_before": scores[0],
        "score_after": scores[-1],
        "improvement": improvement.get("same"),
        "improvement_split": improvement.get("split"),
        "display": {
            "score_steps": [instructions.score_display(s) for s in scores],
            "score_before": instructions.score_display(scores[0]),
            "score_after": instructions.score_display(scores[-1]),
            "improvement": _percent(improvement.get("same")),
            "improvement_split": _percent(improvement.get("split")),
            "values": f"관절 값 {len(chosen)}개",
        },
    }
