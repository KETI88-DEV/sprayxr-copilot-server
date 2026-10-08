#!/usr/bin/env python3
"""입력 품질 게이트 — 자세를 채점하기 전에 그 골격을 믿을 수 있는지 판정한다.

규칙과 임계는 `docs/superpowers/specs/2026-09-29-sprayxr-runtime-alignment-design.md` §4.3에서
왔고, **임계마다 그것을 정한 실측을 같이 적는다.** 근거 없는 임계는 나중에 누가 "조금 느슨하게"
바꾸고 그 이유가 사라진다.

**가장 중요한 교훈 하나**: 사람 해부 기준을 그대로 쓰면 이 코퍼스가 100% 거부된다.
아바타가 고정 리그이고 `shoulder`가 쇄골 기점이라 어깨 사이가 0.0567 m다. 사람 어깨너비
0.25~0.45 m를 임계로 쓰면 정상 데이터가 전부 죽는다. **임계는 사람이 아니라 리그에 맞춘다.**

판정은 셋이다.

- `reject` — 채점하지 않는다. 점수는 `null`이지 `0.0`이 아니다.
- `degrade` — 채점하되 일부를 빼고, 그 사실을 응답에 싣는다.
- `pass` — 그대로 채점한다.

`info`는 판정을 바꾸지 않고 사실만 남긴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Iterable, Literal, Mapping, Sequence

# `posture_sequence`와 같은 이유다 — 불린 이름 공간에서 형제를 잡는다.
if __package__:
    from . import joint_names
else:
    import joint_names

Vec3 = tuple[float, float, float]
GateStatus = Literal["pass", "degrade", "reject"]

# --- 임계 -------------------------------------------------------------------

# R1: 리그는 강체다. CSV 601세션·작업자 6인에서 19개 뼈 길이가 유일값 1개였고
# 샘플 시퀀스 안 변동은 ≤1e-4 m였다. 둘 중 느슨한 쪽을 쓴다 — 짧은 뼈에서 ±5%가
# 1 mm 아래로 내려가면 부동소수 잡음에 걸린다.
BONE_TOLERANCE_RATIO: Final = 0.05
BONE_TOLERANCE_M: Final = 0.003

# R2: **리그 기준**이다. 실측 어깨 사이 0.0567 m 고정. 사람 해부 기준(0.25~0.45 m)을 쓰면
# 코퍼스 100%가 거부된다.
SHOULDER_WIDTH_MIN_M: Final = 0.045
SHOULDER_WIDTH_MAX_M: Final = 0.070

# R3: CSV-FK 28,335프레임에서 발끝↔머리 min 0.411 · max 1.435, 체인 합 1.4645.
# 1.0 m 하한을 쓰면 47.71%가, 1.5 m 상한을 쓰면 100%가 거부된다.
HEIGHT_MIN_M: Final = 0.35
HEIGHT_MAX_M: Final = 1.50
# 2026-10-08: 아바타가 바뀌어 몸통이 길어졌다(머리↔발끝 선 자세 1.6 m 안팎, 뼈 길이 합 1.73 m). 고정 상한 1.50 m가
# 새 아바타의 정상 선 자세 20/32 시퀀스를 거절했다. 호출자가 **골격이 맞춰진 리그**의 뼈 길이 합(발끝→…→머리, 그 몸의
# 물리적 최댓값)을 알면 `height_max_m`으로 넘긴다 — 같은 프레임에서 잰 뼈 길이로 상한을 만들면 삼각부등식 때문에
# 절대 넘을 수 없어 검사가 무의미해진다. 여유는 `HEIGHT_CHAIN_SLACK`. 모르면 위 상수(예전 리그)를 쓴다.
HEIGHT_CHAIN_SLACK: Final = 1.02

# R4: 프레임 간 변화 폭이 이보다 작으면 "얼었다"고 본다. 실물 샘플의 `head`가 정확히 0.0이다.
FROZEN_RANGE_M: Final = 1e-4
# R4a: 골격이 전부 얼어도 **도구까지 정지**했으면 그냥 멈춘 시퀀스다(샘플 SEQ-002가 그렇다).
# 도구가 이만큼 움직였는데 골격이 얼었을 때만 거부한다.
FROZEN_TOOL_DISPLACEMENT_M: Final = 0.030

# R5: 손과 건 사이. CSV p50 0.519 · max 0.860이고 1.30 m를 넘는 창이 957개 중 **0개**다.
# 샘플 SEQ-002는 0.359로 통과, SEQ-001은 **4.961로 거부**된다(건이 손에 없다).
HAND_TOOL_MIN_M: Final = 0.05
HAND_TOOL_MAX_M: Final = 1.30

# R6: 방향 판정의 주축을 PCA로 잡으려면 점이 셋 필요하다. 자세는 한 프레임으로도 낸다.
MIN_FRAMES_FOR_METRICS: Final = 3
MIN_FRAMES_FOR_POSE: Final = 1

# R7: 샘플 8·9 ms(120FPS), CSV 11.111 ms(90FPS). 둘 다 안에 드는 대역으로 잡는다.
FRAME_INTERVAL_MIN_S: Final = 0.004
FRAME_INTERVAL_MAX_S: Final = 0.040
FRAME_INTERVAL_RATIO_MAX: Final = 3.0

# R8: 오른팔 IK가 퇴화했다. CSV-FK에서 신전 비율 p10 = p50 = 1.000(언제나 완전 신전)이다.
# 거부하지 않는다 — 정상 데이터가 전부 이렇다. 교정에서만 뺀다.
ARM_EXTENSION_RATIO: Final = 0.98

# R1b: 등록 리그 대조의 허용 폭. 샘플 리그가 CSV 리그와 세 뼈에서 다르므로
# (×0.72 · ×1.885 · ×7.48) 어느 쪽이 정본인지 확정 전에는 **거부 근거로 쓰지 않는다**.
RIG_TOLERANCE_RATIO: Final = 0.10

# 뼈 19개. 20관절 사슬이고, `Neck`이 없어 `Head`가 `Chest`에 붙는다.
# 개수가 실측(“19개 뼈 길이가 유일값 1개”)과 맞는 것이 이 사슬의 근거다.
BONES: Final[tuple[tuple[str, str], ...]] = (
    ("Spine", "Pelvis"),
    ("Chest", "Spine"),
    ("Head", "Chest"),
    ("LeftShoulder", "Chest"),
    ("LeftUpperArm", "LeftShoulder"),
    ("LeftForeArm", "LeftUpperArm"),
    ("LeftHand", "LeftForeArm"),
    ("RightShoulder", "Chest"),
    ("RightUpperArm", "RightShoulder"),
    ("RightForeArm", "RightUpperArm"),
    ("RightHand", "RightForeArm"),
    ("LeftThigh", "Pelvis"),
    ("LeftCalf", "LeftThigh"),
    ("LeftFoot", "LeftCalf"),
    ("LeftToe", "LeftFoot"),
    ("RightThigh", "Pelvis"),
    ("RightCalf", "RightThigh"),
    ("RightFoot", "RightCalf"),
    ("RightToe", "RightFoot"),
)

# 이것이 없으면 자세를 정규화할 기준점이 없다.
REQUIRED_JOINTS: Final[tuple[str, ...]] = ("Pelvis", "Chest", "LeftShoulder", "RightShoulder")

# 닫힌 집합. 새 사유는 여기 먼저 넣는다 — 호출부에서 문자열을 지어내면 소비자가 모르는 값을 받는다.
REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "skeleton_coordinates_invalid",
        "skeleton_required_joint_missing",
        "skeleton_bone_length_inconsistent",
        "skeleton_rig_mismatch",
        "skeleton_shoulder_width_out_of_range",
        "skeleton_height_out_of_range",
        "skeleton_all_joints_frozen",
        "skeleton_joint_frozen",
        "tool_not_held",
        "insufficient_frames",
        "frame_interval_irregular",
        "right_arm_degenerate",
    }
)


@dataclass(frozen=True)
class GateFrame:
    """게이트가 보는 프레임 하나. 이름은 이미 정본으로 옮겨져 있어야 한다."""

    joints: Mapping[str, Vec3]
    tool_pos: Vec3 | None = None
    time_s: float | None = None


@dataclass(frozen=True)
class GateResult:
    status: GateStatus
    reason_codes: tuple[str, ...] = ()
    excluded_joints: tuple[str, ...] = ()
    frozen_joints: tuple[str, ...] = ()
    info_codes: tuple[str, ...] = ()
    measurements: Mapping[str, float] = field(default_factory=dict)

    @property
    def scorable(self) -> bool:
        return self.status != "reject"


def _distance(a: Vec3, b: Vec3) -> float:
    return math.dist(a, b)


def _finite(value: Vec3) -> bool:
    return all(math.isfinite(component) for component in value)


def _spread(points: Sequence[Vec3]) -> float:
    """점들 사이 최대 거리. 두 점 미만이면 0이다."""
    if len(points) < 2:
        return 0.0
    return max(
        _distance(points[i], points[j])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    )


def _ranges(frames: Sequence[GateFrame], names: Iterable[str]) -> dict[str, float]:
    return {
        name: _spread([frame.joints[name] for frame in frames if name in frame.joints])
        for name in names
    }


def _scorable_names(frames: Sequence[GateFrame]) -> list[str]:
    seen: set[str] = set()
    for frame in frames:
        seen.update(frame.joints)
    return sorted(name for name in seen if joint_names.is_scorable(name))


def _arm_extension(joints: Mapping[str, Vec3]) -> float | None:
    """어깨→손 직선거리 ÷ 마디 합. 1.0이면 완전히 편 팔이다."""
    upper = joints.get("RightUpperArm")
    fore = joints.get("RightForeArm")
    hand = joints.get("RightHand")
    if upper is None or fore is None or hand is None:
        return None
    chain = _distance(upper, fore) + _distance(fore, hand)
    if chain <= 1e-9:
        return None
    return _distance(upper, hand) / chain


def evaluate(
    frames: Sequence[GateFrame],
    *,
    frozen_before: frozenset[str] = frozenset(),
    registered_bone_lengths: Mapping[str, float] | None = None,
    height_max_m: float | None = None,
) -> GateResult:
    """시퀀스 하나의 입력 품질을 판정한다.

    `frozen_before`는 **같은 세션의 앞선 시퀀스에서 이미 얼었던 관절**이다(R4c).
    두 시퀀스에서 같은 관절이 얼면 영구 제외다 — 한 번은 우연이지만 두 번은 그 관절이
    안 오는 것이다. CSV 교차창에서는 473개 중 0개였다.

    `registered_bone_lengths`는 등록 리그의 뼈 길이다(R1b). **없으면 건너뛴다** —
    샘플 리그가 CSV 리그와 세 뼈에서 다르므로 어느 쪽이 정본인지 확정 전에는 거부
    근거로 쓸 수 없다. 건너뛴 사실도 `info`에 적는다.
    """
    reasons: list[str] = []
    infos: list[str] = []
    excluded: list[str] = []
    measurements: dict[str, float] = {}

    # R6 — 프레임 수. 자세는 한 프레임으로도 내지만 지표는 셋이 필요하다.
    if len(frames) < MIN_FRAMES_FOR_POSE:
        return GateResult("reject", ("insufficient_frames",), measurements={"frames": 0.0})
    measurements["frames"] = float(len(frames))
    if len(frames) < MIN_FRAMES_FOR_METRICS:
        infos.append("insufficient_frames")

    # R0 — 좌표가 수인가, 필수 관절이 있는가.
    for frame in frames:
        for value in frame.joints.values():
            if not _finite(value):
                return GateResult(
                    "reject", ("skeleton_coordinates_invalid",), measurements=measurements
                )
    present: set[str] = set()
    for frame in frames:
        present.update(frame.joints)
    if [name for name in REQUIRED_JOINTS if name not in present]:
        return GateResult(
            "reject", ("skeleton_required_joint_missing",), measurements=measurements
        )

    # R1 — 뼈 길이가 시퀀스 안에서 일정한가. 리그는 강체다.
    worst_drift = 0.0
    for child, parent in BONES:
        lengths = [
            _distance(frame.joints[child], frame.joints[parent])
            for frame in frames
            if child in frame.joints and parent in frame.joints
        ]
        if len(lengths) < 2:
            continue
        low = min(lengths)
        drift = max(lengths) - low
        worst_drift = max(worst_drift, drift)
        if drift > max(BONE_TOLERANCE_M, BONE_TOLERANCE_RATIO * low):
            reasons.append("skeleton_bone_length_inconsistent")
            break
    measurements["bone_drift_m"] = worst_drift

    # R1b — 등록 리그 대조. 확정된 리그가 없으면 건너뛴다.
    if registered_bone_lengths is None:
        infos.append("skeleton_rig_mismatch")
    else:
        first = frames[0].joints
        for child, parent in BONES:
            expected = registered_bone_lengths.get(child)
            if expected is None or child not in first or parent not in first:
                continue
            actual = _distance(first[child], first[parent])
            if expected > 0 and abs(actual - expected) > RIG_TOLERANCE_RATIO * expected:
                reasons.append("skeleton_rig_mismatch")
                break

    # R2 — 어깨 너비. 리그 기준이다.
    widths = [
        _distance(frame.joints["LeftShoulder"], frame.joints["RightShoulder"])
        for frame in frames
        if "LeftShoulder" in frame.joints and "RightShoulder" in frame.joints
    ]
    if widths:
        width = sum(widths) / len(widths)
        measurements["shoulder_width_m"] = width
        if not SHOULDER_WIDTH_MIN_M <= width <= SHOULDER_WIDTH_MAX_M:
            return GateResult(
                "reject",
                tuple(dict.fromkeys([*reasons, "skeleton_shoulder_width_out_of_range"])),
                measurements=measurements,
            )

    # R3 — 키. 발끝이 둘이면 더 멀리 뻗은 쪽을 쓴다(더 펴진 다리가 키에 가깝다).
    # 설계 문서가 "발끝↔머리"라고만 적어 정의를 못 박지 않았다. 이 선택을 검사로 고정한다.
    heights: list[float] = []
    for frame in frames:
        head = frame.joints.get("Head")
        if head is None:
            continue
        toes = [frame.joints[name] for name in ("LeftToe", "RightToe") if name in frame.joints]
        if toes:
            heights.append(max(_distance(head, toe) for toe in toes))
    if heights:
        height = sum(heights) / len(heights)
        measurements["height_m"] = height
        height_max = HEIGHT_MAX_M if height_max_m is None else height_max_m
        measurements["height_max_m"] = height_max
        if not HEIGHT_MIN_M <= height <= height_max:
            return GateResult(
                "reject",
                tuple(dict.fromkeys([*reasons, "skeleton_height_out_of_range"])),
                measurements=measurements,
            )

    # R5 — 건이 손에 있는가. 없으면 그 시퀀스의 자세는 도장 자세가 아니다.
    gaps = [
        _distance(frame.joints["RightHand"], frame.tool_pos)
        for frame in frames
        if frame.tool_pos is not None and "RightHand" in frame.joints
    ]
    if gaps:
        gap = sum(gaps) / len(gaps)
        measurements["hand_tool_gap_m"] = gap
        if not HAND_TOOL_MIN_M <= gap <= HAND_TOOL_MAX_M:
            return GateResult(
                "reject",
                tuple(dict.fromkeys([*reasons, "tool_not_held"])),
                measurements=measurements,
            )

    # R4 — 동결.
    scorable = _scorable_names(frames)
    ranges = _ranges(frames, scorable)
    frozen = tuple(name for name in scorable if ranges.get(name, 0.0) < FROZEN_RANGE_M)
    tool_displacement = _spread([f.tool_pos for f in frames if f.tool_pos is not None])
    measurements["tool_displacement_m"] = tool_displacement

    if scorable and len(frozen) == len(scorable):
        if tool_displacement >= FROZEN_TOOL_DISPLACEMENT_M:
            return GateResult(
                "reject",
                tuple(dict.fromkeys([*reasons, "skeleton_all_joints_frozen"])),
                frozen_joints=frozen,
                measurements=measurements,
            )
        # 도구도 정지했다 — 그냥 멈춰 있는 시퀀스다. 거부하지 않는다.
        infos.append("skeleton_all_joints_frozen")
    elif frozen:
        reasons.append("skeleton_joint_frozen")
        excluded.extend(frozen)

    # R4c — 앞선 시퀀스에서도 얼었던 관절은 영구 제외한다.
    repeat = [name for name in frozen if name in frozen_before]
    if repeat:
        if "skeleton_joint_frozen" not in reasons:
            reasons.append("skeleton_joint_frozen")
        excluded.extend(repeat)

    # R7 — 프레임 간격.
    times = [frame.time_s for frame in frames if frame.time_s is not None]
    if len(times) >= 2:
        steps = [b - a for a, b in zip(times, times[1:]) if b - a > 0]
        if steps:
            low, high = min(steps), max(steps)
            measurements["frame_interval_min_s"] = low
            measurements["frame_interval_max_s"] = high
            if (
                low < FRAME_INTERVAL_MIN_S
                or high > FRAME_INTERVAL_MAX_S
                or high / low > FRAME_INTERVAL_RATIO_MAX
            ):
                reasons.append("frame_interval_irregular")

    # R8 — 오른팔 퇴화. 판정을 바꾸지 않고 사실만 남긴다.
    extensions = [
        value for value in (_arm_extension(frame.joints) for frame in frames) if value is not None
    ]
    if extensions:
        ratio = sum(extensions) / len(extensions)
        measurements["right_arm_extension"] = ratio
        if ratio >= ARM_EXTENSION_RATIO:
            infos.append("right_arm_degenerate")

    return GateResult(
        status="degrade" if reasons else "pass",
        reason_codes=tuple(dict.fromkeys(reasons)),
        excluded_joints=tuple(dict.fromkeys(excluded)),
        frozen_joints=frozen,
        info_codes=tuple(dict.fromkeys(infos)),
        measurements=measurements,
    )
