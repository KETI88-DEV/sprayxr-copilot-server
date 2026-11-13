#!/usr/bin/env python3
"""표시용 오른팔: 오른어깨에서 **건 손잡이**까지 두 마디 IK. 채점은 이 경로를 타지 않는다.

왜 CSV 오른팔 관절을 버리는가 — 두 차례 사전등록 측정(17세션 · 작업자 6 · 레벨 1/3/6 · 분사 프레임
90,555)이 그것이 데이터라고 확정했다. 시뮬레이터(벤더)의 오른팔 솔버는 트래커 거리와 무관하게 팔꿈치를
약 91% 편 채 고정하고(트래커가 팔 길이 밖 91.5% · 안 90.7%), 팔 방향은 트래커를 겨누지 않는다(상완→손
대 상완→트래커 81.3°, 5초 밀린 대조군 81.7°). FK 규약 대안 1,338개 중 지금 FK가 1등이었다 — 우리
코드가 아니다. 실측 건은 몸 오른쪽에 77.7%인데 FK 오른손은 29.5%다. 그래서 팔은 **실측 건**으로 세운다.

손→트래커 ~0.6m 장착 오프셋도 쓰지 않는다: 분사 축에서 68° 틀어져 있고 길이가 세션마다 SD 0.127m로
흔들려 분사구 길이가 아니다. 하드웨어(사용자 제공)는 트래커가 **분사구에** 달려 있다는 것이다(끝에서 약
9cm 뒤 — `level_mesh.L_MUZZLE_M`). 그러면 손잡이는 트래커에서 분사 축을 따라 뒤로 간 자리다:

    손잡이 = 트래커 − GRIP_BEHIND_TRACKER_M · 건 전방      (건 전방 = 건 로컬 +Y, `measured_pose` 규약)

손 방향은 건의 실측 회전이다 — 리그에 손 너머 관절이 없으므로 화면에서는 손잡이→트래커 **건 막대**의
방향이 곧 손 방향이다. 뼈 길이는 리그 바인드 오프셋 그대로(상완 = `RightForeArm` 오프셋, 전완 =
`RightHand` 오프셋)이고, 손잡이에 닿지 않으면(사전 확인에서 약 15% 프레임) 팔을 손잡이 쪽으로 곧게
편 채 두고 손끝→손잡이 사이를 비워 둔다 — 화면이 그 사이를 점선으로 잇는다. 늘려서 닿게 하지 않는다.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import measured_pose  # noqa: E402
import simulator_csv  # noqa: E402
import world_axes  # noqa: E402

Vec3 = tuple[float, float, float]

# 트래커(분사구에 달림)에서 손잡이까지 분사 축을 따라 뒤로 간 거리(m). **측정값이 아니다** — 사용자가
# 고른 추정값이다("추정값 0.3m로 먼저 진행"). 데이터로는 정해지지 않는다: 사전등록 밖 확인에서 L = 0.2~0.5m
# 사이 손잡이의 팔 길이 안 비율이 83~85%로 거의 같았고(0.3m에서 85.1%), 몸 오른쪽 비율은 78.0%로 실측 건
# 77.7%와 맞았다. 한길숏건 손잡이→트래커 실측 치수를 받으면 이 한 값을 그 치수로 바꾼다.
GRIP_BEHIND_TRACKER_M: Final = 0.30

# 팔 사슬(리그 이름). 어깨 = 상완 관절 위치, 팔꿈치 = 전완 관절, 손 = 손 관절.
SHOULDER: Final = "RightUpperArm"
ELBOW: Final = "RightForeArm"
HAND: Final = "RightHand"
# 바깥쪽을 정하는 반대편 어깨.
OTHER_SHOULDER: Final = "LeftUpperArm"

# 팔꿈치가 향하는 쪽(폴). 건을 앞으로 든 오른팔의 팔꿈치는 **아래로, 조금 바깥으로** 처진다 — 아래
# 1 : 바깥 0.5로 섞는다. 아래가 주인 이유: 어깨를 들어 팔꿈치를 옆으로 벌린 자세는 분사 중에 오래
# 버티지 않는다. 바깥이 조금 섞이는 이유: 팔꿈치가 몸통 안쪽(가슴 앞)으로 파고드는 해는 해부학적으로
# 드물다. 바깥은 리그의 **왼어깨→오른어깨** 수평 성분이다(외적을 쓰지 않아 좌표계 손대칭과 무관).
POLE_DOWN_WEIGHT: Final = 1.0
POLE_OUT_WEIGHT: Final = 0.5

# 길이가 이보다 짧으면 방향이 정의되지 않는다.
_EPS: Final = 1e-9


@dataclass(frozen=True, slots=True)
class RightArm:
    shoulder: Vec3
    elbow: Vec3
    hand: Vec3
    grip: Vec3
    tracker: Vec3
    gun_forward: Vec3
    reached: bool
    upper_arm_m: float
    forearm_m: float

    def as_view(self) -> dict[str, object]:
        """화면·검사가 읽는 모양. 닿지 못했으면 손끝→손잡이 점선 두 끝점(`reach_gap`)을 싣는다."""
        return {
            "shoulder": list(self.shoulder),
            "elbow": list(self.elbow),
            "hand": list(self.hand),
            "grip": list(self.grip),
            "tracker": list(self.tracker),
            "gun_forward": list(self.gun_forward),
            "reached": self.reached,
            "upper_arm_m": self.upper_arm_m,
            "forearm_m": self.forearm_m,
            "grip_behind_tracker_m": GRIP_BEHIND_TRACKER_M,
            "reach_gap": None if self.reached else {"from": list(self.hand), "to": list(self.grip)},
        }


def _tuple(v: np.ndarray) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < _EPS else v / n


def gun_forward(row: Mapping[str, str], gun: str) -> Vec3 | None:
    """건 전방(분사 축) — `measured_pose`의 회전 규약·로컬 축 그대로. 회전 컬럼이 없으면 None."""
    try:
        angles = simulator_csv.vec3(dict(row), gun, "Rotation")
    except (KeyError, ValueError):
        return None
    return measured_pose.forward_vector((angles[0], angles[1], angles[2]), measured_pose.GUN_FORWARD)


def grip_point(tracker: Sequence[float], forward: Sequence[float]) -> Vec3:
    """손잡이 = 트래커 − `GRIP_BEHIND_TRACKER_M` · 건 전방."""
    t = np.asarray(tracker, dtype=float)
    f = _unit(np.asarray(forward, dtype=float))
    if f is None:
        raise ValueError("grip_point: 건 전방이 영벡터다")
    return _tuple(t - GRIP_BEHIND_TRACKER_M * f)


def pole_direction(joints: Mapping[str, Sequence[float]]) -> np.ndarray:
    """팔꿈치 폴 = 아래 · `POLE_DOWN_WEIGHT` + 바깥 · `POLE_OUT_WEIGHT`(위 상수 주석)."""
    down = -np.asarray(world_axes.UP, dtype=float)
    pole = POLE_DOWN_WEIGHT * down
    if SHOULDER in joints and OTHER_SHOULDER in joints:
        across = np.asarray(world_axes.flatten(tuple(np.asarray(joints[SHOULDER], dtype=float)
                                                     - np.asarray(joints[OTHER_SHOULDER], dtype=float))))
        out = _unit(across)
        if out is not None:
            pole = pole + POLE_OUT_WEIGHT * out
    return pole


def solve_two_bone(
    shoulder: Sequence[float],
    target: Sequence[float],
    upper_m: float,
    fore_m: float,
    pole: Sequence[float],
) -> tuple[Vec3, Vec3, bool]:
    """(팔꿈치, 손, 닿았는가). 뼈 길이는 정확히 `upper_m`·`fore_m`다.

    닿으면 손 = 목표이고 팔꿈치는 어깨→목표 축에서 폴 쪽으로 굽는다. 못 닿으면 목표 쪽으로 곧게 편다.
    너무 가까워(|두 뼈 차| 미만) 접어도 못 맞추면 가장 접힌 자세로 목표 방향에 둔다 — 늘리지도 줄이지도 않는다.
    """
    s = np.asarray(shoulder, dtype=float)
    g = np.asarray(target, dtype=float)
    p = np.asarray(pole, dtype=float)
    to_target = g - s
    distance = float(np.linalg.norm(to_target))
    u = _unit(to_target)
    if u is None:
        u = _unit(p)
        if u is None:
            u = -np.asarray(world_axes.UP, dtype=float)
    if distance >= upper_m + fore_m:
        elbow = s + upper_m * u
        hand = s + (upper_m + fore_m) * u
        return _tuple(elbow), _tuple(hand), False
    reached = distance >= abs(upper_m - fore_m)
    d = max(distance, abs(upper_m - fore_m), _EPS)
    along = (upper_m**2 - fore_m**2 + d**2) / (2.0 * d)
    height = math.sqrt(max(upper_m**2 - along**2, 0.0))
    side = _unit(p - float(np.dot(p, u)) * u)
    if side is None:
        # 폴이 팔 축과 평행하다(팔이 정확히 아래를 향함) — 수평 한 축을 대신 쓴다.
        for axis in np.eye(3):
            side = _unit(axis - float(np.dot(axis, u)) * u)
            if side is not None:
                break
    assert side is not None
    elbow = s + along * u + height * side
    hand = s + d * u
    return _tuple(elbow), _tuple(hand), reached


def rig_lengths(offsets: Mapping[str, Sequence[float]]) -> tuple[float, float] | None:
    """(상완, 전완) 길이 — 리그 바인드 오프셋의 노름. 없으면 None(리터럴로 지어내지 않는다)."""
    if ELBOW not in offsets or HAND not in offsets:
        return None
    return math.hypot(*offsets[ELBOW]), math.hypot(*offsets[HAND])


def right_arm(
    joints: Mapping[str, Sequence[float]],
    offsets: Mapping[str, Sequence[float]],
    row: Mapping[str, str],
    gun: str,
    tracker: Sequence[float] | None,
) -> RightArm | None:
    """배치된 몸(`joints`, 월드)의 오른어깨에서 이 프레임 건 손잡이까지 세운 팔. 재료가 없으면 None."""
    lengths = rig_lengths(offsets)
    forward = gun_forward(row, gun)
    if lengths is None or forward is None or tracker is None or SHOULDER not in joints:
        return None
    unit_forward = _unit(np.asarray(forward, dtype=float))
    if unit_forward is None:
        return None
    grip = grip_point(tracker, unit_forward)
    upper_m, fore_m = lengths
    shoulder = _tuple(np.asarray(joints[SHOULDER], dtype=float))
    elbow, hand, reached = solve_two_bone(shoulder, grip, upper_m, fore_m, pole_direction(joints))
    return RightArm(
        shoulder=shoulder,
        elbow=elbow,
        hand=hand,
        grip=grip,
        tracker=_tuple(np.asarray(tracker, dtype=float)),
        gun_forward=_tuple(unit_forward),
        reached=reached,
        upper_arm_m=upper_m,
        forearm_m=fore_m,
    )
