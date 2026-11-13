#!/usr/bin/env python3
"""자세 시퀀스 — 시퀀스 하나를 자세 한 점으로 줄이고, 세션 안에서 그 점들을 쌓는다.

## 왜 시퀀스당 한 점인가

측정이 그렇게 말한다. **시퀀스 하나 안에서 몸은 고정이다.** 학습모드 코퍼스 77,708패스에서
패스 안 머리 높이 변화 폭이 중앙 0.008 m인데 같은 패스의 착탄 높이는 0.152 m 움직인다.
착탄을 따라가는 기울기가 +0.016이다. 실물 SIM 샘플에서는 더 뚜렷해서 두 시퀀스 모두
`head` 변화 폭이 **정확히 0.0**이다.

그래서 시퀀스 하나를 자세 스칼라 하나로 줄이는 것은 정보를 버리는 것이 아니라
**충실한 요약**이다. 움직이는 것은 몸이 아니라 팔·건이다.

## 자세 시퀀스는 시퀀스들을 가로질러 생긴다

시퀀스를 가로지르면 이야기가 달라진다. 세션 427개에서 착탄 높이에 대한 바닥 기준 머리
높이의 기울기가 **중앙 +0.654(r 0.864)**이고 0 근처인 세션이 하나도 없다. 벽 아래를
칠할 때 낮추고 위를 칠할 때 편다. **그것이 자세 시퀀스다.**

## 바닥은 기울기에 영향을 주지 않는다

바닥 높이는 **레벨당 상수**다(solved 9레벨에서 3.531~3.657 m). 상수를 빼도 기울기는
그대로다 — `(y - c)`를 `x`에 회귀하면 절편만 바뀐다. 실측으로도 절대 폭 중앙 1.092 m와
바닥 기준 폭 중앙 1.092 m가 같았다.

그래서 **기울기는 바닥을 몰라도 낸다.** 바닥이 필요한 곳은 절대 대역 게이트뿐이다
(학습모드 77,708패스 전부가 0.346~1.883 m 안이고 1.9 m 초과가 0건 — 그 대역을 벗어나면
자세가 아니라 이동이다. 실전모드는 4.094~7.388 m를 오간다).

## 지금 채점하지 않는다

추종 기울기를 **재서 싣되 점수로 만들지 않는다.** 레퍼런스 아티팩트가 축 규약 수정으로
버킷의 64.33%가 바뀔 예정이라, 지금 기준에 대고 낸 점수는 곧 무효가 된다.
`reference_quality`를 `unavailable`로 두고 `similarity_score`는 `null`이다 — `0.0`이 아니다.
"""

from __future__ import annotations

import statistics as st
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Final, Sequence

# MCP는 `trainer.scripts.posture_sequence`로, 테스트·스크립트는 scripts를 sys.path에
# 얹고 `posture_sequence`로 부른다. 자기가 불린 이름 공간에서 형제를 잡아야
# `world_axes`가 두 벌로 갈라지지 않는다 — 갈라지면 `UP_INDEX` 몽키패치가 한쪽에만 든다.
if __package__:
    from . import world_axes
else:
    import world_axes

Vec3 = tuple[float, float, float]

# 회귀로 기울기를 낼 최소 표본. 세션 기울기를 잰 측정이 표본 5개 미만을 버렸다.
MIN_SAMPLES_FOR_SLOPE: Final = 5

# 학습모드 77,708패스 전부가 이 안이고 1.9 m 초과가 0건이다. 이 대역을 벗어난 값은
# 자세가 아니라 이동(발판·리프트)이다.
POSTURE_BAND_MIN_M: Final = 0.30
POSTURE_BAND_MAX_M: Final = 1.90

# 바닥은 레벨당 상수다. solved 9레벨의 실측이 3.531~3.657 m이고, 이 밖의 값이 오면 배선이 틀린 것이다.
FLOOR_PLAUSIBLE_MIN_M: Final = 3.0
FLOOR_PLAUSIBLE_MAX_M: Final = 4.5

# 런타임 서비스라 무한히 쌓이면 안 된다.
MAX_SESSIONS: Final = 64
MAX_SAMPLES_PER_SESSION: Final = 512

# 채점 집합 v2. 정본 20종에서 이렇게 줄었다.
#
# - 다리 8종: 트래커가 없어 IK 산출물이다(해부 검사 통과 프레임 37% 대).
# - `Pelvis`: 자세 정규화의 **원점**이다. 정규화하면 항상 (0,0,0)이라 정보가 없는데
#   채점에 넣으면 분모(관절 수)만 키워 점수를 조용히 올린다.
# - `Head`: 조건부다. 실물 SIM 샘플에서 두 시퀀스 모두 변화 폭이 정확히 0.0이다.
#   오면 쓰고 얼어 있으면 뺀다.
#
# 남는 10종이 설계 문서 §4.2의 `SCORING_JOINTS_V2` 10종과 같다.
SCORING_JOINTS_V2: Final[tuple[str, ...]] = (
    "Spine",
    "Chest",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightForeArm",
    "RightHand",
)
CONDITIONAL_JOINTS_V2: Final[tuple[str, ...]] = ("Head",)
ROOT_JOINT_V2: Final = "Pelvis"

# 이보다 적은 관절로 낸 점수는 다른 점수와 비교할 수 없다 — 분모가 관절 수이기 때문이다.
# 모자라면 점수를 만들지 않는다. **`0.0`이 아니라 `null`이다.**
MIN_SCORED_JOINTS: Final = 8

# 채점 집합이 바뀌면 점수의 뜻이 바뀐다. 응답에 이 딱지를 실어, 다른 집합으로 낸 값과
# 나란히 놓이는 것을 막는다.
SCORING_SET_ID: Final = "v2-upper-10"


def scored_joints(available: Sequence[str], excluded: Sequence[str] = ()) -> tuple[str, ...]:
    """이번 시퀀스에서 실제로 채점에 쓴 관절. 응답에 그대로 싣는다.

    선언과 실효가 갈리면 점수의 분모가 조용히 달라진다 — `pose_similarity.SCORING_JOINTS`가
    12종을 선언하고 11종으로 돌던 것이 그 예다.
    """
    have = set(available)
    drop = set(excluded)
    out = [name for name in SCORING_JOINTS_V2 if name in have and name not in drop]
    out.extend(name for name in CONDITIONAL_JOINTS_V2 if name in have and name not in drop)
    return tuple(out)


@dataclass(frozen=True)
class PostureSample:
    """시퀀스 하나를 줄인 한 점.

    높이는 **월드 절대값**이다. 바닥을 빼지 않는다 — 기울기는 상수에 불변이고,
    바닥을 알게 되는 시점이 레벨을 아는 시점이라 여기서 빼면 그 시점을 강제하게 된다.
    """

    sequence_id: str
    head_height_m: float
    impact_height_m: float | None
    frames: int
    gate_status: str
    excluded_joints: tuple[str, ...] = ()


def _vertical(point: Vec3) -> float:
    """월드 수직 성분. 축은 `world_axes`가 호출 시점에 정한다.

    인덱스를 리터럴로 박으면 규약이 바뀌어도 반응하지 않는다 —
    `pass_segmentation.py:9`가 'Z가 수직'이라 적어 둔 채 남아 있는 것이 그 예다.
    """
    return world_axes.vertical_of(point)


def summarize(
    head_positions: Sequence[Vec3],
    impact_points: Sequence[Vec3] = (),
) -> tuple[float, float | None] | None:
    """시퀀스의 머리 높이와 착탄 높이를 각각 한 값으로 줄인다.

    **중앙값을 쓴다.** 시퀀스 안에서 몸이 고정이므로 평균과 거의 같지만, 한 프레임이 튀어도
    값이 흔들리지 않는다. 착탄점이 하나도 없으면(미명중 시퀀스) 착탄 높이는 `None`이다 —
    0.0으로 채우면 "바닥을 칠했다"가 된다.
    """
    if not head_positions:
        return None
    head = st.median([_vertical(point) for point in head_positions])
    impact = st.median([_vertical(point) for point in impact_points]) if impact_points else None
    return head, impact


def in_posture_band(head_height_m: float, floor_height_m: float | None) -> bool | None:
    """바닥 기준 머리 높이가 자세 대역 안인가. 바닥을 모르면 판정하지 않는다(`None`).

    판정하지 않는 것과 통과는 다르다. 모르면서 통과로 적으면 이동 구간을 자세로 읽는다.
    """
    if floor_height_m is None:
        return None
    if not FLOOR_PLAUSIBLE_MIN_M <= floor_height_m <= FLOOR_PLAUSIBLE_MAX_M:
        return None
    relative = head_height_m - floor_height_m
    return POSTURE_BAND_MIN_M <= relative <= POSTURE_BAND_MAX_M


@dataclass
class PostureSession:
    """세션 하나가 쌓아 온 자세 점들.

    `frozen_joints`는 앞선 시퀀스에서 얼었던 관절이다. 게이트의 R4c가 이 집합을 받아
    같은 관절이 두 번 얼면 영구 제외한다 — 한 번은 우연이지만 두 번은 그 관절이 안 오는 것이다.
    """

    session_id: str
    samples: list[PostureSample] = field(default_factory=list)
    frozen_joints: set[str] = field(default_factory=set)

    def add(self, sample: PostureSample, frozen: Sequence[str] = ()) -> None:
        self.samples.append(sample)
        if len(self.samples) > MAX_SAMPLES_PER_SESSION:
            del self.samples[0]
        self.frozen_joints.update(frozen)

    def tracking_slope(self) -> tuple[float, float, int] | None:
        """착탄 높이에 대한 머리 높이의 기울기·상관·표본 수. 못 내면 `None`.

        기울기 1.0은 착탄이 1 m 오를 때 몸도 1 m 오른다는 뜻이고, 0.0은 몸은 그대로 두고
        팔로만 처리한다는 뜻이다. 세션 427개 실측 중앙값이 **+0.654**(r 0.864)다.

        **바닥을 빼지 않아도 된다** — 상수는 기울기를 바꾸지 않는다.
        """
        pairs = [
            (sample.impact_height_m, sample.head_height_m)
            for sample in self.samples
            if sample.impact_height_m is not None
        ]
        if len(pairs) < MIN_SAMPLES_FOR_SLOPE:
            return None
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        n = len(pairs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
        sxx = sum((x - mean_x) ** 2 for x in xs)
        syy = sum((y - mean_y) ** 2 for y in ys)
        if sxx <= 1e-12 or syy <= 1e-12:
            return None
        return sxy / sxx, sxy / (sxx * syy) ** 0.5, n


class SessionStore:
    """세션별 자세 시퀀스를 들고 있는 인메모리 저장소.

    런타임 서비스라 경계가 있어야 한다. 세션 수가 `MAX_SESSIONS`를 넘으면 **가장 오래
    쓰이지 않은 것부터** 버린다. 버린 사실을 호출자가 알 수 있어야 하므로 `evicted`를 센다 —
    조용히 버리면 "시퀀스 3개짜리 세션"이 되어 기울기가 안 나오는 이유가 사라진다.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[str, PostureSession] = OrderedDict()
        self._max = max_sessions
        self.evicted = 0

    def get(self, session_id: str) -> PostureSession | None:
        session = self._sessions.get(session_id)
        if session is not None:
            self._sessions.move_to_end(session_id)
        return session

    def get_or_create(self, session_id: str) -> PostureSession:
        session = self.get(session_id)
        if session is None:
            session = PostureSession(session_id=session_id)
            self._sessions[session_id] = session
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
                self.evicted += 1
        return session

    def drop(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)


def session_report(
    session: PostureSession, floor_height_m: float | None = None
) -> dict[str, object]:
    """세션 누적 자세 시퀀스의 보고.

    **점수를 내지 않는다.** 레퍼런스 아티팩트가 축 규약 수정으로 버킷 64.33%가 바뀔
    예정이라 지금 기준에 대고 낸 점수는 곧 무효다. 재서 싣되 판정은 미룬다 —
    키 집합은 레퍼런스가 준비된 세계와 같고 값만 다르다.
    """
    slope = session.tracking_slope()
    band = None
    if session.samples:
        band = in_posture_band(session.samples[-1].head_height_m, floor_height_m)
    return {
        "sequences": len(session.samples),
        "tracking_slope": None if slope is None else slope[0],
        "tracking_r": None if slope is None else slope[1],
        "tracking_n": None if slope is None else slope[2],
        "posture_band_ok": band,
        "frozen_joints": sorted(session.frozen_joints),
        # 아래 셋은 레퍼런스가 준비되기 전까지 고정이다. 점수는 0.0이 아니라 null이다.
        "similarity_score": None,
        "reference_quality": "unavailable",
        "reason_code": "reference_rebuild_pending",
    }
