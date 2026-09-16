#!/usr/bin/env python3
"""CSV 리그의 몸통을 BVH 실측 뼈 길이 비례로 다시 재고, 빠진 두 마디를 그려 넣는다.

**왜.** 시뮬레이터 CSV 리그에는 `Spine1`과 `Neck`이 없다. 골반→머리가 `Spine` 0.1053 m +
`Chest` 0.1053 m + `Head` 0.1368 m = 0.3474 m뿐인데 팔은 0.6941 m, 다리(골반→발목)는
0.9674 m다. 골반→머리를 1로 잡으면 **1 : 2.00 : 2.79**다. 사람 비례가 아니다 — 화면의 인물이
거미처럼 보인 것은 렌더 버그가 아니라 리그에 관절이 빠진 것이다.

**비교 대상.** 같은 작업자(전형기)의 BVH 79개는 65관절 리그이고 손가락을 빼면 `Spine1`·`Neck`이
있다. OFFSET 노름의 중앙값(79파일에서 최대−최소가 2e-6 m 이하, 즉 파일마다 같은 아바타):

    Spine 0.101824 · Spine1 0.100027 · Chest 0.093221 · Neck 0.168653 · Head 0.097643
      → 골반→가슴 0.295072 · 가슴→머리 0.266296 · 골반→머리 0.561368
    LeftShoulder 0.121194 + LeftUpperArm 0.108377 + LeftForeArm 0.278415 + LeftHand 0.283288
      → 팔 0.791274
    LeftThigh 0.107477 + LeftCalf 0.443714 + LeftFoot 0.445278
      → 다리(골반→발목) 0.996469

`trainer/pose_lab/tests/test_body_proportions.py`가 아래 상수를 **BVH 파일에서 직접 다시 재서**
맞는지 본다(파일이 있을 때만).

**무엇을 바꾸고 무엇을 안 바꾸는가 — 앵커 오차가 정한다.**

배치(`skeleton.place_oriented`)는 강체 변환이라, 앵커 어긋남은 **세 앵커 관절(Head·RightHand·
LeftHand)의 서로에 대한 위치**에만 달려 있다(평행이동 `t`가 공통 변위를 흡수한다). FK 사슬은
`Pelvis → Spine → Chest → {Head, 양 어깨}`다. 그래서:

* `Spine`·`Chest`(골반→가슴)를 늘리면 머리도 양손도 **같은 벡터만큼** 함께 움직인다 →
  앵커 사이 상대 위치가 그대로다 → 앵커 오차가 **정확히 그대로**다. 움직이는 것은 골반(과 다리)이고,
  머리·손이 실측에 붙어 있으므로 골반이 그만큼 **내려간다**. 키가 큰 사람의 엉덩이가 더 낮다.
* `Head`(가슴→머리)를 늘리면 머리가 양손에 대해 움직인다 → 앵커 오차가 **바뀐다**.

실제로 재 봤다. 세 마디를 다 BVH 비례(골반→머리 0.518m)로 늘리고 `fit_skeleton_mounts.py`로
장착 상수까지 다시 적합한 뒤(LOWO 합격), 교차 검증 풀드 중앙 어긋남이

    머리 80.6 → 92.8 mm · 왼손 100.0 → 103.0 mm · 오른손 134.0 → 137.9 mm

로 **전부 나빠졌다**. 검증된 네 패스(김경호 L1·L2·L3·L6)의 앵커 중앙값도 6.96 → 8.55 cm로
나빠졌다. 가슴→머리를 늘리는 것은 실측이 반대한다 — 추적된 HMD는 이 리그가 말하는 자리보다
가슴에 가깝다. 그래서 **가슴→머리는 리그 값 그대로 둔다**. 지어낸 비례를 밀어 넣고 실측을
나빠지게 하는 것보다, 실측이 허락하는 만큼만 고치는 편이 낫다.

**골반→가슴 목표 길이 L.** 팔과 다리 둘 다에 BVH 비례를 맞출 수는 없다. 골반→가슴을 1로 정규화하면
BVH는 1 : 2.68162(팔) : 3.37703(다리)이고 CSV 리그는 1 : 3.296 : 4.593이라, 팔에서 유도한
L_arm = 0.6941/2.68162 = 0.2588 m와 다리에서 유도한 L_leg = 0.9674/3.37703 = 0.2865 m가 다르다.
한쪽을 고르면 다른 쪽 비례 오차를 혼자 다 떠안는다. 그래서 **기하평균**을 쓴다 —
L = sqrt(L_arm · L_leg)는 두 비례의 로그 오차를 크기가 같고 부호가 반대가 되게 하는 유일한 값이다.

    L = sqrt(0.2588 · 0.2865) = 0.2723 m  (리그 0.2106 m의 1.293배)

**마디별 배분.** CSV의 두 마디를 BVH의 세 마디에 짝지어, CSV 한 마디가 BVH에서 갈리는 만큼만
합쳐 배분한다. 지어낸 숫자가 아니라 위 BVH 측정값의 몫이다.

    CSV Pelvis→Spine ≙ BVH Spine            몫 0.101824/0.295072 = 0.345078
    CSV Spine→Chest  ≙ BVH Spine1 + Chest   몫 0.193248/0.295072 = 0.654922

**빠진 마디를 그려 넣기.** 위 배분 뒤 CSV의 `Spine→Chest`가 BVH의 두 마디를 담고 있으므로
`Spine1`은 그 선분 위 0.100027/0.193248 = 0.517607 지점에, `Neck`은 `Chest→Head` 위
0.168653/0.266296 = 0.633329 지점에 놓는다. **선분 위 내분점**이라 방향을 지어내지 않고,
배치(강체 변환) 전후 어디서 계산해도 같다.

**결과 비례.** 골반→머리 0.3474 → 0.4091 m, 1 : 2.00 : 2.79 → **1 : 1.70 : 2.37**
(BVH 1 : 1.41 : 1.78). 남은 차이는 가슴→머리를 실측 때문에 늘리지 않은 몫이다.

**이 마디는 추정이다.** 실측은 머리(HMD)·건·플래쉬 세 강체뿐이다. 새 관절도 나머지 뼈와 똑같이
회색 추정 기하로 그린다(`service.LEGEND_*`), 화면 고지가 출처(BVH에서 잰 비례)를 말한다.

**장착 상수.** `Spine`·`Chest`만 바꾸면 세 앵커의 상대 위치가 그대로라 `skeleton_mounts.json`의
`m_j`를 다시 맞출 필요가 없다 — 위 유도와 `test_body_proportions.py`가 그 불변을 고정한다.
그래도 `trainer/scripts/fit_skeleton_mounts.py`는 서버와 **같은 리그**로 적합하도록 이 모듈을 거친다.
"""

from __future__ import annotations

import math
from typing import Final, Mapping

Vec3 = tuple[float, float, float]

# BVH(전형기 79파일, 65관절) OFFSET 노름 중앙값(m). 파일마다 같은 값이라 산포가 없다.
BVH_TORSO_M: Final[dict[str, float]] = {
    "Spine": 0.101824,
    "Spine1": 0.100027,
    "Chest": 0.093221,
    "Neck": 0.168653,
    "Head": 0.097643,
}
BVH_ARM_M: Final[dict[str, float]] = {
    "LeftShoulder": 0.121194,
    "LeftUpperArm": 0.108377,
    "LeftForeArm": 0.278415,
    "LeftHand": 0.283288,
}
BVH_LEG_M: Final[dict[str, float]] = {
    "LeftThigh": 0.107477,
    "LeftCalf": 0.443714,
    "LeftFoot": 0.445278,
}

# 골반→가슴(BVH Spine+Spine1+Chest). 이것만 다시 잰다 — 가슴→머리는 실측이 반대한다(독스트링).
BVH_LOWER_TORSO_TOTAL_M: Final = BVH_TORSO_M["Spine"] + BVH_TORSO_M["Spine1"] + BVH_TORSO_M["Chest"]
BVH_ARM_TOTAL_M: Final = sum(BVH_ARM_M.values())
BVH_LEG_TOTAL_M: Final = sum(BVH_LEG_M.values())

# 골반→가슴을 1로 정규화한 BVH 비례. 목표 길이가 여기서 나온다.
BVH_ARM_RATIO: Final = BVH_ARM_TOTAL_M / BVH_LOWER_TORSO_TOTAL_M
BVH_LEG_RATIO: Final = BVH_LEG_TOTAL_M / BVH_LOWER_TORSO_TOTAL_M

# 다시 재는 CSV 마디와, 그 마디가 가져갈 BVH 길이의 몫.
RESCALED_BONES: Final[tuple[str, ...]] = ("Spine", "Chest")
LOWER_TORSO_SHARE_M: Final[dict[str, float]] = {
    "Spine": BVH_TORSO_M["Spine"],
    "Chest": BVH_TORSO_M["Spine1"] + BVH_TORSO_M["Chest"],
}
# 길이를 **바꾸지 않는** 마디. 앵커(머리)가 양손에 대해 움직이면 앵커 오차가 바뀐다.
ANCHOR_FIXED_BONES: Final[tuple[str, ...]] = ("Head",)
# 화면 고지가 적는 "골반→머리" 길이를 재는 마디들.
TORSO_BONES: Final[tuple[str, ...]] = ("Spine", "Chest", "Head")

# CSV 팔·다리 사슬(바인드 길이를 재는 데 쓴다). 좌우 평균을 쓴다.
ARM_CHAINS: Final[tuple[tuple[str, ...], ...]] = (
    ("LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand"),
    ("RightShoulder", "RightUpperArm", "RightForeArm", "RightHand"),
)
LEG_CHAINS: Final[tuple[tuple[str, ...], ...]] = (
    ("LeftThigh", "LeftCalf", "LeftFoot"),
    ("RightThigh", "RightCalf", "RightFoot"),
)

# 그려 넣는 마디: (새 관절, 그 관절이 앉을 선분의 시작·끝, 시작에서의 내분 비율).
# 비율은 BVH에서 그 선분이 갈리는 몫이다 — 위 독스트링의 유도.
INSERTED_JOINTS: Final[tuple[tuple[str, str, str, float], ...]] = (
    (
        "Spine1",
        "Spine",
        "Chest",
        BVH_TORSO_M["Spine1"] / (BVH_TORSO_M["Spine1"] + BVH_TORSO_M["Chest"]),
    ),
    (
        "Neck",
        "Chest",
        "Head",
        BVH_TORSO_M["Neck"] / (BVH_TORSO_M["Neck"] + BVH_TORSO_M["Head"]),
    ),
)

# 화면이 그릴 몸통 **줄기** 마디. 그려 넣은 두 관절이 들어간 다섯 마디다.
TORSO_STEM_BONES: Final[tuple[tuple[str, str], ...]] = (
    ("Pelvis", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
)
# 갈아 끼우는 옛 줄기 마디.
OLD_STEM_BONES: Final[frozenset[tuple[str, str]]] = frozenset(
    {("Pelvis", "Spine"), ("Spine", "Chest"), ("Chest", "Head")}
)


def _chain_length(offsets: Mapping[str, Vec3], chain: tuple[str, ...]) -> float | None:
    total = 0.0
    for name in chain:
        if name not in offsets:
            return None
        total += math.hypot(*offsets[name])
    return total


def _mean_chain(offsets: Mapping[str, Vec3], chains: tuple[tuple[str, ...], ...]) -> float | None:
    lengths = [length for chain in chains if (length := _chain_length(offsets, chain)) is not None]
    return sum(lengths) / len(lengths) if lengths else None


def target_lower_torso_m(offsets: Mapping[str, Vec3]) -> float | None:
    """이 리그의 팔·다리에서 유도한 **골반→가슴** 목표 길이(m). 팔도 다리도 없으면 None.

    독스트링 §"골반→가슴 목표 길이 L": `sqrt((팔/2.68162) · (다리/3.37703))`. 둘 중 하나만 있으면
    그 하나로 유도한다 — 기하평균을 흉내 내려고 없는 쪽을 지어내지 않는다.
    """
    arm = _mean_chain(offsets, ARM_CHAINS)
    leg = _mean_chain(offsets, LEG_CHAINS)
    candidates = [
        value
        for value in (
            None if arm is None else arm / BVH_ARM_RATIO,
            None if leg is None else leg / BVH_LEG_RATIO,
        )
        if value is not None and value > 0.0
    ]
    if not candidates:
        return None
    return math.exp(sum(math.log(value) for value in candidates) / len(candidates))


def rig_offsets(offsets: Mapping[str, Vec3]) -> dict[str, Vec3]:
    """골반→가슴 두 마디만 BVH 비례 길이로 바꾼 바인드 오프셋. 나머지는 그대로다.

    **방향은 건드리지 않는다** — 마디 벡터에 스칼라를 곱할 뿐이라 FK 회전·사슬 구조가 그대로이고,
    `skeleton.bone_length_errors`가 재는 "배치된 뼈 길이 == 오프셋 노름"도 그대로 0이다.
    `Head`(가슴→머리)는 건드리지 않으므로 세 앵커의 상대 위치가 보존된다 — 앵커 오차가 그대로다.

    두 마디가 다 있지 않거나 목표 길이를 유도할 수 없으면 **원본을 그대로 돌려준다**.
    """
    out = dict(offsets)
    if any(name not in offsets for name in RESCALED_BONES):
        return out
    target = target_lower_torso_m(offsets)
    if target is None:
        return out
    share_total = sum(LOWER_TORSO_SHARE_M[name] for name in RESCALED_BONES)
    for name in RESCALED_BONES:
        current = math.hypot(*offsets[name])
        if current <= 0.0:
            return dict(offsets)
        wanted = target * LOWER_TORSO_SHARE_M[name] / share_total
        factor = wanted / current
        x, y, z = offsets[name]
        out[name] = (x * factor, y * factor, z * factor)
    return out


def inserted_joints(joints: Mapping[str, Vec3]) -> dict[str, Vec3]:
    """그려 넣는 `Spine1`·`Neck`의 위치. 양 끝 관절이 없으면 그 관절은 싣지 않는다.

    선분 위 내분점이라 배치(강체 변환) 전후 어디서 계산해도 같다.
    """
    out: dict[str, Vec3] = {}
    for name, start, end, ratio in INSERTED_JOINTS:
        if start not in joints or end not in joints:
            continue
        a = joints[start]
        b = joints[end]
        out[name] = tuple(a[i] + (b[i] - a[i]) * ratio for i in range(3))  # type: ignore[misc]
    return out


def display_bones(
    bones: tuple[tuple[str, str], ...] | list[list[str]], joints: Mapping[str, Vec3]
) -> list[list[str]]:
    """리그 마디 목록에서 몸통 줄기를 다섯 마디로 갈아 끼운 **표시용** 마디 목록.

    그려 넣은 관절이 없으면(양 끝이 없어 못 만든 경우) 원래 줄기 마디를 남긴다.
    """
    have = all(name in joints for name in ("Pelvis", "Spine", "Spine1", "Chest", "Neck", "Head"))
    if not have:
        return [list(pair) for pair in bones]
    out: list[list[str]] = []
    inserted = False
    for pair in bones:
        a, b = pair[0], pair[1]
        if (a, b) in OLD_STEM_BONES:
            if not inserted:
                out.extend([list(stem) for stem in TORSO_STEM_BONES])
                inserted = True
            continue
        out.append([a, b])
    return out
