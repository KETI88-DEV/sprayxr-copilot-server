#!/usr/bin/env python3
"""관절 어휘를 한 곳에서 통일한다.

세 어휘가 따로 살아 있었다.

- **SIM 실물**(`docs/SprayXR_CoPilot_샘플_2026-09-21.json`의 `skeleton_joints`):
  소문자 붙여쓰기 20종 — `rightforearm`. `neck`이 없다.
- **CSV 코퍼스**(캡처 헤더): 공백 포함 20종 — `Right Fore Arm`. `Neck`이 없다.
- **코드**(`pose_similarity`): PascalCase 붙여쓰기 — `RightForeArm`. `SCORING_JOINTS`에만
  `Neck`이 들어 있는데 런타임·CSV 어디에도 없어 실효는 11종이다.

그리고 `mcp_servers/pose_estimation.py:29`에 `wrist_r`·`elbow_r` 류 구 별칭 6개가 있다.

**코드 어휘를 정본으로 삼는다.** 런타임이 읽는 이름이고 기존 레퍼런스 아티팩트의 키이기 때문이다.

**모르는 이름에 예외를 던지지 않고 `None`을 돌려준다.** `zone_resolver`와 같은 규약이다 —
새 라벨 하나가 시퀀스 전체를 죽이면 안 된다. 호출자가 `None`을 보고 판정을 내린다.

다리 8관절은 **받아서 보관하되 채점·교정에서 뺀다**. 트래커가 없어 IK 산출물이고,
해부 검사를 모두 통과하는 분사 프레임이 37% 대에 그친다.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

Vec3 = tuple[float, float, float]

# 정본 어휘. SIM·CSV 양쪽에 실제로 존재하는 20종이다(`Neck`은 없다).
CODE_JOINTS: Final[tuple[str, ...]] = (
    "Pelvis",
    "Spine",
    "Chest",
    "Head",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightForeArm",
    "RightHand",
    "LeftThigh",
    "LeftCalf",
    "LeftFoot",
    "LeftToe",
    "RightThigh",
    "RightCalf",
    "RightFoot",
    "RightToe",
)

# 트래커가 없어 IK로만 존재한다. 보관은 하되 채점·교정에서 뺀다.
LEG_JOINTS: Final[frozenset[str]] = frozenset(
    {
        "LeftThigh",
        "LeftCalf",
        "LeftFoot",
        "LeftToe",
        "RightThigh",
        "RightCalf",
        "RightFoot",
        "RightToe",
    }
)

# 폐기 별칭. `mcp_servers/pose_estimation.py`가 쓰던 이름을 흡수만 하고 새로 쓰지 않는다.
DEPRECATED_ALIASES: Final[Mapping[str, str]] = {
    "wrist_r": "RightHand",
    "wrist_l": "LeftHand",
    "elbow_r": "RightForeArm",
    "elbow_l": "LeftForeArm",
    "shoulder_r": "RightShoulder",
    "shoulder_l": "LeftShoulder",
}


def _fold(name: str) -> str:
    """비교용 정규형 — 공백·밑줄을 지우고 소문자로. `Right Fore Arm`·`rightforearm`이 같아진다."""
    return name.replace(" ", "").replace("_", "").lower()


_BY_FOLDED: Final[dict[str, str]] = {_fold(name): name for name in CODE_JOINTS}
_ALIAS_BY_FOLDED: Final[dict[str, str]] = {
    _fold(alias): code for alias, code in DEPRECATED_ALIASES.items()
}
_CANONICAL: Final[frozenset[str]] = frozenset(CODE_JOINTS)


def to_code(name: str) -> str | None:
    """어떤 어휘의 이름이든 정본 이름으로. 모르면 `None`이다(예외 아님).

    `Neck`은 의도적으로 `None`이다 — 런타임·CSV 어디에도 없고, BVH의 `Neck`은
    `body_proportions`가 비율로 끼워 넣은 표시용 값이라 측정량이 아니다.
    측정량이 아닌 값을 채점에 넣으면 σ의 출처가 데이터가 아니라 우리 상수가 된다.
    """
    folded = _fold(name)
    found = _BY_FOLDED.get(folded)
    if found is not None:
        return found
    return _ALIAS_BY_FOLDED.get(folded)


def to_sim(code: str) -> str | None:
    """정본 이름 → SIM 실물 이름(소문자 붙여쓰기). 정본이 아니면 `None`."""
    return code.lower() if code in _CANONICAL else None


def to_csv(code: str) -> str | None:
    """정본 이름 → CSV 헤더 이름(대문자 앞에 공백). 정본이 아니면 `None`.

    `RightForeArm` → `Right Fore Arm`. 첫 글자 앞에는 공백을 넣지 않는다.
    """
    if code not in _CANONICAL:
        return None
    out = [code[0]]
    for character in code[1:]:
        if character.isupper():
            out.append(" ")
        out.append(character)
    return "".join(out)


def is_scorable(code: str) -> bool:
    """채점·교정에 쓸 수 있는 관절인가. 다리는 아니다."""
    return code in _CANONICAL and code not in LEG_JOINTS


def parse_position(value: object) -> Vec3 | None:
    """관절 값에서 위치만 꺼낸다. 3원소 배열과 `{pos, rot}` 객체를 **둘 다** 받는다.

    실물 SIM은 3원소 배열로 보내고(`"chest": [0.4207, 4.1166, 0.6457]`), 계약 문서는
    `{pos, rot}` 객체를 적어 두었다. 어느 쪽이 와도 받는다 — 둘 중 하나를 422로 만들면
    정상 데이터가 죽는다.

    회전은 **여기서 돌려주지 않는다.** 회전순서가 미확정이고(실물 `rot`이 전 프레임 90°
    배수다) 각도 교정은 계약에서 빠졌다. 보관이 필요하면 `parse_rotation`을 따로 쓴다.
    """
    if isinstance(value, Mapping):
        value = value.get("pos")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 3:
        return None
    out: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            return None
        out.append(float(component))
    return (out[0], out[1], out[2])


def parse_rotation(value: object) -> Vec3 | None:
    """관절 값의 회전. **원문 보관용이다** — 각도 산출에 쓰지 않는다.

    3원소 배열로 온 값에는 회전이 없으므로 `None`이다.
    """
    if not isinstance(value, Mapping):
        return None
    return parse_position({"pos": value.get("rot")})


def positions(joints: Mapping[str, object]) -> dict[str, Vec3]:
    """SIM/CSV/코드 어느 어휘로 와도 정본 이름 → 위치 사전으로 바꾼다.

    모르는 이름과 형식이 틀린 값은 **조용히 빠진다.** 빠진 것을 호출자가 세어
    입력 품질로 판정한다 — 여기서 예외를 던지면 프레임 하나가 시퀀스를 죽인다.
    """
    out: dict[str, Vec3] = {}
    for raw_name, raw_value in joints.items():
        code = to_code(raw_name)
        if code is None:
            continue
        position = parse_position(raw_value)
        if position is None:
            continue
        out[code] = position
    return out


def unknown_names(joints: Mapping[str, object]) -> list[str]:
    """정본으로 옮기지 못한 이름들. 보고용이다(계약의 `unknown_fields`와 같은 취지)."""
    return sorted(name for name in joints if to_code(name) is None)
