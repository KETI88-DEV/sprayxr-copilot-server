"""CSV 재생 입력의 좌표 등록(registration).

CSV 코퍼스(`/data/sources/professionals`)는 마운트 아티팩트라 새 클론에 없다. 그래서
프레임을 메모리에서 합성해 순수 함수만 시험한다 — 여기서 고정하려는 것은 CSV 읽기가
아니라 **착탄점을 FK 공간으로 옮기는 규칙**이다.
"""

from __future__ import annotations

import pytest

from trainer.scripts import pose_stages
from trainer.scripts.replay_telemetry import _register_impacts


def _frames(
    *, tools: list[list[float]], raypoints: list[list[float]], guns: list[list[float]]
) -> list[dict]:
    """건은 장면 좌표, 도구는 FK 공간 — 수직 원점이 다른 상황을 만든다."""
    return [
        {
            "tool_pos_rot": {"pos": list(tool)},
            "gun_metrics": {"raypoint": list(ray), "_gun_scene": list(gun)},
        }
        for tool, ray, gun in zip(tools, raypoints, guns, strict=True)
    ]


# 건이 장면에서 y≈3.7 에 있고 FK 공간의 도구는 y≈-1.0 이다(원점이 ~4.7 m 어긋남).
_GUNS = [[1.2, 3.70, -2.9], [1.3, 3.74, -2.8], [1.1, 3.78, -3.0], [1.4, 3.72, -2.7]]
_RAYS = [[1.5, 3.86, -2.5], [1.6, 3.95, -2.4], [1.4, 3.88, -2.6], [1.7, 3.91, -2.3]]
_TOOLS = [[0.3, -1.02, 0.4], [0.4, -0.98, 0.5], [0.2, -1.06, 0.3], [0.5, -1.00, 0.6]]


def _reach(frames: list[dict]) -> float | None:
    return pose_stages.reach_of(
        [frame["gun_metrics"]["raypoint"] for frame in frames],
        [frame["tool_pos_rot"]["pos"] for frame in frames],
    )


def test_registration_preserves_the_reach_measured_in_scene_coordinates() -> None:
    """Given scene gun/impact pairs, When registered, Then reach is unchanged.

    `reach = 착탄 수직 − 건 수직`은 차이라서 평행이동에 불변이고, CSV의 건과 착탄은 둘 다
    장면 좌표라 그 차이가 실측값이다. 등록은 그 값을 옮겨올 뿐 바꾸지 않아야 한다.
    """
    truth = pose_stages.reach_of(_RAYS, _GUNS)
    frames = _frames(tools=_TOOLS, raypoints=_RAYS, guns=_GUNS)

    _register_impacts(frames)

    assert _reach(frames) == pytest.approx(truth, abs=1e-12)


def test_reach_is_in_the_wrong_space_without_registration() -> None:
    """등록하지 않으면 reach 가 두 공간의 차만큼 튄다 — 위 단언이 공허하지 않음을 보인다."""
    frames = _frames(tools=_TOOLS, raypoints=_RAYS, guns=_GUNS)

    unregistered = _reach(frames)

    assert unregistered is not None
    assert abs(unregistered - pose_stages.reach_of(_RAYS, _GUNS)) > 4.0


def test_a_per_frame_offset_would_not_preserve_reach() -> None:
    """프레임마다 다른 델타로 옮기면 보존되지 않는다 — 상수여야 하는 이유.

    `pose_stages.reach_of` 는 착탄과 건의 중앙값을 **각각 따로** 취한다. 두 중앙값이
    독립적으로 움직이면 차이가 유지되지 않는다. 실측으로 한 구간의 reach 가 0.1863 대신
    0.5421 이 되어 밴드가 `+0.10` 이 아닌 `+0.50`(support 202)으로 잘못 잡혔다.
    """
    truth = pose_stages.reach_of(_RAYS, _GUNS)
    per_frame = [
        [ray[axis] + tool[axis] - gun[axis] for axis in range(3)]
        for ray, tool, gun in zip(_RAYS, _TOOLS, _GUNS, strict=True)
    ]

    shifted = pose_stages.reach_of(per_frame, _TOOLS)

    assert shifted is not None
    assert shifted != pytest.approx(truth, abs=1e-6)


def test_registration_removes_the_internal_scene_gun_key() -> None:
    """`_gun_scene` 은 등록에만 쓰는 내부 키다. 페이로드로 새면 계약 밖 필드가 된다."""
    frames = _frames(tools=_TOOLS, raypoints=_RAYS, guns=_GUNS)

    _register_impacts(frames)

    assert all("_gun_scene" not in frame["gun_metrics"] for frame in frames)


def test_registration_of_an_empty_sequence_does_nothing() -> None:
    """프레임이 없으면 중앙값을 낼 수 없다 — 터지지 않고 그대로 둔다."""
    frames: list[dict] = []

    _register_impacts(frames)

    assert frames == []
