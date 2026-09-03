#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

world_axes = importlib.import_module("world_axes")


def test_unity_convention_is_y_up_left_handed() -> None:
    """유니티다. Y가 위, 수평면은 (X, Z), 왼손 좌표계."""
    assert world_axes.UP_INDEX == 1
    assert world_axes.UP == (0.0, 1.0, 0.0)
    assert world_axes.HORIZONTAL_INDICES == (0, 2)
    assert world_axes.HANDEDNESS == "left"


def test_vertical_and_horizontal_split_the_vector() -> None:
    assert world_axes.vertical_of((1.0, 2.0, 3.0)) == 2.0
    assert world_axes.horizontal_of((1.0, 2.0, 3.0)) == (1.0, 3.0)


def test_flatten_removes_only_the_vertical_component() -> None:
    assert world_axes.flatten((1.0, 2.0, 3.0)) == (1.0, 0.0, 3.0)


def test_yaw_is_measured_in_the_horizontal_plane() -> None:
    """수평 두 성분의 atan2다. 전방이 +X면 0, +Z면 90도."""
    assert world_axes.yaw_of((1.0, 0.0, 0.0)) == 0.0
    assert math.isclose(world_axes.yaw_of((0.0, 0.0, 1.0)), math.pi / 2)
    # 수직 성분은 요에 영향을 주지 않는다.
    assert math.isclose(world_axes.yaw_of((1.0, 5.0, 0.0)), 0.0)


def test_yaw_is_none_when_there_is_no_horizontal_component() -> None:
    """정확히 위나 아래를 보면 방위가 정의되지 않는다. 0으로 지어내지 않는다."""
    assert world_axes.yaw_of((0.0, 1.0, 0.0)) is None
    assert world_axes.yaw_of((0.0, -1.0, 0.0)) is None


def test_sign_toward_picks_the_direction_facing_the_reference_point() -> None:
    """왼손 좌표계라 외적의 방향 감각이 오른손계와 반대다. 부호는 기준점으로 정한다."""
    assert world_axes.sign_toward((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)
    assert world_axes.sign_toward((1.0, 0.0, 0.0), (-5.0, 0.0, 0.0)) == (-1.0, 0.0, 0.0)


def test_patching_up_index_alone_moves_every_reader() -> None:
    """UP_INDEX가 유일한 손잡이어야 한다.

    UP과 HORIZONTAL_INDICES가 UP_INDEX와 별도인 리터럴이면, UP_INDEX만 몽키패치했을
    때 그 둘과 vertical_of/horizontal_of/flatten/yaw_of가 따라오지 않는다 — 이
    모듈이 막으려던 결함의 축소판이다. 여기서는 UP_INDEX 하나만 바꾸고 나머지 전부가
    새 축을 따라가는지 검사한다.
    """
    original = world_axes.UP_INDEX
    try:
        world_axes.UP_INDEX = 2  # 수직축을 Z로 옮긴다.
        assert world_axes.UP == (0.0, 0.0, 1.0)
        assert world_axes.HORIZONTAL_INDICES == (0, 1)
        assert world_axes.vertical_of((1.0, 2.0, 3.0)) == 3.0
        assert world_axes.horizontal_of((1.0, 2.0, 3.0)) == (1.0, 2.0)
        assert world_axes.flatten((1.0, 2.0, 3.0)) == (1.0, 2.0, 0.0)
        assert math.isclose(world_axes.yaw_of((1.0, 0.0, 0.0)), 0.0)
        assert math.isclose(world_axes.yaw_of((0.0, 1.0, 0.0)), math.pi / 2)
    finally:
        world_axes.UP_INDEX = original


def test_frame_around_is_orthonormal_for_a_horizontal_forward() -> None:
    frame = world_axes.frame_around((0.0, 0.0, 1.0))
    axes = (frame.forward, frame.up, frame.third)
    for axis in axes:
        assert math.isclose(math.dist(axis, (0.0, 0.0, 0.0)), 1.0, abs_tol=1e-9)
    for i in range(3):
        for j in range(i + 1, 3):
            dot = sum(axes[i][k] * axes[j][k] for k in range(3))
            assert math.isclose(dot, 0.0, abs_tol=1e-9)
    # 접선이 이미 수평이라 위 축 직교화는 무너지지 않는다 — 월드 수직축 그대로다.
    assert frame.up == (0.0, 1.0, 0.0)


def test_frame_around_falls_back_to_a_horizontal_up_when_forward_is_vertical() -> None:
    """접선이 월드 수직축과 평행하면(순수 수직 방향) 직교화가 영벡터다.

    수평 대체축으로 up을 다시 세운다 — 이름은 'up'이지만 이 경로에서 값은 수평이다.
    """
    frame = world_axes.frame_around((0.0, 1.0, 0.0))
    assert frame.up == (1.0, 0.0, 0.0)


def test_frame_around_rejects_a_zero_forward() -> None:
    """영벡터를 대체값 없이 조용히 삼키지 않는다 — 호출자가 자기 도메인의 퇴화를
    먼저 해소해야 한다(예: 제자리 프레임의 대체 접선)."""
    with pytest.raises(ValueError):
        world_axes.frame_around((0.0, 0.0, 0.0))


def test_frame_around_fallback_moves_with_up_index() -> None:
    """퇴화 대체축도 UP_INDEX 하나로 따라와야 한다 — 고정된 (1,0,0) 리터럴이 아니다."""
    original = world_axes.UP_INDEX
    try:
        world_axes.UP_INDEX = 2  # 수직축을 Z로 옮긴다. 수평 축은 이제 (X, Y)다.
        frame = world_axes.frame_around((0.0, 0.0, 1.0))  # forward ∥ 새 수직축.
        assert frame.up == (1.0, 0.0, 0.0)
    finally:
        world_axes.UP_INDEX = original


def test_every_module_follows_the_convention_not_a_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UP_INDEX를 바꾸면 네 소비 모듈의 출력이 **필드마다 각각** 따라 움직여야 한다.

    `world_axes`는 `UP`과 `HORIZONTAL_INDICES`를 모듈 `__getattr__`로 `UP_INDEX`
    하나에서 매번 다시 계산한다(PEP 562) — 그래서 손잡이는 `UP_INDEX` 단 하나다.
    이 테스트는 그 손잡이 하나만 돌려서 네 모듈(`measured_score`, `stroke_frame`,
    `measured_pose`, `motion_key`)이 전부 반응하는지 본다.

    **비교 단위는 개별 필드다, 묶음이 아니다 — 그리고 입력도 그 모듈 자신의
    책임만 재도록 골라야 한다.** 예전 버전은 두 겹으로 가려질 수 있었다.

    (1) `_axis_components`가 낸 `{"wall": (value, unit, word), ...}` 전체를 값
    하나로 비교하면, `vertical`의 `value`만 리터럴로 굳어도 그 필드의 `unit`
    (`world_axes.UP`를 호출 시점에 그대로 읽으므로 항상 움직인다)이 여전히 달라
    "vertical 전체가 달라졌다"는 참이 된다 — 3튜플 안에서 얼어붙은 `value` 하나가
    가려진다. 그래서 `value`/`unit`/`word`를 각각 독립된 키로 비교한다.

    (2) `_axis_components`에 넘기는 `frame`을 `stroke_frame.stroke_basis(...)`의
    결과로 두면(예전 버전이 그랬다), `frame` 자체가 `world_axes.frame_around`를
    통해 UP_INDEX를 따라 움직이므로 `_axis_components`가 내부에서 `world_axes.
    vertical_of`를 쓰든 `world[1]` 리터럴을 쓰든 출력값 자체는 (프레임이 바뀌어서)
    거의 항상 달라진다 — 함수 자신의 리터럴 결함이 "프레임이 움직였다"는 잡음에
    묻힌다. 실제로 확인했다: `_axis_components`의 `world_axes.vertical_of(world)`를
    `world[1]`로 되박고 `stroke_basis`가 낸 프레임을 그대로 넘기면, 예전 버전도
    지금 버전(필드별 비교)도 이 결함을 못 잡는다 — `vertical` 값이 3.2→11(예시)로
    바뀌는데, 그 변화는 `world[1]`이 규약을 따라서가 아니라 프레임의 다른 성분이
    바뀌었기 때문이다. 그래서 `_axis_components`에는 world_axes와 무관한, 리터럴로
    고정한 `_FIXED_FRAME`을 넘긴다 — 이러면 `local`과 `frame`이 둘 다 완전히
    고정되므로, UP_INDEX를 바꿨을 때 `value`가 움직이는 유일한 경로는
    `_axis_components` 자신이 `world_axes.vertical_of`/`horizontal_of`를 호출
    시점에 쓰는 것뿐이다. (`stroke_frame.stroke_basis` 자신의 규약 준수는 따로
    `basis_normal`/`basis_up`으로, 실제 `stroke_basis(...)` 호출을 통해 잰다.)

    `basis_tangent`는 표본에서 뺐다 — `stroke_basis`의 `tangent`는 입력 접선을
    그대로 정규화한 것이라 규약과 무관하게 절대 움직이지 않는, 즉 이 테스트가
    "안 움직이면 결함"이라고 말할 자격이 없는 필드이기 때문이다(넣어 두면 영원히
    고칠 수 없는 거짓 경보가 된다).

    `motion_key.body_placement`(lateral·reach·height)와 합성 키
    `motion_key.motion_key()`도 표본에 넣는다 — 이전 표본은 `heading_word`만 재서
    `body_placement`와 합성 키 문자열이 규약을 따르는지가 비어 있었다(리뷰 A3).

    모든 입력은 이 파일을 쓰면서 직접 실행해, 아래 `stuck`이 실제로 빈 리스트임을
    확인했다 — 우연히 같은 답이 나오는 벡터를 썼다면 그 필드는 이 테스트에서
    아무것도 증명하지 않으면서 통과했을 것이다. `_axis_components`의
    `world_axes.vertical_of(world)`를 `world[1]` 리터럴로 되박아 봤을 때
    `axis_vertical_value`·`axis_vertical_word`만 `stuck`에 이름으로 잡히고
    `axis_wall_*`/`axis_lateral_*`는 그대로 움직이는 것도 확인했다(리뷰 A1).

    하나라도 안 움직이면 그 모듈이 리터럴을 다시 박았거나 임포트 시점에 값을
    복사한 것이다. 그 상태는 지금 우연히 옳은 답을 낼 수 있어서 눈으로는 안 보인다.
    """
    measured_score = importlib.import_module("measured_score")
    stroke_frame = importlib.import_module("stroke_frame")
    measured_pose = importlib.import_module("measured_pose")
    motion_key = importlib.import_module("motion_key")

    # world_axes와 무관한, 완전히 고정된 프레임. `_axis_components`에 이것을 넘기면
    # UP_INDEX를 돌려도 local·frame이 똑같이 유지되므로, 출력이 움직이는 유일한
    # 경로는 `_axis_components` 자신이 world_axes를 호출 시점에 쓰는 것뿐이다.
    _FIXED_FRAME = stroke_frame.Basis(
        origin=(0.0, 0.0, 0.0),
        normal=(0.6, 0.3, 0.8),
        tangent=(0.2, 0.9, 0.4),
        up=(0.1, 0.5, 0.7),
    )

    def sample() -> dict[str, object]:
        basis = stroke_frame.stroke_basis(
            hit=(1.2, 4.5, 0.8), tangent=(0.6, 0.0, 0.8), gun=(0.6, 4.9, -0.3)
        )
        axis = measured_score._axis_components((0.02, -0.15, 0.10), _FIXED_FRAME)
        params = motion_key.KeyParams()
        head = (0.5, 4.5, 0.9)
        hit = (0.8, 5.4, 0.6)
        lateral, reach, height = motion_key.body_placement(
            head=head, yaw=0.3, hit=hit, params=params
        )
        key = motion_key.motion_key(
            displacement=(0.4, -0.6, 0.5),
            yaw=0.3,
            head=head,
            hit=hit,
            uv=(1024.0, 512.0),
            params=params,
        )
        out: dict[str, object] = {
            # `tangent`은 일부러 뺀다 — 입력을 그대로 정규화한 값이라 규약과
            # 무관하게 절대 움직이지 않는다(위 독스트링 참고).
            "basis_normal": basis.normal,
            "basis_up": basis.up,
            "yaw": measured_pose.facing_yaw((17.0, 32.0, -11.0)),
            "heading": motion_key.heading_word(0.3, (0.4, -0.6, 0.5)),
            "body_placement_lateral": lateral,
            "body_placement_reach": reach,
            "body_placement_height": height,
            # `signed_band`를 **직접** 잰다. 합성 키 안에서만 보면 그 문자열의 다른
            # 성분이 움직여 band가 얼어붙은 것을 가린다 — `axis` 딕셔너리에서 고친
            # 번들 가림이 문자열 층에서 반복되는 형태다(리뷰가 재현해 확인).
            "signed_band": motion_key.signed_band((0.4, -0.6, 0.5)),
            "motion_key": key,
        }
        for name, (value, unit, word) in axis.items():
            out[f"axis_{name}_value"] = value
            out[f"axis_{name}_unit"] = unit
            out[f"axis_{name}_word"] = word
        return out

    before = sample()
    monkeypatch.setattr(world_axes, "UP_INDEX", 2)  # 유일한 손잡이만 돌린다.
    after = sample()

    stuck = [k for k in before if before[k] == after[k]]
    assert not stuck, f"규약을 따르지 않는 항목: {stuck}"


def test_world_frame_axes_follow_up_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """`measured_score.WORLD_FRAME`도 규약을 따라 움직여야 한다 — 그런데 위 테스트는
    그걸 한 번도 지나가지 않는다(리뷰 E1).

    위 `test_every_module_follows_the_convention_not_a_literal`은 `_axis_components`에
    world_axes와 무관한 `_FIXED_FRAME`을 넘긴다 — `_axis_components` 자신의 리터럴만
    잡도록 일부러 그렇게 설계했다(그 테스트의 독스트링 (2) 참고). 그래서 `WORLD_FRAME`
    자체는 그 테스트에서 한 번도 쓰이지 않는다.

    `WORLD_FRAME`은 장식이 아니다 — `score_track`/`right_axes`/`corrections_text`/
    `issued_axes`/`apply_measured`의 기본 `frame` 인자이고, `service.
    representative_basis`의 실사용 대체값이다. 리뷰 A2 이전에는 `_WorldFrame`이
    frozen dataclass였고 `up: Vec3 = world_axes.UP` 같은 필드 기본값이 클래스
    임포트 시점에 딱 한 번 평가돼 그 뒤로 `UP_INDEX`가 바뀌어도 굳어 있었다. A2가
    그 세 필드(`normal`/`tangent`/`up`)를 프로퍼티로 바꿔 호출 시점마다 `world_axes`를
    다시 읽게 고쳤는데, 위 테스트가 `WORLD_FRAME`을 지나가지 않으므로 그 결함이
    조용히 되돌아갈 수 있다(직접 확인: `_WorldFrame`을 다시 리터럴 필드로 되돌려도
    전체 673개가 그대로 통과한다).

    **왜 UP_INDEX를 두 표적(0과 2)으로 바꾸는가.** `HORIZONTAL_INDICES`는 up_index를
    뺀 나머지 두 인덱스를 오름차순으로 낸다 — up_index=1이면 (0,2), =2면 (0,1),
    =0이면 (1,2)다. 1→2 전환은 첫 자리가 우연히 0으로 유지돼(`(0,2)`→`(0,1)`)
    `normal`(첫 번째 수평 인덱스의 단위벡터)이 이 특정 전환에서는 안 움직인다 —
    직접 실행해 확인했다: `_axis_unit(HORIZONTAL_INDICES[0])`가 UP_INDEX 1에서도
    2에서도 똑같이 `(1.0, 0.0, 0.0)`이다. 이건 구조적 우연이지 결함이 아니라서, 1→2
    전환 하나만으로 `normal`을 재면 항상 거짓 통과(움직였어야 하는데 안 움직인 것처럼
    보이는 게 아니라, 안 움직이는 게 맞는데 그걸 "결함이다"라고 잘못 말하는 반대
    방향의 거짓)가 나온다. 반대로 1→0 전환은 (0,2)→(1,2)라 이번엔 `tangent`(두
    번째 자리, 둘 다 2로 유지)가 안 움직인다. 세 축 중 어느 하나도 이 두 표적 중
    최소 하나에서는 반드시 움직인다는 것을 직접 실행해 확인했다 — 그래서 두 표적
    각각에서 비교하고, 두 표적 모두에서 안 움직인 축만 결함으로 본다.
    """
    measured_score = importlib.import_module("measured_score")

    def axes() -> tuple[object, object, object]:
        return (
            measured_score.WORLD_FRAME.normal,
            measured_score.WORLD_FRAME.tangent,
            measured_score.WORLD_FRAME.up,
        )

    baseline = axes()  # UP_INDEX == 1, 실제 규약.
    monkeypatch.setattr(world_axes, "UP_INDEX", 2)
    at_z = axes()
    monkeypatch.setattr(world_axes, "UP_INDEX", 0)
    at_x = axes()

    names = ("normal", "tangent", "up")
    stuck = [
        name
        for name, base, z, x in zip(names, baseline, at_z, at_x)
        if base == z and base == x
    ]
    assert not stuck, f"WORLD_FRAME이 규약을 따르지 않는 축: {stuck}"


def test_axis_components_exercises_the_vertical_stroke_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_axis_components`의 `along is None` 퇴화 분기(수직 스트로크)를 표본에 넣는다
    — 위 테스트의 `_FIXED_FRAME.tangent = (0.2, 0.9, 0.4)`는 늘 수평 성분이 있어서
    이 분기가 한 번도 안 돈다(리뷰 E2).

    이 분기는 설계가 가장 자주 다루는 사례다 — 스트로크가 순수 수직이면 진행축이
    없어 `lateral`에 좌우를 지어낼 기준이 없고(`word`는 항상 빈 문자열), 대신 남은
    수평 잔차의 크기·방향만 낸다. 직접 실행해 확인했다: `_axis_components`의
    `lateral_cm = math.hypot(h0, h1) * 100.0`(`h0, h1 = world_axes.horizontal_of(plane)`)
    을 `math.hypot(plane[0], plane[2]) * 100.0`으로(월드 축이 아니라 Y-up 리터럴
    인덱스로) 되박아도 673개 전부가 그대로 통과한다 — 이 분기 자체가 어떤 기존
    표본에서도 안 돈다는 뜻이다.

    **왜 `tangent`을 `(0, 0, 0)`으로 두는가.** 벡터 하나가 두 서로 다른 UP_INDEX
    양쪽에서 동시에 "수직"(그 축 성분만 있고 나머지 두 수평 성분이 0)일 수는
    없다 — 어느 한 축에만 실려 있으면 다른 축이 수평이 되는 순간 그 축이 수평
    성분을 갖기 때문이다. 그래서 `tangent = (0.2, 0.9, 0.4)`처럼 어느 한 방향만
    바꾸면, UP_INDEX를 바꿨을 때 분기 자체가 else로 넘어가 버려서(퇴화가 풀려서)
    "출력이 달라졌다"가 분기 안 리터럴 때문인지 분기가 아예 바뀌어서인지 구별이
    안 된다 — 그러면 `math.hypot(plane[0], plane[2])` 리터럴을 되박아도 그냥
    "통과"해 버린다(직접 실행해 확인). 영벡터는 어느 UP_INDEX에서도
    `world_axes.horizontal_of`가 `(0.0, 0.0)`을 내므로 두 표적 모두에서 퇴화가
    유지된다 — 분기 선택 자체(`along = _flattened(frame.tangent)`가 계속 `None`인지)는
    UP_INDEX와 무관하게 항상 참이고, 분기 안의 계산만 UP_INDEX에 반응해야 한다.

    **비교 단위**: `value`만 잰다. `unit`(`_flattened(plane)`)은 이 분기 안에서도
    독립적으로 `world_axes.horizontal_of`를 호출 시점에 읽으므로 리터럴 결함과
    무관하게 항상 움직인다 — `unit`까지 같이 보면 얼어붙은 `value`가 가려진다
    (위 테스트 독스트링 (1)과 같은 함정, 여기서 반복해서 확인했다). `word`는 이
    분기에서 항상 `""`로 고정된 리터럴이라(수직 스트로크에는 좌우를 지어낼 근거가
    없다는 설계 그 자체) UP_INDEX와 무관하게 절대 안 움직인다 — 넣으면 영원히
    고칠 수 없는 거짓 경보가 된다.

    `normal`/`tangent`을 둘 다 `(0, 0, 0)`으로 둬 `wall`이 `None`이 되게 한다 —
    그래야 `plane`이 wall 투영으로 깎이지 않고 온전히 `up` 하나의 기여만 실어,
    `lateral`이 재는 것이 정확히 이 퇴화 분기의 계산이다.
    """
    measured_score = importlib.import_module("measured_score")
    stroke_frame = importlib.import_module("stroke_frame")

    # world_axes와 무관하게 고정된, 접선이 영벡터인(그래서 모든 UP_INDEX에서
    # 수직으로 퇴화하는) 프레임. `origin`은 `_axis_components`가 안 쓴다.
    vertical_tangent_frame = stroke_frame.Basis(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 0.0),
        tangent=(0.0, 0.0, 0.0),
        up=(0.3, 0.9, 0.0),
    )
    local = (0.05, 0.30, 0.12)

    before_value, _before_unit, before_word = measured_score._axis_components(
        local, vertical_tangent_frame
    )["lateral"]
    assert before_word == ""  # 퇴화 분기에 실제로 들어갔는지 확인.

    monkeypatch.setattr(world_axes, "UP_INDEX", 2)

    after_value, _after_unit, after_word = measured_score._axis_components(
        local, vertical_tangent_frame
    )["lateral"]
    assert after_word == ""  # 패치 후에도 여전히 퇴화 분기다(전제가 안 깨졌는지 확인).

    assert before_value != after_value, (
        f"수직 스트로크 lateral 크기가 UP_INDEX를 바꿔도 그대로다: "
        f"{before_value} == {after_value} — world_axes.horizontal_of 대신 "
        f"리터럴 인덱스를 읽었을 가능성이 있다."
    )
