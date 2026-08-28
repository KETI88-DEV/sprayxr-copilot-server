#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

motion_key = importlib.import_module("motion_key")
sweep = importlib.import_module("motion_key_sweep")
stroke_frame_module = importlib.import_module("stroke_frame")


def _record(
    worker: str = "A",
    displacement: tuple[float, float, float] = (0.0, 0.6, 0.0),
    head: tuple[float, float, float] = (0.5, 4.5, 1.5),
    hit: tuple[float, float, float] = (1.2, 4.5, 0.9),
    uv: tuple[float, float] | None = (1024.0, 512.0),
    rotation: tuple[float, float, float] | None = None,
    yaw: float | None = None,
) -> object:
    return sweep.PassRecord(
        worker=worker,
        level="10",
        session="s",
        date="20250901",
        pass_index=0,
        displacement=displacement,
        head=head,
        hit=hit,
        uv=uv,
        gun="한길숏건",
        rotation=rotation,
        yaw=yaw,
    )


def _spread_records() -> list[object]:
    """축마다 값이 갈리는 표본. 어느 축을 빼면 어떤 것이 합쳐지는지 보이게 만든다."""
    records: list[object] = []
    for uv_x in (100.0, 900.0, 1700.0):
        for uv_y in (100.0, 1700.0):
            for reach in (0.4, 1.4):
                for rise in (0.5, -0.5):
                    records.append(
                        _record(
                            displacement=(0.0, 0.3, rise),
                            head=(1.2 - reach, 4.5, 1.5),
                            hit=(1.2, 4.5, 0.9),
                            uv=(uv_x, uv_y),
                        )
                    )
    return records


def _partition(keys: list[str]) -> set[frozenset[int]]:
    groups: dict[str, set[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, set()).add(index)
    return {frozenset(members) for members in groups.values()}


# --- 기본 설정이 현행 키와 같은 분할을 내는가 -----------------------------------


def test_default_config_partitions_exactly_like_the_live_key() -> None:
    """8×8 열이 "현행 입도의 측정치"라고 말하려면 분할이 같아야 한다.

    문자열은 다르다(`sweep:` vs `m1:`) — 설계 D5가 기존 이름공간을 지키라고 했다.
    같아야 하는 것은 어느 두 패스가 같은 버킷에 드는가다.
    """
    records = [
        _record(
            displacement=record.displacement,
            head=record.head,
            hit=record.hit,
            uv=record.uv,
            yaw=math.radians(35.0),
        )
        for record in _spread_records()
    ]
    params = motion_key.KeyParams()
    live = [
        motion_key.motion_key(
            displacement=record.displacement,
            yaw=record.yaw,
            head=record.head,
            hit=record.hit,
            uv=record.uv,
            params=params,
        )
        for record in records
    ]
    swept = [sweep.sweep_key(record, sweep.SweepConfig(name="기본")) for record in records]
    assert _partition(live) == _partition(swept)
    assert all(key.startswith("m1:") for key in live)
    assert all(key.startswith("sweep:") for key in swept)


# --- 축을 빼고 성기게 하면 무엇이 합쳐지는가 -------------------------------------


def test_dropping_the_uv_axis_merges_strokes_on_different_parts_of_the_surface() -> None:
    """③을 빼면 표면 어디를 칠했는지가 키에서 사라진다. 그것이 잃는 것의 이름이다."""
    left = _record(uv=(100.0, 100.0))
    right = _record(uv=(1900.0, 1900.0))
    fine = sweep.SweepConfig(name="uv8", uv_grid=8)
    off = sweep.SweepConfig(name="uv off", uv_grid=None)
    assert sweep.sweep_key(left, fine) != sweep.sweep_key(right, fine)
    assert sweep.sweep_key(left, off) == sweep.sweep_key(right, off)
    assert ":off:" in sweep.sweep_key(left, off)


def test_coarser_uv_grids_are_strictly_nested() -> None:
    """성기게 할수록 키가 줄기만 한다. 늘어나면 격자 계산이 틀린 것이다."""
    records = _spread_records()
    counts = [
        len(sweep.bucketize(records, sweep.SweepConfig(name=str(grid), uv_grid=grid)))
        for grid in (8, 4, 2)
    ]
    counts.append(
        len(sweep.bucketize(records, sweep.SweepConfig(name="off", uv_grid=None)))
    )
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_dropping_the_reach_split_merges_near_and_far_only() -> None:
    """④에서 near/far만 빠진다. 좌우와 높이대는 남아야 한다."""
    near = _record(head=(0.9, 4.5, 1.5))
    far = _record(head=(0.0, 4.5, 1.5))
    with_reach = sweep.SweepConfig(name="with", reach_split=True)
    without = sweep.SweepConfig(name="without", reach_split=False)
    assert sweep.sweep_key(near, with_reach) != sweep.sweep_key(far, with_reach)
    assert sweep.sweep_key(near, without) == sweep.sweep_key(far, without)
    assert sweep.sweep_key(near, without).endswith("-high") or sweep.sweep_key(
        near, without
    ).endswith("-mid")


# --- 점유 통계 -----------------------------------------------------------------


def test_occupancy_counts_multi_worker_buckets_and_preregistered_thresholds() -> None:
    """교차 작업자 비교가 가능한 키가 몇 개인지가 이 측정의 핵심 열이다."""
    solo = [_record(worker="A", uv=(100.0, 100.0)) for _ in range(25)]
    shared = [
        _record(worker="A" if index % 2 else "B", uv=(1900.0, 1900.0))
        for index in range(30)
    ]
    thin = [_record(worker="C", uv=(900.0, 900.0)) for _ in range(3)]
    row = sweep.occupancy(solo + shared + thin, sweep.SweepConfig(name="t", uv_grid=8))
    assert row.passes == 58
    assert row.keys == 3
    assert row.max_n == 30
    assert row.median_n == 25
    assert row.multi_worker_keys == 1
    assert row.keys_n20 == 2
    assert row.keys_n30 == 1
    # solo는 n=25지만 작업자가 1인이라 기준을 못 넘는다(§0 C2).
    assert row.usable_keys == 1
    assert row.usable_passes == 30


def test_sufficiency_criterion_is_the_one_that_was_preregistered() -> None:
    """기준을 뒤에서 옮기면 음성 결과가 조용히 양성이 된다. 상수를 테스트가 잠근다."""
    assert sweep.MIN_BUCKET_N == 20
    assert sweep.MIN_BUCKET_WORKERS == 2
    assert sweep.MIN_COVERAGE == 0.50


def test_coverage_gates_the_verdict_even_when_a_thick_bucket_exists() -> None:
    """두꺼운 버킷 하나가 있어도 대부분의 질의가 못 맞으면 쓸 수 없다(§0 C3)."""
    thick = [
        _record(worker="A" if index % 2 else "B", uv=(1900.0, 1900.0))
        for index in range(30)
    ]
    scattered = [
        _record(worker="C", uv=(100.0 + 60.0 * index, 100.0)) for index in range(200)
    ]
    row = sweep.occupancy(thick + scattered, sweep.SweepConfig(name="t", uv_grid=8))
    assert row.usable_keys == 1
    assert row.coverage < sweep.MIN_COVERAGE
    assert not row.sufficient


# --- 순환 통계 (dataset_spec.md §4.1) -------------------------------------------


def test_circular_statistics_survive_the_zero_360_wrap() -> None:
    """산술 통계가 무의미해지는 바로 그 입력이다. 순환 std가 작아야 맞다."""
    values = [359.0, 0.0, 1.0, 358.0, 2.0]
    mean = sweep.circular_mean_deg(values)
    assert mean is not None
    assert min(mean, 360.0 - mean) < 1.0
    circular = sweep.circular_std_deg(values)
    arithmetic = sweep.arithmetic_std_deg(values)
    assert circular is not None and arithmetic is not None
    assert circular < 2.0
    assert arithmetic > 100.0
    assert arithmetic / circular > 50.0


def test_circular_std_is_zero_for_identical_angles_and_grows_with_spread() -> None:
    identical = sweep.circular_std_deg([42.0] * 5)
    assert identical is not None
    assert math.isclose(identical, 0.0, abs_tol=1e-9)
    assert math.copysign(1.0, identical) > 0.0
    tight = sweep.circular_std_deg([10.0, 12.0, 14.0, 11.0])
    loose = sweep.circular_std_deg([10.0, 60.0, 110.0, 160.0])
    assert tight is not None and loose is not None
    assert tight < loose


def test_circular_statistics_refuse_to_invent_a_direction() -> None:
    """정반대 두 각의 합벡터는 0이다. 평균 방향이 없으면 각도를 지어내지 않는다."""
    assert sweep.circular_mean_deg([0.0, 180.0]) is None
    assert sweep.circular_std_deg([0.0, 180.0]) is None
    assert sweep.circular_mean_deg([]) is None
    assert sweep.circular_std_deg([12.0]) is None


def test_gun_orientation_circular_means_the_pass_and_names_the_measured_gun() -> None:
    """IK 산출 관절각이 아니라 컨트롤러 6-DoF 실측 자세를 읽는다."""
    simulator_csv = importlib.import_module("simulator_csv")
    span = [
        {
            "한길숏건 Rotation X": value,
            "한길숏건 Rotation Y": "90.0",
            "한길숏건 Rotation Z": "10.0",
        }
        for value in ("359.0", "1.0")
    ]
    gun, rotation = sweep.gun_orientation(span, simulator_csv)
    assert gun == "한길숏건"
    assert rotation is not None
    assert min(rotation[0], 360.0 - rotation[0]) < 1e-6
    assert math.isclose(rotation[1], 90.0, abs_tol=1e-6)


def test_gun_orientation_is_none_when_the_column_is_absent() -> None:
    """각도 컬럼이 없으면 0을 채우지 않는다. 없는 값은 없는 값이다."""
    simulator_csv = importlib.import_module("simulator_csv")
    gun, rotation = sweep.gun_orientation([{"인덱스": "0"}], simulator_csv)
    assert gun is None
    assert rotation is None


def test_records_round_trip_through_the_cache_format() -> None:
    record = _record(uv=None, rotation=(12.0, 340.0, 5.5))
    restored = sweep.PassRecord.from_json(record.as_json())
    assert restored == record


# --- 실측 요가 키에 도달하는가 ---------------------------------------------------


def test_sweep_key_uses_the_measured_yaw_when_the_record_carries_one() -> None:
    """요가 0으로 고정되면 ②④는 몸 기준 이름을 단 월드 방위각이다.

    같은 패스를 요만 바꿔 읽었을 때 키가 달라져야 실측 요가 키에 도달한 것이다.
    달라지지 않으면 이 도구가 재는 점유는 여전히 월드 기준 점유다.
    """
    config = sweep.SweepConfig(name="기본")
    world = sweep.sweep_key(_record(yaw=None), config)
    turned = sweep.sweep_key(_record(yaw=math.radians(80.0)), config)
    assert world != turned
    # ①기울기는 착탄 변위에서만 나오므로 요와 무관하다.
    assert world.split(":")[1] == turned.split(":")[1]


def test_missing_yaw_falls_back_to_the_shared_constant() -> None:
    """대체값은 `pose_lab.service.ASSUMED_YAW`와 같아야 한다. 다르면 여기서 잰 점유가
    실험대가 만드는 버킷의 점유가 아니게 된다."""
    config = sweep.SweepConfig(name="기본")
    assert sweep.sweep_key(_record(yaw=None), config) == sweep.sweep_key(
        _record(yaw=sweep.ASSUMED_YAW), config
    )
    assert sweep.ASSUMED_YAW == 0.0


def test_pass_record_round_trips_the_new_fields_through_the_cache() -> None:
    """캐시가 요·건위치·쿼터니언을 잃으면 스캔을 다시 돌려야 한다(479세션, 수 분)."""
    record = sweep.PassRecord(
        worker="A",
        level="10",
        session="s",
        date="20250901",
        pass_index=0,
        displacement=(0.0, 0.6, 0.0),
        head=(0.5, 4.5, 1.5),
        hit=(1.2, 4.5, 0.9),
        uv=(1024.0, 512.0),
        gun="한길숏건",
        rotation=(10.0, 20.0, 30.0),
        yaw=1.234,
        gun_pos=(0.8, 4.5, 1.2),
        quat=(1.0, 0.0, 0.0, 0.0),
    )
    again = sweep.PassRecord.from_json(record.as_json())
    assert again == record


def test_old_cache_lines_still_load() -> None:
    """새 필드가 없는 캐시(이전 연구가 만든 것)를 읽을 때 터지면 재현이 끊긴다."""
    payload = {
        "worker": "A",
        "level": "10",
        "session": "s",
        "date": "20250901",
        "pass_index": 0,
        "displacement": [0.0, 0.6, 0.0],
        "head": [0.5, 4.5, 1.5],
        "hit": [1.2, 4.5, 0.9],
        "uv": [1024.0, 512.0],
        "gun": "한길숏건",
        "rotation": [10.0, 20.0, 30.0],
    }
    record = sweep.PassRecord.from_json(payload)
    assert record.yaw is None and record.gun_pos is None and record.quat is None


# --- 회전 산포: 네 표현 ---------------------------------------------------------


def _rotated(worker: str, angles: tuple[float, float, float], **kwargs: object) -> object:
    """월드 자세만 다른 패스. 스트로크 기저는 기본값에서 동일하다."""
    rotation_math = importlib.import_module("rotation_math")
    quat = rotation_math.matrix_quaternion(rotation_math.euler_matrix("YXZ", angles))
    return _record(worker=worker, rotation=angles, **kwargs) and sweep.PassRecord(
        worker=worker,
        level="10",
        session="s",
        date="20250901",
        pass_index=0,
        displacement=kwargs.get("displacement", (0.0, 0.6, 0.0)),  # pyright: ignore[reportArgumentType]
        head=(0.5, 4.5, 1.5),
        hit=kwargs.get("hit", (1.2, 4.5, 0.9)),  # pyright: ignore[reportArgumentType]
        uv=(1024.0, 512.0),
        gun="한길숏건",
        rotation=angles,
        yaw=0.0,
        gun_pos=kwargs.get("gun_pos", (0.8, 4.5, 1.1)),  # pyright: ignore[reportArgumentType]
        quat=quat,
    )


def test_gate_threshold_is_the_angle_form_of_req19_and_is_not_moved() -> None:
    """`s/e < 0.5`, `e = 30°`이므로 통과선은 15°다. 이 값을 옮기면 판정이 무의미해진다."""
    assert sweep.GATE_ERROR_DEG == 30.0
    assert sweep.GATE_RATIO == 0.5
    assert sweep.GATE_LIMIT_DEG == 15.0


def test_identical_rotations_have_zero_dispersion_in_every_representation() -> None:
    records = [_rotated(f"W{i}", (10.0, 20.0, 30.0)) for i in range(4)]
    world, local, prior = sweep.bucket_dispersions(records)
    assert world.geodesic_mean is not None and world.geodesic_mean < 1e-6
    assert local.geodesic_mean is not None and local.geodesic_mean < 1e-6
    for axis in range(3):
        assert (world.euler[axis] or 0.0) < 1e-6
        assert (prior[axis] or 0.0) < 1e-6


def test_geodesic_dispersion_is_blind_to_which_euler_axis_moved() -> None:
    """오일러 축별 값은 어느 축이 움직였는지에 따라 갈리지만 측지 산포는 같다.

    이것이 가설 (나)를 재는 근거다 — 회전량이 같으면 표현과 무관하게 같은 값이 나와야
    한다.
    """
    rotation_math = importlib.import_module("rotation_math")
    spreads = []
    for axis in "XYZ":
        quats = [
            rotation_math.matrix_quaternion(rotation_math.axis_matrix(axis, angle))
            for angle in (-20.0, 0.0, 20.0)
        ]
        spread = sweep.dispersion_of(quats)
        assert spread.geodesic_mean is not None
        spreads.append(spread.geodesic_mean)
    assert max(spreads) - min(spreads) < 1e-6


def test_stroke_frame_removes_a_rotation_that_only_followed_the_surface() -> None:
    """가설 (가)를 잠그는 테스트.

    스트로크 방향만 다르고 **기법이 같은** 두 패스를 만든다. 건이 스트로크 프레임에서
    같은 자세를 유지하면 월드 자세는 갈리지만 로컬 자세는 같아야 한다. 스트로크 프레임
    산포가 월드 산포보다 작지 않으면 (가)를 재는 경로가 배선되지 않은 것이다.
    """
    rotation_math = importlib.import_module("rotation_math")
    records: list[object] = []
    for index, tangent in enumerate(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
        hit = (1.2, 4.5, 0.9)
        gun_pos = (0.8, 4.5, 1.1)
        basis = stroke_frame_module.stroke_basis(hit, tangent, gun_pos)
        frame = rotation_math.basis_matrix((basis.normal, basis.tangent, basis.up))
        # 로컬에서 항등 자세. 월드 자세는 기저를 그대로 따라간다.
        quat = rotation_math.matrix_quaternion(frame)
        records.append(
            sweep.PassRecord(
                worker=f"W{index}",
                level="10",
                session="s",
                date="20250901",
                pass_index=index,
                displacement=tangent,
                head=(0.5, 4.5, 1.5),
                hit=hit,
                uv=(1024.0, 512.0),
                gun="한길숏건",
                rotation=None,
                yaw=0.0,
                gun_pos=gun_pos,
                quat=quat,
            )
        )
    world, local, _ = sweep.bucket_dispersions(records)  # pyright: ignore[reportArgumentType]
    assert world.geodesic_mean is not None and world.geodesic_mean > 30.0
    assert local.geodesic_mean is not None and local.geodesic_mean < 1e-6


def test_worker_split_separates_within_from_between() -> None:
    """한 사람 안의 흔들림만 있으면 작업자 간 성분이 0에 붙어야 한다."""
    rotation_math = importlib.import_module("rotation_math")

    def quats(offsets: tuple[float, ...]) -> list[object]:
        return [
            rotation_math.matrix_quaternion(rotation_math.axis_matrix("Z", angle))
            for angle in offsets
        ]

    same_centre = {
        "A": quats((-20.0, 0.0, 20.0, 10.0, -10.0)),
        "B": quats((-20.0, 0.0, 20.0, 10.0, -10.0)),
    }
    split = sweep.worker_split(same_centre)  # pyright: ignore[reportArgumentType]
    assert split is not None
    total, between = split
    assert total > 10.0
    assert between < 1e-6

    apart = {
        "A": quats((-2.0, 0.0, 2.0, 1.0, -1.0)),
        "B": quats((58.0, 60.0, 62.0, 61.0, 59.0)),
    }
    split = sweep.worker_split(apart)  # pyright: ignore[reportArgumentType]
    assert split is not None
    assert split[1] > 25.0


def test_usable_buckets_apply_the_preregistered_criterion() -> None:
    records = [_record(worker="A") for _ in range(19)]
    assert sweep.usable_buckets(records, sweep.SweepConfig(name="기본")) == []  # pyright: ignore[reportArgumentType]
    records = [_record(worker="A") for _ in range(20)]
    assert sweep.usable_buckets(records, sweep.SweepConfig(name="기본")) == []  # pyright: ignore[reportArgumentType]
    records = [_record(worker="A") for _ in range(10)] + [_record(worker="B") for _ in range(10)]
    assert len(sweep.usable_buckets(records, sweep.SweepConfig(name="기본"))) == 1  # pyright: ignore[reportArgumentType]
