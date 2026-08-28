#!/usr/bin/env python3
"""동작 키 입도를 쓸어 버킷 점유를 재고, 잘 찬 버킷에서 건 각도의 순환 통계를 낸다.

이것은 기능이 아니라 **측정 도구**다. 답해야 하는 질문 하나가 있다 — "여기서 숙련자
대비 건 각도가 N도 벗어났습니다"라고 말할 수 있는가. 그 문장은 버킷 안의 분포를
요구하고, 분포는 표본을 요구하며, 4축 키(설계 §4)는 표본을 잘게 쪼갠다. 얼마나 잘게
쪼개는지가 여기서 나오는 숫자다.

**기존 키를 건드리지 않는다.** `motion_key.motion_key`가 내는 `m1:` 문자열은 그대로
두고, 이 모듈은 `sweep:` 접두어로 문자열 공간을 따로 쓴다(설계 D5). 기본 설정
(`SweepConfig()`)은 `m1:`과 **같은 분할**을 내야 하며 그것을 테스트가 지킨다 — 문자열은
다르지만 어느 두 패스가 같은 버킷에 드는지는 같다. 그래야 여기서 잰 8×8 열이 현행
입도의 측정치라고 말할 수 있다.

각도는 반드시 **순환 통계**로 낸다. `dataset_spec.md` §4.1이 기록한 대로 Spine X의
산술 std 143.5°는 0/360 wrap 인공물이었고 순환 std는 5.5°였다. 산술 통계를 쓴 모듈이
그 이유로 무효 판정을 받았다. 여기서는 산술값을 **대조용으로만** 함께 내고 판정에는
쓰지 않는다.

무엇을 측정하지 않는가도 적어 둔다. 이 모듈은 각도를 교정문에 싣지 않는다. REQ-19
`correction_target_gate`는 그대로 유효하고, 여기서 나오는 수치는 그 게이트를 통과시키는
근거가 아니라 게이트에 **입력할 분포가 존재하는지**를 재는 값이다.

사용:

    # 1) 실데이터를 한 번 훑어 패스 레코드를 캐시로 굳힌다 (느리다)
    python3 -m motion_key_sweep scan --root /data/sources/professionals \\
        --levels 1-10 --out /tmp/passes.jsonl

    # 2) 캐시 위에서 입도를 쓸어 표를 낸다 (빠르다, 반복 가능)
    python3 -m motion_key_sweep report --cache /tmp/passes.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence

import measured_pose
import motion_key
import rotation_math
import stroke_frame

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# `m1:`과 문자열 공간을 나눈다. 측정용 키가 레퍼런스 아티팩트로 새어 들어가면 안 된다.
SWEEP_PREFIX: Final = "sweep"

# 요를 못 읽은 패스의 대체값. `pose_lab.service.ASSUMED_YAW`와 같은 값을 써야 여기서
# 잰 점유가 실험대가 실제로 만드는 버킷의 점유가 된다. 실측 요가 있으면 그것을 쓴다.
ASSUMED_YAW: Final = 0.0

# 각도 컬럼. 설계 §2.2가 오른손 컨트롤러로 지목한 건이고, IK 산출 관절각이 아니라
# 컨트롤러의 6-DoF 실측 자세다.
GUN_ROTATION_GROUP: Final = "한길숏건"

# §0 사전등록 기준. 숫자를 보기 전에 고정했고 뒤에서 옮기지 않는다.
MIN_BUCKET_N: Final = 20
MIN_BUCKET_WORKERS: Final = 2
MIN_COVERAGE: Final = 0.50

# REQ-19 `correction_target_gate`의 각도판. `pose_lab.service.DEFAULT_ERROR_M`(0.144 m)이
# "팔꿈치 30도에 해당하는 실측 참조값"이므로 각도판 e는 30°이고, `s/e < 0.5`는 곧
# **버킷 산포 15° 미만**이다. 이 값을 옮기지 않는다(설계 D4).
GATE_ERROR_DEG: Final = 30.0
GATE_RATIO: Final = 0.5
GATE_LIMIT_DEG: Final = GATE_ERROR_DEG * GATE_RATIO

# 작업자 내/간 분해에서 한 작업자를 세는 최소 패스 수. 이전 연구와 같은 값이다.
MIN_WORKER_PASSES: Final = 5


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """키 입도 한 가지. 기본값이 현행 `motion_key.KeyParams`와 같은 분할을 낸다.

    `uv_grid=None`은 ③ 대상 지점 축을 **통째로 뺀다**(격자를 성기게 하는 것과 다르다).
    `reach_split=False`는 ④에서 near/far만 뺀다 — 좌우와 높이대는 남는다.
    """

    name: str
    uv_grid: int | None = 8
    reach_split: bool = True
    lateral_deg: float = 30.0
    reach_m: float = 0.8


@dataclass(frozen=True, slots=True)
class PassRecord:
    """패스 한 건. 키 재료와 건 각도를 함께 들고 있다.

    키 재료(`displacement`/`head`/`hit`/`uv`)만 들고 있는 이유는 입도가 요청마다
    달라지기 때문이다. 키 문자열을 굳혀 두면 격자를 바꿀 때마다 CSV 479개를 다시
    읽어야 한다.
    """

    worker: str
    level: str
    session: str
    date: str
    pass_index: int
    displacement: Vec3
    head: Vec3
    hit: Vec3
    uv: tuple[float, float] | None
    gun: str | None
    rotation: Vec3 | None
    # 실측 요(라디안). None이면 회전을 못 읽어 `ASSUMED_YAW`로 떨어진 패스다.
    yaw: float | None = None
    # 패스 대표 프레임의 건 위치. 스트로크 기저의 법선 **부호**를 고르는 데만 쓴다
    # (`stroke_frame.stroke_basis`). 회전을 스트로크 프레임으로 옮기려면 필요하다.
    gun_pos: Vec3 | None = None
    # 패스 구간의 건 자세를 쿼터니언으로 평균한 값. 오일러 축별 순환 평균과 **다른**
    # 값이며, 축별 평균은 회전공간의 평균이 아니기 때문에 둘을 다 들고 있는다.
    quat: tuple[float, float, float, float] | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "worker": self.worker,
            "level": self.level,
            "session": self.session,
            "date": self.date,
            "pass_index": self.pass_index,
            "displacement": list(self.displacement),
            "head": list(self.head),
            "hit": list(self.hit),
            "uv": None if self.uv is None else list(self.uv),
            "gun": self.gun,
            "rotation": None if self.rotation is None else list(self.rotation),
            "yaw": self.yaw,
            "gun_pos": None if self.gun_pos is None else list(self.gun_pos),
            "quat": None if self.quat is None else list(self.quat),
        }

    @staticmethod
    def from_json(payload: Mapping[str, Any]) -> "PassRecord":
        uv = payload["uv"]
        rotation = payload["rotation"]
        gun_pos = payload.get("gun_pos")
        quat = payload.get("quat")
        yaw = payload.get("yaw")
        return PassRecord(
            worker=str(payload["worker"]),
            level=str(payload["level"]),
            session=str(payload["session"]),
            date=str(payload["date"]),
            pass_index=int(payload["pass_index"]),
            displacement=_vec(payload["displacement"]),
            head=_vec(payload["head"]),
            hit=_vec(payload["hit"]),
            uv=None if uv is None else (float(uv[0]), float(uv[1])),
            gun=None if payload["gun"] is None else str(payload["gun"]),
            rotation=None if rotation is None else _vec(rotation),
            yaw=None if yaw is None else float(yaw),
            gun_pos=None if gun_pos is None else _vec(gun_pos),
            quat=None if quat is None else (
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            ),
        )


def _vec(values: Sequence[Any]) -> Vec3:
    return (float(values[0]), float(values[1]), float(values[2]))


# --- 키 -----------------------------------------------------------------------


def sweep_key(record: PassRecord, config: SweepConfig) -> str:
    """한 패스의 키를 이 입도에서 적는다. 축 구성은 설계 §4의 4축 그대로다.

    `motion_key`의 축 판정 함수를 그대로 부른다. 여기서 판정을 다시 구현하면 측정
    대상과 측정 도구가 갈라져, 잰 숫자가 실험대의 버킷 점유가 아니게 된다.
    """
    params = motion_key.KeyParams(
        uv_grid=config.uv_grid if config.uv_grid else 1,
        lateral_deg=config.lateral_deg,
        reach_m=config.reach_m,
    )
    yaw = ASSUMED_YAW if record.yaw is None else record.yaw
    band = motion_key.signed_band(record.displacement)
    heading = motion_key.heading_word(yaw, record.displacement)
    lateral, reach, height = motion_key.body_placement(record.head, yaw, record.hit, params)
    if config.uv_grid is None:
        cell = "off"
    elif record.uv is None:
        cell = "na"
    else:
        cell = "{}x{}".format(*motion_key.uv_cell(record.uv, params))
    place = f"{lateral}-{reach}-{height}" if config.reach_split else f"{lateral}-{height}"
    return f"{SWEEP_PREFIX}:{band}:{heading}:{cell}:{place}"


def bucketize(
    records: Iterable[PassRecord], config: SweepConfig
) -> dict[str, list[PassRecord]]:
    buckets: dict[str, list[PassRecord]] = defaultdict(list)
    for record in records:
        buckets[sweep_key(record, config)].append(record)
    return dict(buckets)


# --- 점유 ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Occupancy:
    """입도 하나의 점유 요약. `usable_*`가 §0 사전등록 기준을 그대로 잰 값이다."""

    config: str
    passes: int
    keys: int
    median_n: float
    max_n: int
    multi_worker_keys: int
    keys_n20: int
    keys_n30: int
    usable_keys: int
    usable_passes: int

    @property
    def coverage(self) -> float:
        return 0.0 if not self.passes else self.usable_passes / self.passes

    @property
    def sufficient(self) -> bool:
        """§0 C1·C2·C3를 모두 만족하는가. 기준은 상수로 박혀 있고 여기서 옮기지 않는다."""
        return self.usable_keys > 0 and self.coverage >= MIN_COVERAGE


def occupancy(records: Sequence[PassRecord], config: SweepConfig) -> Occupancy:
    buckets = bucketize(records, config)
    sizes = sorted(len(items) for items in buckets.values())
    usable = [
        items
        for items in buckets.values()
        if len(items) >= MIN_BUCKET_N
        and len({item.worker for item in items}) >= MIN_BUCKET_WORKERS
    ]
    return Occupancy(
        config=config.name,
        passes=len(records),
        keys=len(buckets),
        median_n=statistics.median(sizes) if sizes else 0.0,
        max_n=max(sizes) if sizes else 0,
        multi_worker_keys=sum(
            1 for items in buckets.values() if len({item.worker for item in items}) > 1
        ),
        keys_n20=sum(1 for size in sizes if size >= 20),
        keys_n30=sum(1 for size in sizes if size >= 30),
        usable_keys=len(usable),
        usable_passes=sum(len(items) for items in usable),
    )


# --- 순환 통계 ----------------------------------------------------------------
#
# dataset_spec.md §4.1: 오일러 각에 산술 통계를 쓰면 0/360 wrap 때문에 결과가
# 무의미해진다(Spine X 산술 std 143.5° vs 순환 std 5.5°). 아래가 유일한 판정 경로다.


def circular_mean_deg(values: Sequence[float]) -> float | None:
    """원형 평균. `atan2(mean sin θ, mean cos θ)`를 [0,360)으로 적는다.

    각을 단위원 위의 점으로 옮겨 벡터 평균을 내고 그 방향을 되읽는다. 359°와 1°의
    평균이 180°가 아니라 0°가 되는 이유가 이것이다.
    """
    if not values:
        return None
    count = len(values)
    cos_bar = sum(math.cos(math.radians(value)) for value in values) / count
    sin_bar = sum(math.sin(math.radians(value)) for value in values) / count
    if math.hypot(cos_bar, sin_bar) < 1e-12:
        # 합벡터가 0이면 평균 방향이 정의되지 않는다. 각도를 지어내지 않는다.
        return None
    return math.degrees(math.atan2(sin_bar, cos_bar)) % 360.0


def circular_std_deg(values: Sequence[float]) -> float | None:
    """순환 표준편차 `√(−2 ln R)`. R은 합벡터 길이 `√(C̄²+S̄²)`다.

    R=1(완전 일치)이면 0, R→0(균등 분포)이면 발산한다. 산술 std와 달리 wrap에
    영향받지 않는다.
    """
    if len(values) < 2:
        return None
    count = len(values)
    cos_bar = sum(math.cos(math.radians(value)) for value in values) / count
    sin_bar = sum(math.sin(math.radians(value)) for value in values) / count
    resultant = math.hypot(cos_bar, sin_bar)
    if resultant <= 1e-12:
        return None
    # log(1.0)이 -0.0을 내 sqrt가 -0.0이 된다. 0.0으로 못 박는다.
    return math.degrees(math.sqrt(max(0.0, -2.0 * math.log(min(1.0, resultant)))))


def arithmetic_std_deg(values: Sequence[float]) -> float | None:
    """대조용. **판정에 쓰지 않는다.** wrap 인공물의 크기를 눈에 보이게 하려고만 낸다."""
    return statistics.pstdev(values) if len(values) >= 2 else None


# --- 회전 산포: 네 가지 표현 ---------------------------------------------------
#
# 이전 연구(`pose-lab-bucket-study.md` §6.2)가 남긴 두 가설을 갈라 재기 위한 것이다.
#
# (가) **월드 오일러가 표면 방향을 산포로 흡수한다.** 벽이 휘면 작업자의 기법이 똑같아도
#      표면을 따라가는 것만으로 건의 월드 자세가 변한다. 그 변화가 버킷 산포로 나타난다.
#      스트로크 프레임(벽 법선·진행 접선·수직)에서 다시 재면 그 몫이 빠진다.
#
# (나) **오일러 축별 산포는 뜻 있는 요약이 아니다.** 세 축은 독립이 아니고 특이점에서
#      결합한다. X가 8.7°인데 Y·Z가 120°인 버킷은 그 결합의 지문일 수 있다. 표현에
#      무관한 측도 — 쿼터니언 측지 산포 — 로 재면 갈린다.
#
# 두 축을 곱해 네 칸을 전부 낸다. 하나를 골라 통과를 만들지 않기 위해서다.


@dataclass(frozen=True, slots=True)
class Dispersion:
    """버킷 하나의 회전 산포. 네 칸 중 한 칸이다."""

    n: int
    euler: tuple[float | None, float | None, float | None]
    geodesic_mean: float | None
    geodesic_rms: float | None


def stroke_basis_of(record: PassRecord) -> Any | None:
    """패스 대표 스트로크 기저. `service.representative_basis`와 같은 규칙이다.

    접선은 착탄점 순 변위, 수직은 월드 Z를 접선에 직교화한 축, 법선은 둘의 외적이고
    부호만 건 위치가 고른다. 축이 측정 대상(건 자세)에서 나오지 않는다는 것이 이
    기저의 요점이며, 그래서 여기에 건 회전을 옮겨도 순환논법이 되지 않는다.
    """
    if record.gun_pos is None:
        return None
    return stroke_frame.stroke_basis(record.hit, record.displacement, record.gun_pos)


def local_quaternion(record: PassRecord) -> Quat | None:
    """건 자세를 스트로크 프레임으로 옮긴 회전. `R_local = Bᵀ · R_world`."""
    if record.quat is None:
        return None
    basis = stroke_basis_of(record)
    if basis is None:
        return None
    frame = rotation_math.basis_matrix((basis.normal, basis.tangent, basis.up))
    world = rotation_math.quaternion_matrix(record.quat)
    return rotation_math.matrix_quaternion(
        rotation_math.matmul(rotation_math.transpose(frame), world)
    )


def euler_of(quat: Quat) -> Vec3:
    """쿼터니언을 다시 오일러 축으로. 규약은 `measured_pose`가 고른 하나뿐이다."""
    return rotation_math.euler_from_matrix(
        measured_pose.EULER_ORDER,
        rotation_math.quaternion_matrix(quat),
        measured_pose.EULER_SIGN,
    )


def dispersion_of(quats: Sequence[Quat]) -> Dispersion:
    """같은 표본에서 축별 순환 std와 측지 산포를 **함께** 낸다.

    같은 쿼터니언 표본에서 둘 다 뽑는 것이 요점이다. 축별은 `record.rotation`(프레임
    축별 순환 평균)에서, 측지는 쿼터니언 평균에서 뽑으면 대표값이 달라 네 칸이
    비교 가능하지 않게 된다.
    """
    if len(quats) < 2:
        return Dispersion(len(quats), (None, None, None), None, None)
    angles = [euler_of(quat) for quat in quats]
    euler = tuple(
        circular_std_deg([angle[axis] for angle in angles]) for axis in range(3)
    )
    spread = rotation_math.geodesic_spread(quats)
    return Dispersion(
        n=len(quats),
        euler=euler,  # pyright: ignore[reportArgumentType]
        geodesic_mean=None if spread is None else spread[0],
        geodesic_rms=None if spread is None else spread[1],
    )


def bucket_dispersions(
    items: Sequence[PassRecord],
) -> tuple[Dispersion, Dispersion, tuple[float | None, float | None, float | None]]:
    """(월드, 스트로크 프레임, 이전 연구식 월드 축별) 셋.

    세 번째는 `record.rotation` — 프레임마다의 오일러를 **축별로** 순환 평균한 값 — 위의
    순환 std다. 이전 연구 §4.4가 낸 바로 그 통계이며, 재현 대조로만 싣는다.
    """
    world = [item.quat for item in items if item.quat is not None]
    local = [quat for quat in (local_quaternion(item) for item in items) if quat is not None]
    prior = tuple(
        circular_std_deg([item.rotation[axis] for item in items if item.rotation is not None])
        for axis in range(3)
    )
    return dispersion_of(world), dispersion_of(local), prior  # pyright: ignore[reportReturnType]


def usable_buckets(
    records: Sequence[PassRecord], config: SweepConfig
) -> list[tuple[str, list[PassRecord]]]:
    """§0 C1·C2를 만족하는 버킷만. 기준은 상수로 박혀 있고 여기서 옮기지 않는다."""
    return [
        (key, items)
        for key, items in sorted(bucketize(records, config).items())
        if len(items) >= MIN_BUCKET_N
        and len({item.worker for item in items}) >= MIN_BUCKET_WORKERS
    ]


def _median(values: Sequence[float | None]) -> float | None:
    found = [value for value in values if value is not None]
    return statistics.median(found) if found else None


def _pass_fraction(values: Sequence[float | None]) -> float | None:
    """게이트를 통과하는 버킷의 비율. 재지 못한 버킷은 분모에서도 뺀다."""
    found = [value for value in values if value is not None]
    if not found:
        return None
    return sum(1 for value in found if value < GATE_LIMIT_DEG) / len(found)


@dataclass(frozen=True, slots=True)
class DispersionSummary:
    """한 입도에서 네 표현의 산포 요약. 각 칸이 (중앙값, 게이트 통과 비율)이다."""

    config: str
    buckets: int
    coverage: float
    world_euler: tuple[tuple[float | None, float | None], ...]
    world_geodesic_mean: tuple[float | None, float | None]
    world_geodesic_rms: tuple[float | None, float | None]
    local_euler: tuple[tuple[float | None, float | None], ...]
    local_geodesic_mean: tuple[float | None, float | None]
    local_geodesic_rms: tuple[float | None, float | None]
    prior_euler: tuple[tuple[float | None, float | None], ...]


def dispersion_summary(
    records: Sequence[PassRecord], config: SweepConfig
) -> DispersionSummary:
    buckets = usable_buckets(records, config)
    rows = [bucket_dispersions(items) for _, items in buckets]
    total = len(records)
    covered = sum(len(items) for _, items in buckets)

    def axis_cell(
        pick: Any, axis: int
    ) -> tuple[float | None, float | None]:
        values = [pick(row)[axis] for row in rows]
        return (_median(values), _pass_fraction(values))

    def scalar_cell(pick: Any) -> tuple[float | None, float | None]:
        values = [pick(row) for row in rows]
        return (_median(values), _pass_fraction(values))

    return DispersionSummary(
        config=config.name,
        buckets=len(buckets),
        coverage=0.0 if not total else covered / total,
        world_euler=tuple(axis_cell(lambda row: row[0].euler, axis) for axis in range(3)),
        world_geodesic_mean=scalar_cell(lambda row: row[0].geodesic_mean),
        world_geodesic_rms=scalar_cell(lambda row: row[0].geodesic_rms),
        local_euler=tuple(axis_cell(lambda row: row[1].euler, axis) for axis in range(3)),
        local_geodesic_mean=scalar_cell(lambda row: row[1].geodesic_mean),
        local_geodesic_rms=scalar_cell(lambda row: row[1].geodesic_rms),
        prior_euler=tuple(axis_cell(lambda row: row[2], axis) for axis in range(3)),
    )


# --- 산포의 출처: 작업자 내 vs 작업자 간 ----------------------------------------


def worker_split(quats_by_worker: Mapping[str, Sequence[Quat]]) -> tuple[float, float] | None:
    """(버킷 전체 측지 산포, 작업자 평균들 사이의 측지 산포). 둘 다 RMS다.

    `총² ≈ 작업자내² + 작업자간²`이므로 둘의 비가 산포의 출처를 가른다. 이전 연구는
    오일러 축별 순환 std로 이 비를 재서 작업자 내 몫 ≈ 94%를 얻었다. 여기서는 회전
    공간에서 같은 분해를 한다 — 인공물이 그 비를 부풀렸다면 여기서 내려가야 한다.
    """
    pooled = [quat for values in quats_by_worker.values() for quat in values]
    total = rotation_math.geodesic_spread(pooled)
    if total is None:
        return None
    means = [
        mean
        for values in quats_by_worker.values()
        if len(values) >= MIN_WORKER_PASSES
        and (mean := rotation_math.quaternion_mean(values)) is not None
    ]
    between = rotation_math.geodesic_spread(means)
    if between is None:
        return None
    return (total[1], between[1])


# --- 실데이터 훑기 ------------------------------------------------------------


def scan(
    root: Path,
    levels: Sequence[str],
    workers: Sequence[str] | None = None,
    session_limit: int = 8,
    progress: bool = False,
) -> Iterator[PassRecord]:
    """CSV 트리를 패스 단위로 훑는다. 세션 하나가 파싱하면 250 MB 규모라 게으르게 돈다.

    `pose_lab.csv_source`의 색인을 그대로 쓴다. 세그멘테이션 규칙을 여기서 다시 쓰면
    실험대가 만드는 패스와 여기서 세는 패스가 달라진다 — 잰 점유가 실험대의 점유가
    아니게 된다.

    세션마다 `index_of`가 행을 한 번 파싱해 LRU에 넣는다. 각도를 읽으려고 여기서
    `read_rows`를 다시 부르면 479세션을 두 번 파싱하게 되므로, 캐시에 들어간 그 행을
    그대로 꺼내 쓴다.
    """
    csv_source = _load_csv_source()
    source = csv_source.CsvPassSource(root, rows_cache=1)
    wanted_levels = set(levels)
    names = set(workers) if workers else set(source.workers())

    seen: dict[tuple[str, str], int] = defaultdict(int)
    for item in source.sessions():
        if item.worker not in names or item.level not in wanted_levels:
            continue
        if seen[(item.worker, item.level)] >= session_limit:
            continue
        seen[(item.worker, item.level)] += 1
        entries = source.index_of(item)
        rows = source._read(item.path)  # pyright: ignore[reportPrivateUsage]
        if progress:
            print(
                f"  {item.worker} L{item.level} {item.date} {item.session}: "
                f"{len(entries)} passes",
                file=sys.stderr,
                flush=True,
            )
        for entry in entries:
            span = rows[entry.row_start : entry.row_end + 1]
            gun, rotation = gun_orientation(span, _simulator_csv())
            inputs = entry.key_inputs
            yield PassRecord(
                worker=item.worker,
                level=item.level,
                session=item.session,
                date=item.date,
                pass_index=entry.pass_index,
                displacement=tuple(inputs.displacement),
                head=tuple(inputs.head),
                hit=tuple(inputs.hit),
                uv=None if inputs.uv is None else tuple(inputs.uv),
                gun=gun,
                rotation=rotation,
                yaw=inputs.yaw,
                gun_pos=gun_position(span, _simulator_csv()),
                quat=gun_quaternion(span, _simulator_csv()),
            )


def _simulator_csv() -> Any:
    return sys.modules.get("simulator_csv") or importlib.import_module("simulator_csv")


def gun_orientation(
    span: Sequence[Mapping[str, str]], simulator_csv: Any
) -> tuple[str | None, Vec3 | None]:
    """패스 한 건의 대표 건 각도. 축마다 프레임들의 **순환 평균**이다.

    프레임 하나를 뽑지 않고 구간 전체를 순환 평균하는 이유는 단일 프레임이 IK 플립이나
    측정 잡음 하나에 통째로 끌려가기 때문이다. 산술 평균을 쓰면 0/360 경계를 지나는
    패스가 180° 근처로 끌려간다(`dataset_spec.md` §4.1).
    """
    series: list[list[float]] = [[], [], []]
    for row in span:
        try:
            values = simulator_csv.vec3(row, GUN_ROTATION_GROUP, "Rotation")
        except (KeyError, ValueError):
            continue
        for axis in range(3):
            series[axis].append(float(values[axis]))
    if not series[0]:
        return None, None
    means = [circular_mean_deg(axis_values) for axis_values in series]
    if any(value is None for value in means):
        return GUN_ROTATION_GROUP, None
    return GUN_ROTATION_GROUP, (means[0], means[1], means[2])  # pyright: ignore[reportArgumentType]


def gun_position(span: Sequence[Mapping[str, str]], simulator_csv: Any) -> Vec3 | None:
    """구간 가운데 프레임의 건 위치.

    스트로크 기저의 법선 **부호**를 고르는 데만 쓰인다(`stroke_frame.stroke_basis`가
    건을 그 용도로만 받는다). 축 방향 자체는 착탄 궤적의 기하에서만 나오므로 어느
    프레임을 고르든 결과가 같고, 가운데를 쓰는 것은 `service.representative_basis`가
    가운데 격자점을 쓰는 것과 맞추기 위함이다.
    """
    if not span:
        return None
    row = span[len(span) // 2]
    try:
        x, y, z = simulator_csv.vec3(row, GUN_ROTATION_GROUP, "Position")
    except (KeyError, ValueError):
        return None
    return (x, y, z)


def gun_quaternion(
    span: Sequence[Mapping[str, str]], simulator_csv: Any
) -> tuple[float, float, float, float] | None:
    """패스 대표 건 자세를 **회전공간에서** 평균한 값.

    `gun_orientation`의 축별 순환 평균과 다른 값이다. 오일러 세 축을 따로 평균하면
    회전이 아닌 것이 나올 수 있다 — 세 축은 독립이 아니고 특이점에서 결합한다. 여기서는
    프레임마다 회전 행렬을 만들어 쿼터니언으로 옮기고 `rotation_math.quaternion_mean`
    (4×4 산포행렬의 최대 고유벡터)으로 평균한다.

    오일러 규약은 `measured_pose.EULER_ORDER`/`EULER_SIGN` 하나만 쓴다. 그 규약은
    건의 분사 광선이 `hit − gun`과 이루는 3차원 각의 중앙값 4.0°로 실측에서 골랐다.
    """
    quats: list[tuple[float, float, float, float]] = []
    for row in span:
        try:
            values = simulator_csv.vec3(row, GUN_ROTATION_GROUP, "Rotation")
        except (KeyError, ValueError):
            continue
        matrix = rotation_math.euler_matrix(
            measured_pose.EULER_ORDER, (values[0], values[1], values[2]),
            measured_pose.EULER_SIGN,
        )
        quats.append(rotation_math.matrix_quaternion(matrix))
    return rotation_math.quaternion_mean(quats) if quats else None


def _load_csv_source() -> Any:
    """`pose_lab.csv_source`를 늦게 부른다. 순수 함수만 쓰는 테스트가 CSV를 안 타게 한다."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from trainer.pose_lab import csv_source

    return csv_source


# --- 보고 --------------------------------------------------------------------


def default_configs() -> list[SweepConfig]:
    """설계가 요구한 최소 스윕. UV 격자 4단계 × 거리 분할 2단계 + 좌우 경계각 변주."""
    configs = [
        SweepConfig(name="uv8 + near/far (현행)", uv_grid=8, reach_split=True),
        SweepConfig(name="uv4 + near/far", uv_grid=4, reach_split=True),
        SweepConfig(name="uv2 + near/far", uv_grid=2, reach_split=True),
        SweepConfig(name="uv off + near/far", uv_grid=None, reach_split=True),
        SweepConfig(name="uv8, 거리분할 없음", uv_grid=8, reach_split=False),
        SweepConfig(name="uv4, 거리분할 없음", uv_grid=4, reach_split=False),
        SweepConfig(name="uv2, 거리분할 없음", uv_grid=2, reach_split=False),
        SweepConfig(name="uv off, 거리분할 없음", uv_grid=None, reach_split=False),
    ]
    for degrees in (20.0, 45.0):
        configs.append(
            SweepConfig(
                name=f"uv off, 거리분할 없음, 좌우 경계 {degrees:.0f}",
                uv_grid=None,
                reach_split=False,
                lateral_deg=degrees,
            )
        )
    return configs


def occupancy_rows(
    records: Sequence[PassRecord], configs: Sequence[SweepConfig]
) -> list[Occupancy]:
    return [occupancy(records, config) for config in configs]


def format_table(rows: Sequence[Occupancy]) -> str:
    header = (
        "| 설정 | 서로 다른 키 | 키당 중앙값 | 최대 n | 2인 이상 키 | n≥20 | n≥30 "
        "| 기준충족 키 | 커버리지 |\n|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row.config} | {row.keys} | {row.median_n:.0f} | {row.max_n} | "
            f"{row.multi_worker_keys} | {row.keys_n20} | {row.keys_n30} | "
            f"{row.usable_keys} | {row.coverage * 100:.1f}% |"
        )
    return "\n".join(lines)


def bucket_angle_report(
    records: Sequence[PassRecord], config: SweepConfig, top: int = 10
) -> list[dict[str, Any]]:
    """가장 잘 찬 버킷들의 건 각도 순환 통계. 산술 std는 대조용으로만 함께 낸다."""
    buckets = bucketize(records, config)
    ranked = sorted(buckets.items(), key=lambda item: -len(item[1]))
    report: list[dict[str, Any]] = []
    for key, items in ranked[:top]:
        rows = [item for item in items if item.rotation is not None]
        entry: dict[str, Any] = {
            "key": key,
            "n": len(items),
            "n_with_rotation": len(rows),
            "workers": sorted({item.worker for item in items}),
        }
        for axis, name in enumerate("XYZ"):
            values = [item.rotation[axis] for item in rows if item.rotation is not None]
            entry[f"mean_{name}"] = circular_mean_deg(values)
            entry[f"circ_std_{name}"] = circular_std_deg(values)
            entry[f"arith_std_{name}"] = arithmetic_std_deg(values)
        report.append(entry)
    return report


def _cell(cell: tuple[float | None, float | None]) -> str:
    median, fraction = cell
    left = "—" if median is None else f"{median:.1f}°"
    right = "—" if fraction is None else f"{fraction * 100:.1f}%"
    return f"{left} | {right}"


def format_dispersion_table(rows: Sequence[DispersionSummary]) -> str:
    """네 표현 × (중앙 산포, 게이트 통과 비율). 한 칸도 빼지 않는다."""
    header = (
        "| 설정 | 기준충족 키 | 커버리지 "
        "| 월드 오일러 X | 월드 오일러 Y | 월드 오일러 Z "
        "| 월드 측지 mean | 월드 측지 rms "
        "| 스트로크 오일러 X | 스트로크 오일러 Y | 스트로크 오일러 Z "
        "| 스트로크 측지 mean | 스트로크 측지 rms |"
    )
    lines = [header, "|" + "---|" * 13]
    for row in rows:
        cells = [
            *(_cell(row.world_euler[axis]) for axis in range(3)),
            _cell(row.world_geodesic_mean),
            _cell(row.world_geodesic_rms),
            *(_cell(row.local_euler[axis]) for axis in range(3)),
            _cell(row.local_geodesic_mean),
            _cell(row.local_geodesic_rms),
        ]
        lines.append(
            f"| {row.config} | {row.buckets} | {row.coverage * 100:.1f}% | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def worker_split_report(
    records: Sequence[PassRecord], config: SweepConfig, local: bool
) -> dict[str, Any]:
    """기준충족 버킷 전체에서 (총 산포, 작업자 간 산포, 작업자 간 비)의 중앙값.

    `총² ≈ 작업자내² + 작업자간²`이므로 작업자 **내** 몫은 `√(1 − 비²)`다. 이전 연구가
    오일러 축별로 얻은 0.94를 회전공간에서 다시 낸다.
    """
    totals: list[float] = []
    betweens: list[float] = []
    ratios: list[float] = []
    for _, items in usable_buckets(records, config):
        grouped: dict[str, list[Quat]] = defaultdict(list)
        for item in items:
            quat = local_quaternion(item) if local else item.quat
            if quat is not None:
                grouped[item.worker].append(quat)
        split = worker_split(grouped)
        if split is None or split[0] <= 0.0:
            continue
        totals.append(split[0])
        betweens.append(split[1])
        ratios.append(min(1.0, split[1] / split[0]))
    if not ratios:
        return {"buckets": 0}
    ratio = statistics.median(ratios)
    return {
        "buckets": len(ratios),
        "total_rms": statistics.median(totals),
        "between_rms": statistics.median(betweens),
        "between_ratio": ratio,
        "within_share": math.sqrt(max(0.0, 1.0 - ratio * ratio)),
    }


# --- CLI ---------------------------------------------------------------------


def _levels(spec: str) -> list[str]:
    if "-" in spec:
        low, high = spec.split("-", 1)
        return [str(value) for value in range(int(low), int(high) + 1)]
    return [part for part in spec.split(",") if part]


def read_cache(path: Path) -> list[PassRecord]:
    with path.open(encoding="utf-8") as handle:
        return [PassRecord.from_json(json.loads(line)) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan_cmd = sub.add_parser("scan", help="CSV 트리를 훑어 패스 레코드를 캐시로 굳힌다")
    scan_cmd.add_argument("--root", type=Path, required=True)
    scan_cmd.add_argument("--levels", default="1-10")
    scan_cmd.add_argument("--workers", default="")
    scan_cmd.add_argument("--session-limit", type=int, default=8)
    scan_cmd.add_argument("--out", type=Path, required=True)
    scan_cmd.add_argument("--progress", action="store_true")

    report_cmd = sub.add_parser("report", help="캐시 위에서 입도를 쓸어 표를 낸다")
    report_cmd.add_argument("--cache", type=Path, required=True)
    report_cmd.add_argument("--top", type=int, default=10)

    dispersion_cmd = sub.add_parser(
        "dispersion", help="네 표현(월드/스트로크 × 오일러/측지)의 산포와 게이트 통과 비율"
    )
    dispersion_cmd.add_argument("--cache", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "scan":
        workers = [name for name in args.workers.split(",") if name] or None
        count = 0
        with args.out.open("w", encoding="utf-8") as handle:
            for record in scan(
                args.root, _levels(args.levels), workers, args.session_limit, args.progress
            ):
                handle.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
                count += 1
        print(f"{count} passes -> {args.out}")
        return 0

    records = read_cache(args.cache)
    configs = default_configs()
    if args.command == "dispersion":
        print(f"passes={len(records)}")
        print(format_dispersion_table([dispersion_summary(records, c) for c in configs]))
        print()
        for config in configs[:1]:
            for local in (False, True):
                name = "스트로크 프레임" if local else "월드"
                print(f"### 작업자 내/간 분해 — {config.name} · {name}")
                print(json.dumps(worker_split_report(records, config, local), ensure_ascii=False))
        return 0
    print(f"passes={len(records)} workers={sorted({r.worker for r in records})}")
    print(f"levels={sorted({r.level for r in records}, key=int)}")
    print(Counter(record.worker for record in records))
    print()
    print(format_table(occupancy_rows(records, configs)))
    print()
    for config in configs:
        row = occupancy(records, config)
        if not row.sufficient:
            continue
        print(f"### {config.name} — 기준 충족")
        for entry in bucket_angle_report(records, config, args.top):
            print(json.dumps(entry, ensure_ascii=False))
        break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
