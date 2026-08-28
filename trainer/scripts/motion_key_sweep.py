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

import motion_key

Vec3 = tuple[float, float, float]

# `m1:`과 문자열 공간을 나눈다. 측정용 키가 레퍼런스 아티팩트로 새어 들어가면 안 된다.
SWEEP_PREFIX: Final = "sweep"

# 작업자 요는 아직 실측에서 뽑지 않는다. `pose_lab.service.ASSUMED_YAW`와 같은 값을
# 써야 여기서 잰 점유가 실험대가 실제로 만드는 버킷의 점유가 된다.
ASSUMED_YAW: Final = 0.0

# 각도 컬럼. 설계 §2.2가 오른손 컨트롤러로 지목한 건이고, IK 산출 관절각이 아니라
# 컨트롤러의 6-DoF 실측 자세다.
GUN_ROTATION_GROUP: Final = "한길숏건"

# §0 사전등록 기준. 숫자를 보기 전에 고정했고 뒤에서 옮기지 않는다.
MIN_BUCKET_N: Final = 20
MIN_BUCKET_WORKERS: Final = 2
MIN_COVERAGE: Final = 0.50


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
        }

    @staticmethod
    def from_json(payload: Mapping[str, Any]) -> "PassRecord":
        uv = payload["uv"]
        rotation = payload["rotation"]
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
    band = motion_key.signed_band(record.displacement)
    heading = motion_key.heading_word(ASSUMED_YAW, record.displacement)
    lateral, reach, height = motion_key.body_placement(
        record.head, ASSUMED_YAW, record.hit, params
    )
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
