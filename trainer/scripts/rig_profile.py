#!/usr/bin/env python3
"""아바타 리그 치수(부모 기준 뼈 오프셋) — 리그가 바뀌어도 숙련자 기준을 맞는 몸으로 쓰기 위한 것.

2026-10-08 샘플부터 시뮬레이터 아바타가 바뀌었다. 팔·다리 뼈 길이와 축 규약은 예전 리그와 같고, 몸통만 다르다
(골반→척추 0.105→0.198 m, 가슴→머리 0.137→0.341 m, 어깨가 가슴 축으로 0.21 m 위, 엉덩관절이 골반 앞 0.093 m → 0).
머리↔발끝이 예전 리그는 최대 1.43 m, 새 리그는 선 자세에서 1.6 m 안팎이다.

이 모듈은
- 새 형식 CSV(관절 월드 위치·회전)에서 리그 오프셋을 역산해 파일로 저장하고(`derive_from_rows`, `save`),
- 숙련자 기준을 그 리그로 **옮겨** 만들 때 오프셋을 내주고(`build_arm_reference.py --rig`),
- 런타임 시퀀스의 뼈 길이로 어느 리그인지 고른다(`match`).

예전 리그는 파일이 없다 — 예전 CSV의 바인드 오프셋 열이 곧 리그이고 세션마다 같다. 고를 때는 `LEGACY_BONE_LENGTHS`
(예전 코퍼스 바인드 오프셋 노름)를 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

try:  # MCP는 패키지(`trainer.scripts`)로, 스크립트는 경로로 부른다(`skeleton_gate`와 같은 방식).
    from . import simulator_csv
except ImportError:
    import simulator_csv  # type: ignore[no-redef]

# `pose_fk_csv`는 역산(`derive_from_rows`)에만 쓴다. MCP 이미지는 이 모듈로 리그를 **고르기만** 하므로
# FK 의존(measured_pose·rotation_math)을 끌고 가지 않게 그 함수 안에서 불러온다.

Vec3 = tuple[float, float, float]

LEGACY_RIG: Final = "legacy"
# 예전 코퍼스(전형기 L3 바인드 오프셋)의 몸통 뼈 길이. 리그를 가르는 데만 쓴다 — 팔·다리는 두 리그가 같다.
LEGACY_BONE_LENGTHS: Final[dict[str, float]] = {
    "Spine": 0.1053, "Chest": 0.1053, "Head": 0.1368,
    "LeftShoulder": 0.0284, "RightShoulder": 0.0284, "LeftThigh": 0.1337, "RightThigh": 0.1337,
}
# 리그를 가르는 뼈(자식 관절 이름). 두 리그가 다른 곳만 본다.
SIGNATURE_BONES: Final[tuple[str, ...]] = tuple(LEGACY_BONE_LENGTHS)
# 이보다 멀면 아는 리그가 아니다(뼈마다 평균 절대 차이, m). 두 리그 사이 거리는 0.1 m 안팎이다.
MATCH_TOLERANCE_M: Final = 0.02


@dataclass(frozen=True)
class RigProfile:
    name: str
    offsets: dict[str, Vec3]
    source: str = ""

    def bone_lengths(self) -> dict[str, float]:
        return {child: math.sqrt(sum(v * v for v in vector)) for child, vector in self.offsets.items()}

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "derived_at": datetime.now(timezone.utc).isoformat(),
            "offsets": {k: [round(v, 6) for v in vec] for k, vec in self.offsets.items()},
            "bone_lengths_m": {k: round(v, 6) for k, v in self.bone_lengths().items()},
        }


def derive_from_rows(rows: Sequence[Mapping[str, str]], *, name: str, source: str = "", stride: int = 20) -> RigProfile:
    """새 형식 행들에서 뼈마다 역산 오프셋의 성분별 중앙값을 낸다. 새 형식이 아니면 ValueError."""
    import pose_fk_csv

    picked = [row for row in rows[::max(stride, 1)] if pose_fk_csv.is_world_format(row)]
    if not picked:
        raise ValueError("관절 월드 열(새 형식)이 없다 — 예전 형식은 바인드 오프셋 열이 곧 리그다")
    samples: dict[str, list[Vec3]] = {}
    for row in picked:
        for child, vector in pose_fk_csv.derived_offsets(dict(row)).items():
            samples.setdefault(child, []).append(vector)
    offsets: dict[str, Vec3] = {}
    for child, vectors in samples.items():
        offsets[child] = (
            statistics.median(v[0] for v in vectors),
            statistics.median(v[1] for v in vectors),
            statistics.median(v[2] for v in vectors),
        )
    return RigProfile(name=name, offsets=offsets, source=source)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """`.csv`와 `.csv.gz`를 모두 읽는다(시뮬레이터가 압축해 보내기도 한다)."""
    return read_csv_bytes(path.read_bytes())


def read_csv_bytes(raw: bytes) -> list[dict[str, str]]:
    """CSV 바이트(gzip이면 풀어서)를 행 목록으로. 압축 여부는 매직 바이트로 가린다."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def save(profile: RigProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path) -> RigProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    offsets = {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in payload["offsets"].items()}
    return RigProfile(name=str(payload["name"]), offsets=offsets, source=str(payload.get("source", "")))


def signature_distance(lengths: Mapping[str, float], reference: Mapping[str, float]) -> float | None:
    """리그 표지 뼈의 평균 절대 길이 차(m). 공통 뼈가 없으면 None."""
    common = [name for name in SIGNATURE_BONES if name in lengths and name in reference]
    if not common:
        return None
    return sum(abs(lengths[name] - reference[name]) for name in common) / len(common)


def measured_lengths(joints: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """한 프레임의 관절 위치에서 뼈 길이(자식 이름 기준)."""
    out: dict[str, float] = {}
    for parent, child in simulator_csv.BONES:
        if parent in joints and child in joints:
            out[child] = math.dist(joints[parent], joints[child])
    return out


def match(lengths: Mapping[str, float], candidates: Mapping[str, Mapping[str, float]]) -> str | None:
    """측정 뼈 길이에 가장 가까운 리그 이름. 어느 것도 `MATCH_TOLERANCE_M` 안이 아니면 None."""
    best: tuple[float, str] | None = None
    for name, reference in candidates.items():
        distance = signature_distance(lengths, reference)
        if distance is not None and (best is None or distance < best[0]):
            best = (distance, name)
    if best is None or best[0] > MATCH_TOLERANCE_M:
        return None
    return best[1]


def chain_height_m(lengths: Mapping[str, float]) -> float | None:
    """발끝→발→정강이→허벅지→골반→척추→가슴→머리 뼈 길이의 합 — 이 몸으로 머리↔발끝이 가질 수 있는 최댓값."""
    torso = ("Spine", "Chest", "Head")
    best: float | None = None
    for leg in (("LeftToe", "LeftFoot", "LeftCalf", "LeftThigh"), ("RightToe", "RightFoot", "RightCalf", "RightThigh")):
        names = (*leg, *torso)
        if all(name in lengths for name in names):
            total = sum(lengths[name] for name in names)
            best = total if best is None else max(best, total)
    return best


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="새 형식 CSV에서 리그 치수를 역산해 저장한다.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    profile = derive_from_rows(read_csv_rows(args.csv), name=args.name, source=str(args.csv))
    save(profile, args.out)
    print(json.dumps(profile.to_json()["bone_lengths_m"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
