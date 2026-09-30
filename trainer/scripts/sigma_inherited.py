#!/usr/bin/env python3
"""숙련자의 팔 변동은 어디서 오고, 6관절 σ는 몇 개를 재고 있나.

사전등록: `docs/measurements/2026-09-30-sigma-inherited-spread-prereg.md` (개정 포함)

교정 지목이 관절마다 극단적으로 갈린다 — 위팔 0.97 대 왼아래팔 0.08. 순위는 원인이
아니었으므로(`...-correction-ranking-result.md`) 남은 후보는 σ의 정의다.

**처음 물은 것("손의 산포는 물려받았나")은 자료를 보기 전에 폐기했다.** 뼈 길이가
리그 상수라(0.1797/0.2430/0.2430) 손 관절 위치는 어깨 방향 2 + 팔꿈치 각 2로 완전히
결정된다 — 재 보나 마나 물려받은 것이다. 운동학을 자료로 확인하는 꼴이 된다.

고쳐 묻는다.

1. **어디서 오나** — 어깨 마디 방향의 산포 대 팔꿈치 각의 산포. 어깨가 지배하면
   손에 단서를 내는 것은 중복이다(위팔은 이미 0.97로 잡힌다).
2. **몇 개를 재나** — 6관절 σ의 상관과 **유효 차원**(분산 90%를 덮는 성분 수).
   팔은 약 4 자유도인데 현행 σ는 6관절 위치를 한 원점에서 따로 잰다.

**검출 성능은 여기서 재지 않는다.** 지금의 `correction_recovery.perturb`는 관절
하나를 그대로 옮겨 뼈를 깨므로, 사슬을 아는 지표에 유리하게 기울어 있다.

구워진 레퍼런스를 읽지 않는다. 자료 자체의 성질을 재는 것이라 이미 만든 σ가
개입하면 무엇을 쟀는지 모르게 된다.

통제: **작업자 섞기.** 관절 σ를 작업자 안에서 섞어 짝지으면 상관이 무너지고 유효
차원이 6에 가까워져야 한다. 안 무너지면 우리가 잰 상관은 자세가 아니라 계산의
부산물이다.

    docker exec -w /workspace -e PYTHONPATH=/workspace ship-paint-trainer \
      python3 trainer/scripts/sigma_inherited.py --passes 600 \
        --out /output/sigma_inherited.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import arm_reference
import pose_fk_csv
import skeleton_gate

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]

DEFAULT_ROOT: Final = Path("/data/sources/professionals")

# 개정 사전등록한 판정 문턱.
DOMINANCE: Final = 1.5             # 한쪽이 다른 쪽의 이 배 이상이면 "지배한다"
VARIANCE_COVERED: Final = 0.90     # 유효 차원을 세는 분산 비율
REDUNDANT_DIMS: Final = 4          # 유효 차원이 이 이하면 6관절 σ는 중복이다
SHUFFLED_DIMS_MIN: Final = 5.0     # 작업자 섞기 통제: 유효 차원이 이보다 커야 한다

MIN_PASSES: Final = 600

# 부모는 정본 골격에서 읽는다. 여기 다시 적으면 한쪽만 바뀐다.
PARENT: Final[dict[str, str]] = {child: parent for child, parent in skeleton_gate.BONES}


def _load_pose_lab() -> tuple[Any, Any]:
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")
    from trainer.pose_lab import csv_source, service

    return csv_source, service


def _unit(vector: Vec3) -> Vec3 | None:
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-9:
        return None
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _angle_deg(a: Vec3, b: Vec3) -> float:
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


def angular_spread_deg(points: Sequence[Vec3]) -> float | None:
    """평균 방향에서 벌어진 각의 표준편차(도).

    평균 방향은 단위벡터의 합을 정규화한 것이다(구면 평균). 각의 산술 표준편차를
    쓰는 이유는 레퍼런스의 `angular_sd`가 같은 정의이기 때문이다 — 두 수를 나란히
    놓으려면 같은 자여야 한다.
    """
    units = [unit for unit in (_unit(point) for point in points) if unit is not None]
    if len(units) < 3:
        return None
    total = tuple(sum(unit[axis] for unit in units) for axis in range(3))
    mean = _unit(total)  # type: ignore[arg-type]
    if mean is None:
        return None
    angles = [_angle_deg(unit, mean) for unit in units]
    return st.pstdev(angles)


def radial_spread_m(points: Sequence[Vec3]) -> float | None:
    lengths = [math.sqrt(sum(value * value for value in point)) for point in points]
    return st.pstdev(lengths) if len(lengths) >= 3 else None


def _pass_pose(rows: list[dict[str, str]]) -> dict[str, Vec3] | None:
    """패스 하나의 대표 자세 — 프레임별 정규 좌표의 **축별 중앙값**.

    평균이 아니라 중앙값이다. 한 프레임의 트래킹 튐이 패스 전체의 대표값을 끌고
    가면 산포가 실제보다 커진다.
    """
    prepared = pose_fk_csv.pass_inputs(rows)
    if prepared is None:
        return None
    frames, _impacts, _tools = prepared
    normalized = [
        pose for pose in (arm_reference.normalize(frame) for frame in frames)
        if pose is not None
    ]
    if len(normalized) < 3:
        return None
    names = set(normalized[0])
    for pose in normalized[1:]:
        names &= set(pose)
    wanted = [name for name in names if name in PARENT or name in arm_reference.FRAME_JOINTS]
    return {
        name: tuple(  # type: ignore[misc]
            st.median([pose[name][axis] for pose in normalized]) for axis in range(3)
        )
        for name in wanted
    }


def collect(root: Path, passes: int, seed: int) -> list[dict[str, Any]]:
    """패스마다 (작업자, 대표 자세). 작업자 균형은 앞선 실험들과 같은 규칙이다."""
    csv_source, service = _load_pose_lab()
    source = csv_source.CsvPassSource(root)

    by_worker: dict[str, list[Any]] = {}
    for item in source.sessions():
        by_worker.setdefault(item.worker, []).append(item)
    rng = random.Random(seed)
    for group in by_worker.values():
        rng.shuffle(group)

    order = sorted(by_worker)
    sessions: list[Any] = []
    index = 0
    while any(index < len(by_worker[worker]) for worker in order):
        for worker in order:
            if index < len(by_worker[worker]):
                sessions.append(by_worker[worker][index])
        index += 1
    per_worker_cap = max(1, passes // max(len(order), 1))

    out: list[dict[str, Any]] = []
    used_by_worker: dict[str, int] = {}
    for item in sessions:
        if len(out) >= passes:
            break
        if used_by_worker.get(item.worker, 0) >= per_worker_cap:
            continue
        try:
            entries = source.index_of(item)
        except (OSError, ValueError, KeyError):
            continue
        for entry in entries:
            if len(out) >= passes or used_by_worker.get(item.worker, 0) >= per_worker_cap:
                break
            ref = service.PassRef(
                worker=item.worker, date=item.date, level=item.level,
                session=item.session, pass_index=entry.pass_index,
            )
            try:
                raw = source.load(ref)
            except (LookupError, OSError, ValueError):
                continue
            pose = _pass_pose(raw.rows)
            if pose is None:
                continue
            used_by_worker[item.worker] = used_by_worker.get(item.worker, 0) + 1
            out.append({"worker": item.worker, "pose": pose})
            if len(out) % 50 == 0:
                logger.info("%d패스 · 작업자 %d인", len(out), len(used_by_worker))
    return out


def _percentile(values: Sequence[float], point: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * point / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def segment_angles(pose: Mapping[str, Vec3], side: str) -> tuple[Vec3, float] | None:
    """(어깨 마디 방향, 팔꿈치 각°).

    어깨 마디는 **위팔 → 아래팔** 단위벡터다(몸 기준). 팔꿈치 각은 위팔 마디와
    아래팔 마디(아래팔 → 손) 사이의 각이다. 둘 다 **위치가 아니라 방향**이라
    뼈 길이가 상수인 이 리그에서 실제 자유도에 대응한다.
    """
    upper = pose.get(f"{side}UpperArm")
    fore = pose.get(f"{side}ForeArm")
    hand = pose.get(f"{side}Hand")
    if upper is None or fore is None or hand is None:
        return None
    upper_bone = _unit(tuple(fore[i] - upper[i] for i in range(3)))  # type: ignore[arg-type]
    fore_bone = _unit(tuple(hand[i] - fore[i] for i in range(3)))  # type: ignore[arg-type]
    if upper_bone is None or fore_bone is None:
        return None
    return upper_bone, _angle_deg(upper_bone, fore_bone)


def _sigma_matrix(rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    """패스 × 6관절의 **몸 기준 위치 이탈**. 구워진 σ를 읽지 않고 여기서 만든다.

    각 관절에 대해 전체 평균 위치에서 벌어진 거리(m)다. 현행 σ가 무엇들을 나란히
    세우는지를 재는 것이므로, 단위는 σ가 아니라 m여도 상관의 구조는 같다.
    """
    names = list(arm_reference.SCORING_JOINTS_ARM)
    usable = [row for row in rows if all(name in row["pose"] for name in names)]
    if len(usable) < 10:
        return []
    means = {
        name: tuple(
            sum(row["pose"][name][axis] for row in usable) / len(usable) for axis in range(3)
        )
        for name in names
    }
    return [
        [
            math.dist(row["pose"][name], means[name])
            for name in names
        ]
        for row in usable
    ]


def _correlation(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    columns = list(zip(*matrix))
    means = [sum(column) / len(column) for column in columns]
    sds = [st.pstdev(column) for column in columns]
    size = len(columns)
    out = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if sds[i] < 1e-12 or sds[j] < 1e-12:
                out[i][j] = 0.0
                continue
            cov = sum(
                (matrix[k][i] - means[i]) * (matrix[k][j] - means[j]) for k in range(len(matrix))
            ) / len(matrix)
            out[i][j] = cov / (sds[i] * sds[j])
    return out


def _eigenvalues(matrix: Sequence[Sequence[float]], rounds: int = 400) -> list[float]:
    """대칭 행렬의 고윳값 — 야코비 회전. numpy를 요구 의존성으로 만들지 않는다."""
    size = len(matrix)
    work = [list(row) for row in matrix]
    for _ in range(rounds):
        best, p, q = 0.0, 0, 1
        for i in range(size):
            for j in range(i + 1, size):
                if abs(work[i][j]) > best:
                    best, p, q = abs(work[i][j]), i, j
        if best < 1e-12:
            break
        theta = 0.5 * math.atan2(2.0 * work[p][q], work[p][p] - work[q][q])
        cos, sin = math.cos(theta), math.sin(theta)
        for k in range(size):
            kp, kq = work[k][p], work[k][q]
            work[k][p] = cos * kp + sin * kq
            work[k][q] = -sin * kp + cos * kq
        for k in range(size):
            pk, qk = work[p][k], work[q][k]
            work[p][k] = cos * pk + sin * qk
            work[q][k] = -sin * pk + cos * qk
    return sorted((work[i][i] for i in range(size)), reverse=True)


def effective_dimensions(matrix: Sequence[Sequence[float]]) -> int | None:
    """분산의 `VARIANCE_COVERED`를 덮는 데 필요한 성분 수."""
    if not matrix:
        return None
    values = [max(value, 0.0) for value in _eigenvalues(_correlation(matrix))]
    total = sum(values)
    if total <= 0.0:
        return None
    running = 0.0
    for index, value in enumerate(values, start=1):
        running += value
        if running / total >= VARIANCE_COVERED:
            return index
    return len(values)


def _shuffle_columns(
    matrix: Sequence[Sequence[float]], workers: Sequence[str], seed: int
) -> list[list[float]]:
    """**작업자 안에서** 열마다 따로 섞는다 — 통제다.

    같은 패스의 관절들이 함께 움직이는 것이 상관의 원인이라면, 짝을 흩으면 상관이
    무너지고 유효 차원이 6에 가까워져야 한다. 안 무너지면 우리가 잰 상관은 자세가
    아니라 계산의 부산물이다. 작업자 안에서 섞는 이유는 작업자 간 차이를 상관으로
    되살리지 않기 위해서다.
    """
    rng = random.Random(seed)
    by_worker: dict[str, list[int]] = {}
    for index, worker in enumerate(workers):
        by_worker.setdefault(worker, []).append(index)
    out = [list(row) for row in matrix]
    for column in range(len(matrix[0])):
        for indices in by_worker.values():
            values = [matrix[index][column] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values):
                out[index][column] = value
    return out


def measure(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    shoulder_dirs: list[Vec3] = []
    elbow_angles: list[float] = []
    # **좌우를 나눠서도 낸다(사후 기술).** 합쳐 재니 팔꿈치 각이 양봉이었다
    # (p25 0.2° · p75 117.5°). 오른팔은 건을 쥐어 거의 곧게 펴져 있고(게이트의
    # `right_arm_degenerate`가 보는 그 성질이다) 왼팔은 접혀 있다. 합친 산포는
    # 팔 안의 변동이 아니라 **좌우 차이**를 재고 있을 수 있다.
    by_side: dict[str, dict[str, list[Any]]] = {
        side: {"dirs": [], "elbows": []} for side in ("Left", "Right")
    }
    for row in rows:
        for side in ("Left", "Right"):
            found = segment_angles(row["pose"], side)
            if found is None:
                continue
            shoulder_dirs.append(found[0])
            elbow_angles.append(found[1])
            by_side[side]["dirs"].append(found[0])
            by_side[side]["elbows"].append(found[1])

    sides = {
        side: {
            "arms": len(block["elbows"]),
            "shoulder_direction_sd_deg": angular_spread_deg(block["dirs"]),
            "elbow_angle_sd_deg": (
                st.pstdev(block["elbows"]) if len(block["elbows"]) >= 3 else None
            ),
            "elbow_angle_median_deg": (
                st.median(block["elbows"]) if block["elbows"] else None
            ),
        }
        for side, block in by_side.items()
    }

    names = list(arm_reference.SCORING_JOINTS_ARM)
    matrix = _sigma_matrix(rows)
    workers = [
        str(row["worker"]) for row in rows
        if all(name in row["pose"] for name in names)
    ]
    correlation = _correlation(matrix) if matrix else []
    shuffled = _shuffle_columns(matrix, workers, seed + 3) if matrix else []
    return {
        "shoulder_direction_sd_deg": angular_spread_deg(shoulder_dirs),
        "elbow_angle_sd_deg": st.pstdev(elbow_angles) if len(elbow_angles) >= 3 else None,
        "elbow_angle_median_deg": st.median(elbow_angles) if elbow_angles else None,
        # **사후 기술이다 — 판정에 들어가지 않는다.** 중앙 9.3°에 산포 58.6°가 나와
        # 표준편차가 이 분포의 요약으로 쓸 수 없다는 것을 보고, 모양을 적기로 했다.
        # `verdict`는 건드리지 않았다.
        "elbow_angle_ladder_deg": (
            {f"p{point:g}": _percentile(elbow_angles, point)
             for point in (5, 25, 50, 75, 95)} if len(elbow_angles) >= 3 else {}
        ),
        "arms": len(elbow_angles),
        "by_side": sides,
        "joints": names,
        "correlation": correlation,
        "mean_abs_correlation": (
            None if not correlation else
            sum(abs(correlation[i][j]) for i in range(len(names)) for j in range(len(names))
                if i != j) / (len(names) * (len(names) - 1))
        ),
        "effective_dimensions": effective_dimensions(matrix),
        "shuffled_effective_dimensions": effective_dimensions(shuffled),
        "matrix_rows": len(matrix),
    }


def verdict(found: Mapping[str, Any]) -> str:
    """개정 사전등록한 판정. 결과를 보고 고치지 않는다."""
    shoulder = found.get("shoulder_direction_sd_deg")
    elbow = found.get("elbow_angle_sd_deg")
    if shoulder is None or elbow is None or elbow <= 0.0 or shoulder <= 0.0:
        return "미측정"
    if shoulder >= DOMINANCE * elbow:
        return "①′ 어깨가 지배한다 — 손 단서는 중복이다"
    if elbow >= DOMINANCE * shoulder:
        return "②′ 팔꿈치가 지배한다 — 팔꿈치를 따로 재야 한다"
    return "③′ 비슷하다 — 둘 다 필요하다"


def run(root: Path, passes: int, seed: int) -> dict[str, Any]:
    rows = collect(root, passes, seed)
    found = measure(rows, seed)
    return {
        "prereg": "docs/measurements/2026-09-30-sigma-inherited-spread-prereg.md",
        "passes": len(rows),
        "workers": len({str(row["worker"]) for row in rows}),
        **found,
        "verdict": verdict(found),
        "enough_passes": len(rows) >= MIN_PASSES,
    }


def summarize(payload: Mapping[str, Any]) -> list[str]:
    def _fmt(value: Any, digits: int = 1) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [f"패스 {payload['passes']}개 · 작업자 {payload['workers']}인 · "
             f"팔 {payload.get('arms', 0)}개", ""]
    lines.append(f"어깨 마디 방향 산포 {_fmt(payload.get('shoulder_direction_sd_deg'))}°")
    lines.append(f"팔꿈치 각 산포     {_fmt(payload.get('elbow_angle_sd_deg'))}°"
                 f"  (중앙 {_fmt(payload.get('elbow_angle_median_deg'))}°)")
    ladder = payload.get("elbow_angle_ladder_deg") or {}
    if ladder:
        lines.append("  팔꿈치 각 사다리 " +
                     " · ".join(f"{k} {v:.1f}°" for k, v in ladder.items()) +
                     "   ← 사후 기술, 판정에 안 들어감")
    for side, block in (payload.get("by_side") or {}).items():
        lines.append(
            f"  [{side}] 팔 {block['arms']} · 어깨 마디 "
            f"{_fmt(block['shoulder_direction_sd_deg'])}° · 팔꿈치 각 "
            f"{_fmt(block['elbow_angle_sd_deg'])}° (중앙 "
            f"{_fmt(block['elbow_angle_median_deg'])}°)   ← 사후 기술"
        )
    lines.append("")

    names = payload.get("joints") or []
    correlation = payload.get("correlation") or []
    if correlation:
        short = {"RightUpperArm": "RUp", "RightForeArm": "RFo", "RightHand": "RHa",
                 "LeftUpperArm": "LUp", "LeftForeArm": "LFo", "LeftHand": "LHa"}
        lines.append("6관절 이탈 상관 — 같은 변동을 몇 번 세고 있나")
        lines.append("        " + " ".join(f"{short.get(n, n):>5s}" for n in names))
        for index, name in enumerate(names):
            lines.append(f"  {short.get(name, name):>5s} " +
                         " ".join(f"{correlation[index][j]:>5.2f}" for j in range(len(names))))
        lines.append(f"  평균 |상관| {_fmt(payload.get('mean_abs_correlation'), 3)}")
    lines.append("")
    lines.append(f"유효 차원(분산 {VARIANCE_COVERED:.0%}) : "
                 f"{payload.get('effective_dimensions')} / {len(names)}")
    lines.append(f"  작업자 섞기 통제           : {payload.get('shuffled_effective_dimensions')}")
    lines.append("")
    lines.append(f"사전등록 판정: {payload['verdict']}")

    dims = payload.get("effective_dimensions")
    if dims is not None:
        lines.append(
            f"  유효 차원 {dims} ≤ {REDUNDANT_DIMS} — 6관절 σ는 중복이다"
            if dims <= REDUNDANT_DIMS
            else f"  유효 차원 {dims} > {REDUNDANT_DIMS} — 여섯이 서로 다른 것을 잰다"
        )
    shuffled = payload.get("shuffled_effective_dimensions")
    if shuffled is not None and shuffled < SHUFFLED_DIMS_MIN:
        lines.append("  ★ 짝을 흩었는데도 차원이 안 늘었다 — 상관이 자세가 아니라 계산의"
                     " 부산물이다. 결론을 내지 않는다.")
    if not payload.get("enough_passes"):
        lines.append(f"  주의 — 등록 하한 {MIN_PASSES}패스에 못 미친다.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--passes", type=int, default=MIN_PASSES)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    payload = run(args.root, args.passes, args.seed)
    print("\n".join(summarize(payload)))
    if args.out is not None:
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
