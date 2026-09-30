#!/usr/bin/env python3
"""스트로크 굽이가 채점 차원이어야 하는가 — 통제군을 붙인 측정.

사전등록: `docs/measurements/2026-09-30-stroke-curvature-dimension-prereg.md`

숙련자 패스마다 굽이 `s`와 자기 밴드 레퍼런스 대비 σ를 재고, 굽은 쪽(s ≥ p90)과
곧은 쪽(s ≤ p50)의 σ 중앙값 차이 Δ를 작업자 클러스터 부트스트랩으로 구간 추정한다.

`build_pass_reference.py`의 `straightness_rms_mm`과 **다른 것을 잰다.** 저쪽은 궤적이
얼마나 곧은가(직선 적합 잔차)이고, 여기서는 그 굽이가 **팔 자세 편차**와 연결되는지다.

**통제군 둘을 함께 돌린다.** 난수 그룹이 같은 효과를 내면 측정 장치가 고장난 것이고,
패스 길이 그룹이 같은 크기를 내면 우리가 잰 것은 굽이가 아니라 길이다.

측정 단위는 **런타임이 채점하는 단위와 같다** — 코퍼스의 패스는 `split_pass_spans`가
만든 것이고(`trainer/pose_lab/csv_source.py:158`), 런타임의 스트로크도 같은 함수가
만든다. 그래서 "패스의 굽이가 1에 가깝다"에서 "굽이를 다룰 필요가 없다"로 가는 추론은
순환이다 — 사전등록 문서에 함정으로 적어 두었다.

    docker exec -w /workspace -e PYTHONPATH=/workspace ship-paint-trainer \
      python3 trainer/scripts/curvature_dimension.py --passes 600 \
        --out /output/curvature_dimension.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import measured_pose
import pose_fk_csv
import pose_similarity
import pose_stages
import sequence_dtw

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]

DEFAULT_ROOT: Final = Path("/data/sources/professionals")
DEFAULT_REFERENCE: Final = Path("/output/expert_reference/expert_arm_reference.json")

# 사전등록한 그룹 경계. **순위**다 — 절댓값으로 두면 표본이 바뀔 때 그룹 크기가
# 조용히 달라져 검정력이 변한다.
STRAIGHT_PERCENTILE: Final = 50.0
CURVED_PERCENTILE: Final = 90.0

# 사전등록한 판정 문턱(σ 단위).
DELTA_DIMENSION: Final = 0.50
DELTA_NEGLIGIBLE: Final = 0.25

BOOTSTRAP_ROUNDS: Final = 2000
CI_LOW: Final = 2.5
CI_HIGH: Final = 97.5

# 밴드 하나가 결론을 좌우하지 않도록 최소 표본을 등록해 둔다.
MIN_PASSES: Final = 600
MIN_PER_BAND: Final = 60

# 길이 대조가 굽이의 이만큼이면 "굽이 단독 효과"를 주장하지 않는다.
CONFOUND_RATIO: Final = 0.8


def _load_pose_lab() -> tuple[Any, Any]:
    """`trainer.pose_lab`는 패키지 경로로만 import된다 — 스크립트 경로와 섞지 않는다."""
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")
    from trainer.pose_lab import csv_source, service

    return csv_source, service


def _pass_inputs(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Vec3]], list[Vec3], list[Vec3]] | None:
    """패스 하나에서 (프레임별 월드 관절, 착탄점, 건 위치). 못 만들면 `None`.

    `correction_recovery._pass_inputs`와 같은 규칙이다 — 두 실험이 같은 패스를 같은
    방식으로 읽어야 결과를 나란히 놓을 수 있다.
    """
    if len(rows) < 6:
        return None
    offsets = pose_fk_csv.bind_offsets(rows[0])
    if not offsets:
        return None
    frames: list[dict[str, Vec3]] = []
    keep: list[int] = []
    for index, row in enumerate(rows):
        try:
            frames.append(dict(pose_fk_csv.forward_kinematics(row, offsets)))
        except (KeyError, ValueError):
            continue
        keep.append(index)
    if len(frames) < 6:
        return None
    measured = measured_pose.measured_frames(rows)
    impacts: list[Vec3] = []
    tools: list[Vec3] = []
    for index in keep:
        if index >= len(measured):
            continue
        frame = measured[index]
        if frame.hit is not None:
            impacts.append(tuple(frame.hit))  # type: ignore[arg-type]
        if frame.right is not None:
            tools.append(tuple(frame.right))  # type: ignore[arg-type]
    if len(impacts) != len(frames) or not tools:
        return None
    return frames, impacts, tools


def collect(root: Path, reference: Path, passes: int, seed: int) -> list[dict[str, Any]]:
    """패스마다 (작업자, 밴드, 굽이, 길이, σ). 채점에 실패한 패스는 조용히 빠진다."""
    csv_source, service = _load_pose_lab()
    buckets = json.loads(reference.read_text(encoding="utf-8"))["buckets"]
    source = csv_source.CsvPassSource(root)

    # **작업자별로 균형을 맞춘다.** σ가 작업자별로 군집돼 있어(위양성률 17배) 섞어
    # 뽑기만 하면 추정치가 누가 뽑혔는지에 좌우된다.
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

    rows_out: list[dict[str, Any]] = []
    used_by_worker: dict[str, int] = {}

    for item in sessions:
        if len(rows_out) >= passes:
            break
        if used_by_worker.get(item.worker, 0) >= per_worker_cap:
            continue
        try:
            entries = source.index_of(item)
        except (OSError, ValueError, KeyError):
            continue
        for entry in entries:
            if len(rows_out) >= passes:
                break
            if used_by_worker.get(item.worker, 0) >= per_worker_cap:
                break
            ref = service.PassRef(
                worker=item.worker, date=item.date, level=item.level,
                session=item.session, pass_index=entry.pass_index,
            )
            try:
                raw = source.load(ref)
            except (LookupError, OSError, ValueError):
                continue
            prepared = _pass_inputs(raw.rows)
            if prepared is None:
                continue
            frames, impacts, tools = prepared

            straightness, span_m, length_m = pose_stages._straightness(impacts)
            if span_m < sequence_dtw.MIN_PATH_LENGTH_M:
                # 움직이지 않은 패스는 굽이가 정의되지 않는다(1.0으로 채워 나온다).
                continue

            resolved = pose_stages.resolve(pose_stages.reach_of(impacts, tools), buckets)
            if not resolved.ok or resolved.band is None:
                continue
            aligned = pose_stages.align(impacts, resolved, list(range(len(impacts))))
            scored = pose_stages.score(
                frames, aligned, buckets[resolved.band], labels=pose_similarity.JOINT_LABELS
            )
            if not scored.ok or not scored.sigmas:
                continue

            used_by_worker[item.worker] = used_by_worker.get(item.worker, 0) + 1
            rows_out.append({
                "worker": item.worker,
                "band": resolved.band,
                "straightness": straightness,
                "span_m": span_m,
                "length_m": length_m,
                # 교정이 실제로 보는 값 — 6관절 σ의 **최댓값**이다.
                "sigma": max(entry_.sigma for entry_ in scored.sigmas),
                "score": scored.score,
            })
            if len(rows_out) % 50 == 0:
                logger.info("%d패스 · 작업자 %d인", len(rows_out), len(used_by_worker))

    return rows_out


def _percentile(values: Sequence[float], point: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("빈 표본")
    position = (len(ordered) - 1) * point / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _delta(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any] | None:
    """`key`의 p50 이하 / p90 이상으로 나눈 두 무리의 σ 중앙값 차이."""
    values = [float(row[key]) for row in rows]
    if len(values) < 4:
        return None
    low_edge = _percentile(values, STRAIGHT_PERCENTILE)
    high_edge = _percentile(values, CURVED_PERCENTILE)
    low = [float(row["sigma"]) for row in rows if float(row[key]) <= low_edge]
    high = [float(row["sigma"]) for row in rows if float(row[key]) >= high_edge]
    if not low or not high:
        return None
    return {
        "delta": st.median(high) - st.median(low),
        "low_n": len(low),
        "high_n": len(high),
        "low_median_sigma": st.median(low),
        "high_median_sigma": st.median(high),
        "low_edge": low_edge,
        "high_edge": high_edge,
    }


def bootstrap(rows: Sequence[Mapping[str, Any]], key: str, seed: int) -> dict[str, Any] | None:
    """**작업자 단위** 재표집. 패스 단위로 뽑으면 군집을 무시해 구간이 좁게 나온다."""
    point = _delta(rows, key)
    if point is None:
        return None
    by_worker: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_worker.setdefault(str(row["worker"]), []).append(row)
    workers = sorted(by_worker)
    if len(workers) < 2:
        return {**point, "ci_low": None, "ci_high": None, "workers": len(workers)}

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample: list[Mapping[str, Any]] = []
        for _ in workers:
            sample.extend(by_worker[workers[rng.randrange(len(workers))]])
        drawn = _delta(sample, key)
        if drawn is not None:
            draws.append(drawn["delta"])
    if len(draws) < BOOTSTRAP_ROUNDS // 2:
        return {**point, "ci_low": None, "ci_high": None, "workers": len(workers)}
    return {
        **point,
        "ci_low": _percentile(draws, CI_LOW),
        "ci_high": _percentile(draws, CI_HIGH),
        "workers": len(workers),
        "draws": len(draws),
    }


def verdict(measured: Mapping[str, Any] | None) -> str:
    """사전등록한 판정. 결과를 보고 고치지 않는다."""
    if measured is None:
        return "미측정"
    low, high = measured.get("ci_low"), measured.get("ci_high")
    excludes_zero = low is not None and high is not None and (low > 0.0 or high < 0.0)
    delta = float(measured["delta"])
    if delta >= DELTA_DIMENSION and excludes_zero:
        return "① 차원으로 넣는다"
    if delta < DELTA_NEGLIGIBLE or not excludes_zero:
        return "② 싣기만 한다"
    return "③ 미결 — 넣지 않는다"


def run(root: Path, reference: Path, passes: int, seed: int) -> dict[str, Any]:
    rows = collect(root, reference, passes, seed)
    rng = random.Random(seed + 1)
    # **난수 대조.** 굽이와 같은 파이프라인을 타야 대조가 된다 — 따로 계산하면
    # 파이프라인의 결함이 본 결과에만 실린다.
    for row in rows:
        row["random"] = rng.random()

    by_band: dict[str, int] = {}
    for row in rows:
        by_band[str(row["band"])] = by_band.get(str(row["band"]), 0) + 1

    arms = {
        key: bootstrap(rows, key, seed + offset)
        for offset, key in enumerate(("straightness", "length_m", "random"), start=2)
    }
    straight_values = [float(row["straightness"]) for row in rows]
    return {
        "prereg": "docs/measurements/2026-09-30-stroke-curvature-dimension-prereg.md",
        "passes": len(rows),
        "workers": len({str(row["worker"]) for row in rows}),
        "by_band": by_band,
        "straightness_ladder": (
            {f"p{point:g}": _percentile(straight_values, point)
             for point in (50, 75, 90, 95, 99)} if straight_values else {}
        ),
        "arms": arms,
        "verdict": verdict(arms.get("straightness")),
        # 표본이 등록한 하한에 못 미치면 판정을 그대로 믿지 않는다.
        "enough_passes": len(rows) >= MIN_PASSES,
        "bands_with_enough": sorted(b for b, n in by_band.items() if n >= MIN_PER_BAND),
        "rows": rows,
    }


def summarize(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        f"패스 {payload['passes']}개 · 작업자 {payload['workers']}인 · 밴드 {payload['by_band']}",
    ]
    ladder = payload.get("straightness_ladder") or {}
    if ladder:
        lines.append("굽이 사다리 " + " · ".join(f"{k} {v:.3f}" for k, v in ladder.items()))
    label = {"straightness": "굽이", "length_m": "길이(대조)", "random": "난수(대조)"}
    for key, arm in (payload.get("arms") or {}).items():
        if arm is None:
            lines.append(f"  {label.get(key, key)}: 미측정")
            continue
        ci = (
            "CI 미산출"
            if arm.get("ci_low") is None
            else f"95% CI [{arm['ci_low']:+.3f}, {arm['ci_high']:+.3f}]"
        )
        lines.append(
            f"  {label.get(key, key):10s} Δ {arm['delta']:+.3f}σ  {ci}"
            f"  (낮은쪽 n={arm['low_n']} 중앙 {arm['low_median_sigma']:.3f},"
            f" 높은쪽 n={arm['high_n']} 중앙 {arm['high_median_sigma']:.3f})"
        )
    lines.append("")
    lines.append(f"사전등록 판정: {payload['verdict']}")
    if not payload.get("enough_passes"):
        lines.append(f"  주의 — 등록 하한 {MIN_PASSES}패스에 못 미친다. 판정을 그대로 믿지 않는다.")
    random_arm = (payload.get("arms") or {}).get("random")
    if random_arm is not None and verdict(random_arm) == "① 차원으로 넣는다":
        lines.append("  ★ 난수 대조가 ①을 통과했다 — 측정 장치가 고장났다. 본 결과를 폐기한다.")
    length_arm = (payload.get("arms") or {}).get("length_m")
    curve_arm = (payload.get("arms") or {}).get("straightness")
    if length_arm is not None and curve_arm is not None:
        if abs(length_arm["delta"]) >= abs(curve_arm["delta"]) * CONFOUND_RATIO:
            lines.append("  ★ 길이 대조가 굽이와 비슷한 크기다 — 굽이 단독 효과를 주장하지 않는다.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--passes", type=int, default=MIN_PASSES)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    payload = run(args.root, args.reference, args.passes, args.seed)
    print("\n".join(summarize(payload)))
    if args.out is not None:
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
