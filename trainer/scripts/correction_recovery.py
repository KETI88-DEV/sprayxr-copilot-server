#!/usr/bin/env python3
"""교정 지목 정확도 — 통제 교란 회복 실험.

사전등록: `docs/measurements/2026-09-30-correction-recovery-prereg.md`

실제 숙련자 패스에서 관절 하나를 통제된 크기로 옮기고, 교정이 **그 관절을 지목하는지**
본다. 정답을 우리가 만들었으므로 순환이 아니다 — `/verify`의 `gain_error`가 구성상 0인
것과 다르다.

반경(지표가 잡도록 설계된 성분)과 **접선**(못 잡을 것으로 예측한 성분)을 둘 다 잰다.
대조가 없으면 "무엇이든 잡는다"와 구별되지 않는다.

    docker exec -w /workspace -e PYTHONPATH=/workspace ship-paint-trainer \
      python3 trainer/scripts/correction_recovery.py --passes 200 --out /output/correction_recovery.json
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
import measured_pose
import pose_fk_csv
import pose_similarity
import pose_stages
import sequence_dtw
import simulator_csv

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]

DEFAULT_ROOT: Final = Path("/data/sources/professionals")
DEFAULT_REFERENCE: Final = Path("/output/expert_reference/expert_arm_reference.json")
# 사전등록한 교란 크기.
MAGNITUDES: Final[tuple[float, ...]] = (0.05, 0.10, 0.15, 0.25)
DIRECTIONS: Final[tuple[str, ...]] = ("radial", "tangential")
# 사전등록한 판정 기준.
VERDICT_MAGNITUDE: Final = 0.15
VERDICT_HIT_MIN: Final = 0.70
VERDICT_FALSE_POSITIVE_MAX: Final = 0.20


def _load_pose_lab() -> tuple[Any, Any]:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from trainer.pose_lab import csv_source, service

    return csv_source, service


def _unit(v: Vec3) -> Vec3 | None:
    length = math.sqrt(sum(c * c for c in v))
    return None if length < 1e-9 else (v[0] / length, v[1] / length, v[2] / length)


def _orthogonal(axis: Vec3) -> Vec3:
    """`axis`에 수직인 단위벡터 하나. 접선 교란의 방향이다."""
    helper = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
    cross = (
        axis[1] * helper[2] - axis[2] * helper[1],
        axis[2] * helper[0] - axis[0] * helper[2],
        axis[0] * helper[1] - axis[1] * helper[0],
    )
    return _unit(cross) or (1.0, 0.0, 0.0)


def perturb(
    frames: Sequence[Mapping[str, Vec3]], joint: str, direction: str, metres: float
) -> list[dict[str, Vec3]]:
    """프레임마다 `joint`를 옮긴다. **그 관절만** 옮긴다 — 귀속을 모호하게 하지 않는다.

    방향은 어깨 중점 기준이다. `radial`은 멀어지는 쪽, `tangential`은 그에 수직.
    """
    out: list[dict[str, Vec3]] = []
    for joints in frames:
        moved = dict(joints)
        frame = arm_reference.body_frame(joints)
        point = joints.get(joint)
        if frame is None or point is None:
            out.append(moved)
            continue
        offset = (
            point[0] - frame.origin[0],
            point[1] - frame.origin[1],
            point[2] - frame.origin[2],
        )
        radial = _unit(offset)
        if radial is None:
            out.append(moved)
            continue
        axis = radial if direction == "radial" else _orthogonal(radial)
        moved[joint] = (
            point[0] + axis[0] * metres,
            point[1] + axis[1] * metres,
            point[2] + axis[2] * metres,
        )
        out.append(moved)
    return out


def _pass_inputs(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Vec3]], list[Vec3], list[Vec3]] | None:
    """정본은 `pose_fk_csv.pass_inputs`다 — 실험과 pose_lab이 **같은 재료**를
    써야 같은 패스가 도구마다 다른 값을 내지 않는다."""
    return pose_fk_csv.pass_inputs(rows)


def _score(
    frames: Sequence[Mapping[str, Vec3]],
    aligned: pose_stages.AlignResult,
    bucket: Mapping[str, Any],
) -> pose_stages.ScoreResult:
    return pose_stages.score(frames, aligned, bucket, labels=pose_similarity.JOINT_LABELS)


def _silent_over(result: pose_stages.ScoreResult, bucket: Mapping[str, Any]) -> bool:
    """자기 범위를 벗어난 관절이 있는데 교정이 **안 나간** 경우.

    사전등록 ⑤다. 생 σ 순위에서는 1등이 아닌 관절의 이탈이 조용히 사라진다 —
    실측 222패스 중 이탈 65건에서 18건(28%)이 그랬다.
    """
    if result.corrections:
        return False
    ladders = bucket.get("joint_sigma_ladder")
    ladders = ladders if isinstance(ladders, Mapping) else None
    return any(
        item.sigma >= arm_reference.sigma_threshold(item.joint, ladders)
        for item in result.sigmas
    )


def run(root: Path, reference: Path, passes: int, seed: int) -> dict[str, Any]:
    csv_source, service = _load_pose_lab()
    buckets = json.loads(reference.read_text(encoding="utf-8"))["buckets"]
    source = csv_source.CsvPassSource(root)
    # **작업자별로 균형을 맞춘다.** 섞어 뽑기만 하면 위양성 추정치가 어느 작업자가
    # 뽑혔는지에 좌우된다 — σ가 작업자별로 군집돼 있어 위양성률이 0.015~0.259로
    # 17배 벌어진다(`docs/measurements/2026-09-30-per-joint-threshold-prereg.md` 결과).
    # 균형을 안 맞춘 120패스 표본이 0.225를 냈고 전량은 0.095였다.
    by_worker: dict[str, list[Any]] = {}
    for item in source.sessions():
        by_worker.setdefault(item.worker, []).append(item)
    rng = random.Random(seed)
    for group in by_worker.values():
        rng.shuffle(group)
    sessions = []
    order = sorted(by_worker)
    index = 0
    while any(index < len(by_worker[w]) for w in order):
        for worker in order:
            if index < len(by_worker[worker]):
                sessions.append(by_worker[worker][index])
        index += 1
    # 세션 단위 라운드로빈만으로는 부족하다 — 세션 안 패스 수가 제각각이라 한 작업자가
    # 몰릴 수 있다(실측에서 60패스 중 41개가 한 사람이었다). **패스 수로 상한**을 건다.
    per_worker_cap = max(1, passes // max(len(order), 1))

    # 결과: (direction, magnitude, perturbed_joint) -> {named_joint or None: count}
    tally: dict[tuple[str, float, str], dict[str, int]] = {}
    baseline = {"total": 0, "corrected": 0, "silent_over": 0,
                "by_joint": {}, "by_worker": {}}
    modes = {"phase": 0, "fold": 0, "none": 0}
    used = 0
    used_by_worker: dict[str, int] = {}
    workers: set[str] = set()

    for item in sessions:
        if used >= passes:
            break
        if used_by_worker.get(item.worker, 0) >= per_worker_cap:
            continue
        try:
            entries = source.index_of(item)
        except (OSError, ValueError, KeyError):
            continue
        for entry in entries[::5]:
            if used >= passes or used_by_worker.get(item.worker, 0) >= per_worker_cap:
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

            reach = pose_stages.reach_of(impacts, tools)
            resolved = pose_stages.resolve(reach, buckets)
            if not resolved.ok:
                continue
            bucket = buckets[resolved.band]
            aligned = pose_stages.align(impacts, resolved)
            if aligned.ok:
                modes["phase"] += 1
            else:
                # 접기 경로: 정렬 없이 시퀀스 전체를 한 점으로 본다.
                modes["fold"] += 1
                continue  # 위상 레퍼런스가 있는 버킷만 쓴다 — 두 채점을 섞지 않는다.

            clean = _score(frames, aligned, bucket)
            if not clean.ok:
                continue
            used += 1
            used_by_worker[item.worker] = used_by_worker.get(item.worker, 0) + 1
            workers.add(item.worker)

            baseline["total"] += 1
            slot = baseline["by_worker"].setdefault(item.worker, [0, 0])
            slot[0] += 1
            if clean.corrections:
                slot[1] += 1
            if _silent_over(clean, bucket):
                baseline["silent_over"] += 1
            if clean.corrections:
                baseline["corrected"] += 1
                name = clean.corrections[0].joint
                baseline["by_joint"][name] = baseline["by_joint"].get(name, 0) + 1

            for direction in DIRECTIONS:
                for metres in MAGNITUDES:
                    for sign in (1.0, -1.0):
                        for joint in arm_reference.SCORING_JOINTS_ARM:
                            moved = perturb(frames, joint, direction, sign * metres)
                            result = _score(moved, aligned, bucket)
                            key = (direction, metres, joint)
                            slot = tally.setdefault(key, {})
                            if not result.ok or not result.corrections:
                                slot["__miss__"] = slot.get("__miss__", 0) + 1
                            else:
                                named = result.corrections[0].joint
                                slot[named] = slot.get(named, 0) + 1
        if used and used % 25 == 0:
            logger.info("%d passes", used)

    return {
        "prereg": "docs/measurements/2026-09-30-correction-recovery-prereg.md",
        "reference": str(reference),
        "passes": used,
        "workers": sorted(workers),
        "scoring_modes": modes,
        "seed": seed,
        "magnitudes": list(MAGNITUDES),
        "baseline": baseline,
        "tally": {f"{d}|{m}|{j}": v for (d, m, j), v in tally.items()},
    }


def summarize(payload: Mapping[str, Any]) -> list[str]:
    """사람이 읽는 요약. 사전등록한 기준으로 판정한다."""
    lines: list[str] = []
    base = payload["baseline"]
    fp = base["corrected"] / base["total"] if base["total"] else float("nan")
    lines.append(
        f"패스 {payload['passes']}개 · 작업자 {len(payload['workers'])}인 · "
        f"채점 {payload['scoring_modes']}"
    )
    lines.append(f"기준선(d=0) 위양성률: {fp:.3f}  ({base['corrected']}/{base['total']})")
    silent = base.get("silent_over", 0)
    over = silent + base["corrected"]
    lines.append(
        f"⑤ 조용한 이탈: {silent}/{over} = "
        f"{silent / over if over else 0.0:.3f}  (범위를 벗어났는데 교정이 없음)"
    )
    if base.get("by_worker"):
        lines.append("  작업자별 위양성률 — **합계만 보면 군집이 숨는다**")
        for worker, (n, fired) in sorted(base["by_worker"].items()):
            lines.append(f"    {worker:10s} n={n:4d}  {fired/n if n else float('nan'):.3f}")
    if base["by_joint"]:
        lines.append("  교란 없이 지목된 관절: " + ", ".join(
            f"{k} {v}" for k, v in sorted(base["by_joint"].items(), key=lambda x: -x[1])
        ))

    rates: dict[tuple[str, float], tuple[float, float, float]] = {}
    for direction in DIRECTIONS:
        lines.append("")
        lines.append(f"[{direction}]  크기별 hit / confusion / miss")
        for metres in MAGNITUDES:
            hit = conf = miss = 0
            for joint in arm_reference.SCORING_JOINTS_ARM:
                slot = payload["tally"].get(f"{direction}|{metres}|{joint}", {})
                for named, count in slot.items():
                    if named == "__miss__":
                        miss += count
                    elif named == joint:
                        hit += count
                    else:
                        conf += count
            total = hit + conf + miss
            if not total:
                continue
            rates[(direction, metres)] = (hit / total, conf / total, miss / total)
            lines.append(
                f"  {metres:.2f} m   hit {hit/total:.3f}   confusion {conf/total:.3f}   miss {miss/total:.3f}"
            )

    # 혼동행렬 — 논문 표로 그대로 쓴다. 지목이 정확한데 민감도가 낮은 것과,
    # 지목이 엉뚱한 것은 다른 결함이고 다르게 고친다.
    short = {j: j.replace("Right", "R").replace("Left", "L").replace("UpperArm", "Up")
                 .replace("ForeArm", "Fo").replace("Hand", "Ha") for j in arm_reference.SCORING_JOINTS_ARM}
    for direction in DIRECTIONS:
        lines.append("")
        lines.append(f"[{direction}] 혼동행렬 — 교란 관절(행) × 지목 관절(열), 전 크기 합산")
        header = "  " + f"{'교란':>8s}" + "".join(f"{short[c]:>6s}" for c in arm_reference.SCORING_JOINTS_ARM) + f"{'miss':>7s}"
        lines.append(header)
        for row in arm_reference.SCORING_JOINTS_ARM:
            counts = {c: 0 for c in arm_reference.SCORING_JOINTS_ARM}
            miss = 0
            for metres in MAGNITUDES:
                slot = payload["tally"].get(f"{direction}|{metres}|{row}", {})
                for named, n in slot.items():
                    if named == "__miss__":
                        miss += n
                    elif named in counts:
                        counts[named] += n
            total = sum(counts.values()) + miss
            if not total:
                continue
            cells = "".join(f"{counts[c]/total:6.2f}" for c in arm_reference.SCORING_JOINTS_ARM)
            lines.append(f"  {short[row]:>8s}{cells}{miss/total:7.2f}")

    lines.append("")
    lines.append("사전등록 판정")
    radial = rates.get(("radial", VERDICT_MAGNITUDE))
    tangential = rates.get(("tangential", VERDICT_MAGNITUDE))
    checks: list[tuple[str, bool]] = []
    if radial:
        checks.append((f"① 반경 {VERDICT_MAGNITUDE} m hit {radial[0]:.3f} ≥ {VERDICT_HIT_MIN}",
                       radial[0] >= VERDICT_HIT_MIN))
    checks.append((f"② 위양성률 {fp:.3f} < {VERDICT_FALSE_POSITIVE_MAX}",
                   fp < VERDICT_FALSE_POSITIVE_MAX))
    if radial and tangential:
        checks.append((f"③ 접선 hit {tangential[0]:.3f} < 반경 hit {radial[0]:.3f}",
                       tangential[0] < radial[0]))
    for text, ok in checks:
        lines.append(f"  {'통과' if ok else '미달'}  {text}")
    lines.append(f"  → {'쓸 만하다' if all(ok for _t, ok in checks) else '미달 — 그대로 보고한다'}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--passes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    payload = run(args.root, args.reference, args.passes, args.seed)
    text = "\n".join(summarize(payload))
    print(text)
    if args.out:
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
