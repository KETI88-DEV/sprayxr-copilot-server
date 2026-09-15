#!/usr/bin/env python3
"""스켈레톤 장착 상수(C_j, m_j) 적합과 작업자 교차 검증(LOWO).

사전등록: `docs/measurements/2026-09-15-skeleton-mounts-prereg.md`. 이 스크립트는 그 문서의
§2 표본, §3 적합, §4 폴드, §5 합격 기준을 그대로 구현한다. 문턱은 여기서 옮기지 않는다.

출력: `trainer/pose_lab/skeleton_mounts.json`. `validation.passed`가 false면 서버는 전역 상수를
쓰지 않는다(`service.skeleton_block`).

사용(컨테이너):
    python3 trainer/scripts/fit_skeleton_mounts.py --root /data/sources/professionals \\
        --out trainer/pose_lab/skeleton_mounts.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import measured_pose  # noqa: E402
import pose_fk_csv  # noqa: E402
import simulator_csv  # noqa: E402

from trainer.pose_lab import csv_source, skeleton  # noqa: E402

logger = logging.getLogger("fit_skeleton_mounts")

SEED: Final = 20260915
SESSIONS_PER_WORKER: Final = 10
FRAMES_PER_SESSION: Final = 400
MIN_GROUP_SESSIONS: Final = 3

POOLED_MEDIAN_MAX_MM: Final = 150.0
FOLD_MEDIAN_MAX_MM: Final = 200.0
BASELINE_RATIO_MAX: Final = 0.7
BONE_ERROR_MAX_M: Final = 1e-9

GRIP_LIMIT_M: Final[dict[str, float]] = {
    "한길숏건": 0.25,
    "반도숏건": 0.25,
    "한길폴건": 1.2,
    "반도폴건": 1.2,
}

PREREG_DOC: Final = "docs/measurements/2026-09-15-skeleton-mounts-prereg.md"
METHOD: Final = "fk_csv_yxz_orient_gravity"


def pick_sessions(sessions: Sequence[csv_source.SessionFile]) -> list[csv_source.SessionFile]:
    """§2: (worker, date, level, session) 정렬 뒤 작업자별 `rng.sample(·, min(10, n))`."""
    rng = random.Random(SEED)
    by_worker: dict[str, list[csv_source.SessionFile]] = defaultdict(list)
    for item in sorted(sessions, key=lambda s: (s.worker, s.date, s.level, s.session)):
        by_worker[item.worker].append(item)
    picked: list[csv_source.SessionFile] = []
    for worker in sorted(by_worker):
        pool = by_worker[worker]
        picked.extend(rng.sample(pool, min(SESSIONS_PER_WORKER, len(pool))))
    return picked


def session_samples(item: csv_source.SessionFile) -> tuple[list[skeleton.FrameSample], dict[str, str], dict[str, tuple]]:
    rows = simulator_csv.read_rows(item.path)
    gun = measured_pose.active_gun(rows)
    meta = {"worker": item.worker, "date": item.date, "level": item.level, "session": item.session, "group": gun or ""}
    if gun is None or not rows:
        return [], meta, {}
    offsets = pose_fk_csv.bind_offsets(rows[0])
    if not offsets:
        return [], meta, {}
    paired = [
        (frame, row)
        for frame, row in measured_pose.measured_frames_with_rows(rows)
        if frame.spraying and frame.hit is not None
    ]
    if not paired:
        return [], meta, offsets
    chosen = sorted(set(np.linspace(0, len(paired) - 1, min(FRAMES_PER_SESSION, len(paired))).astype(int).tolist()))
    samples: list[skeleton.FrameSample] = []
    for i in chosen:
        frame, row = paired[i]
        sample = skeleton.frame_sample(frame, row, gun, offsets)
        if sample is not None:
            samples.append(sample)
    return samples, meta, offsets


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


def _p90(values: Sequence[float]) -> float | None:
    return float(np.percentile(values, 90)) if len(values) else None


def fit_table(sessions: Sequence[Sequence[skeleton.FrameSample]]) -> dict[str, skeleton.AnchorMount]:
    table = skeleton.fit_mounts(sessions)
    group_sessions = Counter(
        session[0].group for session in sessions if session
    )
    return {
        key: mount
        for key, mount in table.items()
        if not key.startswith("right:") or group_sessions[key.split(":", 1)[1]] >= MIN_GROUP_SESSIONS
    }


def evaluate(
    sessions: Sequence[tuple[list[skeleton.FrameSample], dict, dict]],
    table: dict[str, skeleton.AnchorMount],
) -> dict[str, Any]:
    miss: dict[str, list[float]] = defaultdict(list)
    base: dict[str, list[float]] = defaultdict(list)
    bone_errors: list[float] = []
    leg_tilt: list[float] = []
    gravity: list[float] = []
    skipped = 0
    frames = 0
    for samples, meta, offsets in sessions:
        if not samples:
            continue
        if skeleton.mount_key("right", meta["group"]) not in table:
            skipped += 1
            continue
        mounts = skeleton.mounts_for(table, meta["group"])
        identity = skeleton.identity_mounts()
        for sample in samples:
            placed = skeleton.place_oriented(sample.local_pos, sample.local_rot, sample.measured, mounts)
            baseline = skeleton.place_oriented(sample.local_pos, sample.local_rot, sample.measured, identity)
            if placed is None or baseline is None:
                continue
            frames += 1
            for fit in placed.anchors:
                miss[fit.point].append(fit.miss_m * 1000.0)
            for fit in baseline.anchors:
                base[fit.point].append(fit.miss_m * 1000.0)
            bone_errors.extend(skeleton.bone_length_errors(placed.joints, offsets))
            if placed.leg_tilt_deg is not None:
                leg_tilt.append(placed.leg_tilt_deg)
            gravity.append(placed.gravity_correction_deg)
    return {
        "frames": frames,
        "skipped_sessions_group_missing": skipped,
        "miss": miss,
        "base": base,
        "max_bone_error_m": max(bone_errors) if bone_errors else None,
        "leg_tilt": leg_tilt,
        "gravity": gravity,
    }


def _serialize_mount(mount: skeleton.AnchorMount) -> dict[str, Any]:
    return {"rotation": [list(r) for r in mount.rotation], "offset": list(mount.offset)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    source = csv_source.CsvPassSource(args.root)
    picked = pick_sessions(source.sessions())
    logger.info("picked %d sessions", len(picked))
    loaded: list[tuple[list[skeleton.FrameSample], dict, dict]] = []
    for n, item in enumerate(picked, 1):
        samples, meta, offsets = session_samples(item)
        meta["frames"] = len(samples)
        loaded.append((samples, meta, offsets))
        logger.info("[%d/%d] %s %s L%s %s group=%s frames=%d", n, len(picked), item.worker, item.date, item.level, item.session, meta["group"], len(samples))

    anchors = list(skeleton.ANCHOR_ORDER)
    workers = sorted({meta["worker"] for _s, meta, _o in loaded})
    folds: list[dict[str, Any]] = []
    pooled_miss: dict[str, list[float]] = defaultdict(list)
    pooled_base: dict[str, list[float]] = defaultdict(list)
    all_leg: list[float] = []
    all_gravity: list[float] = []
    max_bone = 0.0
    for worker in workers:
        train = [s for s, meta, _o in loaded if meta["worker"] != worker and s]
        held = [entry for entry in loaded if entry[1]["worker"] == worker]
        table = fit_table(train)
        result = evaluate(held, table)
        fold_median = {a: _median(result["miss"][a]) for a in anchors}
        fold_base = {a: _median(result["base"][a]) for a in anchors}
        for a in anchors:
            pooled_miss[a].extend(result["miss"][a])
            pooled_base[a].extend(result["base"][a])
        all_leg.extend(result["leg_tilt"])
        all_gravity.extend(result["gravity"])
        if result["max_bone_error_m"] is not None:
            max_bone = max(max_bone, result["max_bone_error_m"])
        folds.append(
            {
                "held_out_worker": worker,
                "eval_frames": result["frames"],
                "skipped_sessions_group_missing": result["skipped_sessions_group_missing"],
                "median_miss_mm": fold_median,
                "baseline_no_mount_median_miss_mm": fold_base,
                "ratio_to_baseline": {
                    a: (fold_median[a] / fold_base[a]) if fold_median[a] is not None and fold_base[a] else None
                    for a in anchors
                },
                "n": {a: len(result["miss"][a]) for a in anchors},
                "m_right_norm_by_group": {
                    key.split(":", 1)[1]: float(np.linalg.norm(mount.offset))
                    for key, mount in table.items()
                    if key.startswith("right:")
                },
            }
        )
        logger.info("fold %s: median %s baseline %s", worker, fold_median, fold_base)

    pooled = {a: _median(pooled_miss[a]) for a in anchors}
    pooled_baseline = {a: _median(pooled_base[a]) for a in anchors}
    c1_pooled = all(v is not None and v <= POOLED_MEDIAN_MAX_MM for v in pooled.values())
    c1_fold = all(
        fold["median_miss_mm"][a] is not None and fold["median_miss_mm"][a] <= FOLD_MEDIAN_MAX_MM
        for fold in folds
        for a in anchors
    )
    c2 = all(
        fold["ratio_to_baseline"][a] is not None and fold["ratio_to_baseline"][a] <= BASELINE_RATIO_MAX
        for fold in folds
        for a in anchors
    )
    c3 = max_bone <= BONE_ERROR_MAX_M
    passed = bool(c1_pooled and c1_fold and c2 and c3)

    final = fit_table([s for s, _m, _o in loaded if s])
    group_counts = Counter(meta["group"] for s, meta, _o in loaded if s)
    output = {
        "version": 1,
        "method": METHOD,
        "prereg": PREREG_DOC,
        "seed": SEED,
        "fitted_sessions": [meta for _s, meta, _o in loaded],
        "group_session_counts": dict(group_counts),
        "validation": {
            "folds": folds,
            "pooled": {"median_miss_mm": pooled, "n": {a: len(pooled_miss[a]) for a in anchors}},
            "baseline_no_mount": {"median_miss_mm": pooled_baseline},
            "max_bone_error_m": max_bone,
            "criteria": {
                "pooled_median_le_150mm": c1_pooled,
                "fold_median_le_200mm": c1_fold,
                "fold_ratio_to_baseline_le_0.7": c2,
                "bone_error_le_1e-9m": c3,
            },
            "passed": passed,
        },
        "reported": {
            "leg_tilt_deg": {"median": _median(all_leg), "p90": _p90(all_leg), "n": len(all_leg)},
            "gravity_correction_deg": {"median": _median(all_gravity), "p90": _p90(all_gravity), "n": len(all_gravity)},
            "m_right_norm_by_group": {
                key.split(":", 1)[1]: float(np.linalg.norm(mount.offset))
                for key, mount in final.items()
                if key.startswith("right:")
            },
        },
        "grip_limit_m": GRIP_LIMIT_M,
        "mounts": {
            "head": _serialize_mount(final["head"]) if "head" in final else None,
            "left": _serialize_mount(final["left"]) if "left" in final else None,
            "right": {
                key.split(":", 1)[1]: _serialize_mount(mount)
                for key, mount in final.items()
                if key.startswith("right:")
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("passed=%s pooled=%s baseline=%s criteria=%s", passed, pooled, pooled_baseline, output["validation"]["criteria"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
