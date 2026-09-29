#!/usr/bin/env python3
"""숙련자 CSV 코퍼스에서 팔 자세 레퍼런스를 굽는다.

자세는 `pose_fk_csv`의 FK에서 온다 — CSV의 관절 Position 컬럼은 Pelvis를 뺀 19관절이
프레임 불변 바인드 오프셋이라 월드 좌표가 아니다(`simulator_csv.joint_world_positions`
독스트링). 회전에서 FK로 풀어야 자세가 나온다.

버킷 조건은 **reach = 착탄 수직 − 건 월드 수직**이다. 건은 활성 건 강체
(`measured_pose.right`), 착탄은 Ray point 컬럼 — 둘 다 실측 월드라 바닥 상수가 지워진다.

**머리를 쓰지 않는다.** CSV의 머리는 카메라(HMD) 강체인데 런타임이 보내는 머리는 골격
`head` 관절이라 같은 점이 아니다. 건은 양쪽 모두 같은 물건이다(`arm_reference` 독스트링).

**패스 하나가 표본 하나다.** 프레임을 다 넣으면 같은 패스의 120프레임이 독립 표본인 척해
n이 부풀고 산포가 좁아진다 — 좁은 산포는 모든 σ를 키워 교정을 과발부한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import arm_reference
import measured_pose
import sequence_dtw
import pose_fk_csv
import simulator_csv
import world_axes

logger = logging.getLogger(__name__)

DEFAULT_ROOT: Final = Path("/data/sources/professionals")
DEFAULT_OUT: Final = Path("/output/expert_reference/expert_arm_reference.json")


def _load_pose_lab() -> tuple[Any, Any]:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from trainer.pose_lab import csv_source, service

    return csv_source, service


def _pass_sample(
    rows: list[dict[str, str]], *, with_frames: bool = False
) -> tuple[dict[str, Any], float] | tuple[dict[str, Any], float, list, list] | None:
    """패스 하나를 (정규화된 팔 자세, reach)로 줄인다. 못 줄이면 `None`.

    `with_frames=True`면 **위상 레퍼런스용**으로 솎은 프레임별 자세와 착탄 경로를 더 낸다.
    전량을 메모리에 얹을 수 없어(밴드당 최대 14,245패스) 호출자가 표본을 정한다.
    """
    if not rows:
        return None
    offsets = pose_fk_csv.bind_offsets(rows[0])
    if not offsets:
        return None

    # 자세: `arm_reference.normalize_sequence`가 프레임별 정규화 후 중앙값을 낸다.
    # **재는 쪽과 같은 함수여야 한다** — 순서가 갈리면 숙련자를 숙련자 기준에 대고도 0점이 나온다.
    poses: list[dict[str, tuple[float, float, float]]] = []
    for row in rows:
        try:
            poses.append(pose_fk_csv.forward_kinematics(row, offsets))
        except (KeyError, ValueError):
            continue
    pose = arm_reference.normalize_sequence(poses)
    if not pose:
        return None
    normalized_frames = [n for n in (arm_reference.normalize(p) for p in poses) if n]

    try:
        frames = measured_pose.measured_frames(rows)
    except (KeyError, ValueError):
        return None
    guns = [world_axes.vertical_of(tuple(frame.right)) for frame in frames if frame.right]
    impacts: list[float] = []
    for row in rows:
        try:
            impacts.append(
                world_axes.vertical_of(
                    tuple(float(row[column]) for column in simulator_csv.RAY_POINT_COLUMNS)
                )
            )
        except (KeyError, ValueError):
            continue
    if not guns or not impacts:
        return None
    reach = st.median(impacts) - st.median(guns)
    if not with_frames:
        return pose, reach

    # 위상용: 착탄 경로(월드 3D)와 프레임별 자세를 **같은 인덱스로** 솎는다.
    hits: list[tuple[float, float, float]] = []
    for row in rows:
        try:
            hits.append(tuple(float(row[column]) for column in simulator_csv.RAY_POINT_COLUMNS))
        except (KeyError, ValueError):
            hits.append(None)  # type: ignore[arg-type]
    paired = [
        (frame, hit)
        for frame, hit in zip(normalized_frames, hits[: len(normalized_frames)])
        if hit is not None
    ]
    if len(paired) < 4:
        return pose, reach
    shape = sequence_dtw.path_shape([hit for _, hit in paired])
    if shape is None:
        return pose, reach
    indices = sequence_dtw.decimate(shape)
    return (
        pose,
        reach,
        [paired[i][0] for i in indices],
        [shape[i] for i in indices],
    )


def build(root: Path, levels: set[str] | None = None) -> dict[str, object]:
    csv_source, service = _load_pose_lab()
    source = csv_source.CsvPassSource(root)
    buckets: dict[str, arm_reference.BucketAccum] = {}
    samples: list[tuple[str, dict[str, tuple[float, float, float]]]] = []

    # 위상 레퍼런스용 표본. 밴드마다 `PHASE_SAMPLE_PER_BAND`까지만 들고 있는다 —
    # 전량(밴드당 최대 14,245패스)은 메모리에 안 들어가고, 위상마다 수백이면 SD가 안정적이다.
    phase_samples: dict[str, list[tuple[list, list]]] = {}

    sessions = [s for s in source.sessions() if levels is None or s.level in levels]
    workers: set[str] = set()
    passes = 0
    skipped = 0
    for number, item in enumerate(sessions, start=1):
        try:
            entries = source.index_of(item)
        except (OSError, ValueError, KeyError):
            continue
        for entry in entries:
            ref = service.PassRef(
                worker=item.worker, date=item.date, level=item.level,
                session=item.session, pass_index=entry.pass_index,
            )
            try:
                raw = source.load(ref)
            except (LookupError, OSError, ValueError):
                skipped += 1
                continue
            # 밴드를 알기 전에는 위상 표본이 필요한지 모른다. 그래서 먼저 싸게 재고,
            # 그 밴드가 아직 덜 찼을 때만 프레임까지 뽑는다 — 64,098패스 전부에서
            # 프레임을 뽑으면 빌드 시간이 배로 든다.
            cheap = _pass_sample(raw.rows)
            if cheap is None:
                skipped += 1
                continue
            pose, reach = cheap[0], cheap[1]
            band = arm_reference.reach_band(reach)
            buckets.setdefault(band, arm_reference.BucketAccum()).add(pose, item.worker)
            samples.append((band, pose))
            slot = phase_samples.setdefault(band, [])
            if len(slot) < arm_reference.PHASE_SAMPLE_PER_BAND:
                rich = _pass_sample(raw.rows, with_frames=True)
                if rich is not None and len(rich) == 4:
                    slot.append((rich[2], rich[3]))
            workers.add(item.worker)
            passes += 1
        if number % 25 == 0:
            logger.info("%d/%d sessions · %d passes · %d skipped", number, len(sessions), passes, skipped)

    # 2차 통과: 숙련자 자신의 점수 분포를 잰다. 절대값이 해석 불가능하므로 백분위 사다리를
    # 같이 실어야 "이 점수가 숙련자 중 어디쯤인가"를 말할 수 있다.
    #
    # **표본내다** — 각 패스가 자기 버킷의 평균에 기여했다. n이 수백이라 한 패스의 영향은
    # 작지만 사다리가 미세하게 낙관적이라는 사실은 남는다. 응답에 `in_sample`로 적는다.
    payloads = {band: accum.payload() for band, accum in sorted(buckets.items())}

    # 위상 레퍼런스: 밴드마다 DBA 프로토타입을 굽고, 그것에 정렬해 위상별 분포를 쌓는다.
    for band, entry in payloads.items():
        kept = phase_samples.get(band, [])
        if len(kept) < arm_reference.MIN_PHASE_SAMPLES:
            continue
        shapes = [shape for _, shape in kept]
        prototype = sequence_dtw.barycenter(
            shapes[:: max(1, len(shapes) // arm_reference.PROTOTYPE_SAMPLE)][
                : arm_reference.PROTOTYPE_SAMPLE
            ],
            phases=arm_reference.PHASES,
        )
        if prototype is None:
            continue
        to_prototype, between = sequence_dtw.barycenter_support(prototype, shapes)
        phase_accums = [arm_reference.PhaseAccum() for _ in range(arm_reference.PHASES)]
        for frames, shape in kept:
            path, _ = sequence_dtw.align(shape, prototype)
            for query_index, phase in path:
                if query_index < len(frames):
                    phase_accums[phase].add(frames[query_index], "w")
        # 위상 점수는 접기 점수와 **다른 분포**다(실측 p50 0.704 vs 0.763). 접기용 사다리에
        # 대고 백분위를 내면 틀린다 — 위상 전용 사다리를 따로 굽는다.
        phase_payload = [accum.payload() for accum in phase_accums]
        phase_scores: list[float] = []
        for frames, shape in kept:
            path, _ = sequence_dtw.align(shape, prototype)
            per_phase: list[dict[str, tuple[float, float, float]] | None] = [
                None for _ in range(arm_reference.PHASES)
            ]
            grouped: dict[int, list[dict]] = {}
            for query_index, phase in path:
                if query_index < len(frames):
                    grouped.setdefault(phase, []).append(frames[query_index])
            for phase, poses_at in grouped.items():
                names = set(poses_at[0])
                per_phase[phase] = {
                    name: tuple(
                        sum(pose[name][axis] for pose in poses_at) / len(poses_at)
                        for axis in range(3)
                    )
                    for name in names
                }
            sigmas = arm_reference.phase_sigmas(per_phase, phase_payload)
            value = arm_reference.similarity_score(sigmas)
            if value is not None:
                phase_scores.append(value)
        ladder = arm_reference.percentile_ladder(phase_scores)
        if ladder:
            entry["expert_scores_phase"] = {
                **ladder,
                "reference_pass_count": len(kept),
                "in_sample": True,
            }

        entry["prototype"] = [list(point) for point in prototype]
        entry["prototype_support"] = {
            "to_prototype": to_prototype,
            "between": between,
            "n": len(kept),
        }
        entry["phases"] = phase_payload
        logger.info(
            "%s 위상 레퍼런스 · 표본 %d · 지지도 %.4f/%.4f",
            band, len(kept), to_prototype, between,
        )
    scored: dict[str, list[float]] = {}
    for band, pose in samples:
        entry = payloads.get(band)
        if entry is None:
            continue
        value = arm_reference.similarity_score(arm_reference.joint_sigmas(pose, entry))
        if value is not None:
            scored.setdefault(band, []).append(value)
    for band, entry in payloads.items():
        ladder = arm_reference.percentile_ladder(scored.get(band, []))
        if ladder:
            entry["expert_scores"] = {
                **ladder,
                "reference_pass_count": entry["n"],
                "in_sample": True,
            }

    return {
        "version": 1,
        "scoring_set_id": arm_reference.SCORING_SET_ID,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "reach_band_m": arm_reference.REACH_BAND_M,
        "sessions": len(sessions),
        "passes": passes,
        "skipped": skipped,
        "workers": sorted(workers),
        "buckets": payloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--levels", default="", help="쉼표로 구분. 비우면 전부.")
    parser.add_argument("--limit-sessions", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    levels = {part for part in args.levels.split(",") if part} or None
    payload = build(args.root, levels)
    if args.limit_sessions:
        logger.info("limit-sessions는 build 안에서 쓰지 않는다 — 전수로 돌았다")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "wrote %s · passes=%d buckets=%d workers=%d",
        args.out, payload["passes"], len(payload["buckets"]), len(payload["workers"]),
    )


if __name__ == "__main__":
    main()
