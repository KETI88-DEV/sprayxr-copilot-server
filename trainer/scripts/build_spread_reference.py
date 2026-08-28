#!/usr/bin/env python3
"""전형기 패스에서 버킷별·점별·축별 산포를 뽑아 아티팩트로 굽는다.

산포를 전형기 한 사람의 변동으로 재는 이유는 6인으로 재면 작업자 간 차이가 잣대에
들어가기 때문이다. 그것이 바로 교정하려는 대상이다.

축은 wall/vertical/lateral이다. 교정 후보를 고르는 축과 같아야 sigma로 축을 고를 수 있다.

축 분해는 `measured_score._axis_magnitudes_cm`을 쓴다 — `_correction_candidates`가 쓰는
`MIN_CORRECTION_CM` 임계를 여기서 쓰면 안 된다. 그 임계는 "문장으로 낼 만큼 크다"는
질문이고, 산포는 "이 축의 산포에 비춰 이상한가"라는 다른 질문의 재료다. 임계로 작은
성분을 지우면 그 축의 산포가 큰 편차를 낸 패스로만 채워져 조용히 부풀고, 부푼 산포는
이후 모든 σ를 낮춰 그 축의 교정이 점점 덜 발동하게 만든다.

날짜별 부분합을 함께 실어 질의 날짜 제외와 개선율의 A/B 분리를 전수 재빌드 없이 한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import measured_score
import motion_key
import spread_stats

logger = logging.getLogger(__name__)

GRANULARITIES: Final[tuple[tuple[str, int], ...]] = (
    ("uv8", 8), ("uv4", 4), ("uv2", 2), ("uv_off", 0),
)
# 이전 연구에서 사전등록한 값. 이번 수치를 보고 정하지 않는다.
N_MIN: Final = 20


def key_without_uv(key: str) -> str:
    """UV 셀 성분을 na로 바꾼다. motion_key가 uv=None일 때 내는 형태와 같다."""
    parts = key.split(":")
    if len(parts) < 5:
        return key
    parts[3] = "na"
    return ":".join(parts)


def pass_axis_values(
    tracks: Mapping[str, Any], frame: Any
) -> dict[str, dict[str, float]]:
    """한 패스의 점별·축별 값(m). 임계 없는 축 분해(`_axis_magnitudes_cm`)를 쓴다.

    `_correction_candidates`를 쓰면 안 된다 — 그것은 `MIN_CORRECTION_CM` 미만인 축을
    후보에서 뺀다. 산포 계산에 그 필터를 쓰면 작은 편차를 낸 패스가 그 축의 표본에서
    빠져 산포가 큰 편차 쪽으로만 부푼다.
    """
    output: dict[str, dict[str, float]] = {}
    for point in measured_score.POINT_ORDER:
        points = [p for p in (tracks.get(point) or []) if p is not None]
        if not points:
            continue
        mean = (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
            sum(p[2] for p in points) / len(points),
        )
        axes = {
            axis: magnitude_cm / 100.0
            for axis, magnitude_cm in measured_score._axis_magnitudes_cm(mean, frame).items()
        }
        output[point] = axes
    return output


def build_payload(
    worker: str,
    levels: Sequence[str],
    sessions: int,
    buckets: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Mapping[str, spread_stats.AxisAccum]]]]],
) -> dict[str, object]:
    """buckets: layer -> key -> date -> point -> axis -> AxisAccum"""
    granularities: dict[str, object] = {}
    for layer, _grid in GRANULARITIES:
        layer_out: dict[str, object] = {}
        for key, by_date in (buckets.get(layer) or {}).items():
            dates = sorted(by_date)
            merged: dict[str, dict[str, spread_stats.AxisAccum]] = {}
            for date in dates:
                for point, axes in by_date[date].items():
                    slot = merged.setdefault(point, {})
                    for axis, accum in axes.items():
                        slot[axis] = spread_stats.merge(
                            [slot.get(axis, spread_stats.AxisAccum()), accum]
                        )
            points_out: dict[str, dict[str, dict[str, float]]] = {}
            for point, axes in merged.items():
                points_out[point] = {}
                for axis, accum in axes.items():
                    mean, spread = spread_stats.summarize(accum)
                    points_out[point][axis] = {"mean": mean, "spread": spread}
            counts = [accum.n for axes in merged.values() for accum in axes.values()]
            layer_out[key] = {
                "n": max(counts) if counts else 0,
                "dates": dates,
                "by_date": {
                    date: {
                        point: {axis: spread_stats.to_json(accum) for axis, accum in axes.items()}
                        for point, axes in by_date[date].items()
                    }
                    for date in dates
                },
                "points": points_out,
            }
        granularities[layer] = layer_out

    return {
        "version": 1,
        "worker": worker,
        "levels": list(levels),
        "sessions": sessions,
        "n_min": N_MIN,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "granularities": granularities,
    }


# --- 실데이터 훑기 --------------------------------------------------------------


def _load_pose_lab() -> tuple[Any, Any]:
    """`trainer.pose_lab.{csv_source,service}`를 늦게 부른다.

    순수 함수(위)만 쓰는 테스트는 이 경로를 안 타므로 트리 구조가 달라도(패키지
    루트를 못 찾아도) 임포트가 깨지지 않는다. `motion_key_sweep._load_csv_source`와
    같은 패턴이다.
    """
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from trainer.pose_lab import csv_source, service

    return csv_source, service


def _levels(spec: str) -> list[str]:
    if "-" in spec:
        low, high = spec.split("-", 1)
        return [str(value) for value in range(int(low), int(high) + 1)]
    return [part for part in spec.split(",") if part]


def _accumulate_pass(
    buckets: dict[str, dict[str, dict[str, dict[str, dict[str, spread_stats.AxisAccum]]]]],
    layer: str,
    key: str,
    date: str,
    axis_values: Mapping[str, Mapping[str, float]],
) -> None:
    by_date = buckets[layer].setdefault(key, {}).setdefault(date, {})
    for point, axes in axis_values.items():
        slot = by_date.setdefault(point, {})
        for axis, value in axes.items():
            addition = spread_stats.accumulate([value])
            slot[axis] = spread_stats.merge([slot.get(axis, spread_stats.AxisAccum()), addition])


def build_from_root(root: Path, worker: str, levels: Sequence[str]) -> dict[str, object]:
    """`root` 아래 `worker`의 학습모드 패스를 전부 훑어 아티팩트 페이로드를 만든다."""
    csv_source, service = _load_pose_lab()
    source = csv_source.CsvPassSource(root)
    wanted_levels = set(levels)
    matched_sessions = [
        item for item in source.sessions()
        if item.worker == worker and item.level in wanted_levels
    ]

    buckets: dict[str, dict[str, dict[str, dict[str, dict[str, spread_stats.AxisAccum]]]]] = {
        name: {} for name, _grid in GRANULARITIES
    }

    total_passes = 0
    for session_number, item in enumerate(matched_sessions, start=1):
        entries = source.index_of(item)
        session_passes = 0
        for entry in entries:
            ref = service.PassRef(
                worker=item.worker, date=item.date, level=item.level,
                session=item.session, pass_index=entry.pass_index,
            )
            raw = source.load(ref)
            frames = service.pass_frames(raw)
            inputs = service.key_inputs(frames)
            if inputs is None:
                continue
            basis = service.representative_basis(frames)
            tracks = service._local_tracks(frames)
            axis_values = pass_axis_values(tracks, basis)
            if not axis_values:
                continue

            keys_by_layer: dict[str, str] = {}
            for layer, grid in GRANULARITIES:
                if layer == "uv_off":
                    keys_by_layer[layer] = key_without_uv(keys_by_layer["uv8"])
                else:
                    keys_by_layer[layer] = service.key_from(
                        inputs, motion_key.KeyParams(uv_grid=grid)
                    )

            for layer, _grid in GRANULARITIES:
                _accumulate_pass(buckets, layer, keys_by_layer[layer], item.date, axis_values)

            session_passes += 1
            total_passes += 1

        logger.info(
            "[%d/%d] %s L%s %s %s: %d/%d passes accumulated",
            session_number, len(matched_sessions),
            item.worker, item.level, item.date, item.session,
            session_passes, len(entries),
        )

    logger.info(
        "worker=%s sessions=%d passes=%d", worker, len(matched_sessions), total_passes
    )
    return build_payload(
        worker=worker,
        levels=sorted(wanted_levels, key=lambda level: int(level) if level.isdigit() else 0),
        sessions=len(matched_sessions),
        buckets=buckets,
    )


# --- CLI -------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="세션 CSV 트리의 루트.")
    parser.add_argument("--worker", required=True, help="레퍼런스를 뽑을 전문가 이름.")
    parser.add_argument("--out", type=Path, required=True, help="아티팩트 JSON 출력 경로.")
    parser.add_argument("--levels", default="1-10", help="레벨 범위(예: 1-10) 또는 콤마 목록.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    payload = build_from_root(args.root, args.worker, _levels(args.levels))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
