#!/usr/bin/env python3
"""자세를 그림과 영상으로 견준다.

숫자만으로는 "오른쪽 팔꿈치를 15° 올리세요"가 실제로 어떤 모습인지 알기 어렵다. 같은
좌표계에 두 자세를 겹쳐 그리면 어디가 다른지 한눈에 보인다.

정규화 공간에서 그린다. 루트가 원점이고 +X가 작업자의 오른쪽, +Y가 앞, +Z가 위다.
그래서 정면도는 X-Z, 측면도는 Y-Z가 된다.

그림 안의 글자는 ASCII만 쓴다. 컨테이너에 한글 폰트가 없어 한글을 넘기면 경고 없이
두부(네모)로 렌더링된다. 한국어 설명은 그림이 아니라 리포트에 적는다.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Final, Mapping, Sequence

from simulator_csv import BONES

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]

VIEWS: Final = (("front (X-Z)", 0), ("side (Y-Z)", 1))


def ascii_label(text: str) -> str:
    """그림용 라벨. 한글 폰트가 없어 비 ASCII는 두부로 나오므로 걸러서 알린다."""
    if text.isascii():
        return text
    logger.warning("non-ascii label would render as boxes, replaced: %r", text)
    return "".join(char if char.isascii() else "?" for char in text)


def drawable_bones(pose: Mapping[str, Vec3]) -> list[tuple[Vec3, Vec3]]:
    """자세에 양쪽 관절이 다 있는 뼈만 낸다. 없는 관절을 그리면 조용히 선이 빠진다."""
    segments: list[tuple[Vec3, Vec3]] = []
    for start, end in BONES:
        first = pose.get(start)
        second = pose.get(end)
        if first is not None and second is not None:
            segments.append((first, second))
    return segments


def _plot(axis: object, pose: Mapping[str, Vec3], plane: int, color: str, label: str) -> None:
    drawn = False
    for start, end in drawable_bones(pose):
        axis.plot(  # type: ignore[attr-defined]
            [start[plane], end[plane]],
            [start[2], end[2]],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=label if not drawn else None,
        )
        drawn = True


def _limits(poses: Sequence[Mapping[str, Vec3]], plane: int) -> tuple[float, float, float, float]:
    values = [point[plane] for pose in poses for point in pose.values()]
    heights = [point[2] for pose in poses for point in pose.values()]
    pad = 0.15
    return min(values) - pad, max(values) + pad, min(heights) - pad, max(heights) + pad


def render_comparison(
    first: Mapping[str, Vec3],
    second: Mapping[str, Vec3],
    path: str | Path,
    labels: tuple[str, str],
    title: str = "",
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for axis, (name, plane) in zip(axes, VIEWS):
        _plot(axis, second, plane, "#9aa0a6", ascii_label(labels[1]))
        _plot(axis, first, plane, "#d93025", ascii_label(labels[0]))
        left, right, bottom, top = _limits((first, second), plane)
        axis.set_xlim(left, right)
        axis.set_ylim(bottom, top)
        axis.set_aspect("equal")
        axis.set_title(name)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(target, dpi=110)
    plt.close(figure)
    return target


def render_animation(
    first: Sequence[Mapping[str, Vec3]],
    second: Sequence[Mapping[str, Vec3]],
    path: str | Path,
    labels: tuple[str, str],
    fps: int = 12,
    title: str = "",
) -> Path:
    """두 자세 흐름을 나란히 재생한다. 길이가 다르면 짧은 쪽에 맞춘다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = min(len(first), len(second))
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    bounds = [_limits([*first, *second], plane) for _, plane in VIEWS]

    # 컨테이너 ffmpeg의 libopenh264가 깨져 있어 h264는 실패한다. mpeg4는 동작한다.
    writer = FFMpegWriter(fps=fps, codec="mpeg4", metadata={"title": title or "pose comparison"})
    with writer.saving(figure, str(target), dpi=110):
        for index in range(count):
            for axis, (name, plane), limit in zip(axes, VIEWS, bounds):
                axis.clear()
                _plot(axis, second[index], plane, "#9aa0a6", ascii_label(labels[1]))
                _plot(axis, first[index], plane, "#d93025", ascii_label(labels[0]))
                axis.set_xlim(limit[0], limit[1])
                axis.set_ylim(limit[2], limit[3])
                axis.set_aspect("equal")
                axis.set_title(f"{name}  frame {index + 1}/{count}")
                axis.grid(alpha=0.25)
                axis.legend(loc="upper right", fontsize=8)
            if title:
                figure.suptitle(title)
            writer.grab_frame()
    plt.close(figure)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a pose comparison image or video.")
    parser.add_argument("--session", type=Path, required=True, help="납품 형식 월드 좌표 CSV.")
    parser.add_argument("--expert-session", type=Path, required=True, help="숙련자 월드 좌표 CSV. 그림에는 평균이 아니라 실제 프레임을 쓴다.")
    parser.add_argument("--reference", type=Path, required=True, help="숙련자 레퍼런스 아티팩트.")
    parser.add_argument("--level", default="10")
    parser.add_argument("--out", type=Path, required=True, help="출력 PNG 또는 MP4 경로.")
    parser.add_argument("--sequence", type=int, default=None, help="특정 시퀀스만 그린다.")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    import build_pose_reference
    import pose_similarity

    payload = json.loads(args.reference.read_text(encoding="utf-8"))
    entry = payload.get(args.level)
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        logger.error("level %s has no usable reference", args.level)
        return 1
    buckets = entry["buckets"]

    expert_frames: dict[str, list[Mapping[str, Vec3]]] = {}
    for _, bucket, frames in build_pose_reference.iter_csv_pass_frames(args.expert_session):
        expert_frames.setdefault(bucket, frames)

    scored = {
        sequence: (bucket, pose)
        for sequence, bucket, pose in build_pose_reference.iter_csv_passes(args.session)
    }

    for sequence, bucket, frames in build_pose_reference.iter_csv_pass_frames(args.session):
        if args.sequence is not None and sequence != args.sequence:
            continue
        found = buckets.get(bucket)
        reference_frames = expert_frames.get(bucket)
        if not isinstance(found, dict) or reference_frames is None:
            continue
        mean = scored.get(sequence, (None, None))[1]
        if mean is None:
            continue
        similarity = pose_similarity.similarity_score(pose_similarity.joint_sigmas(mean, found))
        caption = f"seq {sequence}  {bucket}  similarity {similarity:.3f}"

        if args.out.suffix.lower() == ".mp4":
            render_animation(frames, reference_frames, args.out, ("trainee", "expert"), title=caption)
        else:
            render_comparison(
                frames[len(frames) // 2],
                reference_frames[len(reference_frames) // 2],
                args.out,
                ("trainee", "expert"),
                title=caption,
            )
        print(f"{caption} | frames {len(frames)} vs {len(reference_frames)} | {args.out}")
        return 0

    logger.error("no bucket shared by both sessions")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
