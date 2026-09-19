#!/usr/bin/env python3
"""패스 한 개의 스켈레톤을 여러 프레임 **격자**로 뽑는 감사 도구(D5, 사용자 제안).

화면(`/api/compare`)은 패스마다 대표 프레임 **하나**만 그린다. 그래서 어느 구간에서 스켈레톤이
무너지는지 — 다리가 꺾이는지, 앵커가 벌어지는지, 팔꿈치가 상수로 굳는지 — 가 안 보인다. 여기서는
한 패스의 분사 프레임을 균등하게 뽑아 한 장의 SVG에 나란히 늘어놓고, 칸마다 시각과 핵심 지표를
작게 적는다.

**표시 전용이다.** 채점·교정은 이 파일을 부르지 않는다. 화면과 같은 배치 경로
(`pose_lab.skeleton.place_oriented`, 같은 장착 상수)를 쓰므로, 여기서 이상한 것이 보이면
화면에서도 같은 것이 이상하다.

의존성은 저장소가 이미 쓰는 것뿐이다 — 표준 라이브러리로 SVG를 직접 쓰고, 기하는 이미 numpy를
쓰는 `pose_lab` 모듈을 그대로 부른다. 새 그림 라이브러리를 들이지 않는다.

결정적이다. 프레임 선택은 균등 추출(`round(i * step)`)이고, 좌표·수치는 모두 고정 자리수로 적는다 —
같은 입력이면 바이트가 같다(`test_pose_contact_sheet.py`가 고정한다).

투영은 **직교**다. 화면 가로축은 머리 실측 마커→착탄점의 수평 방향, 세로축은 월드 수직(`world_axes.UP`)이다.
칸마다 카메라를 새로 맞추면 프레임끼리 못 비교하므로 **패스 전체에 하나의 축척**을 쓴다.

한 줄 실행:

    docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/pose_contact_sheet.py \\
      --worker 김경호 --level 3 --session 1000391_1 --pass-index 0 --frames 12 \\
      --out /tmp/contact-kkh-L3.svg
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pose_fk_csv  # noqa: E402
import simulator_csv  # noqa: E402
import world_axes  # noqa: E402

from trainer.pose_lab import body_proportions, composite, joint_angles, service, skeleton  # noqa: E402

DEFAULT_ROOT: Final = Path("/data/sources/professionals")
DEFAULT_FRAMES: Final = 12
DEFAULT_COLUMNS: Final = 4
CELL_W: Final = 300
CELL_H: Final = 330
PAD: Final = 14
HEADER_H: Final = 56
# 그림 영역이 칸에서 차지하는 높이. 나머지는 지표 글자다.
PLOT_H: Final = 210

BG: Final = "#0b0f16"
INK: Final = "#e8edf5"
DIM: Final = "#93a1b5"
BONE: Final = "#c9d4e4"
PLACEHOLDER: Final = "#8a97a8"
MARKER: Final = "#4f9fe8"
BAD: Final = "#e2685f"
OK: Final = "#6fcf8b"


@dataclass(frozen=True, slots=True)
class Cell:
    """한 프레임 칸. `joints`가 None이면 배치에 실패한 프레임이고 사유를 적는다."""

    index: int
    time_sec: float
    joints: dict[str, tuple[float, float]] | None
    placeholder: dict[str, tuple[float, float]] | None
    markers: list[tuple[str, tuple[float, float], tuple[float, float], float]]
    metrics: list[tuple[str, str, bool]]  # (이름, 값, 눈에 띄게 할까)
    reason: str | None


def _basis(origin: Sequence[float], hit: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """직교 투영 축 (가로, 세로). 가로는 `origin`→착탄점의 수평 방향, 세로는 월드 수직."""
    up = np.asarray(world_axes.UP, dtype=float)
    direction = np.asarray(hit, dtype=float) - np.asarray(origin, dtype=float)
    direction = direction - float(np.dot(direction, up)) * up
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        first, _second = world_axes.HORIZONTAL_INDICES
        direction = np.zeros(3)
        direction[first] = 1.0
        norm = 1.0
    return direction / norm, up


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def pick_frames(count: int, wanted: int) -> list[int]:
    """균등 추출한 프레임 번호. 결정적이다."""
    if count <= 0 or wanted <= 0:
        return []
    if count <= wanted:
        return list(range(count))
    step = (count - 1) / (wanted - 1) if wanted > 1 else 0.0
    return sorted({round(i * step) for i in range(wanted)})


def build_cells(
    raw: service.RawPass,
    frames: int = DEFAULT_FRAMES,
    mounts_table: service.MountsTable | None = None,
    floor: float | None = None,
) -> tuple[list[Cell], dict[str, str]]:
    """프레임마다 투영된 관절·마커·지표. 두 번째 값은 시트 머리글에 적을 패스 요약이다."""
    representative = service.representative_frame(raw)
    if representative is None or not raw.rows:
        return [], {"error": "분사 중 착탄 프레임이 없어 컨택트시트를 만들 수 없습니다."}
    # 서버(`service.skeleton_block`)와 **같은 리그**로 그린다 — 몸통 두 마디를 BVH 비례로 다시 잰 값.
    offsets = body_proportions.rig_offsets(pose_fk_csv.bind_offsets(raw.rows[0]))
    if not offsets:
        return [], {"error": "이 패스에는 관절 컬럼이 없어 컨택트시트를 만들 수 없습니다."}

    table = mounts_table if mounts_table is not None else service.load_mounts()
    mounts = skeleton.mounts_for(table.table, representative.gun)
    kind = "global"
    if not (table.passed and mounts.right is not None and mounts.head is not None):
        samples = service.pass_fit_samples(representative, offsets)
        if not samples:
            return [], {"error": "검증된 장착 상수도, 이 패스로 맞출 프레임도 없어 만들 수 없습니다."}
        mounts = skeleton.mounts_for(skeleton.fit_mounts([samples]), representative.gun)
        kind = "pass_fit"

    spraying = representative.spraying
    picks = pick_frames(len(spraying), frames)
    axis_x, axis_y = _basis(
        np.zeros(3) if representative.frame.head is None else representative.frame.head,
        representative.frame.hit,
    )

    placed_by_pick: list[tuple[int, object, skeleton.Placement | None]] = []
    for pick in picks:
        frame, row = spraying[pick]
        sample = skeleton.frame_sample(frame, row, representative.gun, offsets)
        placed = None if sample is None else skeleton.place_oriented(
            sample.local_pos, sample.local_rot, sample.measured, mounts
        )
        placed_by_pick.append((pick, sample, placed))

    def flat(point: Sequence[float]) -> tuple[float, float]:
        p = np.asarray(point, dtype=float)
        return (float(np.dot(p, axis_x)), float(np.dot(p, axis_y)))

    cells: list[Cell] = []
    for pick, sample, placed in placed_by_pick:
        frame, _row = spraying[pick]
        if placed is None or sample is None:
            cells.append(
                Cell(
                    index=int(frame.index),
                    time_sec=float(frame.time_sec),
                    joints=None,
                    placeholder=None,
                    markers=[],
                    metrics=[],
                    reason="이 프레임에서는 실측 강체가 2개 미만이라 세우지 않았습니다.",
                )
            )
            continue
        legs = skeleton.leg_check(sample.local_pos)
        angles = {
            spec.key: joint_angles._angle_at(placed.joints, spec)  # noqa: SLF001
            for spec in joint_angles.ANGLE_SPECS
        }
        # 진단용이라 **리그 제 길이로 곧게** 내린다(`reach_floor` 없음). 화면은 엉덩이가 리그
        # 다리보다 낮을 때만 무릎을 접어 세우고(늘리는 경로는 없다), 접촉 시트는 그 굽힘조차 없이
        # 발이 고른 면보다 얼마나 아래인지를 보는 자리다 — 두 경로 다 뼈 길이는 리그 그대로다.
        placeholder = skeleton.standing_legs(
            placed.joints,
            offsets,
            None if frame.hit is None else [frame.hit[i] - placed.joints["Pelvis"][i] for i in range(3)],
            floor=floor,
        )
        metrics = [
            ("앵커 RMS", f"{placed.residual_m * 100.0:.1f}cm", placed.residual_m * 100.0 > 10.0),
            (
                "앵커별",
                " ".join(f"{fit.point[0]}{fit.miss_m * 100.0:.0f}" for fit in placed.anchors),
                False,
            ),
            ("수직 보정", f"{placed.gravity_correction_deg:.0f}°", placed.gravity_correction_deg > 30.0),
            ("다리 기울기", f"{_fmt(placed.leg_tilt_deg, 0)}°", (placed.leg_tilt_deg or 0.0) > 25.0),
            (
                "IK 다리",
                "범위 밖" if legs is not None and not legs.plausible else ("범위 안" if legs else "—"),
                legs is not None and not legs.plausible,
            ),
            ("오른 팔꿈치", f"{_fmt(angles['right_elbow'], 0)}°", False),
            ("왼 팔꿈치", f"{_fmt(angles['left_elbow'], 0)}°", False),
            (
                "자리표시 다리",
                "리그 길이"
                if placeholder is None or placeholder.feet_below_floor_m is None
                else f"리그 길이 · 바닥면 아래 {placeholder.feet_below_floor_m:+.2f}m",
                False,
            ),
        ]
        cells.append(
            Cell(
                index=int(frame.index),
                time_sec=float(frame.time_sec),
                joints={name: flat(p) for name, p in placed.joints.items()},
                placeholder=None if placeholder is None else {n: flat(p) for n, p in placeholder.joints.items()},
                markers=[
                    (fit.point, flat(fit.predicted_marker), flat(fit.measured_marker), fit.miss_m)
                    for fit in placed.anchors
                ],
                metrics=metrics,
                reason=None,
            )
        )

    summary = {
        "floor": "—" if floor is None else f"{floor:.2f}m",
        "mounts": "작업자 교차 검증 장착 상수" if kind == "global" else "이 패스로 맞춘 장착 상수",
        "gun": representative.gun,
        "spraying_frames": str(len(spraying)),
        "shown": str(len(cells)),
    }
    return cells, summary


def _bounds(cells: Sequence[Cell]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for cell in cells:
        for source in (cell.joints, cell.placeholder):
            for x, y in (source or {}).values():
                xs.append(x)
                ys.append(y)
        for _point, predicted, measured, _miss in cell.markers:
            xs += [predicted[0], measured[0]]
            ys += [predicted[1], measured[1]]
    if not xs:
        return (0.0, 1.0, 0.0, 1.0)
    return (min(xs), max(xs), min(ys), max(ys))


def render_svg(cells: Sequence[Cell], summary: Mapping[str, str], title: str, columns: int = DEFAULT_COLUMNS) -> str:
    """칸마다 같은 축척으로 그린 SVG. 축척이 칸마다 다르면 프레임끼리 비교할 수 없다."""
    if not cells:
        message = summary.get("error", "그릴 프레임이 없습니다.")
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="80" viewBox="0 0 640 80">'
            f'<rect width="640" height="80" fill="{BG}"/>'
            f'<text x="16" y="46" fill="{INK}" font-family="sans-serif" font-size="15">'
            f"{html.escape(message)}</text></svg>"
        )
    rows = math.ceil(len(cells) / columns)
    width = PAD + columns * (CELL_W + PAD)
    height = HEADER_H + PAD + rows * (CELL_H + PAD)
    x0, x1, y0, y1 = _bounds(cells)
    span = max(x1 - x0, y1 - y0, 1e-6)
    inner = min(CELL_W - 2 * PAD, PLOT_H - 2 * PAD)
    scale = inner / span
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    def place(point: tuple[float, float], ox: float, oy: float) -> tuple[float, float]:
        # SVG는 아래가 +y다. 월드 수직을 위로 두려면 부호를 뒤집는다.
        return (
            ox + CELL_W / 2.0 + (point[0] - cx) * scale,
            oy + PLOT_H / 2.0 - (point[1] - cy) * scale,
        )

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        f'<text x="{PAD}" y="24" fill="{INK}" font-size="16">{html.escape(title)}</text>',
        f'<text x="{PAD}" y="44" fill="{DIM}" font-size="12">'
        + html.escape(
            f"건 {summary.get('gun', '—')} · 분사 프레임 {summary.get('spraying_frames', '—')}개 중 "
            f"{summary.get('shown', '—')}개 · {summary.get('mounts', '—')} · 벽 메시 바닥면 "
            f"{summary.get('floor', '—')} · 직교 투영(가로=머리→착탄 수평, 세로=월드 수직) · "
            "모든 칸 같은 축척 · 표시 전용(채점 아님)"
        )
        + "</text>",
    ]

    for i, cell in enumerate(cells):
        ox = PAD + (i % columns) * (CELL_W + PAD)
        oy = HEADER_H + PAD + (i // columns) * (CELL_H + PAD)
        out.append(
            f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{CELL_W}" height="{CELL_H}" rx="6" '
            f'fill="#10161f" stroke="#1e2733"/>'
        )
        out.append(
            f'<text x="{ox + 10:.1f}" y="{oy + 18:.1f}" fill="{INK}" font-size="12">'
            + html.escape(f"#{cell.index} · {cell.time_sec:.2f}s")
            + "</text>"
        )
        if cell.joints is None:
            out.append(
                f'<text x="{ox + 10:.1f}" y="{oy + 44:.1f}" fill="{BAD}" font-size="11">'
                + html.escape(cell.reason or "세우지 못했습니다.")
                + "</text>"
            )
            continue
        for a, b in simulator_csv.BONES:
            if a not in cell.joints or b not in cell.joints:
                continue
            if a in skeleton.LEG_JOINTS or b in skeleton.LEG_JOINTS:
                continue  # IK 다리는 기본 화면과 같게 빼고, 자리표시 다리를 끊어 그린다.
            ax, ay = place(cell.joints[a], ox, oy)
            bx, by = place(cell.joints[b], ox, oy)
            out.append(
                f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                f'stroke="{BONE}" stroke-width="2.4" stroke-linecap="round"/>'
            )
        for a, b in simulator_csv.BONES:
            source = cell.placeholder or {}
            if a not in source or b not in source:
                continue
            ax, ay = place(source[a], ox, oy)
            bx, by = place(source[b], ox, oy)
            out.append(
                f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                f'stroke="{PLACEHOLDER}" stroke-width="1.6" stroke-dasharray="3 3"/>'
            )
        for _point, predicted, measured, miss in cell.markers:
            px, py = place(predicted, ox, oy)
            mx, my = place(measured, ox, oy)
            out.append(
                f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{mx:.1f}" y2="{my:.1f}" '
                f'stroke="{BAD if miss > 0.1 else OK}" stroke-width="1" stroke-dasharray="2 2"/>'
            )
            out.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="3.4" fill="{MARKER}"/>')
        for j, (name, value, loud) in enumerate(cell.metrics):
            ty = oy + PLOT_H + 16 + j * 14
            out.append(
                f'<text x="{ox + 10:.1f}" y="{ty:.1f}" fill="{DIM}" font-size="10.5">'
                + html.escape(name)
                + "</text>"
            )
            out.append(
                f'<text x="{ox + CELL_W - 10:.1f}" y="{ty:.1f}" text-anchor="end" '
                f'fill="{BAD if loud else INK}" font-size="10.5" font-family="monospace">'
                + html.escape(value)
                + "</text>"
            )
    out.append("</svg>")
    return "\n".join(out)


def sheet_for(
    root: Path, ref: service.PassRef, frames: int = DEFAULT_FRAMES, columns: int = DEFAULT_COLUMNS
) -> str:
    from trainer.pose_lab import csv_source

    raw = csv_source.CsvPassSource(root).load(ref)
    mesh = service.placed_mesh(ref.level)
    floor = None
    if mesh is not None and mesh.mesh is not None:
        floor = composite.floor_vertical(
            composite.MeshGeometry.from_mesh(mesh.mesh.vertices, mesh.mesh.triangles)
        )
    cells, summary = build_cells(raw, frames=frames, floor=floor)
    title = f"{ref.worker} · L{ref.level} · {ref.session} #{ref.pass_index} — 스켈레톤 컨택트시트"
    return render_svg(cells, summary, title, columns=columns)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="패스 하나의 스켈레톤을 프레임 격자 SVG로 뽑는다(감사 도구).")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--date", default="")
    parser.add_argument("--pass-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from trainer.pose_lab import csv_source

    date = args.date
    if not date:
        source = csv_source.CsvPassSource(args.root)
        found, _total = source.passes(args.worker, level=args.level, limit=10_000)
        matches = [item for item, _entry in found if item.session == args.session]
        if not matches:
            parser.error(f"패스를 찾지 못했다: {args.worker} L{args.level} {args.session} #{args.pass_index}")
        date = matches[0].date

    ref = service.PassRef(
        worker=args.worker,
        date=date,
        level=args.level,
        session=args.session,
        pass_index=args.pass_index,
    )
    svg = sheet_for(args.root, ref, frames=args.frames, columns=args.columns)
    args.out.write_text(svg, encoding="utf-8")
    # 한글이 섞여 있어 문자 수와 바이트 수가 다르다 — 파일에 실제로 쓰인 바이트를 찍는다.
    print(written_line(args.out, svg))
    return 0


def written_line(out: Path, svg: str) -> str:
    """쓴 파일의 경로와 **바이트** 수. `len(svg)`는 문자 수라서 UTF-8 바이트와 다르다."""
    return f"{out} ({len(svg.encode('utf-8'))} bytes)"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
