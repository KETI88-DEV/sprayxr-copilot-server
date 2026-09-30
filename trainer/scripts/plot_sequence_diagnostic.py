#!/usr/bin/env python3
"""시퀀스 **하나**의 비교가 맞는지 보는 진단 그림.

    docker exec -w /workspace -e PYTHONPATH=/workspace ship-paint-trainer \
      python3 trainer/scripts/plot_sequence_diagnostic.py --level 1 --index 12

`plot_correction_recovery.py`와 목적이 다르다. 저쪽은 **지표의 성능**(250패스 집계)이고
여기는 **이 시퀀스의 판정이 맞는가**다. 논문에서는 정성적 예시로 쓴다.

다섯 패널이 각자 다른 실패 모드를 반증한다. **한 장이다** — 교정이 별도 파일로
나가면 독자가 (c)의 관절과 (a)의 구간을 이어 붙일 수 없다.

(a) **정렬** — 착탄 경로가 프로토타입에 맞았나. 틀리면 뒤 전부가 무의미하다.
    제목에 몇 번째 스트로크인지 적는다 — 채점 단위가 버스트가 아니라 스트로크다.
(b) **결정 평면** — 각 관절이 문턱 안인가 밖인가, 그리고 **반경 때문인가 방향 때문인가**.
(c) **교정 목표** — 무엇을 어디로 옮기라는 것이고, 그러면 뼈가 얼마나 깨지나.
(d) **숙련자 분포 속 위치** — 이 점수가 좋은 값인지 혼자서는 안 읽힌다.
(e) **위상별 σ와 표본** — σ가 튀는 곳이 표본 얇은 곳인가.

`diagnose`는 **배포된 경로 그대로**(`pose_stages.score_burst`) 돈다. 그림이 파이프라인과
다른 계산을 하면 그림은 파이프라인을 증명하지 못한다.

패널 3의 축은 **관절별 문턱으로 정규화**한다. 문턱이 관절마다 다르므로(p97.5) 그렇게
해야 **하나의 단위원**이 모든 관절의 결정 경계가 된다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

import arm_reference  # noqa: E402
import correction_recovery as recovery  # noqa: E402
import pose_similarity  # noqa: E402
import pose_stages  # noqa: E402
import sequence_dtw  # noqa: E402
from plot_correction_recovery import JOINTS, LABEL, register_fonts, style  # noqa: E402

Vec3 = tuple[float, float, float]
# 결정 경계. 관절 σ = RMS(반경, 방향)이므로 σ ≥ T는 (r² + a²) ≥ 2T² — 반지름 T√2의 원이다.
# 축을 T√2로 나누면 **단위원**이 경계가 된다.
BOUNDARY: Final = 1.0


def _principal_2d(points: Sequence[Vec3]) -> list[tuple[float, float]]:
    """3D 경로를 자기 주평면에 투영한다. 축을 임의로 고르면 경로 모양이 왜곡된다."""
    n = len(points)
    mean = [sum(p[a] for p in points) / n for a in range(3)]
    centred = [[p[a] - mean[a] for a in range(3)] for p in points]
    # 공분산의 상위 두 고유벡터를 거듭제곱법으로 뽑는다 — numpy 없이.
    cov = [[sum(c[i] * c[j] for c in centred) / n for j in range(3)] for i in range(3)]

    def _power(matrix: list[list[float]], skip: list[list[float]] | None) -> list[float]:
        v = [1.0, 0.3, -0.2]
        for _ in range(200):
            w = [sum(matrix[i][j] * v[j] for j in range(3)) for i in range(3)]
            if skip is not None:
                for u in skip:
                    dot = sum(w[i] * u[i] for i in range(3))
                    w = [w[i] - dot * u[i] for i in range(3)]
            norm = math.sqrt(sum(x * x for x in w))
            if norm < 1e-12:
                return [1.0, 0.0, 0.0]
            v = [x / norm for x in w]
        return v

    first = _power(cov, None)
    second = _power(cov, [first])
    return [
        (sum(c[i] * first[i] for i in range(3)), sum(c[i] * second[i] for i in range(3)))
        for c in centred
    ]


def panel_alignment(ax, shape: Sequence[Vec3], prototype: Sequence[Vec3],
                    aligned: pose_stages.AlignResult, *, offset: int = 0,
                    stroke: int = 0, strokes: int = 1, tag: str = "a") -> None:
    """착탄 경로 ↔ 프로토타입. 대응선이 꼬이거나 한쪽에 몰리면 정렬이 틀린 것이다.

    **레퍼런스는 패스 하나에서 구워졌다.** 런타임은 버스트를 보내므로 채점은 스트로크
    단위로 쪼개 돈다 — 여기 그리는 것은 `strokes`개 중 `stroke`번째다.
    """
    combined = _principal_2d(list(shape) + list(prototype))
    query = combined[: len(shape)]
    proto = combined[len(shape):]
    palette = sns.color_palette("colorblind")

    # `phase_of`의 프레임 번호는 **버스트 전체 기준**이다(인덱스 어긋남을 고친 뒤로).
    # 이 패널은 진단 대상 스트로크의 국소 좌표를 그리므로 `offset`만큼 빼서 맞춘다.
    for frame_index, phase in aligned.phase_of:
        local = frame_index - offset
        if not 0 <= local < len(query) or phase >= len(proto):
            continue
        q = query[local]
        p = proto[phase]
        ax.plot([q[0], p[0]], [q[1], p[1]], color="#bbbbbb", linewidth=0.45, zorder=1)

    ax.plot([p[0] for p in proto], [p[1] for p in proto], "-o", color=palette[1],
            markersize=3.0, linewidth=1.3, label="expert prototype", zorder=3)
    ax.plot([p[0] for p in query], [p[1] for p in query], "-o", color=palette[0],
            markersize=2.4, linewidth=1.1, label="this sequence", zorder=2)
    ax.plot(*query[0], marker="*", markersize=8, color=palette[0], zorder=4)
    ax.plot(*proto[0], marker="*", markersize=8, color=palette[1], zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("impact path, principal plane (m)")
    ax.legend(frameon=False, loc="best")
    ax.set_title(
        f"({tag}) DTW alignment, stroke {stroke + 1}/{strokes}"
        f" — warp {aligned.warp:.3f}, cost {aligned.cost:.3f} m", pad=6
    )


def panel_phase_sigma(ax, ax_bar, phase_sigmas: list[list[float | None]],
                      phase_n: list[int], *, tag: str = "e") -> None:
    """위상별 관절 σ. 아래 막대가 그 위상의 표본 수다 — **σ가 튀는 곳이 얇은 곳인지** 본다."""
    # 리스트로 넘기면 seaborn이 pandas로 감싸면서 mask의 축을 잘못 잡는다(실제로 깨졌다).
    data = np.array([[0.0 if v is None else v for v in row] for row in phase_sigmas], dtype=float)
    mask = np.array([[v is None for v in row] for row in phase_sigmas], dtype=bool)
    sns.heatmap(data, ax=ax, cmap=sns.color_palette("rocket_r", as_cmap=True),
                vmin=0, vmax=max(3.0, float(data.max()) if data.size else 3.0),
                mask=mask, cbar_kws={"label": "σ", "pad": 0.02, "shrink": 0.85},
                linewidths=0.4, linecolor="white",
                yticklabels=[LABEL[j] for j in JOINTS],
                xticklabels=[str(i + 1) for i in range(len(phase_n))])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(length=0)
    ax.set_title(f"({tag}) per-phase joint σ", pad=6)
    ax.set_xticks([])

    ax_bar.bar(range(len(phase_n)), phase_n, color="#9fb8d0", width=0.82)
    ax_bar.set_xlim(-0.5, len(phase_n) - 0.5)
    ax_bar.set_xticks(range(len(phase_n)))
    ax_bar.set_xticklabels([str(i + 1) for i in range(len(phase_n))])
    ax_bar.set_xlabel("stroke phase")
    ax_bar.set_ylabel("n")
    ax_bar.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    sns.despine(ax=ax_bar)


def panel_decision(ax, sigmas: Sequence[arm_reference.JointSigma],
                   ladders: Mapping[str, Mapping[str, float]] | None,
                   *, tag: str = "b") -> None:
    """결정 평면. **관절별 문턱으로 정규화**해서 단위원 하나가 모든 경계가 되게 한다."""
    palette = sns.color_palette("colorblind", len(JOINTS))
    markers = ["o", "s", "^", "v", "D", "P"]
    circle = plt.Circle((0, 0), BOUNDARY, fill=False, color="#b5651d",
                        linestyle="--", linewidth=1.1)
    ax.add_patch(circle)
    ax.text(BOUNDARY * 0.72, BOUNDARY * 0.72, "cue threshold", fontsize=6.5,
            color="#b5651d", rotation=-45, ha="center", va="center")

    limit = BOUNDARY * 1.15
    for item, colour, marker in zip(sigmas, palette, markers):
        if item.radial_sigma is None or item.angular_sigma is None:
            continue
        scale = arm_reference.sigma_threshold(item.joint, ladders) * math.sqrt(2.0)
        x = item.radial_sigma / scale
        y = item.angular_sigma / scale
        limit = max(limit, math.hypot(x, y) * 1.12)
        ax.plot(x, y, marker=marker, markersize=5.5, color=colour,
                markeredgecolor="white", markeredgewidth=0.5, label=LABEL[item.joint])
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("reach deviation  (÷ threshold)")
    ax.set_ylabel("direction deviation  (÷ threshold)")
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.2, fontsize=6.4)
    ax.set_title(f"({tag}) decision plane — outside the arc fires a cue", pad=6)
    sns.despine(ax=ax)


def panel_correction(ax, correction: arm_reference.Correction | None,
                     plan: Sequence[Mapping[str, Any]], *, tag: str = "c") -> None:
    """교정 목표. **깨지는 뼈를 점선으로** 그린다 — 대입 가능한 좌표가 아님을 선 모양으로."""
    if correction is None or not plan:
        ax.text(0.5, 0.5, "no cue issued\n(all joints within expert range)",
                ha="center", va="center", fontsize=8, color="#555555",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({tag}) correction target", pad=6)
        sns.despine(ax=ax, left=True, bottom=True)
        return

    step = plan[len(plan) // 2]
    points = [tuple(step["current"]), tuple(step["target"])]
    if "parent" in step:
        points.append(tuple(step["parent"]))
    flat = _principal_2d(points)
    current, target = flat[0], flat[1]
    palette = sns.color_palette("colorblind")

    if len(flat) > 2:
        parent = flat[2]
        ax.plot([parent[0], current[0]], [parent[1], current[1]], "-",
                color="#555555", linewidth=1.6, label="current bone")
        ax.plot([parent[0], target[0]], [parent[1], target[1]], "--",
                color="#b5651d", linewidth=1.4, label="bone if applied literally")
        ax.plot(*parent, marker="o", markersize=5, color="#555555")
    ax.plot(*current, marker="o", markersize=6.5, color=palette[0], label="current")
    ax.plot(*target, marker="X", markersize=7.5, color=palette[3], label="target")
    ax.annotate("", xy=target, xytext=current,
                arrowprops={"arrowstyle": "->", "color": palette[3], "linewidth": 1.1})
    ax.set_aspect("equal")
    # `best`는 현재 뼈 위에 앉는다 — 데이터가 가운데 띠를 채워서 자리가 없다.
    # 범례를 축 아래로 내보내 선을 가리지 않게 한다.
    # 종횡비가 같아 축 상자가 납작하다 — 범례가 안에 들어갈 세로 자리가 없고,
    # 살짝만 내리면 x축 이름과 겹친다(실제로 겹쳤다). 이름 아래로 완전히 내린다.
    ax.set_xlabel("world, principal plane (m)")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.58), handlelength=1.4, columnspacing=1.1)
    broken = "" if not correction.target else ""
    ax.set_title(f"({tag}) correction: {LABEL.get(correction.joint, correction.joint)}"
                 f" — {correction.dominant}{broken}", pad=6)
    sns.despine(ax=ax)


def panel_ladder(ax, score: float | None, ladder: Mapping[str, Any] | None,
                 mode: str, *, tag: str = "d") -> None:
    """이 시퀀스의 점수가 **숙련자 분포 어디인가**.

    점수의 절댓값은 축 수·산포 하한·가중치가 스케일을 건드려 혼자서는 읽히지 않는다.
    0.53이 나쁜 값인지 보통인지는 이 사다리에 올려 봐야 안다.

    사다리는 **패스 하나**에서 구워졌다(`reference_pass_count`). 버스트를 스트로크
    RMS로 합친 점수는 이 분포 위에 있지 않으므로 그 경우 올리지 않는다.
    """
    points = [("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90)]
    values = [(name, rank, float(ladder[name])) for name, rank in points
              if isinstance(ladder, Mapping) and isinstance(ladder.get(name), (int, float))]
    if not values:
        ax.text(0.5, 0.5, "no expert score ladder in this bucket",
                ha="center", va="center", fontsize=8, color="#555555",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({tag}) score vs expert range", pad=6)
        sns.despine(ax=ax, left=True, bottom=True)
        return

    palette = sns.color_palette("colorblind")
    xs = [value for _, _, value in values]
    ys = [rank for _, rank, _ in values]
    ax.plot(xs, ys, "-o", color="#8a8a8a", markersize=3.6, linewidth=1.1,
            label="expert passes")
    ax.fill_betweenx([ys[1], ys[3]], xs[1], xs[3], color="#8a8a8a", alpha=0.12)
    for (name, rank, value) in values:
        ax.annotate(name, (value, rank), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6.0, color="#6a6a6a")

    if score is not None and mode != "burst":
        # 점수가 p10 아래/p90 위로 나가는 일이 흔하다(여기 예시가 0.476, p10이 0.554).
        # 축을 사다리에만 맞추면 표시선이 축과 겹쳐 눈금처럼 보인다 — 실제로 그랬다.
        low, high = min(xs + [score]), max(xs + [score])
        pad = max((high - low) * 0.08, 0.01)
        ax.set_xlim(low - pad, high + pad)
        ax.axvline(score, color=palette[3], linewidth=1.3, zorder=1)
        ax.plot([score], [50], marker="v", markersize=7.5, color=palette[3],
                clip_on=False, label="this sequence", zorder=4)
        ax.legend(frameon=False, loc="lower right", fontsize=6.4)
    elif mode == "burst":
        # 합친 점수는 패스 사다리 위에 있지 않다 — 올리지 않는 이유를 그림에 적는다.
        ax.text(0.5, 0.06, "burst score not placed —\nladder is per pass",
                ha="center", va="bottom", fontsize=6.4, color=palette[3],
                transform=ax.transAxes)

    count = (ladder or {}).get("reference_pass_count")
    in_sample = bool((ladder or {}).get("in_sample"))
    note = f"n={int(count):,}" if isinstance(count, (int, float)) else "n=?"
    if in_sample:
        note += ", in-sample"
    ax.set_ylim(0, 100)
    ax.set_xlabel("similarity score")
    ax.set_ylabel("expert percentile")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_title(f"({tag}) score vs expert range ({note})", pad=6)
    sns.despine(ax=ax)


def diagnose(frames, impacts, tools, buckets) -> dict[str, Any] | None:
    """**배포된 경로 그대로** 진단한다 — `score_burst`가 쪼개고 합친다.

    그림이 파이프라인과 다른 계산을 하면 그림은 파이프라인을 증명하지 못한다.
    예전에는 여기서 `align`을 한 번만 불렀는데, 그때는 채점도 그랬다.
    """
    burst = pose_stages.score_burst(
        frames, impacts, list(range(len(impacts))), tools, buckets,
        labels=pose_similarity.JOINT_LABELS,
    )
    scored = burst.result
    resolved = burst.strokes[0].resolve if burst.strokes else pose_stages.resolve(
        pose_stages.reach_of(impacts, tools), buckets
    )
    if not resolved.ok:
        return {"error": resolved.reason, "resolve": resolved}
    if not scored.ok:
        return {"error": scored.reason, "resolve": resolved, "burst": burst}

    # 교정이 나온 스트로크를 그린다 — 그래야 패널 (a)·(c)가 같은 구간을 말한다.
    chosen = burst.strokes[burst.worst_stroke or 0] if burst.strokes else None
    if chosen is None or not chosen.align.ok:
        return {"error": "alignment_failed", "resolve": resolved, "burst": burst}
    aligned = chosen.align
    bucket = buckets[chosen.resolve.band]

    phase_poses, reference = pose_stages._phase_view(frames, aligned, bucket["phases"])
    grid: list[list[float | None]] = []
    for joint in JOINTS:
        row: list[float | None] = []
        for pose, entry in zip(phase_poses, reference):
            value = None
            if pose is not None and isinstance(entry, Mapping):
                if int(entry.get("n", 0) or 0) >= arm_reference.MIN_PHASE_SAMPLES:
                    found = [item for item in arm_reference.joint_sigmas(pose, entry)
                             if item.joint == joint]
                    value = found[0].sigma if found else None
            row.append(value)
        grid.append(row)
    phase_n = [int(entry.get("n", 0) or 0) if isinstance(entry, Mapping) else 0
               for entry in reference]

    local = list(chosen.impacts) if hasattr(chosen, "impacts") else [
        impacts[i] for i in chosen.frames
    ]
    return {
        "resolve": chosen.resolve, "align": aligned, "score": scored, "bucket": bucket,
        "burst": burst, "grid": grid, "phase_n": phase_n,
        "offset": chosen.frames[0] if chosen.frames else 0,
        "stroke": chosen.index, "strokes": len(burst.strokes),
        "shape": sequence_dtw.path_shape(local),
        "prototype": [tuple(point) for point in bucket["prototype"]],
    }


def render(diag: Mapping[str, Any], out: Path, header: str) -> None:
    """다섯 패널이 각자 다른 실패 모드를 반증한다. **한 장에 넣는다** — 교정이
    따로 나가면 독자가 (c)의 관절과 (a)의 구간을 이어 붙일 수 없다."""
    fig = plt.figure(figsize=(7.2, 7.6))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.86],
                            hspace=0.46, wspace=0.30)
    ax_align = fig.add_subplot(grid[0, 0])
    ax_decision = fig.add_subplot(grid[0, 1])
    ax_correction = fig.add_subplot(grid[1, 0])
    ax_ladder = fig.add_subplot(grid[1, 1])
    # 히트맵과 표본 막대는 x축을 공유해 읽는다 — 사이를 벌리면 몇 번째 위상인지 못 잇는다.
    bottom = grid[2, :].subgridspec(2, 1, height_ratios=[0.80, 0.20], hspace=0.06)
    ax_sigma = fig.add_subplot(bottom[0])
    ax_bar = fig.add_subplot(bottom[1])

    scored = diag["score"]
    ladders = diag["bucket"].get("joint_sigma_ladder")

    panel_alignment(ax_align, diag["shape"], diag["prototype"], diag["align"],
                    offset=diag["offset"], stroke=diag["stroke"],
                    strokes=diag["strokes"], tag="a")
    panel_decision(ax_decision, scored.sigmas,
                   ladders if isinstance(ladders, Mapping) else None, tag="b")
    panel_correction(ax_correction,
                     scored.corrections[0] if scored.corrections else None,
                     scored.plans[0] if scored.plans else (), tag="c")
    panel_ladder(ax_ladder, scored.score,
                 diag["bucket"].get("expert_scores_phase"), scored.mode, tag="d")
    panel_phase_sigma(ax_sigma, ax_bar, diag["grid"], diag["phase_n"], tag="e")

    fig.suptitle(header, y=0.992, fontsize=8.5)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=recovery.DEFAULT_ROOT)
    parser.add_argument("--reference", type=Path, default=recovery.DEFAULT_REFERENCE)
    parser.add_argument("--level", default="1")
    parser.add_argument("--index", type=int, default=0, help="조건을 만족하는 패스 중 몇 번째")
    parser.add_argument("--out-dir", type=Path, default=Path("/output/figures"))
    parser.add_argument("--name", default="fig_sequence_diagnostic")
    args = parser.parse_args()

    buckets = json.loads(args.reference.read_text(encoding="utf-8"))["buckets"]
    csv_source, service = recovery._load_pose_lab()
    source = csv_source.CsvPassSource(args.root)

    seen = 0
    chosen = None
    for item in [s for s in source.sessions() if s.level == args.level]:
        for entry in source.index_of(item)[::3]:
            raw = source.load(service.PassRef(
                worker=item.worker, date=item.date, level=item.level,
                session=item.session, pass_index=entry.pass_index))
            prepared = recovery._pass_inputs(raw.rows)
            if prepared is None:
                continue
            diag = diagnose(*prepared, buckets)
            if diag is None or "error" in diag:
                continue
            if seen == args.index:
                chosen = (item, entry.pass_index, diag)
                break
            seen += 1
        if chosen:
            break
    if chosen is None:
        raise SystemExit(f"index {args.index}에 해당하는 패스를 찾지 못했다 (본 것 {seen}개)")

    item, pass_index, diag = chosen
    scored, resolved = diag["score"], diag["resolve"]
    burst = diag["burst"]
    percentile = "—" if scored.percentile is None else str(scored.percentile)
    header = (f"level {item.level} · session {item.session} · pass {pass_index}  |  "
              f"band {resolved.band} (n={resolved.support:,}, {resolved.quality})  |  "
              f"strokes {burst.shape.strokes}, straightness {burst.shape.straightness:.2f}"
              f"  |  {scored.mode} score {scored.score:.3f}, percentile {percentile}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = register_fonts()
    with plt.rc_context(style(font)):
        render(diag, args.out_dir / args.name, header)
    print(f"font: {font}")
    print(header)
    for path in sorted(args.out_dir.glob(f"{args.name}*")):
        print(f"  {path.name:40s} {path.stat().st_size:>9,} B")


if __name__ == "__main__":
    main()
