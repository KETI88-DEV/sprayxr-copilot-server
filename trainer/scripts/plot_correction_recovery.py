#!/usr/bin/env python3
"""교란 회복 실험 결과를 논문 그림으로 굽는다.

    docker exec -w /workspace -e PYTHONPATH=/workspace ship-paint-trainer \
      python3 trainer/scripts/plot_correction_recovery.py \
        --result /output/correction_recovery_v2.json --out-dir /output/figures

PDF(벡터)와 PNG(미리보기)를 같이 낸다. 라벨은 **영문**이다 — 한글 폰트가 없는 환경에서
사각형으로 깨지면 그림이 쓸모없어진다.

그림은 넷이고 각자 하나의 주장을 지지한다.

1. `confusion` — 지목이 정확한가. 반경(설계 성분)과 접선(대조)을 나란히.
2. `sensitivity` — 얼마나 벗어나야 잡히는가. 관절별 곡선.
3. `reference_geometry` — 왜 손이 둔감한가. 관절별 반경 SD와 방향 산포.
4. `worker_false_positive` — 같은 문턱이 작업자마다 다르게 걸리는가.

`__miss__`(교정 미발부)를 혼동행렬의 마지막 열로 **반드시** 넣는다. 빼면 행의 합이 1이
아니게 되고, "지목을 안 한 것"이 "틀리게 지목한 것"으로 읽힌다 — 이 실험의 핵심 결과가
바로 그 구분이다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import arm_reference  # noqa: E402

JOINTS: Final[tuple[str, ...]] = arm_reference.SCORING_JOINTS_ARM
# 논문 라벨. 코드 이름을 그대로 쓰면 축이 너무 길어진다.
LABEL: Final[dict[str, str]] = {
    "RightUpperArm": "R-Upper", "RightForeArm": "R-Fore", "RightHand": "R-Hand",
    "LeftUpperArm": "L-Upper", "LeftForeArm": "L-Fore", "LeftHand": "L-Hand",
}
MISS: Final = "no cue"

# CVPR 본문이 Times라 그림도 serif로 맞춘다. 없으면 기본 sans로 떨어진다(경고만).
RC: Final[dict[str, Any]] = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def _matrix(result: Mapping[str, Any], direction: str) -> tuple[list[list[float]], list[int]]:
    """행 정규화한 (6 × 7) 행렬과 행별 시행 수. 마지막 열이 `no cue`다."""
    rows: list[list[float]] = []
    totals: list[int] = []
    for row in JOINTS:
        counts = {name: 0 for name in JOINTS}
        miss = 0
        for metres in result["magnitudes"]:
            slot = result["tally"].get(f"{direction}|{metres}|{row}", {})
            for named, n in slot.items():
                if named == "__miss__":
                    miss += n
                elif named in counts:
                    counts[named] += n
        total = sum(counts.values()) + miss
        totals.append(total)
        rows.append([counts[c] / total for c in JOINTS] + [miss / total] if total else [0.0] * 7)
    return rows, totals


def plot_confusion(result: Mapping[str, Any], out: Path) -> None:
    columns = [LABEL[j] for j in JOINTS] + [MISS]
    blues = plt.get_cmap("Blues")
    # `no cue` 열은 다른 계열로 둔다 — 오분류와 미발부는 다른 사건이다.
    greys = LinearSegmentedColormap.from_list("cue", ["#ffffff", "#9a9a9a"])

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), width_ratios=[1, 1])
    for ax, direction, tag in zip(axes, ("radial", "tangential"), ("a", "b")):
        rows, totals = _matrix(result, direction)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                cmap = greys if c == len(JOINTS) else blues
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=cmap(min(value, 1.0)),
                                           edgecolor="white", linewidth=0.7))
                if value >= 0.005:
                    ax.text(c + 0.5, r + 0.5, f"{value:.2f}".lstrip("0") or "0",
                            ha="center", va="center",
                            color="white" if value > 0.55 else "#222222",
                            fontsize=6.4,
                            fontweight="bold" if c == r else "normal")
        ax.set_ylim(len(JOINTS), 0)
        ax.set_xticks([i + 0.5 for i in range(len(columns))])
        ax.set_xticklabels(columns, rotation=45, ha="right")
        ax.set_yticks([i + 0.5 for i in range(len(JOINTS))])
        ax.set_yticklabels([LABEL[j] for j in JOINTS])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        # 미발부 열을 시각적으로 뗀다.
        ax.axvline(len(JOINTS), color="#333333", linewidth=0.8)
        ax.set_xlabel("cued joint")
        if tag == "a":
            ax.set_ylabel("perturbed joint")
        ax.set_title(f"({tag}) {direction}", pad=4)
        # 비율만 두면 근거의 두께가 안 보인다. 행별 시행 수를 오른쪽에 적는다.
        for r, total in enumerate(totals):
            ax.text(len(columns) + 0.25, r + 0.5, f"{total:,}", ha="left", va="center",
                    fontsize=6.0, color="#666666")
        ax.text(len(columns) + 0.25, -0.35, "n", ha="left", va="center",
                fontsize=6.0, color="#666666")
        ax.set_xlim(0, len(columns) + 1.6)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def plot_sensitivity(result: Mapping[str, Any], out: Path) -> None:
    """관절별 hit 곡선. '정밀하지만 둔감하다'를 이 그림이 말한다."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True)
    markers = ["o", "s", "^", "v", "D", "P"]
    for ax, direction, tag in zip(axes, ("radial", "tangential"), ("a", "b")):
        for joint, marker in zip(JOINTS, markers):
            xs, ys = [], []
            for metres in result["magnitudes"]:
                slot = result["tally"].get(f"{direction}|{metres}|{joint}", {})
                total = sum(slot.values())
                if not total:
                    continue
                xs.append(metres * 100)
                ys.append(slot.get(joint, 0) / total)
            ax.plot(xs, ys, marker=marker, markersize=3.2, linewidth=1.0, label=LABEL[joint])
        ax.set_xlabel("displacement (cm)")
        ax.set_ylim(0, 1.0)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.set_title(f"({tag}) {direction}", pad=4)
        if tag == "a":
            ax.set_ylabel("hit rate")
    handles, labels = axes[0].get_legend_handles_labels()
    # 범례를 패널 안에 두면 곡선을 가린다(접선 패널에서 실제로 겹쳤다). 아래로 뺀다.
    fig.subplots_adjust(bottom=0.30)
    fig.legend(handles, labels, frameon=False, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), handlelength=1.4, columnspacing=1.1)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def plot_reference_geometry(reference: Mapping[str, Any], out: Path) -> None:
    """왜 손이 둔감한가 — 반경은 좁고 방향은 넓다."""
    bands = [b for b, entry in reference["buckets"].items()
             if entry.get("joints") and int(entry.get("n", 0) or 0) >= 100]
    radius_sd: dict[str, list[float]] = {j: [] for j in JOINTS}
    angular_sd: dict[str, list[float]] = {j: [] for j in JOINTS}
    for band in bands:
        for joint, entry in reference["buckets"][band]["joints"].items():
            if joint not in radius_sd:
                continue
            if isinstance(entry.get("radius"), Mapping):
                radius_sd[joint].append(float(entry["radius"]["sd"]))
            if isinstance(entry.get("heading"), Mapping):
                angular_sd[joint].append(math.degrees(float(entry["heading"]["angular_sd"])))

    fig, ax1 = plt.subplots(figsize=(3.4, 2.2))
    positions = range(len(JOINTS))
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0.0
    bars = [median(radius_sd[j]) * 100 for j in JOINTS]
    ax1.bar([p - 0.2 for p in positions], bars, width=0.4,
            color="#3b6ea5", label="radius SD (cm)")
    ax1.set_ylabel("radius SD (cm)", color="#3b6ea5")
    ax1.tick_params(axis="y", labelcolor="#3b6ea5")
    ax2 = ax1.twinx()
    ax2.bar([p + 0.2 for p in positions], [median(angular_sd[j]) for j in JOINTS],
            width=0.4, color="#b5651d", label="heading SD (deg)")
    ax2.set_ylabel("heading SD (deg)", color="#b5651d")
    ax2.tick_params(axis="y", labelcolor="#b5651d")
    ax1.set_xticks(list(positions))
    ax1.set_xticklabels([LABEL[j] for j in JOINTS], rotation=45, ha="right")
    ax1.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def plot_worker_false_positive(result: Mapping[str, Any], out: Path) -> None:
    """같은 문턱이 작업자마다 다르게 걸린다. 익명화한다 — 개인을 지목하지 않는다."""
    by_worker = result["baseline"].get("by_worker") or {}
    if not by_worker:
        return
    rates = sorted((fired / n if n else 0.0) for n, fired in by_worker.values())
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    labels = [f"W{i+1}" for i in range(len(rates))]
    ax.bar(labels, rates, color="#3b6ea5", width=0.6)
    target = arm_reference.TARGET_FALSE_POSITIVE
    ax.axhline(target, color="#b5651d", linewidth=1.0, linestyle="--")
    ax.text(len(rates) - 0.4, target, f" target {target:g}", va="bottom", ha="right",
            fontsize=6.5, color="#b5651d")
    ax.set_ylabel("false-cue rate")
    ax.set_xlabel("operator (sorted, anonymised)")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("/output/correction_recovery_v2.json"))
    parser.add_argument("--reference", type=Path,
                        default=Path("/output/expert_reference/expert_arm_reference.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("/output/figures"))
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(RC):
        plot_confusion(result, args.out_dir / "fig_confusion")
        plot_sensitivity(result, args.out_dir / "fig_sensitivity")
        plot_reference_geometry(reference, args.out_dir / "fig_reference_geometry")
        plot_worker_false_positive(result, args.out_dir / "fig_worker_false_positive")
    for path in sorted(args.out_dir.glob("fig_*")):
        print(f"{path}  {path.stat().st_size:,} B")


if __name__ == "__main__":
    main()
