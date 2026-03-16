"""Task 12 최종 리포트 시각화 차트 생성기.

5개 차트를 matplotlib+seaborn으로 PNG 렌더링:
  1. score_progression.png   — Iter별 점수 추이 (line)
  2. radar_comparison.png    — 5차원 Iter11 vs Iter13 (radar)
  3. category_scores.png     — 카테고리별 점수 (horizontal bar)
  4. score_distribution.png  — 30문항 점수 분포 (histogram)
  5. sft_composition.png     — SFT 데이터 구성 (donut)

Usage:
    python3 generate_report_charts.py [--outdir /output/eval_results/charts]
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch, FancyBboxPatch
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)


def _setup_korean_font() -> None:
    """Try to find a Korean-capable font; fall back to Inter/system sans."""
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            logger.info("Korean font loaded: %s", path)
            return
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    logger.warning("No Korean font found — labels may render as boxes")


# ---------------------------------------------------------------------------
# Modern palette — Linear/shadcn/Tailwind hybrid
# ---------------------------------------------------------------------------

C = {
    "primary":    "#5E6AD2",
    "secondary":  "#4EA7FC",
    "success":    "#2EB88A",
    "warning":    "#E88C30",
    "danger":     "#E23670",
    "purple":     "#AF57DB",
    "teal":       "#14B8A6",
    "pink":       "#EC4899",
    "muted":      "#8A8F98",
    "text":       "#18181B",
    "subtext":    "#71717A",
    "grid":       "#F4F4F5",
    "bg":         "#FAFAFA",
    "white":      "#FFFFFF",
    "iter11":     "#4EA7FC",
    "iter13":     "#5E6AD2",
}

SEQUENTIAL_9 = [
    "#5E6AD2", "#4EA7FC", "#2EB88A", "#E88C30", "#E23670",
    "#AF57DB", "#14B8A6", "#EC4899", "#0EA5E9",
]

TARGETS = {
    "poc":        {"val": 4.0, "label": "PoC (4.0)",        "color": "#2EB88A", "ls": "--"},
    "mvp":        {"val": 6.0, "label": "MVP/Beta (6.0)",   "color": "#E88C30", "ls": "-."},
    "production": {"val": 7.5, "label": "Production (7.5)", "color": "#E23670", "ls": ":"},
}


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette=SEQUENTIAL_9)
    plt.rcParams.update({
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["white"],
        "axes.edgecolor": C["grid"],
        "grid.color": C["grid"],
        "grid.linewidth": 0.8,
        "text.color": C["text"],
        "axes.labelcolor": C["text"],
        "xtick.color": C["subtext"],
        "ytick.color": C["subtext"],
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.4,
    })


def _draw_target_lines(ax: plt.Axes, show_labels: bool = True) -> None:
    """Draw PoC / MVP / Production horizontal target lines."""
    for t in TARGETS.values():
        ax.axhline(y=t["val"], color=t["color"], linestyle=t["ls"],
                   linewidth=1.8, alpha=0.75, zorder=2)
        if show_labels:
            ax.text(ax.get_xlim()[1], t["val"] + 0.15, t["label"],
                    ha="right", va="bottom", fontsize=9, fontweight="bold",
                    color=t["color"], alpha=0.85)


def _draw_target_vlines(ax: plt.Axes, show_labels: bool = True) -> None:
    for key, t in TARGETS.items():
        ax.axvline(x=t["val"], color=t["color"], linestyle=t["ls"],
                   linewidth=1.8, alpha=0.75, zorder=2)
        if show_labels:
            ax.text(t["val"] + 0.08, ax.get_ylim()[1] * 0.95, t["label"],
                    rotation=90, ha="left", va="top", fontsize=8,
                    fontweight="bold", color=t["color"], alpha=0.85)


# ---------------------------------------------------------------------------
# 1. Score Progression
# ---------------------------------------------------------------------------

def chart_score_progression(history: list[dict[str, Any]], outdir: Path) -> Path:
    """Iter별 agentic 점수 추이 꺾은선 그래프."""
    data = [(h["iter"], h["agentic"], h.get("tc_made", 0), h.get("decision", ""))
            for h in history if h.get("agentic") is not None]
    labels, scores, tc_counts, decisions = zip(*data)

    fig, ax1 = plt.subplots(figsize=(14, 6.5))

    ax2 = ax1.twinx()
    ax2.bar(labels, tc_counts, alpha=0.10, color=C["muted"], width=0.55, zorder=1)
    ax2.set_ylabel("Tool Calls", color=C["muted"], fontsize=11)
    ax2.tick_params(axis="y", labelcolor=C["muted"])
    ax2.set_ylim(0, max(tc_counts) * 2.2)

    ax1.plot(labels, scores, "o-", color=C["primary"], linewidth=2.8,
             markersize=9, zorder=5, markeredgecolor=C["white"], markeredgewidth=2,
             label="Agentic Score")

    gradient_x = range(len(labels))
    ax1.fill_between(gradient_x, scores, alpha=0.06, color=C["primary"])

    for i, (lbl, sc, tc, dec) in enumerate(zip(labels, scores, tc_counts, decisions)):
        vert = 14 if i % 2 == 0 else -18
        color = C["success"] if dec == "TARGET_MET" else C["text"]
        ax1.annotate(f"{sc:.1f}" if sc != int(sc) else f"{sc:.2f}",
                     (lbl, sc), textcoords="offset points",
                     xytext=(0, vert), ha="center", fontsize=10,
                     fontweight="bold", color=color)

    _draw_target_lines(ax1, show_labels=True)

    best_idx = list(scores).index(max(scores))
    ax1.scatter([labels[best_idx]], [scores[best_idx]], s=280,
                facecolors="none", edgecolors=C["success"], linewidths=2.5, zorder=6)

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Agentic Score (/10)", fontsize=12)
    ax1.set_title("Agentic Score Progression", fontsize=17, fontweight="bold", pad=12)
    ax1.set_ylim(0, 10.5)
    ax1.tick_params(axis="x", rotation=30)

    lines1, labels1 = ax1.get_legend_handles_labels()
    target_patches = [
        plt.Line2D([0], [0], color=t["color"], linestyle=t["ls"], linewidth=1.5,
                   label=t["label"])
        for t in TARGETS.values()
    ]
    ax1.legend(handles=lines1 + target_patches, loc="upper left", fontsize=9,
               framealpha=0.9, edgecolor=C["grid"])

    path = outdir / "score_progression.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# 2. Radar Comparison
# ---------------------------------------------------------------------------

def chart_radar_comparison(history: list[dict[str, Any]], outdir: Path) -> Path:
    """5차원 레이더 차트 — Iter13 vs Iter14."""
    dims = ["tool_selection", "param_correctness", "reasoning_depth",
            "context_integration", "response_structure"]
    dim_labels = ["Tool\nSelection", "Parameter\nCorrectness", "Reasoning\nDepth",
                  "Context\nIntegration", "Response\nStructure"]

    iter13 = next((h for h in history if h["iter"] == "Iter13"), None)
    iter14 = next((h for h in history if h["iter"] == "Iter14"), None)
    if not iter13 or not iter14:
        logger.warning("Missing Iter13 or Iter14 data for radar chart")
        return outdir / "radar_comparison.png"

    vals_13 = [iter13.get(d, 0) for d in dims]
    vals_14 = [iter14.get(d, 0) for d in dims]
    mvp_target = [1.5, 1.4, 1.4, 1.4, 1.3]
    prod_target = [1.8, 1.7, 1.6, 1.7, 1.7]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    vals_13 += vals_13[:1]
    vals_14 += vals_14[:1]
    mvp_target += mvp_target[:1]
    prod_target += prod_target[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, prod_target, alpha=0.06, color=C["danger"])
    ax.plot(angles, prod_target, "--", color=C["danger"], linewidth=1.2,
            alpha=0.5, label="Production (7.5)")

    ax.fill(angles, mvp_target, alpha=0.06, color=C["warning"])
    ax.plot(angles, mvp_target, "-.", color=C["warning"], linewidth=1.2,
            alpha=0.5, label="MVP (6.0)")

    ax.fill(angles, vals_13, alpha=0.12, color=C["iter11"])
    ax.plot(angles, vals_13, "o-", color=C["iter11"], linewidth=2,
            markersize=7, markeredgecolor=C["white"], markeredgewidth=1.5,
            label="Iter13 (5.43)")

    ax.fill(angles, vals_14, alpha=0.20, color=C["iter13"])
    ax.plot(angles, vals_14, "o-", color=C["iter13"], linewidth=2.5,
            markersize=8, markeredgecolor=C["white"], markeredgewidth=1.5,
            label="Iter14 (4.97)")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=11, color=C["text"])
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0.5", "1.0", "1.5", "2.0"], fontsize=9, color=C["muted"])
    ax.spines["polar"].set_color(C["grid"])

    ax.set_title("5-Dimension Comparison", fontsize=15, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=10,
              framealpha=0.9, edgecolor=C["grid"])

    for i, (v13, v14) in enumerate(zip(vals_13[:-1], vals_14[:-1])):
        delta = v14 - v13
        if abs(delta) > 0.01:
            sign = "+" if delta > 0 else ""
            color = C["success"] if delta > 0 else C["danger"]
            ax.annotate(f"{sign}{delta:.2f}", xy=(angles[i], v14),
                        textcoords="offset points", xytext=(14, 8),
                        fontsize=9, fontweight="bold", color=color)

    path = outdir / "radar_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# 3. Category Scores
# ---------------------------------------------------------------------------

def chart_category_scores(outdir: Path) -> Path:
    """카테고리별 점수 수평 막대 그래프."""
    cat_labels = ["Tool\nCalling", "XR\nContent", "Anomaly\nInterpretation",
                  "Sequence\nDecision", "CoT\nReasoning"]
    scores = [5.67, 5.33, 5.17, 5.00, 3.67]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bar_colors = [SEQUENTIAL_9[i] for i in range(len(scores))]
    bars = ax.barh(cat_labels, scores, color=bar_colors, height=0.55,
                   edgecolor=C["white"], linewidth=1.5, zorder=3)

    for t in TARGETS.values():
        ax.axvline(x=t["val"], color=t["color"], linestyle=t["ls"],
                   linewidth=1.5, alpha=0.6, zorder=2)

    for bar, score, color in zip(bars, scores, bar_colors):
        label_color = C["success"] if score >= 4.0 else C["danger"]
        ax.text(bar.get_width() + 0.12, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=12, fontweight="bold",
                color=label_color)

    ax.set_xlim(0, 9.5)
    ax.set_xlabel("Score (/10)", fontsize=12)
    ax.set_title("Category Scores — Iter14", fontsize=16, fontweight="bold", pad=12)
    ax.invert_yaxis()

    target_patches = [
        plt.Line2D([0], [0], color=t["color"], linestyle=t["ls"], linewidth=1.5,
                   label=t["label"])
        for t in TARGETS.values()
    ]
    ax.legend(handles=target_patches, fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor=C["grid"])

    path = outdir / "category_scores.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# 4. Score Distribution
# ---------------------------------------------------------------------------

def chart_score_distribution(outdir: Path) -> Path:
    """30문항 점수 분포 히스토그램."""
    score_counts = {2: 1, 3: 8, 4: 6, 5: 4, 6: 5, 7: 4, 8: 1, 9: 1}
    scores_list = []
    for score, count in score_counts.items():
        scores_list.extend([score] * count)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bins = np.arange(1.5, 10.5, 1)
    n, _, patches = ax.hist(scores_list, bins=bins, edgecolor=C["white"],
                            linewidth=2, alpha=0.90, zorder=3)

    threshold_colors = {
        range(0, 4): C["danger"],
        range(4, 7): C["primary"],
        range(7, 10): C["success"],
    }
    for patch, left_edge in zip(patches, bins[:-1]):
        score_val = int(left_edge + 0.5)
        for rng, color in threshold_colors.items():
            if score_val in rng:
                patch.set_facecolor(color)
                break

    for i, count in enumerate(n):
        if count > 0:
            ax.text(bins[i] + 0.5, count + 0.25, str(int(count)),
                    ha="center", fontsize=11, fontweight="bold", color=C["text"])

    mean_score = np.mean(scores_list)
    ax.axvline(x=mean_score, color=C["warning"], linestyle="-.",
               linewidth=2.5, zorder=4, label=f"Mean: {mean_score:.2f}")

    ax.set_xlabel("Score (/10)", fontsize=12)
    ax.set_ylabel("Question Count", fontsize=12)
    ax.set_title("Score Distribution — Iter14 (30 Questions)", fontsize=16,
                 fontweight="bold", pad=12)
    ax.set_xticks(range(2, 10))
    ax.set_ylim(0, max(n) + 2)

    legend_elements = [
        Patch(facecolor=C["success"], label="High (7-9)"),
        Patch(facecolor=C["primary"], label="Mid (4-6)"),
        Patch(facecolor=C["danger"], label="Low (2-3)"),
        plt.Line2D([0], [0], color=C["warning"], linestyle="-.",
                   linewidth=2, label=f"Mean: {mean_score:.2f}"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right",
              framealpha=0.9, edgecolor=C["grid"])

    path = outdir / "score_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# 5. SFT Composition
# ---------------------------------------------------------------------------

def chart_sft_composition(outdir: Path) -> Path:
    """SFT 데이터 구성 도넛 차트."""
    labels = [
        "Document QA", "Original FC", "Location Tracking", "Pose Estimation",
        "Sequence Validation", "Timeseries", "XR Content", "Multi-tool",
        "Multi-turn (aug)",
    ]
    sizes = [433, 254, 15, 15, 15, 15, 10, 15, 12]

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=SEQUENTIAL_9[:len(sizes)], autopct="",
        startangle=90, pctdistance=0.8,
        wedgeprops=dict(width=0.42, edgecolor=C["white"], linewidth=2.5)
    )

    legend_labels = [f"{l} ({s})" for l, s in zip(labels, sizes)]
    legend = ax.legend(wedges, legend_labels, loc="center left",
                       bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=False)

    total = sum(sizes)
    ax.text(0, 0.03, f"{total}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=C["primary"])
    ax.text(0, -0.08, "samples", ha="center", va="center",
            fontsize=12, color=C["muted"])

    original = 433 + 254
    augmented = total - original
    ax.text(0, -0.22, f"Original {original}  |  Augmented {augmented}",
            ha="center", va="center", fontsize=10, color=C["subtext"],
            transform=ax.transData)

    ax.set_title("SFT Training Data Composition", fontsize=16,
                 fontweight="bold", pad=20)

    path = outdir / "sft_composition.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report visualization charts")
    parser.add_argument("--outdir", type=str, default="/output/eval_results/charts",
                        help="Output directory for PNG files")
    parser.add_argument("--history", type=str,
                        default="/output/eval_results/iteration_history.json",
                        help="Path to iteration_history.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _setup_korean_font()
    _apply_style()

    with open(args.history) as f:
        history = json.load(f)

    paths = []
    paths.append(chart_score_progression(history, outdir))
    paths.append(chart_radar_comparison(history, outdir))
    paths.append(chart_category_scores(outdir))
    paths.append(chart_score_distribution(outdir))
    paths.append(chart_sft_composition(outdir))

    logger.info("=" * 50)
    logger.info("All %d charts generated in %s", len(paths), outdir)
    for p in paths:
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
