#!/usr/bin/env python3
"""교란 회복 실험 결과를 논문 그림으로 굽는다.

    docker exec -w /workspace -e PYTHONPATH=/workspace ship-paint-trainer \
      python3 trainer/scripts/plot_correction_recovery.py

PDF(벡터)와 PNG(미리보기)를 같이 내고, 폴더에 `README.md`(그림 설명)를 함께 쓴다.

## 폰트

CVPR 본문이 Times New Roman인데 **리눅스 컨테이너에 그 폰트가 없다**(MS 독점).
**Liberation Serif**를 쓴다 — Times New Roman과 **메트릭 호환**이라 자폭·행 길이가
같게 설계된 대체 폰트다. `/output/fonts/`에 넣어 두고 여기서 등록한다. 없으면 기본
serif로 떨어지고 경고만 낸다(그림은 나온다).

## 라벨

전부 **영문**이다. 한글 폰트가 없는 환경에서 사각형으로 깨지면 그림이 쓸모없어진다.

## `no cue` 열을 반드시 넣는다

빼면 혼동행렬의 행 합이 1이 아니게 되고, **"지목을 안 한 것"이 "틀리게 지목한 것"으로
읽힌다** — 이 실험의 핵심 결과가 바로 그 구분이다. 오분류와 다른 색 계열로 떼어 놓는다.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import arm_reference  # noqa: E402

JOINTS: Final[tuple[str, ...]] = arm_reference.SCORING_JOINTS_ARM
LABEL: Final[dict[str, str]] = {
    "RightUpperArm": "R-Upper", "RightForeArm": "R-Fore", "RightHand": "R-Hand",
    "LeftUpperArm": "L-Upper", "LeftForeArm": "L-Fore", "LeftHand": "L-Hand",
}
MISS: Final = "no cue"
FONT_DIR: Final = Path("/output/fonts")
# Times New Roman 메트릭 호환 순서. 앞의 것부터 찾는다.
SERIF_STACK: Final[list[str]] = [
    "Times New Roman", "Liberation Serif", "Nimbus Roman", "Tinos", "DejaVu Serif",
]


def register_fonts() -> str:
    """`/output/fonts`의 TTF를 matplotlib에 등록하고 실제로 쓸 폰트 이름을 돌려준다."""
    for path in sorted(glob.glob(str(FONT_DIR / "*.ttf"))):
        try:
            font_manager.fontManager.addfont(path)
        except (OSError, RuntimeError):
            continue
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in SERIF_STACK:
        if name in available:
            return name
    return "serif"


def style(font: str) -> dict[str, Any]:
    """seaborn paper 테마에 논문 규격을 얹는다."""
    sns.set_theme(style="whitegrid", context="paper")
    return {
        "font.family": "serif",
        "font.serif": [font] + SERIF_STACK,
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # TrueType 埋め込み — 심사 시스템이 Type3를 거부하는 일이 있다
        "ps.fonttype": 42,
    }


def _matrix(result: Mapping[str, Any], direction: str) -> tuple[list[list[float]], list[int]]:
    """행 정규화한 (6 × 7). 마지막 열이 `no cue`다."""
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
        rows.append(
            [counts[c] / total for c in JOINTS] + [miss / total] if total else [0.0] * 7
        )
    return rows, totals


def _fmt(value: float) -> str:
    return "" if value < 0.005 else f"{value:.2f}".lstrip("0")


def plot_confusion(result: Mapping[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(
        1, 4, figsize=(7.0, 2.7),
        gridspec_kw={"width_ratios": [6, 1, 6, 1], "wspace": 0.05},
    )
    main = sns.color_palette("Blues", as_cmap=True)
    cue = sns.color_palette("Greys", as_cmap=True)
    totals_seen: set[int] = set()

    for index, (direction, tag) in enumerate((("radial", "a"), ("tangential", "b"))):
        ax, ax_miss = axes[index * 2], axes[index * 2 + 1]
        rows, totals = _matrix(result, direction)
        totals_seen.update(totals)
        body = [row[: len(JOINTS)] for row in rows]
        miss = [[row[-1]] for row in rows]

        sns.heatmap(
            body, ax=ax, cmap=main, vmin=0, vmax=1, cbar=False, square=True,
            linewidths=0.6, linecolor="white",
            annot=[[_fmt(v) for v in row] for row in body], fmt="",
            annot_kws={"fontsize": 6.6},
            xticklabels=[LABEL[j] for j in JOINTS],
            yticklabels=[LABEL[j] for j in JOINTS] if index == 0 else False,
        )
        # `vmax`를 1보다 크게 둔다 — 0.9가 검정이 되면 미발부 열이 시각적으로 지배해
        # 대각선(이 그림의 주제)보다 눈에 먼저 들어온다.
        sns.heatmap(
            miss, ax=ax_miss, cmap=cue, vmin=0, vmax=1.9, cbar=False, square=True,
            linewidths=0.6, linecolor="white",
            annot=[[_fmt(v) for v in row] for row in miss], fmt="",
            annot_kws={"fontsize": 6.6},
            xticklabels=[MISS], yticklabels=False,
        )
        for target in (ax, ax_miss):
            target.tick_params(length=0)
            target.set_xticklabels(target.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("cued joint")
        if index == 0:
            ax.set_ylabel("perturbed joint")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(f"({tag}) {direction}", x=0.54, pad=6)

    # 행별 n은 싣지 않는다 — 모든 행이 같은 수라 6번 반복하면 군더더기다. README가 적는다.
    if len(totals_seen) == 1:
        fig.text(0.5, -0.13, f"n = {totals_seen.pop():,} trials per row",
                 ha="center", fontsize=6.8, color="#666666")
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def plot_sensitivity(result: Mapping[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    palette = sns.color_palette("colorblind", len(JOINTS))
    markers = ["o", "s", "^", "v", "D", "P"]
    for ax, direction, tag in zip(axes, ("radial", "tangential"), ("a", "b")):
        for joint, colour, marker in zip(JOINTS, palette, markers):
            xs, ys = [], []
            for metres in result["magnitudes"]:
                slot = result["tally"].get(f"{direction}|{metres}|{joint}", {})
                total = sum(slot.values())
                if not total:
                    continue
                xs.append(metres * 100)
                ys.append(slot.get(joint, 0) / total)
            ax.plot(xs, ys, marker=marker, markersize=3.6, color=colour,
                    markeredgecolor="white", markeredgewidth=0.4, label=LABEL[joint])
        ax.set_xlabel("imposed displacement (cm)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"({tag}) {direction}", pad=6)
        sns.despine(ax=ax)
    axes[0].set_ylabel("hit rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.32)
    fig.legend(handles, labels, frameon=False, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), handlelength=1.5, columnspacing=1.2)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def plot_reference_geometry(reference: Mapping[str, Any], out: Path) -> None:
    """왜 손이 둔감한가 — 반경은 좁고 방향은 넓다."""
    radius: dict[str, list[float]] = {j: [] for j in JOINTS}
    heading: dict[str, list[float]] = {j: [] for j in JOINTS}
    for entry in reference["buckets"].values():
        if int(entry.get("n", 0) or 0) < 100:
            continue
        for joint, stats in (entry.get("joints") or {}).items():
            if joint not in radius:
                continue
            if isinstance(stats.get("radius"), Mapping):
                radius[joint].append(float(stats["radius"]["sd"]) * 100)
            if isinstance(stats.get("heading"), Mapping):
                heading[joint].append(math.degrees(float(stats["heading"]["angular_sd"])))

    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0.0
    order = [LABEL[j] for j in JOINTS]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, data, ylabel, colour, tag, title in (
        (axes[0], [median(radius[j]) for j in JOINTS], "radius SD (cm)",
         sns.color_palette("Blues", 6)[4], "a", "reach spread"),
        (axes[1], [median(heading[j]) for j in JOINTS], "heading SD (deg)",
         sns.color_palette("Oranges", 6)[4], "b", "direction spread"),
    ):
        sns.barplot(x=order, y=data, ax=ax, color=colour, width=0.62)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({tag}) {title}", pad=6)
        # `set_xticklabels`만 부르면 눈금 위치가 고정돼 있지 않다는 경고가 난다.
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right")
        for patch, value in zip(ax.patches, data):
            ax.text(patch.get_x() + patch.get_width() / 2, value, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=6.2, color="#444444")
        sns.despine(ax=ax)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def plot_worker_false_positive(result: Mapping[str, Any], out: Path) -> None:
    """같은 문턱이 작업자마다 다르게 걸린다. **익명화한다** — 개인을 지목하지 않는다."""
    by_worker = result["baseline"].get("by_worker") or {}
    if not by_worker:
        return
    rates = sorted((fired / n if n else 0.0) for n, fired in by_worker.values())
    labels = [f"W{i + 1}" for i in range(len(rates))]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    sns.barplot(x=labels, y=rates, ax=ax, color=sns.color_palette("Blues", 6)[4], width=0.6)
    target = arm_reference.TARGET_FALSE_POSITIVE
    ax.axhline(target, color=sns.color_palette("Oranges", 6)[4], linewidth=1.1, linestyle="--")
    # 목표 라벨을 **왼쪽**에 둔다 — 오른쪽에 두면 가장 높은 막대의 값 라벨과 겹친다.
    ax.text(-0.42, target, f" target {target:g}", va="bottom", ha="left",
            fontsize=6.8, color=sns.color_palette("Oranges", 6)[5])
    for patch, value in zip(ax.patches, rates):
        ax.text(patch.get_x() + patch.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom", fontsize=6.2, color="#444444")
    ax.set_ylabel("false-cue rate")
    ax.set_xlabel("operator (sorted, anonymised)")
    ax.set_ylim(0, max(max(rates), target) * 1.25)
    sns.despine(ax=ax)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def write_readme(result: Mapping[str, Any], out_dir: Path, font: str) -> None:
    base = result["baseline"]
    fp = base["corrected"] / base["total"] if base["total"] else float("nan")
    rows_radial, _ = _matrix(result, "radial")
    hit = {}
    for direction in ("radial", "tangential"):
        for metres in result["magnitudes"]:
            h = c = m = 0
            for joint in JOINTS:
                slot = result["tally"].get(f"{direction}|{metres}|{joint}", {})
                for named, n in slot.items():
                    if named == "__miss__":
                        m += n
                    elif named == joint:
                        h += n
                    else:
                        c += n
            total = h + c + m
            if total:
                hit[(direction, metres)] = (h / total, c / total, m / total)
    r15 = hit.get(("radial", 0.15))
    precision = r15[0] / (r15[0] + r15[1]) if r15 and (r15[0] + r15[1]) else float("nan")

    (out_dir / "README.md").write_text(f"""# 그림 — 교정 지목 정확도 (통제 교란 회복 실험)

`trainer/scripts/plot_correction_recovery.py`가 굽는다. **PDF가 논문용**(벡터,
`pdf.fonttype=42`로 TrueType 임베딩)이고 PNG는 미리보기다.

- 사전등록: `docs/measurements/2026-09-30-correction-recovery-prereg.md`,
  `docs/measurements/2026-09-30-per-joint-threshold-prereg.md`
- 원자료: `/output/correction_recovery_v2.json`
- 재현: `python3 trainer/scripts/correction_recovery.py --passes 250 --seed 20260930
  --out /output/correction_recovery_v2.json` 뒤에 위 그림 스크립트

**폰트**: `{font}`. CVPR 본문은 Times New Roman인데 리눅스 컨테이너에 그 폰트(MS 독점)가
없어 **메트릭 호환 대체폰트**를 쓴다 — 자폭·행 길이가 Times와 같게 설계된 것이라 본문과
섞여도 어색하지 않다. 라벨은 전부 영문이다(한글 폰트 없는 환경에서 깨지지 않게).

---

## 실험 요약

실제 숙련자 패스에서 팔 관절 하나를 통제된 크기로 옮기고, 시스템이 **그 관절을
지목하는지** 본다. 정답을 우리가 만들었으므로 순환이 아니다.

- 패스 **{result['passes']}개**, 작업자 **{len(result['workers'])}인**(패스 수 균등)
- 교란 방향 둘: **radial**(어깨 중점에서 멀어지는 쪽 — 지표가 잡도록 설계된 성분)과
  **tangential**(그에 수직 — **못 잡을 것으로 예측한 대조군**)
- 크기 {', '.join(f'{m*100:.0f}' for m in result['magnitudes'])} cm, ±양방향 평균
- 기준선 `d = 0`: 교란 없이 교정이 나가는 비율 = **위양성률 {fp:.3f}**

---

## `fig_confusion` — 혼동행렬

행이 **교란한 관절**, 열이 **시스템이 지목한 관절**. 행 정규화(합 = 1). 오른쪽 끝의
**`no cue`**는 교정이 **나가지 않은** 비율이고, 회색 계열로 떼어 놓았다.

**`no cue`를 빼면 안 된다.** 빼면 행 합이 1이 아니게 되고 "지목을 안 한 것"이 "틀리게
지목한 것"으로 읽힌다 — 이 실험의 핵심 결과가 바로 그 구분이다.

읽는 법:

- **대각선이 hit** — 옳게 지목. 비대각이 **오분류**.
- (a) radial: 위팔 **0.97**, 아래팔·손 0.08~0.33. 비대각은 전부 0.00~0.02다.
  → **지목하면 거의 항상 맞고, 문제는 지목을 안 하는 것이다.**
- (b) tangential(대조군): 위팔 0.53~0.59, 나머지 0.01~0.03, `no cue` 0.90 안팎.
  → 방향 성분은 설계대로 **잡지 않는다.** 이 대조가 없으면 "무엇이든 잡는다"와 구별되지 않는다.
- `n`은 행별 시행 수다. 비율만 두면 근거의 두께가 보이지 않는다.

**지목 정밀도** = hit / (hit + 오분류) = **{precision:.2f}** (radial 15 cm).

---

## `fig_sensitivity` — 검출 감도 곡선

가로축이 **가한 변위**, 세로축이 hit 비율. 관절마다 한 선.

- (a) radial: 위팔 둘은 **10 cm에서 1.00으로 포화**, 아래팔·손은 25 cm에서도 0.21~0.76.
  두 무리가 완전히 갈린다.
- (b) tangential: 위팔만 올라가고 나머지는 바닥에 붙는다.

이 그림이 **"정밀하지만 둔감하다"**를 말한다. 그리고 둔감도가 관절마다 다르다는 것이
단일 성능 수치로 보고하면 안 되는 이유다.

---

## `fig_reference_geometry` — 왜 손이 둔감한가

위 결과의 **원인**이다. 숙련자 레퍼런스에서 관절별 산포를 두 성분으로 나눠 보면:

- (a) **반경 SD** — 어깨 중점까지 거리의 산포.
- (b) **방향 SD** — 평균 방향과 이루는 각의 산포(원형통계).

위팔은 어깨에 뼈로 묶여 있어 둘 다 작고, 손·아래팔은 **방향 산포가 45~73°**다.
숙련자끼리도 그만큼 다르므로 **잡을 것이 없다** — 문턱 문제가 아니라 레퍼런스의 성질이다.

이 그림 덕분에 `fig_sensitivity`의 결과가 **한계가 아니라 측정된 성질**이 된다.

---

## `fig_worker_false_positive` — 작업자별 위양성

교란 없이 교정이 나가는 비율을 작업자별로. **익명화**(W1~W6)했고 위양성률로 정렬했다 —
개인을 지목하는 그림을 논문에 넣지 않는다.

점선이 사전등록한 목표({arm_reference.TARGET_FALSE_POSITIVE:g}). 전체는 **{fp:.3f}**로
목표를 만족하지만 **작업자 간에 벌어진다.**

σ가 작업자별로 군집돼 있어, 여러 명을 풀링해 백분위 문턱을 정하면 개인마다 다른
위양성률을 받는다. 운영상 의미가 있다 — 어떤 작업자는 다른 작업자보다 훨씬 자주
"틀렸다"는 말을 듣는데 **실력 차이가 아니다.** 남은 과제로 문서에 적어 두었다.

---

## 이 그림들이 증명하지 **못하는** 것

**교정의 효능** — "교정을 받은 학습자가 나아지는가"는 재지 않는다. 그것은 학습자
pre/post + 대조 조건(무교정 또는 위교정) + **우리 지표와 독립된 결과 변수**(도막 두께
균일도 등)가 필요하다. 같은 지표로 재면 순환논증이다.
""", encoding="utf-8")


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

    font = register_fonts()
    with plt.rc_context(style(font)):
        plot_confusion(result, args.out_dir / "fig_confusion")
        plot_sensitivity(result, args.out_dir / "fig_sensitivity")
        plot_reference_geometry(reference, args.out_dir / "fig_reference_geometry")
        plot_worker_false_positive(result, args.out_dir / "fig_worker_false_positive")
    write_readme(result, args.out_dir, font)
    print(f"font: {font}")
    for path in sorted(args.out_dir.glob("*")):
        if path.suffix in (".pdf", ".png", ".md"):
            print(f"  {path.name:34s} {path.stat().st_size:>9,} B")


if __name__ == "__main__":
    main()
