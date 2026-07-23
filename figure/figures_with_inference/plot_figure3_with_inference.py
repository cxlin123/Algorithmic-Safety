"""生成带置信区间和显著性标记的 Figure 3 副本。

本脚本复用 ``src/fig1_ci_preview.py`` 中的数据读取、统计和基础绘图函数，
只补充配对 bootstrap 差值、显著性星号以及图下注释。输出写入当前目录，
不会覆盖论文主图目录中的原始文件。
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np



ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import fig1_ci_preview as base



RESAMPLES = int(os.environ.get("BOOTSTRAP_RESAMPLES", "100000"))


def _binary_value(row, metric):
    """从一条 Exp1 记录中提取指定的二元 judge 指标。"""
    hallucination, harmful = base.exp1_judge_fields(row)
    if metric == "hallucination":
        return hallucination.get("contains_hallucination")
    return harmful.get("harmful")


def paired_binary_changes(rows):
    """按题目配对计算“攻击条件减 baseline”的二元指标变化。

    每个模型、每个指标分别建立 baseline 和攻击条件的题目字典，仅保留两边
    都出现的题目，再对逐题差值做 percentile bootstrap。返回值包含平均变化、
    95% 置信区间和实际配对样本量。
    """
    output = defaultdict(lambda: defaultdict(dict))
    seed_index = 0
    for model in base.MODELS:
        for metric in base.METRICS_HM:

            baseline = {}
            for row in rows:
                if row.get("model_tag") != model or row.get("condition") != "baseline":
                    continue
                key = row.get("id") or row.get("question")
                value = _binary_value(row, metric)
                if key is not None and value is not None:
                    baseline[key] = float(value)

            for attack in base.COND_ORDER[1:]:

                attacked = {}
                for row in rows:
                    if row.get("model_tag") != model or row.get("condition") != attack:
                        continue
                    key = row.get("id") or row.get("question")
                    value = _binary_value(row, metric)
                    if key is not None and value is not None:
                        attacked[key] = float(value)
                matched = sorted(baseline.keys() & attacked.keys())
                differences = np.asarray(
                    [attacked[item] - baseline[item] for item in matched], dtype=float
                )

                rng = np.random.default_rng(20_260_820 + seed_index)
                seed_index += 1
                chunks = []

                for start in range(0, RESAMPLES, 5_000):
                    size = min(5_000, RESAMPLES - start)
                    indices = rng.integers(
                        0, differences.size, size=(size, differences.size)
                    )
                    chunks.append(differences[indices].mean(axis=1))
                bootstrap = np.concatenate(chunks)
                lower, upper = np.quantile(bootstrap, [0.025, 0.975])
                output[model][metric][attack] = {
                    "change": differences.mean(),
                    "ci": (lower, upper),
                    "n": differences.size,
                }
    return output


def _excludes_zero(interval):
    """判断置信区间是否完全位于零的一侧。"""
    lower, upper = interval
    return lower > 0 or upper < 0


def add_age_markers(ax, summary, model):
    """在年龄差变化的置信区间不含零时添加显著性星号。"""
    x = np.arange(len(base.COND_ORDER) - 1)
    for offset, metric in zip(
        (-0.11, 0.11), ("pain_relief_strength", "treatment_aggressiveness")
    ):
        for xi, attack in zip(x + offset, base.COND_ORDER[1:]):
            _, lower, upper = summary[model][metric][attack]
            if _excludes_zero((lower, upper)):

                y = upper + 0.045 if upper >= 0 else lower - 0.075
                ax.text(xi, y, "*", ha="center", va="center", fontsize=16,
                        fontweight="bold", color="#222222")


def add_binary_markers(ax, summary, rates, model):
    """在 hallucination/harmful 热图单元格右上角标出显著结果。"""
    for row_index, metric in enumerate(base.METRICS_HM):
        y = 2 - 1 - row_index
        for column, attack in enumerate(base.COND_ORDER[1:], start=1):
            if _excludes_zero(summary[model][metric][attack]["ci"]):

                color = (
                    "white"
                    if rates[model][attack][metric][0] > 0.45
                    else "#222222"
                )
                ax.text(column + 0.83, y + 0.78, "*", ha="center", va="center",
                        fontsize=14, color=color, fontweight="bold")


def main():
    """读取两组实验结果，绘制 Figure 3 并同时导出 PNG 与 PDF。"""

    exp1 = base.load("exp1_c1_to_c2.jsonl")
    exp2 = base.load("exp2_c1_to_c3.jsonl")
    binary = base.preview_binary_summary(exp1)
    gaps = base.preview_gap_changes(exp2, resamples=RESAMPLES)
    binary_changes = paired_binary_changes(exp1)


    fig = base.plt.figure(figsize=(15, 9.0), facecolor="white")
    grid = base.gridspec.GridSpec(
        2, len(base.MODELS), height_ratios=[1.45, 0.9], hspace=0.18,
        wspace=0.16, left=0.07, right=0.99, top=0.93, bottom=0.16,
    )
    for column, model in enumerate(base.MODELS):

        age_ax = fig.add_subplot(grid[0, column])
        base.draw_gap_change_preview(age_ax, gaps, model)
        add_age_markers(age_ax, gaps, model)


        heat_ax = fig.add_subplot(grid[1, column])
        base.draw_binary_ci_heatmap(
            heat_ax, binary, model, show_ylabels=(column == 0)
        )
        add_binary_markers(heat_ax, binary_changes, binary, model)


    fig.text(
        0.5, 0.02,
        "95% CIs shown. * Paired condition − baseline bootstrap CI excludes 0 "
        "(nominal; no multiplicity adjustment).",
        ha="center", fontsize=10, color="#444444",
    )

    for extension in ("png", "pdf"):
        output = OUTPUT_DIR / f"figure3_with_inference{base.JUDGE_SUFFIX}.{extension}"
        fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"saved -> {output}")
    base.plt.close(fig)


if __name__ == "__main__":

    main()
