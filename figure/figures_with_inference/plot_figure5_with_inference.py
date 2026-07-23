"""生成带置信区间和配对变化检验的 Figure 5 副本。

脚本复用 ``src/fig3_ci_preview.py`` 的汇总与绘图函数，并按论文设定重新计算
Blind→Rubric Shown 的配对 p_safe/harm 变化。新版图片写入当前目录，不覆盖
原始 Figure 5。
"""

from __future__ import annotations

import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np



ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import fig3_ci_preview as base



RESAMPLES = int(os.environ.get("BOOTSTRAP_RESAMPLES", "100000"))


def primary_paired_psafe_summary(rows):
    """复现论文中固定随机种子的 p_safe 配对 percentile bootstrap。

    只比较同一模型、同一题目在 blind 与 rubric_shown 两个条件下的结果，
    输出 rubric_shown 减 blind 的均值、95% 区间及配对样本量。
    """
    rng = random.Random(20_260_711)
    output = {}
    for model in base.MODELS:
        by_condition = {}
        for condition in ("blind", "rubric_shown"):

            by_condition[condition] = {
                row["question"]: float(row["p_safe"])
                for row in rows
                if row.get("model_tag") == model
                and row.get("condition") == condition
                and row.get("question")
                and row.get("p_safe") is not None
            }
        matched = sorted(
            by_condition["blind"].keys() & by_condition["rubric_shown"].keys()
        )
        differences = [
            by_condition["rubric_shown"][item] - by_condition["blind"][item]
            for item in matched
        ]

        bootstrap = sorted(
            sum(rng.choices(differences, k=len(differences))) / len(differences)
            for _ in range(RESAMPLES)
        )
        lower = bootstrap[int(0.025 * (RESAMPLES - 1))]
        upper = bootstrap[int(0.975 * (RESAMPLES - 1))]
        output[model] = {
            "change": statistics.mean(differences),
            "ci": (lower, upper),
            "n": len(differences),
        }
    return output


def paired_harm_summary(rows):
    """计算 Blind→Rubric Shown 的 harmful 二元指标配对变化。"""
    output = {}
    for model_index, model in enumerate(base.MODELS):
        by_condition = {}
        for condition in ("blind", "rubric_shown"):

            by_condition[condition] = {
                row["question"]: int(row["harmful"])
                for row in rows
                if row.get("model_tag") == model
                and row.get("condition") == condition
                and row.get("question")
                and row.get("harmful") is not None
            }
        matched = sorted(
            by_condition["blind"].keys() & by_condition["rubric_shown"].keys()
        )
        differences = np.asarray(
            [
                by_condition["rubric_shown"][item]
                - by_condition["blind"][item]
                for item in matched
            ],
            dtype=float,
        )

        lower, upper = base.bootstrap_mean_interval(
            differences, 20_260_860 + model_index, resamples=RESAMPLES
        )
        output[model] = {
            "change": statistics.mean(differences),
            "ci": (lower, upper),
            "n": differences.size,
        }
    return output


def _marker(interval):
    """把区间转换为图中使用的显著性标记。"""
    lower, upper = interval
    return "*" if lower > 0 or upper < 0 else "ns"


def main():
    """汇总 Exp8，绘制各模型透明度结果并导出 PNG/PDF。"""
    rows = base.load("exp8_principle3.jsonl")
    summary = base.preview_exp8_summary(rows)

    primary_psafe = primary_paired_psafe_summary(rows)
    for model in base.MODELS:
        summary[model]["paired"] = primary_psafe[model]
    harm_paired = paired_harm_summary(rows)


    fig = base.plt.figure(figsize=(12.8, 5.8), facecolor="white")
    grid = base.gridspec.GridSpec(
        1, len(base.MODELS), wspace=0.12,
        left=0.07, right=0.99, top=0.88, bottom=0.18,
    )
    for index, model in enumerate(base.MODELS):
        ax = fig.add_subplot(grid[0, index])
        base.draw_exp8_ci_preview(
            ax, summary, model, chr(ord("A") + index)
        )

        ax.set_xticklabels(["Blind", "Rubric\nShown", "Explicit\nscore max."])
        ax.axvspan(1.67, 2.33, color="#D9D9D9", alpha=0.25, zorder=0)
        ax.text(2, 1.245, "positive control", ha="center", va="top",
                fontsize=9.5, color="#555555")
        ax.set_ylim(-0.08, 1.34)


        paired_text = next(
            text for text in ax.texts if text.get_text().startswith("paired Δ=")
        )
        psafe = summary[model]["paired"]
        harm = harm_paired[model]
        paired_text.set_text(
            f"p_safe Δ={psafe['change']:+.3f} "
            f"[{psafe['ci'][0]:+.3f}, {psafe['ci'][1]:+.3f}] {_marker(psafe['ci'])}\n"
            f"harm Δ={harm['change']:+.3f} "
            f"[{harm['ci'][0]:+.3f}, {harm['ci'][1]:+.3f}] {_marker(harm['ci'])}; "
            f"N={psafe['n']}"
        )
        paired_text.set_fontsize(9.2)


    fig.text(
        0.5, 0.035,
        "Error bars show bootstrap/Wilson 95% CIs. * Matched Blind→Rubric Shown change CI excludes 0; "
        "ns includes 0 (nominal, unadjusted).",
        ha="center", fontsize=10, color="#444444",
    )

    for extension in ("png", "pdf"):
        output = OUTPUT_DIR / f"figure5_with_inference{base.JUDGE_SUFFIX}.{extension}"
        fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"saved -> {output}")
    base.plt.close(fig)


if __name__ == "__main__":

    main()
