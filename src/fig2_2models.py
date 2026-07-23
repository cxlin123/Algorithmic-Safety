"""绘制论文 Figure 4：compound agent risk 与 Principle 2。

输入：
- results/exp4_agent_compound.jsonl：五步 agent 的逐步 p_safe。
- results/exp5_agent_jailbreak.jsonl：clean vs attacked compound p_safe。
- results/exp9_step_injection.jsonl：不同步骤注入后的 final harmful rate。

三个面板：
- A：累计 p_safe 随 agent 步骤衰减。
- B：clean 与 attacked agent 的 compound p_safe 对比。
- C：按注入步骤统计 final harm rate，并叠加 naive irreversibility 预测线。
"""
from __future__ import annotations
import json, os, statistics, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16, "axes.titlesize": 16,
                     "axes.labelsize": 16, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "legend.fontsize": 16})
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR


JUDGE_VIEW = os.environ.get("JUDGE_VIEW", "").strip()
JUDGE_SUFFIX = f"_{JUDGE_VIEW}" if JUDGE_VIEW else ""

MODELS       = [m.strip() for m in os.environ.get("FIG_MODELS", "gpt,claude").split(",") if m.strip()]
MODEL_LABELS = {"gpt": "GPT-5", "claude": "Claude-Sonnet-4-6", "llama": "LLaMA-3.1-70B"}
MODEL_SHORT  = {"gpt": "GPT-5", "claude": "Claude", "llama": "LLaMA"}


MODEL_PAIR = {
    "gpt":    ("#BEA6BE", "#523852"),
    "claude": ("#D4BD92", "#7A5D28"),
    "llama":  ("#AFBFA5", "#3F5D30"),
}

STEP_NAMES = ["understand", "pubmed_search", "differential", "treatment", "report"]
STEP_SHORT = ["S1", "S2", "S3",
              "S4", "S5"]
STEP_IRR   = [1, 1, 3, 4, 2]


def load(fname):
    """读取 results/ 下的 JSONL，跳过 error 行。"""
    rows = []
    for line in (RESULTS_DIR / fname).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)


        if "error" not in row:
            rows.append(row)
    return rows



def compute_cumulative(recs):
    """计算每个模型五步平均 p_safe 的累计乘积。"""
    out = {}
    for m in MODELS:
        per_step = []
        for r in recs:
            if r.get("model_tag") != m:
                continue
            if JUDGE_VIEW:
                steps = (r.get("judges") or {}).get(JUDGE_VIEW) or []
                psafes = [
                    step.get("p_safe")
                    for step in steps
                    if isinstance(step, dict) and step.get("p_safe") is not None
                ]
            else:
                psafes = r.get("per_step_psafe", [])
            if len(psafes) == 5:
                per_step.append(psafes)
        if not per_step:
            continue
        arr = np.array(per_step)
        step_means = arr.mean(axis=0)
        out[m] = np.cumprod(step_means)
    return out


def draw_panel_A(ax, cum):
    """绘制 cumulative p_safe 随步骤下降的折线图。"""
    xs = np.arange(1, 6)
    for m in MODELS:
        if m not in cum:
            continue
        _, dark = MODEL_PAIR[m]
        ys = cum[m]
        ax.plot(xs, ys, color=dark, linewidth=2.6,
                marker="o", markersize=10,
                markerfacecolor="white", markeredgewidth=1.8,
                label=MODEL_LABELS[m])
        label_offset = {
            "gpt": 0.018,
            "claude": -0.020,
            "llama": -0.038 if len(MODELS) > 2 else -0.020,
        }.get(m, 0.018)
        for x, y in zip(xs, ys):                                      
            va = "bottom" if label_offset > 0 else "top"
            ax.text(x, y + label_offset, f"{y:.3f}",
                    ha="center", va=va,
                    fontsize=16, color=dark, fontweight="bold")

    ax.axhline(y=1.0, color="#888", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(5.05, 1.005, "single-step ceiling", ha="right",
            fontsize=16, color="#666", style="italic")

    ax.set_xlim(0.6, 5.4)
    ax.set_ylim(0.78, 1.04)
    ax.set_xticks(xs)
    ax.set_xticklabels(STEP_SHORT, fontsize=16)
    ax.set_ylabel("Cumulative $P_{safe}$", fontsize=16)
    ax.set_xlabel("Agent Step", fontsize=16, labelpad=4)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    ax.legend(loc="lower left", fontsize=16, frameon=False)
    ax.set_title("(A)  Compound risk leakage",
                 fontsize=16, fontweight="bold", pad=8, loc="left")



def compute_exp5(recs):
    """计算 clean 和 attacked agent 的平均 compound p_safe。"""
    out = {}
    for m in MODELS:
        clean_vals, attacked_vals = [], []
        for r in recs:
            if r.get("model_tag") != m:
                continue
            if JUDGE_VIEW:
                payload = (r.get("judges") or {}).get(JUDGE_VIEW) or {}
                clean = payload.get("clean_psafe")
                atk = payload.get("attacked_psafe")
            else:
                clean = r.get("clean_psafe")
                atk = r.get("attacked_psafe")
            if clean is not None and atk is not None:
                clean_vals.append(clean)
                attacked_vals.append(atk)
        if not clean_vals:
            continue
        clean = statistics.mean(clean_vals)
        atk   = statistics.mean(attacked_vals)
        out[m] = (clean, atk, len(clean_vals))
    return out


def draw_panel_B(ax, exp5):
    """绘制 clean/attacked dumbbell，对比攻击前后 compound p_safe。"""
    xs = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        if m not in exp5:
            continue
        light, dark = MODEL_PAIR[m]
        clean, atk, n = exp5[m]
        drop_pct = 100 * (clean - atk) / clean


        ax.plot([i, i], [clean, atk],
                color=dark, linewidth=2.6, alpha=0.75, zorder=2)
        ax.scatter(i, clean, s=240, marker="o",
                   color=light, edgecolors="#222", linewidths=1.4, zorder=4)
        ax.scatter(i, atk, s=240, marker="s",
                   color=dark, edgecolors="white", linewidths=1.4, zorder=5)

        ax.text(i + 0.20, clean, f"{clean:.2f}",
                ha="left", va="center", fontsize=16, color="#444")
        ax.text(i + 0.20, atk, f"{atk:.2f}",
                ha="left", va="center", fontsize=16,
                color=dark, fontweight="bold")

        mid = (clean + atk) / 2
        ax.text(i, mid, f"↓{drop_pct:.0f}%",
                ha="right", va="center", fontsize=16,
                color="#C0392B", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS],
                       fontsize=14 if len(MODELS) > 2 else 16,
                       rotation=12 if len(MODELS) > 2 else 0,
                       ha="right" if len(MODELS) > 2 else "center")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(-0.7, len(MODELS) - 0.3)
    ax.set_ylabel("Compound $P_{safe}$", fontsize=16, labelpad=4)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#CCC", markeredgecolor="#222",
               markersize=11, label="Clean"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor="#444", markeredgecolor="white",
               markersize=11, label="Attacked"),
    ]
    ax.legend(handles=handles, loc="upper right",
              fontsize=13, frameon=True, framealpha=0,
              edgecolor="#CCC")
    ax.set_title("(B)  C1 → C4 attack",
                 fontsize=16, fontweight="bold", pad=8, loc="left")



def compute_exp9(recs):
    """按模型和注入步骤统计 final_harmful 均值。"""
    bucket = defaultdict(lambda: defaultdict(list))
    for r in recs:
        m = r.get("model_tag")
        s = r.get("inject_at")
        if m in MODELS and s in range(5):
            if JUDGE_VIEW:
                payload = (r.get("judges") or {}).get(JUDGE_VIEW) or {}
                harm = (payload.get("final_harmful") or {}).get("harmful")
            else:





                judges = r.get("judges") or {}
                judge_values = []
                for tag in ("gemini_judge", "qwen_judge"):
                    payload = judges.get(tag) or {}
                    value = (payload.get("final_harmful") or {}).get("harmful")
                    if value is not None:
                        judge_values.append(int(value))
                harm = int(any(judge_values)) if judge_values else r.get("final_harmful")
            if harm is not None:
                bucket[m][s].append(harm)
    out = {}
    for m in MODELS:
        out[m] = np.array([
            (statistics.mean(bucket[m][s]) if bucket[m][s] else 0)
            for s in range(5)
        ])
    return out


def draw_panel_C(ax, exp9):
    """绘制不同注入点的 final harm rate，并叠加 irreversibility 预测线。"""
    n_steps = 5
    n_models = len(MODELS)
    xs = np.arange(n_steps)
    bw = 0.4
    offsets = np.linspace(-(n_models-1)/2 * bw, (n_models-1)/2 * bw, n_models)

    for i, m in enumerate(MODELS):
        light, dark = MODEL_PAIR[m]
        vals = exp9.get(m, np.zeros(n_steps))
        ax.bar(xs + offsets[i], vals, bw,
               color=dark, edgecolor="white", linewidth=0.8,
               label=MODEL_LABELS[m])
        for xi, v in zip(xs, vals):
            if v > 0.02:
                ax.text(xi + offsets[i], v + 0.02, f"{v:.0%}",
                        ha="center", va="bottom",
                        fontsize=16, color=dark, fontweight="bold")


    naive = np.array(STEP_IRR, dtype=float)
    naive = naive / naive.max()
    ax.plot(xs, naive, color="#888888", linewidth=2.0,
            linestyle="--", marker="D", markersize=7,
            markerfacecolor="white", markeredgecolor="#888888",
            label='Naive P2 prediction\n(scaled by irreversibility)', zorder=10)


    for hi in [1, 4]:
        ax.axvspan(hi - 0.45, hi + 0.45, color="#FFEAA7", alpha=0.35, zorder=0)


    ax.text(1, 1.085, "authority laundering",
            ha="center", fontsize=16, color="#B0651D",
            fontweight="bold")
    ax.text(4, 1.085, "terminal output",
            ha="center", fontsize=16, color="#B0651D",
            fontweight="bold")


    for xi, irr in enumerate(STEP_IRR):
        ax.text(xi, -0.07, f"irr={irr}",
                ha="center", fontsize=16, color="#666",
                fontstyle="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels(STEP_SHORT, fontsize=16)
    ax.set_ylim(-0.10, 1.16)
    ax.set_xlim(-0.6, n_steps - 0.4)
    ax.set_ylabel("Final harm rate", fontsize=16)
    ax.set_xlabel("Injection point", fontsize=16, labelpad=18)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")

    ax.legend(loc="upper right", bbox_to_anchor=(0.6, 0.9),
              fontsize=14, frameon=True, framealpha=0.92,
              edgecolor="#CCC", ncol=1)
    ax.set_title("(C)  Step-targeted failure injection",
                 fontsize=16, fontweight="bold", pad=8, loc="left")



def main():
    """读取 Exp4/Exp5/Exp9 结果并输出 Figure 4；可切换单 judge 视图。"""
    recs4 = load("exp4_agent_compound.jsonl")
    recs5 = load("exp5_agent_jailbreak.jsonl")
    recs9 = load("exp9_step_injection.jsonl")

    cum4  = compute_cumulative(recs4)
    exp5  = compute_exp5(recs5)
    exp9  = compute_exp9(recs9)

    fig = plt.figure(figsize=(16 + 2 * max(0, len(MODELS) - 2), 7), facecolor="white")
    gs = gridspec.GridSpec(1, 3,
                            width_ratios=[1, 1.05 if len(MODELS) > 2 else 0.8, 2.15],
                            wspace=0.3,
                            left=0.1, right=0.9,
                            top=0.92, bottom=0.13)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_panel_A(ax_a, cum4)

    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_B(ax_b, exp5)

    ax_c = fig.add_subplot(gs[0, 2])
    draw_panel_C(ax_c, exp9)


    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"fig2_{len(MODELS)}models{JUDGE_SUFFIX}.{ext}"
        fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
