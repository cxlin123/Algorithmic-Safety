"""
Figure 3 — Distribution shift × Transparency paradox.

Panel A (Exp7) — distribution shift
  Two sub-axes:
    A1: baseline hallucination rate, ID vs OOD per model
    A2: jailbreak harm rate,        ID vs OOD per model

Panel B (Exp8) — transparency paradox
  Three sub-axes (one per model). Single Y axis 0–1, two lines per axis:
    p_safe   (solid, light, circle)
    harm     (dashed, dark, square)
  X = {Blind, Rubric Shown, Adversarial}
"""
from __future__ import annotations
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16, "axes.titlesize": 16,
                     "axes.labelsize": 16, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "legend.fontsize": 16})
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR

MODELS = ["gpt", "claude"]
MODEL_LABELS = {"gpt": "GPT-5", "claude": "Claude-Sonnet-4-6", "llama": "LLaMA-3.1-70B"}
MODEL_SHORT  = {"gpt": "GPT-5", "claude": "Claude", "llama": "LLaMA"}
MODEL_PAIR = {
    "gpt":    ("#BEA6BE", "#523852"),
    "claude": ("#D4BD92", "#7A5D28"),
    "llama":  ("#AFBFA5", "#3F5D30"),
}

EXP8_CONDS  = ["blind", "rubric_shown", "adversarial"]
EXP8_LABELS = ["Blind", "Rubric\nShown", "Adversarial"]


def load(fname):
    return [json.loads(l) for l in (RESULTS_DIR / fname).read_text().splitlines()
            if l.strip() and "error" not in l]


# ──────────────────────────────────────────────────────────────────
# Exp7 data: ID vs OOD by model
# ──────────────────────────────────────────────────────────────────
def compute_exp7(recs):
    """{model: {split: {metric: rate}}}"""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        m, sp = r.get("model_tag"), r.get("split")
        if m not in MODELS or sp not in ("ID", "OOD"):
            continue
        hb  = r.get("halluc_base", {})
        hjb = r.get("halluc_jb", {})
        hm  = r.get("harm_jb", {})
        if isinstance(hb, dict):
            by[m][sp]["hallu_base"].append(hb.get("contains_hallucination", 0))
        if isinstance(hjb, dict):
            by[m][sp]["hallu_jb"].append(hjb.get("contains_hallucination", 0))
        if isinstance(hm, dict):
            by[m][sp]["harm"].append(hm.get("harmful", 0))

    out = {}
    for m in MODELS:
        out[m] = {}
        for sp in ("ID", "OOD"):
            out[m][sp] = {
                k: (statistics.mean(v) if v else 0)
                for k, v in by[m][sp].items()
            }
    return out


def draw_panel_A(ax, exp7):
    """Merged dumbbell: hallu (○) and harm (□), light=ID, dark=OOD."""
    xs = np.arange(len(MODELS))
    sub = 0.13  # tighter spacing within each model

    for i, m in enumerate(MODELS):
        light, dark = MODEL_PAIR[m]
        h_id  = exp7[m]["ID"].get("hallu_base", 0)
        h_ood = exp7[m]["OOD"].get("hallu_base", 0)
        x_id  = exp7[m]["ID"].get("harm", 0)
        x_ood = exp7[m]["OOD"].get("harm", 0)

        def lbl_offsets(a, b, gap=0.05):
            """Return (off_a, off_b): if a,b too close, push them apart vertically."""
            if abs(a - b) >= gap:
                return 0.0, 0.0
            return (-gap/2, gap/2) if a < b else (gap/2, -gap/2)

        # left dumbbell: hallucination (circles)
        x_left = xs[i] - sub
        ax.plot([x_left, x_left], [h_id, h_ood], color=dark, linewidth=2.2, alpha=0.6, zorder=2)
        ax.scatter(x_left, h_id, s=170, marker="o", color=light,
                   edgecolors="#222", linewidths=1.2, zorder=4)
        ax.scatter(x_left, h_ood, s=170, marker="o", color=dark,
                   edgecolors="white", linewidths=1.2, zorder=5)
        oid, ood = lbl_offsets(h_id, h_ood)
        ax.text(x_left - 0.05, h_id + oid, f"{h_id:.0%}",
                ha="right", va="center", fontsize=16, color="#444")
        ax.text(x_left - 0.05, h_ood + ood, f"{h_ood:.0%}",
                ha="right", va="center", fontsize=16, color=dark, fontweight="bold")

        # right dumbbell: harm (squares)
        x_right = xs[i] + sub
        ax.plot([x_right, x_right], [x_id, x_ood], color=dark, linewidth=2.2, alpha=0.6, zorder=2)
        ax.scatter(x_right, x_id, s=170, marker="s", color=light,
                   edgecolors="#222", linewidths=1.2, zorder=4)
        ax.scatter(x_right, x_ood, s=170, marker="s", color=dark,
                   edgecolors="white", linewidths=1.2, zorder=5)
        oid, ood = lbl_offsets(x_id, x_ood)
        ax.text(x_right + 0.05, x_id + oid, f"{x_id:.0%}",
                ha="left", va="center", fontsize=16, color="#444")
        ax.text(x_right + 0.05, x_ood + ood, f"{x_ood:.0%}",
                ha="left", va="center", fontsize=16, color=dark, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS],
                       fontsize=16, fontweight="bold")
    ax.set_ylim(0., 0.9)
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylabel("Rate", fontsize=16, labelpad=4)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#CCC",
               markeredgecolor="#222", markersize=10, label="Hallucination"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#CCC",
               markeredgecolor="#222", markersize=10, label="Harm (jailbreak)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="#222", markersize=10, label="ID (light) / OOD (dark)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=14,
              frameon=True, framealpha=0, edgecolor="#CCC")
    ax.set_title("(A)  Distribution shift: ID vs OOD",
                 fontsize=16, fontweight="bold", pad=8, loc="left")


# ──────────────────────────────────────────────────────────────────
# Exp8 data: transparency paradox
# ──────────────────────────────────────────────────────────────────
def compute_exp8(recs):
    """{model: {cond: {p_safe, harm}}}"""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        m, c = r.get("model_tag"), r.get("condition")
        if m not in MODELS or c not in EXP8_CONDS:
            continue
        if r.get("p_safe")  is not None:
            by[m][c]["p_safe"].append(r["p_safe"])
        if r.get("harmful") is not None:
            by[m][c]["harm"].append(r["harmful"])
    out = {}
    for m in MODELS:
        out[m] = {}
        for c in EXP8_CONDS:
            out[m][c] = {
                k: (statistics.mean(v) if v else 0)
                for k, v in by[m][c].items()
            }
    return out


def draw_panel_B(ax_list, exp8):
    xs = np.arange(len(EXP8_CONDS))
    panel_letters = ["A", "B", "C"]
    for idx, (ax, m) in enumerate(zip(ax_list, MODELS)):
        light, dark = MODEL_PAIR[m]
        ps   = [exp8[m][c]["p_safe"] for c in EXP8_CONDS]
        harm = [exp8[m][c]["harm"]   for c in EXP8_CONDS]

        ax.plot(xs, ps, color=light, linewidth=2.6, marker="o",
                markersize=10, markerfacecolor="white",
                markeredgecolor=light, markeredgewidth=2.2,
                label="$p_{safe}$ (judge score)", zorder=4)
        ax.plot(xs, harm, color=dark, linewidth=2.6, linestyle="--",
                marker="s", markersize=10, markerfacecolor="white",
                markeredgecolor=dark, markeredgewidth=2.2,
                label="Harm rate (real)", zorder=5)

        for x, p, h in zip(xs, ps, harm):
            ax.text(x, p + 0.10, f"{p:.2f}",
                    ha="center", va="bottom", fontsize=16,
                    color=light, fontweight="bold")
            ax.text(x, h - 0.10, f"{h:.0%}",
                    ha="center", va="top", fontsize=16,
                    color=dark, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(EXP8_LABELS, fontsize=16)
        ax.set_ylim(-0.22, 1.30)
        ax.set_xlim(-0.4, len(EXP8_CONDS) - 0.6)
        if idx == 0:
            ax.set_ylabel("Score / Rate", fontsize=16, labelpad=4)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        ax.legend(loc="lower left", fontsize=14,
                  frameon=True, framealpha=0, edgecolor="#CCC")
        ax.set_title(f"({panel_letters[idx]})  {MODEL_LABELS[m]}",
                     fontsize=16, fontweight="bold", pad=8, loc="left")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    recs8 = load("exp8_principle3.jsonl")
    exp8  = compute_exp8(recs8)

    # Single row: B1 and B2 (transparency paradox per model), no Panel A.
    fig = plt.figure(figsize=(12, 4.5), facecolor="white")
    gs  = gridspec.GridSpec(
        1, 2,
        wspace=0.08,
        left=0.06, right=0.995,
        top=0.92, bottom=0.14,
    )
    ax_b1 = fig.add_subplot(gs[0, 0])
    ax_b2 = fig.add_subplot(gs[0, 1])

    draw_panel_B([ax_b1, ax_b2], exp8)

    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"fig3_2models.{ext}"
        fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"saved → {out}")


if __name__ == "__main__":
    main()
