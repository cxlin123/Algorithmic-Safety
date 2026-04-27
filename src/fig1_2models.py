"""
Figure 1 (final): Cross-class safety interactions C1 × C2 × C3.

3 columns (one per model), each column has 3 stacked panels sharing the X axis
(5 jailbreak conditions):

  • Top    — dumbbell:  Pain Relief Strength (W vs B)   — Exp2
  • Middle — dumbbell:  Tone Dismissive      (W vs B)   — Exp2
  • Bottom — heatmap:   Hallucination / Harmful  × condition  — Exp1
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR

# ── constants ────────────────────────────────────────────────────────────
COND_ORDER  = ["baseline", "dan", "role_play", "pair", "indirect"]
COND_LABELS = ["Baseline", "DAN", "Role\nPlay", "PAIR", "Indirect"]
MODELS      = ["gpt", "claude"]
MODEL_LABELS = {"gpt": "GPT-5", "claude": "Claude-Sonnet-4-6", "llama": "LLaMA-3.1-70B"}

METRICS_HM   = ["hallucination", "harmful"]
METRIC_LBL_HM = ["Hallucination", "Harmful"]

# colour palettes
cmap_qwen = LinearSegmentedColormap.from_list("q",
    ['#523852','#6D4E6D','#886688','#A385A3','#BEA6BE','#D8C8D8','#F0EBF0'], N=256)
cmap_llama = LinearSegmentedColormap.from_list("l",
    ['#7A5D28','#967535','#B08D4A','#C5A56B','#D4BD92','#E3D5BA','#F2EDE3'], N=256)
cmap_mistral = LinearSegmentedColormap.from_list("m",
    ['#3F5D30','#587648','#718F62','#90A783','#AFBFA5','#CED8C8','#EDF0EB'], N=256)
MODEL_CMAPS = {
    "gpt":    cmap_qwen.reversed(),
    "claude": cmap_llama.reversed(),
    "llama":  cmap_mistral.reversed(),
}
# (light, dark) per model — for dumbbell W/B dots
MODEL_PAIR = {
    "gpt":    ("#BEA6BE", "#523852"),
    "claude": ("#D4BD92", "#7A5D28"),
    "llama":  ("#AFBFA5", "#3F5D30"),
}

# ── data loading ─────────────────────────────────────────────────────────
def load(fname):
    return [json.loads(l) for l in (RESULTS_DIR / fname).read_text().splitlines()
            if l.strip() and "error" not in l]


def compute_exp1_matrix(recs):
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        m = r.get("model_tag")
        c = r.get("condition")
        if not (m and c in COND_ORDER):
            continue
        jh    = r.get("judge_hallucination", {}) or {}
        jharm = r.get("judge_harmful", {}) or {}
        bucket[m][c]["hallucination"].append(jh.get("contains_hallucination", 0))
        bucket[m][c]["harmful"].append(jharm.get("harmful", 0))

    out = {}
    for m in MODELS:
        mat = np.zeros((len(METRICS_HM), len(COND_ORDER)))
        for ci, c in enumerate(COND_ORDER):
            for mi, met in enumerate(METRICS_HM):
                vals = bucket[m][c][met]
                mat[mi, ci] = statistics.mean(vals) if vals else 0.0
        out[m] = mat
    return out


def compute_exp2_age(recs, key):
    """Returns {model: {cond: {age: mean}}} for one bias key, by age group."""
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        m = r.get("model_tag")
        c = r.get("condition")
        age = r.get("age")
        if not (m and c in COND_ORDER and age in ("pediatric", "elderly")):
            continue
        jb = r.get("judge_bias", {}) or {}
        if key in jb:
            bucket[m][c][age].append(jb[key])
    out = {}
    for m in MODELS:
        out[m] = {}
        for c in COND_ORDER:
            out[m][c] = {
                age: statistics.mean(bucket[m][c][age]) if bucket[m][c][age] else None
                for age in ("pediatric", "elderly")
            }
    return out


# ── drawing helpers ──────────────────────────────────────────────────────
def draw_grouped_bars(ax, pain_data, treat_data, model):
    """
    Grouped bar chart: 5 condition groups × 2 bars per group.
    Bar height = (elderly − pediatric gap at this condition)
                 / (gap at baseline)
    so baseline = 1.0 by construction; jailbreak bars show fold-change.
    """
    light, dark = MODEL_PAIR[model]

    def gap(d, cond):
        p = d[cond]["pediatric"]
        e = d[cond]["elderly"]
        return (e - p) if (p is not None and e is not None) else 0.0

    pain_base  = gap(pain_data,  "baseline") or 1e-9
    treat_base = gap(treat_data, "baseline") or 1e-9
    pain_ratio  = [gap(pain_data,  c) / pain_base  for c in COND_ORDER]
    treat_ratio = [gap(treat_data, c) / treat_base for c in COND_ORDER]

    # align with heatmap cell centres (x = 0.5, 1.5, ..., 4.5)
    xs = np.arange(len(COND_ORDER)) + 0.5
    w  = 0.46

    bars1 = ax.bar(xs - w/2, pain_ratio,  w,
                    color=light, edgecolor="white", linewidth=0.9,
                    label="Pain Relief")
    bars2 = ax.bar(xs + w/2, treat_ratio, w,
                    color=dark,  edgecolor="white", linewidth=0.9,
                    label="Treatment Aggressiveness")

    # value labels on top of each bar
    for b in list(bars1):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.04,
                f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=16, color="#444")
    for b in list(bars2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.04,
                f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=16, color=dark,
                fontweight="bold")

    # baseline = 1.0 reference line
    ax.axhline(y=1.0, color="#888888", linewidth=0.9,
               linestyle="--", alpha=0.7, zorder=0)
    ax.set_xlim(0, len(COND_ORDER))
    ymax = max(max(pain_ratio), max(treat_ratio), 1.3) * 1.2
    ax.set_ylim(0, ymax+0.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([])
    if model == "gpt":
        ax.set_ylabel("Age Gap\n(× baseline)", fontsize=16, labelpad=4)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.set_facecolor("#FAFAFA")

    ax.legend(fontsize=16, loc="upper left", frameon=False, ncol=1)


def draw_dumbbell_combined(ax, pain_data, treat_data, model):
    """
    Combined panel: two dumbbell groups stacked on the same Y axis (0-10).
    For each condition x: pain pediatric vs elderly  +  treatment pediatric vs elderly.
    Shape: pediatric = circle, elderly = square.
    Pain dumbbell = lighter colour, Treatment dumbbell = darker colour.
    """
    light, dark = MODEL_PAIR[model]
    xs = np.arange(len(COND_ORDER))

    for xi, cond in enumerate(COND_ORDER):
        # ── Pain Relief (lower band of Y axis) ─────────────────────────
        p_ped = pain_data[cond]["pediatric"]
        p_eld = pain_data[cond]["elderly"]
        if p_ped is not None and p_eld is not None:
            ax.plot([xi, xi], [p_ped, p_eld],
                    color=light, linewidth=2.6, alpha=0.75, zorder=2)
            ax.scatter(xi, p_ped, s=180, marker="o",
                       color=light, edgecolors="#333333",
                       linewidths=1.2, zorder=4)
            ax.scatter(xi, p_eld, s=180, marker="s",
                       color=light, edgecolors="#333333",
                       linewidths=1.2, zorder=4)
            ax.text(xi - 0.22, p_ped, f"{p_ped:.1f}",
                    ha="right", va="center", fontsize=16, color="#444")
            ax.text(xi - 0.22, p_eld, f"{p_eld:.1f}",
                    ha="right", va="center", fontsize=16, color="#444")

        # ── Treatment Aggressiveness (upper band) ──────────────────────
        t_ped = treat_data[cond]["pediatric"]
        t_eld = treat_data[cond]["elderly"]
        if t_ped is not None and t_eld is not None:
            ax.plot([xi, xi], [t_ped, t_eld],
                    color=dark, linewidth=2.6, alpha=0.9, zorder=3)
            ax.scatter(xi, t_ped, s=180, marker="o",
                       color=dark, edgecolors="white",
                       linewidths=1.2, zorder=5)
            ax.scatter(xi, t_eld, s=180, marker="s",
                       color=dark, edgecolors="white",
                       linewidths=1.2, zorder=5)
            ax.text(xi + 0.22, t_ped, f"{t_ped:.1f}",
                    ha="left", va="center", fontsize=16,
                    color=dark, fontweight="bold")
            ax.text(xi + 0.22, t_eld, f"{t_eld:.1f}",
                    ha="left", va="center", fontsize=16,
                    color=dark, fontweight="bold")

    # ── divider line between the two bands ────────────────────────────
    ax.axhline(y=4.7, color="#BBBBBB", linewidth=0.6, linestyle="--", alpha=0.7)

    # ── band labels at top-left of each band ──────────────────────────
    ax.text(0.02, 0.97, "Treatment Aggressiveness",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=16, color=dark, fontweight="bold")
    ax.text(0.02, 0.46, "Pain Relief Strength",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=16, color="#555555", fontweight="bold")

    ax.set_xlim(-0.7, len(COND_ORDER) - 0.3)
    ax.set_ylim(2.5, 7.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([])
    ax.set_ylabel("Score (0-10)", fontsize=16, labelpad=4)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.set_facecolor("#FAFAFA")


def draw_heatmap(ax, matrix, cmap, show_ylabels=True):
    n_rows, n_cols = matrix.shape
    cell = 0.86
    pad  = (1 - cell) / 2

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            color = cmap(val)
            ax.add_patch(FancyBboxPatch(
                (j + pad, (n_rows - 1 - i) + pad),
                cell, cell,
                boxstyle="round,pad=0.0,rounding_size=0.18",
                facecolor=color, edgecolor="white", linewidth=1.4,
            ))
            text_color = "white" if val > 0.45 else "#222222"
            ax.text(j + 0.5, (n_rows - 1 - i) + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=16, color=text_color, fontweight="bold")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("auto")

    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(COND_LABELS, fontsize=16)
    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set_yticks([(n_rows - 1 - i) + 0.5 for i in range(n_rows)])
    if show_ylabels:
        ax.set_yticklabels(METRIC_LBL_HM, fontsize=16, fontweight="bold")
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# ── main figure ──────────────────────────────────────────────────────────
def main():
    recs1 = load("exp1_c1_to_c2.jsonl")
    recs2 = load("exp2_c1_to_c3.jsonl")

    hm_data    = compute_exp1_matrix(recs1)
    pain_data  = compute_exp2_age(recs2, "pain_relief_strength")
    treat_data = compute_exp2_age(recs2, "treatment_aggressiveness")

    fig = plt.figure(figsize=(15, 9), facecolor="white")
    FIG_OUT = "fig1_2models"
    outer = gridspec.GridSpec(
        2, 2,
        height_ratios=[2, 1.0],
        width_ratios=[1, 1],
        hspace=0.05, wspace=0.08,
        left=0.06, right=0.995,
        top=0.94, bottom=0.07,
    )

    db_axes, hm_axes = [], []
    for col, model in enumerate(MODELS):
        # ── bottom heatmap ──────────────────────────────────────────────
        ax_hm = fig.add_subplot(outer[1, col])
        draw_heatmap(ax_hm, hm_data[model], MODEL_CMAPS[model],
                     show_ylabels=(col == 0))
        ax_hm.set_xlabel("Jailbreak Condition", fontsize=16, labelpad=6)

        sm = plt.cm.ScalarMappable(cmap=MODEL_CMAPS[model],
                                    norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_hm, fraction=0.045, pad=0.02, aspect=12)
        cb.set_label("Rate", fontsize=16)
        cb.ax.tick_params(labelsize=8)
        # cb.outline.set_visible(False)

        # ── top: grouped bars ───────────────────────────────────────────
        ax_db = fig.add_subplot(outer[0, col])
        draw_grouped_bars(ax_db, pain_data[model], treat_data[model], model)
        ax_db.set_title(
            f"{MODEL_LABELS[model]}",
            fontsize=16, fontweight="bold", pad=12,
        )

        db_axes.append(ax_db)
        hm_axes.append(ax_hm)

    # ── force each top axes to share the bottom heatmap's x-extent ──────
    fig.canvas.draw()
    for ax_db, ax_hm in zip(db_axes, hm_axes):
        hm_pos = ax_hm.get_position()
        db_pos = ax_db.get_position()
        ax_db.set_position([hm_pos.x0, db_pos.y0, hm_pos.width, db_pos.height])


    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"fig1_2models.{ext}"
        fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"saved → {out}")


if __name__ == "__main__":
    main()
