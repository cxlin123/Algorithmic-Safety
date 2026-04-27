"""
Figure 2 — Compound Agent Risk and Unrecoverability-aware Principle 2.

Three panels:
  (A) Exp4  Cumulative Psafe decay         — C4 mathematical baseline
  (B) Exp5  Clean → Attacked Psafe dumbbell — C1 → C4 (76-98% collapse)
  (C) Exp9  Step-targeted injection harm    — Revised P2 evidence
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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR

MODELS       = ["gpt", "claude"]
MODEL_LABELS = {"gpt": "GPT-5", "claude": "Claude-Sonnet-4-6", "llama": "LLaMA-3.1-70B"}
MODEL_SHORT  = {"gpt": "GPT-5", "claude": "Claude", "llama": "LLaMA"}

# (light, dark) per model
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
    return [json.loads(l) for l in (RESULTS_DIR / fname).read_text().splitlines()
            if l.strip() and "error" not in l]


# ──────────────────────────────────────────────────────────────────
# Panel A — Cumulative Psafe (Exp4)
# ──────────────────────────────────────────────────────────────────
def compute_cumulative(recs):
    out = {}
    for m in MODELS:
        per_step = [r["per_step_psafe"] for r in recs
                    if r.get("model_tag") == m and len(r.get("per_step_psafe", [])) == 5]
        if not per_step:
            continue
        arr = np.array(per_step)
        step_means = arr.mean(axis=0)
        out[m] = np.cumprod(step_means)
    return out


def draw_panel_A(ax, cum):
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
        # for x, y in zip(xs, ys):
        #     ax.text(x, y - 0.022, f"{y:.3f}",
        #             ha="center", va="top",
        #             fontsize=16, color=dark, fontweight="bold")
        for x, y in zip(xs, ys):                                      
            if m == "gpt":            # 紫色 → 上
                ax.text(x, y + 0.015, f"{y:.3f}",                     
                        ha="center", va="bottom",                     
                        fontsize=16, color=dark, fontweight="bold")   
            else:                      # 其他 → 下                    
                ax.text(x, y - 0.015, f"{y:.3f}",                    
                        ha="center", va="top",                        
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


# ──────────────────────────────────────────────────────────────────
# Panel B — Clean vs Attacked Psafe dumbbell (Exp5)
# ──────────────────────────────────────────────────────────────────
def compute_exp5(recs):
    out = {}
    for m in MODELS:
        data = [r for r in recs if r.get("model_tag") == m]
        if not data:
            continue
        clean = statistics.mean(r["clean_psafe"] for r in data)
        atk   = statistics.mean(r["attacked_psafe"] for r in data)
        out[m] = (clean, atk, len(data))
    return out


def draw_panel_B(ax, exp5):
    xs = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        if m not in exp5:
            continue
        light, dark = MODEL_PAIR[m]
        clean, atk, n = exp5[m]
        drop_pct = 100 * (clean - atk) / clean

        # vertical line connecting clean (top) and attacked (bottom)
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
        # drop annotation in middle of the dumbbell line
        mid = (clean + atk) / 2
        ax.text(i, mid, f"↓{drop_pct:.0f}%",
                ha="right", va="center", fontsize=16,
                color="#C0392B", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS],
                       fontsize=16)
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


# ──────────────────────────────────────────────────────────────────
# Panel C — Step-targeted injection harm (Exp9)
# ──────────────────────────────────────────────────────────────────
def compute_exp9(recs):
    bucket = defaultdict(lambda: defaultdict(list))
    for r in recs:
        m = r.get("model_tag")
        s = r.get("inject_at")
        if m in MODELS and s in range(5):
            bucket[m][s].append(r["final_harmful"])
    out = {}
    for m in MODELS:
        out[m] = np.array([
            (statistics.mean(bucket[m][s]) if bucket[m][s] else 0)
            for s in range(5)
        ])
    return out


def draw_panel_C(ax, exp9):
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

    # ── naive irreversibility prediction (normalised) ────────────────
    naive = np.array(STEP_IRR, dtype=float)
    naive = naive / naive.max()
    ax.plot(xs, naive, color="#888888", linewidth=2.0,
            linestyle="--", marker="D", markersize=7,
            markerfacecolor="white", markeredgecolor="#888888",
            label='Naive P2 prediction\n(scaled by irreversibility)', zorder=10)

    # ── highlight S2 and S5 ──────────────────────────────────────────
    for hi in [1, 4]:
        ax.axvspan(hi - 0.45, hi + 0.45, color="#FFEAA7", alpha=0.35, zorder=0)

    # highlight labels positioned just under the title baseline
    ax.text(1, 1.085, "authority laundering",
            ha="center", fontsize=16, color="#B0651D",
            fontweight="bold")
    ax.text(4, 1.085, "terminal output",
            ha="center", fontsize=16, color="#B0651D",
            fontweight="bold")

    # irr labels
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
    # legend in the empty top-right area (around S3-S4, where bars are tiny)
    ax.legend(loc="upper right", bbox_to_anchor=(0.6, 0.9),
              fontsize=14, frameon=True, framealpha=0.92,
              edgecolor="#CCC", ncol=1)
    ax.set_title("(C)  Step-targeted failure injection",
                 fontsize=16, fontweight="bold", pad=8, loc="left")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    recs4 = load("exp4_agent_compound.jsonl")
    recs5 = load("exp5_agent_jailbreak.jsonl")
    recs9 = load("exp9_step_injection.jsonl")

    cum4  = compute_cumulative(recs4)
    exp5  = compute_exp5(recs5)
    exp9  = compute_exp9(recs9)

    fig = plt.figure(figsize=(16, 7), facecolor="white")
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.8, 2],
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
        out = FIGURES_DIR / f"fig2_2models.{ext}"
        fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"saved → {out}")


if __name__ == "__main__":
    main()
