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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR


JUDGE_VIEW = os.environ.get("JUDGE_VIEW", "").strip()
JUDGE_SUFFIX = f"_{JUDGE_VIEW}" if JUDGE_VIEW else ""

MODELS = [m.strip() for m in os.environ.get("FIG_MODELS", "gpt,claude").split(",") if m.strip()]
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
    rows = []
    for line in (RESULTS_DIR / fname).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" not in row:
            rows.append(row)
    return rows



def compute_exp7(recs):
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        m, sp = r.get("model_tag"), r.get("split")
        if m not in MODELS or sp not in ("ID", "OOD"):
            continue
        if JUDGE_VIEW:
            payload = (r.get("judges") or {}).get(JUDGE_VIEW) or {}
            hb = payload.get("halluc_base", {})
            hjb = payload.get("halluc_jb", {})
            hm = payload.get("harm_jb", {})
        else:
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
    xs = np.arange(len(MODELS))
    sub = 0.13

    for i, m in enumerate(MODELS):
        light, dark = MODEL_PAIR[m]
        h_id  = exp7[m]["ID"].get("hallu_base", 0)
        h_ood = exp7[m]["OOD"].get("hallu_base", 0)
        x_id  = exp7[m]["ID"].get("harm", 0)
        x_ood = exp7[m]["OOD"].get("harm", 0)

        def lbl_offsets(a, b, gap=0.05):
            if abs(a - b) >= gap:
                return 0.0, 0.0
            return (-gap/2, gap/2) if a < b else (gap/2, -gap/2)


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



def compute_exp8(recs):
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        m, c = r.get("model_tag"), r.get("condition")
        if m not in MODELS or c not in EXP8_CONDS:
            continue
        if JUDGE_VIEW:
            payload = (r.get("judges") or {}).get(JUDGE_VIEW) or {}
            p_safe = (payload.get("step") or {}).get("p_safe")
            harmful = (payload.get("harm") or {}).get("harmful")
        else:
            p_safe = r.get("p_safe")
            harmful = r.get("harmful")
        if p_safe is not None:
            by[m][c]["p_safe"].append(p_safe)
        if harmful is not None:
            by[m][c]["harm"].append(harmful)
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
                label="Harm rate", zorder=5)

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



def main():
    recs8 = load("exp8_principle3.jsonl")
    exp8  = compute_exp8(recs8)


    fig = plt.figure(figsize=(6 * len(MODELS), 4.5), facecolor="white")
    gs  = gridspec.GridSpec(
        1, len(MODELS),
        wspace=0.08,
        left=0.06, right=0.995,
        top=0.92, bottom=0.14,
    )
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(len(MODELS))]

    draw_panel_B(axes, exp8)

    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"fig3_{len(MODELS)}models{JUDGE_SUFFIX}.{ext}"
        fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"saved -> {out}")


def preview_wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def bootstrap_mean_interval(values, seed, resamples=100_000):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    chunks = []
    chunk_size = 5_000
    for start in range(0, resamples, chunk_size):
        size = min(chunk_size, resamples - start)
        indices = rng.integers(0, array.size, size=(size, array.size))
        chunks.append(array[indices].mean(axis=1))
    bootstrap = np.concatenate(chunks)
    return tuple(np.quantile(bootstrap, [0.025, 0.975]))


def preview_exp8_summary(recs):
    output = defaultdict(dict)
    for model_index, model in enumerate(MODELS):
        for condition_index, condition in enumerate(EXP8_CONDS):
            selected = [
                row
                for row in recs
                if row.get("model_tag") == model and row.get("condition") == condition
            ]
            psafe = [float(row["p_safe"]) for row in selected if row.get("p_safe") is not None]
            harmful = [int(row["harmful"]) for row in selected if row.get("harmful") is not None]
            output[model][condition] = {
                "psafe": statistics.mean(psafe),
                "psafe_ci": bootstrap_mean_interval(
                    psafe, 20_260_730 + model_index * 10 + condition_index
                ),
                "harm": statistics.mean(harmful),
                "harm_ci": preview_wilson(sum(harmful), len(harmful)),
                "n": len(selected),
            }

        blind = {
            row["question"]: float(row["p_safe"])
            for row in recs
            if row.get("model_tag") == model
            and row.get("condition") == "blind"
            and row.get("question")
            and row.get("p_safe") is not None
        }
        shown = {
            row["question"]: float(row["p_safe"])
            for row in recs
            if row.get("model_tag") == model
            and row.get("condition") == "rubric_shown"
            and row.get("question")
            and row.get("p_safe") is not None
        }
        matched = sorted(blind.keys() & shown.keys())
        differences = np.asarray([shown[item] - blind[item] for item in matched])
        output[model]["paired"] = {
            "change": differences.mean(),
            "ci": bootstrap_mean_interval(differences, 20_260_750 + model_index),
            "n": differences.size,
        }
    return output


def draw_exp8_ci_preview(ax, summary, model, panel_letter):
    light, dark = MODEL_PAIR[model]
    x = np.arange(len(EXP8_CONDS))
    psafe = np.asarray([summary[model][condition]["psafe"] for condition in EXP8_CONDS])
    psafe_lower = np.asarray([summary[model][condition]["psafe_ci"][0] for condition in EXP8_CONDS])
    psafe_upper = np.asarray([summary[model][condition]["psafe_ci"][1] for condition in EXP8_CONDS])
    harm = np.asarray([summary[model][condition]["harm"] for condition in EXP8_CONDS])
    harm_lower = np.asarray([summary[model][condition]["harm_ci"][0] for condition in EXP8_CONDS])
    harm_upper = np.asarray([summary[model][condition]["harm_ci"][1] for condition in EXP8_CONDS])

    ax.plot(x, psafe, color=light, linewidth=2.5, zorder=2)
    ax.errorbar(
        x,
        psafe,
        yerr=np.vstack((psafe - psafe_lower, psafe_upper - psafe)),
        fmt="o",
        color=light,
        markerfacecolor="white",
        markeredgewidth=2,
        markersize=9,
        capsize=4,
        elinewidth=1.6,
        label="$p_{safe}$ (bootstrap 95% CI)",
        zorder=4,
    )
    ax.plot(x, harm, color=dark, linewidth=2.5, linestyle="--", zorder=2)
    ax.errorbar(
        x,
        harm,
        yerr=np.vstack((harm - harm_lower, harm_upper - harm)),
        fmt="s",
        color=dark,
        markerfacecolor="white",
        markeredgewidth=2,
        markersize=9,
        capsize=4,
        elinewidth=1.6,
        label="Harmful response (Wilson 95% CI)",
        zorder=4,
    )

    for xi, value in zip(x, psafe):
        ax.text(xi, value + 0.085, f"{value:.2f}", ha="center",
                fontsize=12, color=light, fontweight="bold")
    for xi, value in zip(x, harm):
        ax.text(xi, max(-0.01, value - 0.14), f"{value:.0%}", ha="center",
                fontsize=12, color=dark, fontweight="bold")

    paired = summary[model]["paired"]
    lower, upper = paired["ci"]
    bracket_y = 1.13
    ax.plot([0, 0, 1, 1], [bracket_y - 0.025, bracket_y, bracket_y, bracket_y - 0.025],
            color="#555", linewidth=1.2)
    ax.text(
        0.5,
        bracket_y + 0.018,
        f"paired Δ={paired['change']:+.3f} [{lower:+.3f}, {upper:+.3f}]\nN={paired['n']}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333",
    )

    ax.set_xticks(x, EXP8_LABELS)
    ax.set_xlim(-0.42, 2.42)
    ax.set_ylim(-0.08, 1.27)
    ax.set_ylabel("Score / rate")
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.set_title(f"({panel_letter}) {MODEL_LABELS[model]}", loc="left", fontweight="bold")


def main():
    summary = preview_exp8_summary(load("exp8_principle3.jsonl"))
    fig = plt.figure(figsize=(12.5, 5.3), facecolor="white")
    grid = gridspec.GridSpec(
        1, len(MODELS), wspace=0.12,
        left=0.07, right=0.99, top=0.91, bottom=0.14,
    )
    for index, model in enumerate(MODELS):
        draw_exp8_ci_preview(
            fig.add_subplot(grid[0, index]), summary, model, chr(ord("A") + index)
        )
    for extension in ("png", "pdf"):
        output = FIGURES_DIR / f"fig3_ci_preview.{extension}"
        fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
        print(f"saved -> {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
