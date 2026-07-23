

import os
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import fig2_ci_preview as base



RESAMPLES = int(os.environ.get("BOOTSTRAP_RESAMPLES", "100000"))


def _excludes_zero(interval):
    lower, upper = interval
    return lower > 0 or upper < 0


def main():

    cumulative = base.preview_cumulative(
        base.load("exp4_agent_compound.jsonl"), resamples=RESAMPLES
    )

    changes = base.preview_agent_change(
        base.load("exp5_agent_jailbreak.jsonl"), resamples=RESAMPLES
    )

    injections = base.preview_step_injection(base.load("exp9_step_injection.jsonl"))


    fig = base.plt.figure(figsize=(16, 6.8), facecolor="white")
    grid = base.gridspec.GridSpec(
        1, 3, width_ratios=[1.05, 0.85, 2.0], wspace=0.27,
        left=0.055, right=0.99, top=0.91, bottom=0.16,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    base.draw_cumulative_preview(ax_a, cumulative)
    base.draw_agent_change_preview(ax_b, changes)
    base.draw_step_injection_preview(ax_c, injections)


    change_labels = [text for text in ax_b.texts if text.get_text().startswith("Δ=")]
    for label, model in zip(change_labels, base.MODELS):
        marker = " *" if _excludes_zero(changes[model]["change_ci"]) else " ns"
        label.set_text(label.get_text().replace("\n", f"{marker}\n", 1))


    fig.text(
        0.5, 0.03,
        "Shading/error bars show bootstrap or Wilson 95% CIs. * Paired change CI excludes 0; "
        "ns includes 0 (nominal, unadjusted).",
        ha="center", fontsize=10, color="#444444",
    )

    for extension in ("png", "pdf"):
        output = OUTPUT_DIR / f"figure4_with_inference{base.JUDGE_SUFFIX}.{extension}"
        fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"saved -> {output}")
    base.plt.close(fig)


if __name__ == "__main__":

    main()
