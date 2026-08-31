"""The three prompt versions side by side, ordered by how visible the target is.

Reads analysis/target_visibility/results.json, which already holds the shift between the two arms and
its 95% interval for every prompt family. Nothing is recomputed here and no API
calls are made, so this is free to re-render.

  uv run python analysis/target_visibility/plot_ladder.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import palette as PAL  # noqa: E402
from value_leakage.plot import INK, INK_MUTED  # noqa: E402

TARGET_VISIBILITY_DIR = HERE

# Ordered by how plainly the model can see what it is supposed to aim at.
# Names match the prompt-versions figure exactly.
FAMILIES = [
    ("instructed", "Instructed (told to aim)"),
    ("bet", "Bet (threshold shown)"),
    ("hidden", "Bet (threshold withheld)"),
]
# One neutral hue for every bar, and the two models told apart by fill.
#
# Deliberately NOT indigo-for-instructed / amber-for-bet. In the palette those
# two hues mean below pays and above pays, and the quantity plotted here is
# computed from both of those sides at once, so neither hue belongs to any bar.
# Borrowing them for prompt conditions would give the same two colours a third
# meaning across the figure set. Indigo alone is the primary mark, matching the
# bias dots in the disclosure figure, which plot this same quantity.
BAR = PAL.BELOW

MODEL_LABEL = {"qwen3.5-122b-a10b": "qwen3.5-122b-a10b",
               "glm-5p2": "glm-5p2"}


def main():
    res = json.loads((TARGET_VISIBILITY_DIR / "results.json").read_text())
    models = [m for m in ("qwen3.5-122b-a10b", "glm-5p2") if m in res]

    fig, ax = plt.subplots(figsize=(10.6, 6.0))

    width = 0.34
    xs = np.arange(len(FAMILIES))
    for j, model in enumerate(models):
        fams = res[model]["families"]
        off = (j - (len(models) - 1) / 2) * width
        for k, (fam, _) in enumerate(FAMILIES):
            e = fams.get(fam)
            if not e:
                continue
            x = xs[k] + off
            # Second model drawn hollow, so the two are never told apart by
            # colour alone.
            solid = (j == 0)
            ax.bar(x, e["separation"], width=width * 0.88,
                   color=BAR if solid else "white",
                   edgecolor=BAR, linewidth=1.4,
                   hatch=None if solid else "///", zorder=2)
            yerr = np.array([[e["separation"] - e["ci95_low"]],
                             [e["ci95_high"] - e["separation"]]])
            ax.errorbar(x, e["separation"], yerr=yerr, fmt="none",
                        ecolor=BAR, elinewidth=1.3, capsize=4,
                        capthick=1.1, alpha=0.55, zorder=3)
            # Print the measured shift on the bar itself so this summary maps
            # directly onto the same values unpacked in the trajectory figure.
            # Values at zero sit just above the baseline because there is no
            # bar interior available for a label.
            value = e["separation"]
            if value > 0.08:
                label_y, va = value - 0.035, "top"
            else:
                label_y, va = value + 0.025, "bottom"
            ax.text(
                x, label_y, f"{value:.2f}", ha="center", va=va,
                fontsize=9.5, fontweight="bold",
                color="white" if solid else BAR, zorder=4,
                bbox=dict(facecolor=BAR if solid else "white",
                          edgecolor="none", alpha=0.92, pad=1.2),
            )
            # Above the whisker, not below the bar — the intervals reach under
            # zero and would sit on top of the label there.
            n = e["n_above"] + e["n_below"]
            ax.text(x, e["ci95_high"] + 0.035, f"n={n}",
                    ha="center", va="bottom", fontsize=8.5, color=INK_MUTED)

    ax.axhline(0, color=INK_MUTED, linewidth=1.1, linestyle="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in FAMILIES],
                       fontsize=11, color=INK)
    ax.tick_params(axis="x", length=0, pad=14)
    ax.set_xlim(-0.62, len(FAMILIES) - 0.38)
    ax.set_ylim(-0.18, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # "Shift" rather than a new word: the prompt-versions figure already labels
    # this same quantity that way in its panel subtitles.
    ax.set_ylabel("Shift in answers\n"
                  "(0 = same either way, 1 = every run follows the prompt)",
                  fontsize=11, color=INK)

    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=11,
               markerfacecolor=INK_MUTED, markeredgecolor=INK_MUTED,
               label=MODEL_LABEL[models[0]]),
    ]
    if len(models) > 1:
        handles.append(Line2D([], [], marker="s", linestyle="none",
                              markersize=11, markerfacecolor="white",
                              markeredgecolor=INK_MUTED,
                              label=MODEL_LABEL[models[1]]))
    ax.legend(handles=handles, fontsize=9.5, frameon=False, loc="upper right")

    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK_MUTED)

    # The title says what is plotted, not how the bars are arranged. The method
    # text that used to sit in a footnote lives in the report caption instead.
    fig.suptitle("How much the answers shift when the prompt flips from below "
                 "to above the threshold,\nin three versions of the Donation "
                 "Bet",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985,
                 linespacing=1.5)
    fig.subplots_adjust(top=0.845, bottom=0.115, left=0.105, right=0.985)
    out = TARGET_VISIBILITY_DIR / "ladder.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight",
                pad_inches=0.3)
    fig.savefig(TARGET_VISIBILITY_DIR / "ladder.svg", facecolor="white", bbox_inches="tight",
                pad_inches=0.3)
    plt.close(fig)
    print(f"saved {out.relative_to(TARGET_VISIBILITY_DIR.parent)}")

    for model in models:
        fams = res[model]["families"]
        row = "  ".join(f"{f}={fams[f]['separation']:+.3f}"
                        for f, _ in FAMILIES if f in fams)
        print(f"  {model:20s} {row}")


if __name__ == "__main__":
    main()
