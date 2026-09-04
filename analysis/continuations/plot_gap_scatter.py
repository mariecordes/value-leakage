"""The interruption scatter, on its own — for the split two-figure layout.

Same panel as the left half of analysis/continuations/analysis.py, pulled out so it can stand
alone next to the box-plot figure (analysis/continuations/plot_gap_bars.py) instead of being welded
to it. One dot per interruption: what the model stated when its reasoning was
stopped on the horizontal axis, where the continuation from that same point
ended up on the vertical. A dot on the diagonal means the two agree.

Reads analysis/continuations/results.json, so it costs nothing to re-render.

  uv run python analysis/continuations/plot_gap_scatter.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "disclosure"))
import palette as PAL  # noqa: E402
from value_leakage.plot import INK, INK_MUTED  # noqa: E402

CONTINUATION_DIR = HERE

# Four series: arm x which side pays. Reuses the palette's two poles and their
# lighter steps, so nothing new is introduced.
# This figure's primary comparison is bet against instructed — the instructed arm is
# the control that licenses reading the bet arm's result — so that goes in hue. The
# side each prompt rewards is the second split, and takes fill. See analysis/palette.py.
SERIES_COLOR = PAL.FAMILY
SERIES_LABEL = {("bet", "above"): "bet, above pays",
                ("bet", "below"): "bet, below pays",
                ("instructed", "above"): "instructed, aim above",
                ("instructed", "below"): "instructed, aim below"}


def main(model: str = "qwen3.5-122b-a10b"):
    res = json.loads((CONTINUATION_DIR / "results.json").read_text())
    rows = res["rows"]

    # Squarer than the two-panel version: on its own the diagonal is the point
    # of the plot, and a diagonal only reads correctly on equal-ish axes.
    fig, ax = plt.subplots(figsize=(8.6, 7.4))

    for arm in ("bet", "instructed"):
        for side in ("above", "below"):
            sel = [r for r in rows if r["arm"] == arm
                   and r["version"].endswith("_" + side)]
            if not sel:
                continue
            # Hue is the prompt family; the side each prompt rewards is the
            # second split and takes a lighter fill. Same size and the same white
            # edge throughout, so no series is visually heavier than another.
            fill = (SERIES_COLOR[arm] if side == "above"
                    else PAL.FAMILY_LIGHT[arm])
            ax.scatter([r["stated_rel"] for r in sel],
                       [r["landing_rel"] for r in sel],
                       s=78, facecolor=fill, edgecolor="white",
                       linewidth=1.0, alpha=0.8,
                       label=SERIES_LABEL[(arm, side)])

    # Symlog on both axes: linear where the bulk lives, compressed in the tail,
    # so the handful of very large estimates stay on the plot instead of being
    # cut. Excluding them would be actively misleading — those runs agree with
    # themselves better than average, so hiding them understates the result.
    lim = [-0.35, 7.0]
    for axis in ("x", "y"):
        getattr(ax, f"set_{axis}scale")("symlog", linthresh=0.3, linscale=1.1)
    ticks = [-0.2, 0, 0.2, 0.5, 1, 2, 5]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax.plot(lim, lim, color=INK_MUTED, linewidth=1.1, linestyle="--", zorder=0)
    ax.text(4.4, 4.4, "no difference  ", fontsize=9.5, color=INK_MUTED,
            va="bottom", ha="right", rotation=39, rotation_mode="anchor")
    ax.axhline(0, color=INK_MUTED, linewidth=0.7, alpha=0.45, zorder=0)
    ax.axvline(0, color=INK_MUTED, linewidth=0.7, alpha=0.45, zorder=0)
    ax.axvspan(-0.3, 0.3, color=INK_MUTED, alpha=0.045, zorder=0)
    ax.axhspan(-0.3, 0.3, color=INK_MUTED, alpha=0.045, zorder=0)
    ax.text(0.02, 0.988, "shaded band: both axes are linear inside ±0.3, "
            "compressed outside", transform=ax.transAxes, va="top",
            fontsize=8.5, color=INK_MUTED)

    ax.set_xlabel("Normalized estimate the model stated when stopped\n"
                  "(0 = threshold)",
                  fontsize=11, color=INK)
    ax.set_ylabel("Normalized estimate after continuation\n(same scale)",
                  fontsize=11, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK_MUTED)

    fig.suptitle("Each interruption: the estimate the model stated when "
                 "stopped,\nand the estimate it reached after continuing",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.985)
    fig.subplots_adjust(top=0.87, bottom=0.125, left=0.105, right=0.985)
    out = CONTINUATION_DIR / f"stated_vs_resampled_scatter_{model}.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight",
                pad_inches=0.3)
    fig.savefig(out.with_suffix(".svg"), facecolor="white", bbox_inches="tight",
                pad_inches=0.3)
    plt.close(fig)
    print(f"saved {out.relative_to(CONTINUATION_DIR.parent)}")


if __name__ == "__main__":
    main()
