"""Alternative layout for the interruption figure.

Two subplots, one per arm. Within each, a pair of boxes at every cut: the
estimate the model stated when its reasoning was stopped there, and the
estimate its continuations reached from that same point. Plus a 100% column
showing where the original, uninterrupted traces ended — one box, because at
100% there is nothing left to state, the trace is over.

This keeps all four series the earlier line version had, and shows their spread
instead of a single median, so one extreme run cannot pull a line into a peak.

Stands on its own, and also works as the second half of the split layout, with
analysis/continuations/plot_gap_scatter.py as the first.

  uv run python analysis/continuations/plot_gap_bars.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "disclosure"))
sys.path.insert(0, str(HERE.parent / "target_visibility"))
import palette as PAL  # noqa: E402
from value_leakage.plot import INK, INK_MUTED  # noqa: E402

CONTINUATION_DIR = HERE
ARM_COLOR = {"bet": PAL.ABOVE, "instructed": PAL.BELOW}
ARM_LIGHT = {"bet": PAL.AMBER_LT, "instructed": PAL.INDIGO_LT}
ARM_TITLE = {"bet": "Bet", "instructed": "Instructed (told to aim)"}


def sgn(version: str) -> float:
    """Sign everything toward whichever side that prompt rewards, so the two
    directions add instead of cancelling."""
    return 1.0 if version.endswith("_above") else -1.0


def main(model: str = "qwen3.5-122b-a10b", clip: float = 6.0):
    res = json.loads((CONTINUATION_DIR / "results.json").read_text())
    rows, thr = res["rows"], res["threshold"]
    sel_file = json.loads((CONTINUATION_DIR / "selection.json").read_text())
    cuts = sorted({r["cut_frac"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)

    for ax, arm in zip(axes, ("bet", "instructed")):
        data, positions, kinds = [], [], []
        for k, c in enumerate(cuts):
            sel = [r for r in rows if r["arm"] == arm and r["cut_frac"] == c]
            if not sel:
                continue
            data.append([r["stated_rel"] * sgn(r["version"]) for r in sel])
            positions.append(k - 0.17)
            kinds.append("said")
            data.append([r["landing_rel"] * sgn(r["version"]) for r in sel])
            positions.append(k + 0.17)
            kinds.append("landed")
        # 100%: where the original traces ended, uninterrupted. One box only —
        # at 100% there is nothing left to say, the trace is finished.
        finals = [r["final_rel"] * sgn(r["version"])
                  for a, e in sel_file["arms"].items() if a == arm
                  for r in e["runs"]]
        if finals:
            data.append(finals)
            positions.append(len(cuts))
            kinds.append("landed")

        bp = ax.boxplot(data, positions=positions, widths=0.3,
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="white", linewidth=2),
                        flierprops=dict(marker="o", markersize=4.5,
                                        markerfacecolor="none", alpha=0.75),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
        for patch, kind in zip(bp["boxes"], kinds):
            col = ARM_LIGHT[arm] if kind == "said" else ARM_COLOR[arm]
            patch.set_facecolor(col)
            patch.set_edgecolor(ARM_COLOR[arm])
            patch.set_alpha(0.9)
        for part in ("whiskers", "caps"):
            for artist in bp[part]:
                artist.set_color(ARM_COLOR[arm])
        for flier in bp["fliers"]:
            flier.set_markeredgecolor(ARM_COLOR[arm])

        ax.axhline(0, color=INK_MUTED, linewidth=1.1, linestyle="--", zorder=0)
        ax.axvline(len(cuts) - 0.5, color=INK_MUTED, linewidth=0.8,
                   alpha=0.4, zorder=0)
        ax.set_xticks(list(range(len(cuts) + 1)))
        ax.set_xticklabels([f"{c:.0%}\ncut" for c in cuts]
                           + ["100%\nuninterrupted"], fontsize=9.5)
        ax.set_xlim(-0.6, len(cuts) + 0.6)
        # Symlog, matching the scatter in the main figure. The bet 50% box
        # genuinely runs from -1.9 to +5.1 — with n=7 the spread is wide enough
        # that nothing is flagged as an outlier — so a linear axis that showed
        # it whole would flatten every other box to a sliver. Linear inside
        # +/-0.3, compressed outside.
        ax.set_yscale("symlog", linthresh=0.3, linscale=1.1)
        ax.set_ylim(-clip, clip)
        yt = [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 5]
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{t:g}" for t in yt])
        ax.axhspan(-0.3, 0.3, color=INK_MUTED, alpha=0.05, zorder=0)
        ax.set_title(ARM_TITLE[arm], fontsize=12, color=INK, loc="left")
        ax.set_xlabel("Where the reasoning was cut", fontsize=11, color=INK)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK_MUTED)
        # Legend inside each panel, in that panel's own colour, so the reader
        # never has to look elsewhere to decode it.
        ax.legend(handles=[
            Patch(facecolor=ARM_LIGHT[arm], edgecolor=ARM_COLOR[arm],
                  label="estimate when stopped at the cut"),
            Patch(facecolor=ARM_COLOR[arm], edgecolor=ARM_COLOR[arm],
                  label="estimate after continuation")],
            fontsize=9.5, frameon=False, loc="lower right")

    axes[0].set_ylabel("Normalized estimate, toward the rewarded side\n"
                       "(0 = threshold, 0.2 = 20% past it)",
                       fontsize=11, color=INK)

    fig.suptitle("The spread of estimates at each cut: when the model was "
                 "stopped, and after it continued",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.98)
    fig.text(0.012, 0.015,
             "The vertical axis is linear inside the shaded band (±0.3) and "
             "compressed outside it, so the wide 50% boxes fit without "
             "flattening the rest. Nothing is excluded.",
             fontsize=8.5, color=INK_MUTED, ha="left")
    fig.subplots_adjust(top=0.845, bottom=0.155, left=0.075, right=0.985,
                        wspace=0.08)
    out = CONTINUATION_DIR / f"stated_vs_resampled_bars_{model}.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight",
                pad_inches=0.3)
    plt.close(fig)
    print(f"saved {out.relative_to(CONTINUATION_DIR.parent)}")


if __name__ == "__main__":
    main()
