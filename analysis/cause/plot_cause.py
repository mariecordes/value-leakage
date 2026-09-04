"""The three CAUSE figures. Reads cause_results.json, so re-rendering costs nothing.

  uv run python analysis/cause/plot_cause.py

House conventions, matching analysis/continuations and analysis/target_visibility:
a factual title naming what each mark is, short axis labels that state the unit
("normalized estimate (0 = threshold)"), reference lines labelled in place, the
shared CVD-validated palette, and no footnotes — anything that cannot sit on the
plot belongs in the figure caption in the write-up.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/value-leakage-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
import palette as PAL  # noqa: E402
from value_leakage.plot import INK, INK_MUTED  # noqa: E402

SIDE = PAL.SIDE
SIDE_LABEL = {"above": "above the threshold pays",
              "below": "below the threshold pays"}
Y_CUT = 1.5


def load() -> dict:
    return json.loads((HERE / "cause_results.json").read_text())


def alternatives(trace: dict) -> list[dict]:
    """Variants other than the unchanged original, ordered by inserted figure."""
    return sorted((v for v in trace["variants"] if v["variant"] != "original"),
                  key=lambda v: v["inserted_value"])


def frame(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def retention(trace: dict) -> float | None:
    """Observed change over the change a proportional response would have given.

    Anchored on the trace's own lowest inserted figure: if the answer tracked the
    premise, doubling the figure would double the total.
    """
    alts = alternatives(trace)
    if len(alts) < 2:
        return None
    low, high = alts[0], alts[-1]
    projected = low["median_final"] * high["inserted_value"] / low["inserted_value"]
    if projected == low["median_final"]:
        return None
    return (high["median_final"] - low["median_final"]) / (projected - low["median_final"])


def strip(ax, values: list[tuple[str, float]], seed: int) -> None:
    """One dot per trace, two rows by arm, jittered so nothing hides."""
    rng = np.random.default_rng(seed)
    for arm, row in (("above", 1), ("below", 0)):
        points = [v for a, v in values if a == arm]
        # Semi-transparent so coincident traces are visible as a darker dot.
        ax.plot(points, row + rng.uniform(-0.11, 0.11, len(points)), "o", ms=9,
                color=SIDE[arm], alpha=0.55, mec="white", mew=1.0, zorder=3)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([SIDE_LABEL["above"], SIDE_LABEL["below"]], fontsize=10,
                       color=INK)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(axis="y", alpha=0)


# --------------------------------------------------------------------------
def main_figure(results: dict | None = None) -> None:
    """Inserted figure against the final estimate, one line per trace."""
    results = results or load()
    threshold = results["threshold"]
    rows = results["rows"]
    fig, ax = plt.subplots(figsize=(12.2, 6.4))

    hidden = 0
    for trace in results["per_trace"]:
        alts = alternatives(trace)
        if len(alts) < 2:
            continue
        colour = SIDE[trace["direction"]]
        for variant in alts:
            points = [(r["final_value"] - threshold) / threshold for r in rows
                      if r["trace_id"] == trace["trace_id"]
                      and r["variant"] == variant["variant"] and r["included"]]
            hidden += sum(1 for p in points if p > Y_CUT)
            ax.plot([variant["inserted_value"]] * len(points), points, "o", ms=6,
                    color=colour, alpha=0.28, mew=0, zorder=1)
        ax.plot([v["inserted_value"] for v in alts],
                [(v["median_final"] - threshold) / threshold for v in alts],
                "-o", lw=1.8, ms=8, color=colour, alpha=0.9, mec="white", mew=1.1,
                zorder=3)

    ax.axhline(0, color=INK_MUTED, lw=0.9, alpha=0.5, zorder=0)
    ax.set_ylim(-0.55, Y_CUT)
    ax.set_xlim(180, 1570)
    ax.text(0.015, 0.985, f"{hidden} of 437 continuations lie above this range",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color=INK_MUTED)
    ax.set_xlabel("Number of spots per giraffe inserted at the cut", fontsize=11)
    ax.set_ylabel("Normalized final estimate (0 = threshold)", fontsize=11)
    frame(ax)

    # Column-major fill puts the two arm colours in the left column and the two
    # mark meanings in the right, so identity and encoding read as separate pairs.
    handles = [plt.Line2D([], [], color=SIDE[a], lw=2.6, label=SIDE_LABEL[a])
               for a in ("above", "below")]
    handles += [
        plt.Line2D([], [], color=INK_MUTED, lw=1.8, marker="o", ms=8, mec="white",
                   mew=1.1, label="median of six continuations"),
        plt.Line2D([], [], color=INK_MUTED, lw=0, marker="o", ms=6, alpha=0.35,
                   label="one continuation"),
    ]
    ax.legend(handles=handles, fontsize=9.5, frameon=False, loc="lower right",
              ncol=2, columnspacing=1.6)

    fig.suptitle("Comparing the number of spots inserted at the cut with the final "
                 "estimate\nthe model reached from there, per trace",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.985)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.12, right=0.98)
    fig.savefig(HERE / "cause.png", dpi=180, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    fig.savefig(HERE / "cause.svg", facecolor="white", bbox_inches="tight",
                pad_inches=0.3)
    plt.close(fig)


# --------------------------------------------------------------------------
def consistency_figure(results: dict | None = None) -> None:
    """How consistent the direction was, and how much of the change survived."""
    results = results or load()
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.4))

    ax = axes[0]
    strip(ax, [(t["direction"], t["spearman_rho"]) for t in results["per_trace"]
               if t["spearman_rho"] is not None], seed=11)
    ax.axvline(0, color=INK_MUTED, lw=1.0, alpha=0.7, zorder=1)
    ax.axvline(1.0, color=INK_MUTED, lw=1.0, ls="--", alpha=0.8, zorder=1)
    ax.set_xlim(-0.12, 1.08)
    ax.text(0.005, 1.42, "no relationship", fontsize=9, color=INK_MUTED, ha="left")
    ax.text(0.985, 1.42, "perfect relationship", fontsize=9, color=INK_MUTED,
            ha="right")
    ax.set_xlabel("Correlation between inserted spots and final estimate",
                  fontsize=11)
    frame(ax)

    ax = axes[1]
    strip(ax, [(t["direction"], retention(t)) for t in results["per_trace"]
               if retention(t) is not None], seed=29)
    ax.axvline(1.0, color=INK_MUTED, lw=1.0, ls="--", alpha=0.8, zorder=1)
    ax.axvline(0, color=INK_MUTED, lw=1.0, alpha=0.7, zorder=1)
    ax.set_xlim(-0.12, 1.08)
    ax.text(0.985, 1.42, "answer changed in proportion", fontsize=9,
            color=INK_MUTED, ha="right")
    ax.text(0.005, 1.42, "answer did not move", fontsize=9, color=INK_MUTED,
            ha="left")
    ax.set_xlabel("Share of the inserted change that reached the final estimate",
                  fontsize=11)
    frame(ax)

    fig.suptitle("How consistently the inserted number carried through to the "
                 "final estimate,\nand how much of it survived, within each trace",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.99)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.20, right=0.98, hspace=0.85)
    fig.savefig(HERE / "cause_consistency.png", dpi=180, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    fig.savefig(HERE / "cause_consistency.svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


# --------------------------------------------------------------------------
def noise_figure(results: dict | None = None) -> None:
    """Where continuations ended, by which number was inserted and by arm.

    Box style copied from analysis/continuations/plot_gap_bars.py: filled boxes in
    the arm colour, a white median line, hollow fliers. Each trace is centred on
    its own original number, so the question "did the answer move further than it
    varies anyway" compares distributions with distributions.
    """
    results = results or load()
    threshold = results["threshold"]
    rows = results["rows"]

    groups = {k: {"above": [], "below": []}
              for k in ("lowest", "original", "highest")}
    for trace in results["per_trace"]:
        alts = alternatives(trace)
        base = next((v for v in trace["variants"] if v["variant"] == "original"), None)
        if len(alts) < 2 or base is None:
            continue
        centre = base["median_final"]
        arm = trace["direction"]
        for key, variant in (("lowest", alts[0]["variant"]),
                             ("original", "original"),
                             ("highest", alts[-1]["variant"])):
            groups[key][arm] += [
                (r["final_value"] - centre) / threshold for r in rows
                if r["trace_id"] == trace["trace_id"]
                and r["variant"] == variant and r["included"]
            ]

    order = [("lowest", 0, "lowest number of spots"),
             ("original", 1, "the trace's own number"),
             ("highest", 2, "highest number of spots")]
    fig, ax = plt.subplots(figsize=(11.0, 5.2))

    for key, row, _ in order:
        for arm, offset in (("above", 0.19), ("below", -0.19)):
            bp = ax.boxplot([groups[key][arm]], positions=[row + offset],
                            widths=0.3, orientation="horizontal", patch_artist=True,
                            showfliers=True,
                            medianprops=dict(color="white", linewidth=2),
                            flierprops=dict(marker="o", markersize=4.5,
                                            markerfacecolor="none", alpha=0.75),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2))
            for patch in bp["boxes"]:
                patch.set_facecolor(SIDE[arm])
                patch.set_edgecolor(SIDE[arm])
                patch.set_alpha(0.9)
            for part in ("whiskers", "caps"):
                for artist in bp[part]:
                    artist.set_color(SIDE[arm])
            for flier in bp["fliers"]:
                flier.set_markeredgecolor(SIDE[arm])

    ax.axvline(0, color=INK_MUTED, lw=1.1, ls="--", alpha=0.6, zorder=0)
    ax.set_yticks([r for _, r, _ in order])
    ax.set_yticklabels([label for _, _, label in order], fontsize=11, color=INK)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("Difference in normalized final estimate from the same trace's "
                  "own-number continuations" + chr(10) + "(0 = no difference)", fontsize=11)
    frame(ax)
    ax.grid(axis="y", alpha=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SIDE[a], alpha=0.9,
                             label=SIDE_LABEL[a]) for a in ("above", "below")]
    ax.legend(handles=handles, fontsize=9.5, frameon=False, ncol=1,
              loc="lower right", columnspacing=1.6)

    fig.suptitle("Difference in final estimate between each trace's original number of "
                 "spots and the lowest and highest numbers" + chr(10) +
                 "inserted at the cut, pooled across "
                 "traces",
                 fontsize=13, color=INK, x=0.012, ha="left", y=1.0)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.20, right=0.98)
    fig.savefig(HERE / "cause_spread.png", dpi=180, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    main_figure(data)
    consistency_figure(data)
    noise_figure(data)
    for name in ("cause.png", "cause_consistency.png", "cause_spread.png"):
        print(f"saved analysis/cause/{name}")
