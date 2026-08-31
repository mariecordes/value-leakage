"""How often the disclosure judge gives the same label twice.

Reads analysis/disclosure/judge_stability_random.json — 40 reasoning traces sent to the judge a
second time with the identical prompt. The grid pairs each trace's first label
with its second, so the diagonal is agreement and everything off it is the
judge changing its mind on unchanged input. No API calls, free to re-render.

  uv run python analysis/disclosure/plot_judge_matrix.py
"""

import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import palette as PAL  # noqa: E402
from value_leakage.plot import INK, INK_MUTED  # noqa: E402

DISCLOSURE_DIR = HERE

# Same order and the same wording used in the disclosure figure, so the two
# can be read side by side.
LABELS = ["ADMITS", "MENTIONS", "NO_MENTION", "DENIES"]
NICE = {"ADMITS": "Admits the bet\ninfluenced it",
        "MENTIONS": "Mentions the bet,\nno influence claimed",
        "NO_MENTION": "Never raises\nthe bet",
        "DENIES": "Denies the bet\ninfluenced it"}
SHORT = {"ADMITS": "Admits", "MENTIONS": "Mentions",
         "NO_MENTION": "Never raises", "DENIES": "Denies"}


def main(source: str = "judge_stability_random.json"):
    s = json.loads((DISCLOSURE_DIR / source).read_text())
    rows = s["rows"]
    counts = collections.Counter(
        (r["label_first_pass"], r["label_second_sample"]) for r in rows)
    m = np.array([[counts[(a, b)] for b in LABELS] for a in LABELS], float)
    n = int(m.sum())
    agreed = int(np.trace(m))

    # A single-hue ramp from white to the palette's indigo. One hue, because
    # the cells carry one quantity; a diverging or rainbow map would imply a
    # midpoint that does not exist here.
    cmap = LinearSegmentedColormap.from_list(
        "count", ["#ffffff", PAL.INDIGO_LT, PAL.BELOW])

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=m.max())

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = int(m[i, j])
            # White ink on the dark cells, ink on the pale ones.
            col = "white" if v > m.max() * 0.55 else (INK if v else INK_MUTED)
            # Every cell styled identically. The diagonal carries no emphasis
            # of its own: the shading is the only thing distinguishing cells.
            ax.text(j, i, f"{v:d}", ha="center", va="center",
                    fontsize=15 if v else 12, color=col)

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([SHORT[x] for x in LABELS], fontsize=10.5, color=INK)
    ax.set_yticklabels([NICE[x] for x in LABELS], fontsize=10, color=INK)
    ax.set_xlabel("Label on the second pass", fontsize=11, color=INK,
                  labelpad=10)
    ax.set_ylabel("Label on the first pass", fontsize=11, color=INK,
                  labelpad=10)
    ax.tick_params(colors=INK_MUTED, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", length=0)

    # No colour bar: every cell already carries its count, so a bar would only
    # add a second thing to decode. The shading is an accent, not the reading.

    fig.suptitle(f"Using the disclosure judge to label whether a reasoning "
                 f"trace admits, mentions or denies the bet,\nwith every trace "
                 f"labelled twice (n={n})",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985,
                 linespacing=1.5)
    fig.subplots_adjust(top=0.845, bottom=0.105, left=0.235, right=0.985)
    out = DISCLOSURE_DIR / "judge_matrix.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight",
                pad_inches=0.3)
    fig.savefig(DISCLOSURE_DIR / "judge_matrix.svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"saved {out.relative_to(DISCLOSURE_DIR.parent)}")
    for a in LABELS:
        tot = int(m[LABELS.index(a)].sum())
        same = int(m[LABELS.index(a), LABELS.index(a)])
        if tot:
            print(f"  first pass {a:11s} n={tot:3d}  same again {same:3d} "
                  f"({same / tot:.0%})")


if __name__ == "__main__":
    main()
