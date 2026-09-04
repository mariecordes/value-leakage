"""How one trace was cut, given a different number, and continued — a worked example.

Nothing here is drawn from imagination. The bar is an actual reasoning trace at its
actual length, the cut is the character offset that was really used, the excerpt is
the model's own text immediately before that cut, every quoted replacement line was
written by the model at the same cut on a different sample, and the endpoints are the
final estimates recorded in cause_results.json for that trace.

The example is bet_below__074, chosen before looking at the endpoints on two grounds:
its rank correlation, 0.70, is the median of the fifteen traces, so it is a typical
case rather than the strongest one; and its replacement lines are short enough to
print verbatim. It also happens to carry the same-number control in a visible form —
two of its six lines commit to 300 spots in different words.

Reads frozen_final.json, inserts.json, cause_results.json and the source rollout.
The quoted reasoning is verified by SHA-256 against the pinned sample before it is
shown, so the figure cannot drift from the trace the experiment actually ran on.
No API calls.

  uv run python analysis/cause/plot_cause_method.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/value-leakage-matplotlib")
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
# The quoted lines are monospace and the committed number inside them is set bold, so
# mathtext has to be monospace too. Left on the default fontset the three bold digits
# would render in a different typeface from the words around them. Matplotlib prints
# "findfont: Failed to find font weight normal" while resolving this; the bold face is
# found and used regardless, so the message is noise rather than a fallback.
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "DejaVu Sans Mono"
matplotlib.rcParams["mathtext.it"] = "DejaVu Sans Mono"
matplotlib.rcParams["mathtext.bf"] = "DejaVu Sans Mono:bold"
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
import palette as PAL  # noqa: E402
from value_leakage.plot import INK, INK_MUTED  # noqa: E402

EXAMPLE = "bet_below__074"
CONTEXT_LINES = 3          # lines of the model's own reasoning shown before the cut
WRAP_EXCERPT = 74          # characters, for the pre-cut excerpt
WRAP_LABEL = 64            # characters; wide enough that every quoted line fits on one
# How far left of the lower axes the shaded band reaches, in axes widths. Enough to
# cover the quoted lines, so the band marks the whole row and not just its dots.
LABEL_EDGE = -0.45

SIDE_LABEL = {"above": "bet, above the threshold pays",
              "below": "bet, below the threshold pays"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tidy(line: str) -> str:
    """One line of reasoning, with its markdown removed but its wording untouched.

    The leading bullet and the ** around the committed number are markdown syntax,
    not something the model said; printed literally they read as typographic noise.
    No word is added, dropped or reordered.
    """
    return line.strip().lstrip("*").replace("**", "").strip()


def emphasise(line: str, value: int) -> str:
    """The same line with the committed number set bold, for scanning down the rows.

    Only the first occurrence is marked, and only the digits — nothing else about the
    wording changes. If the number cannot be found the line is returned untouched
    rather than guessed at.
    """
    for written in (f"{value:,}", str(value)):
        if written in line:
            head, _, tail = line.partition(written)
            digits = written.replace(",", "{,}")
            return head + "$\\bf{" + digits + "}$" + tail
    return line


def load():
    frozen = json.loads((HERE / "frozen_final.json").read_text())
    inserts = json.loads((HERE / "inserts.json").read_text())
    results = json.loads((HERE / "cause_results.json").read_text())

    pinned = next(t for t in frozen["traces"] if t["trace_id"] == EXAMPLE)
    built = next(t for t in inserts["traces"] if t["trace_id"] == EXAMPLE)
    scored = next(t for t in results["per_trace"] if t["trace_id"] == EXAMPLE)

    source = json.loads((ROOT / pinned["source"]).read_text())
    reasoning = source["rows"][pinned["i"]]["reasoning"]

    # The figure quotes the trace, so check it is the trace that was actually run.
    if sha256(reasoning) != pinned["reasoning_sha256"]:
        raise SystemExit(f"{EXAMPLE}: reasoning does not match the pinned sample")
    if built["cut_char"] != pinned["cut_char"]:
        raise SystemExit(f"{EXAMPLE}: cut point differs between pin and inserts")

    model = inserts["model"].split("/")[-1]
    return pinned, built, scored, results, reasoning, model


def rows_for(results: dict, variant: str) -> list[float]:
    return [r["final_value"] / 1e6 for r in results["rows"]
            if r["trace_id"] == EXAMPLE and r["variant"] == variant and r["included"]]


def main() -> None:
    pinned, built, scored, results, reasoning, model = load()
    cut = pinned["cut_char"]
    n_chars = pinned["reasoning_chars"]
    threshold = results["threshold"] / 1e6
    colour = PAL.SIDE[pinned["direction"]]

    # Ordered by the number inserted, with the trace's own line sitting at its own
    # value rather than being held apart. That places the two lines committing to
    # 300 spots next to each other, which is where the same-number control shows.
    variants = sorted(built["variants"],
                      key=lambda v: (v["inserted_value"], v["variant"] != "original"))

    # Explicit rects rather than a shared gridspec: the lower panel needs a wide left
    # margin for the quoted lines, and the trace bar should still span the full width.
    fig = plt.figure(figsize=(17.0, 8.4))
    tx = fig.add_axes([0.030, 0.615, 0.955, 0.235])
    bx = fig.add_axes([0.243, 0.095, 0.742, 0.450])

    # ---- top: the trace, and where it was cut -----------------------------
    tx.barh(0, cut, height=0.34, color=colour, alpha=0.35, edgecolor=colour,
            linewidth=1.4, zorder=2)
    tx.barh(0, n_chars - cut, left=cut, height=0.34, facecolor="none",
            edgecolor=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    tx.plot([cut, cut], [-0.17, 0.17], color=INK, linewidth=2.2, zorder=3)

    # The model's own last few lines before the cut, kept as separate lines: run
    # together they read as one paragraph and the deliberation is lost.
    tail = [tidy(ln) for ln in reasoning[:cut].split("\n") if tidy(ln)][-CONTEXT_LINES:]
    excerpt = "\n".join("\n".join(textwrap.wrap(ln, WRAP_EXCERPT)) for ln in tail)
    tx.annotate(f"…{excerpt}", xy=(cut, 0.22), xytext=(cut, 0.62), fontsize=9.5,
                family="monospace", color=INK_MUTED, ha="left", va="bottom",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8,
                                shrinkA=0, shrinkB=2))
    tx.text(cut / 2, 0, "kept unchanged", ha="center", va="center", fontsize=10,
            color=INK, zorder=4)
    tx.text((cut + n_chars) / 2, 0,
            "next line replaced, everything after it regenerated",
            ha="center", va="center", fontsize=10, color=INK_MUTED, zorder=4)
    tx.text(cut, -0.3, f"cut at {pinned['cut_fraction']:.0%}\n{cut:,} chars",
            ha="center", va="top", fontsize=10, color=INK)
    tx.text(n_chars, -0.3, f"{n_chars:,} chars", ha="right", va="top", fontsize=10,
            color=INK_MUTED)
    tx.set_xlim(-n_chars * 0.02, n_chars * 1.02)
    tx.set_ylim(-0.9, 2.0)
    tx.axis("off")

    # ---- bottom: each replacement line, and where its continuations ended ---
    labels = []
    for row, variant in enumerate(variants):
        finals = rows_for(results, variant["variant"])
        median = next(v["median_final"] for v in scored["variants"]
                      if v["variant"] == variant["variant"]) / 1e6

        # One shaded band marks the unchanged row, so the rows themselves can all be
        # the same colour and colour is left to mean only "this experiment's arm".
        if variant["variant"] == "original":
            # Extended left of the axes and unclipped, so the band runs behind the
            # quoted line as well as behind its dots: the row is one object.
            band = bx.axhspan(row - 0.46, row + 0.46, xmin=LABEL_EDGE, xmax=1.0,
                              color=INK_MUTED, alpha=0.075, zorder=0)
            band.set_clip_on(False)
            bx.annotate("the trace's original line", xy=(23.5, row), ha="left",
                        va="center", fontsize=10, color=INK_MUTED, zorder=4)

        # A hairline through each row, so six scattered dots read as one run's spread
        # rather than as six unrelated marks.
        bx.plot([min(finals), max(finals)], [row, row], "-", lw=0.8, color=colour,
                alpha=0.45, zorder=1)
        bx.plot(finals, [row] * len(finals), "o", ms=8, color=colour, alpha=0.45,
                mew=0, zorder=2)
        bx.plot([median], [row], "o", ms=13, color=colour, mec="white", mew=1.4,
                zorder=3)
        bx.annotate(f"{median:,.1f}M", xy=(median, row), xytext=(0, 14),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, color=INK)

        # Wrapped first, emphasised second: textwrap counts the mathtext markup as
        # characters and would otherwise break a line in the middle of it.
        labels.append(emphasise(
            "\n".join(textwrap.wrap(tidy(variant["step"]), WRAP_LABEL)),
            variant["inserted_value"]))

    # Two of the six lines commit to 300 spots in different words, and their medians
    # differ by more than the whole inserted range does. That is the most interesting
    # thing here, but the quoted lines say it plainly enough on their own; an
    # annotation pointing it out was tried and removed as clutter. It belongs in the
    # figure caption instead.
    bx.axvline(threshold, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    bx.annotate(f"threshold {threshold:.0f}M", xy=(threshold, len(variants) - 0.35),
                xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                fontsize=10, color=INK_MUTED)

    bx.set_yticks(range(len(variants)))
    bx.set_yticklabels(labels, fontsize=10.5, family="monospace", color=INK,
                       linespacing=1.45)
    bx.set_ylim(-1.35, len(variants) - 0.3)
    bx.set_xlim(23.2, 42.0)
    bx.set_xlabel("Final estimate, millions of spots", fontsize=11.5, color=INK)
    bx.grid(True, axis="x", alpha=0.22, linewidth=0.6)
    bx.grid(False, axis="y")
    bx.set_axisbelow(True)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.tick_params(colors=INK_MUTED, length=0, pad=8)

    # Every row is drawn in the one arm colour, so the legend only has to say what the
    # two mark sizes mean. Which row is the unchanged one is said by the shaded band
    # and its label, in place, rather than by a second colour and a legend entry.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8, color=colour,
                   alpha=0.45, label="one continuation"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=13, color=colour,
                   mec="white", mew=1.4, label="median of six continuations"),
    ]
    bx.legend(handles=handles, fontsize=10.5, frameon=False, loc="lower right",
              ncol=2, columnspacing=1.8, handletextpad=0.6)

    title = fig.suptitle(
        "Example: how one reasoning trace was cut, given a different number of spots "
        "per giraffe, and continued\n"
        f"({model}, {SIDE_LABEL[pinned['direction']]}, "
        f"{n_chars:,} characters; rank correlation "
        f"{scored['spearman_rho']:.2f}, the median of the fifteen traces)",
        fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985, linespacing=1.5)

    # Flush the title with the start of the longest quoted line. Where that falls
    # depends on the font metrics of the lines themselves, so it is measured from the
    # drawn figure rather than guessed at and then nudged by hand.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left = min(label.get_window_extent(renderer).x0
               for label in bx.get_yticklabels())
    title.set_x(left / fig.bbox.width)

    for suffix in ("png", "svg"):
        fig.savefig(HERE / f"cause_method.{suffix}", dpi=180, facecolor="white",
                    bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    print("saved analysis/cause/cause_method.png")
    print(f"  {EXAMPLE}: cut at {cut:,} of {n_chars:,} chars "
          f"({pinned['cut_fraction']:.0%}), rho {scored['spearman_rho']:.2f}")
    for variant in variants:
        median = next(v["median_final"] for v in scored["variants"]
                      if v["variant"] == variant["variant"]) / 1e6
        print(f"  {variant['inserted_value']:>5} spots "
              f"({variant['variant']:<8}) -> median {median:,.1f}M")


if __name__ == "__main__":
    main()
