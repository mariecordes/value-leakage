"""How one trace was interrupted — a worked example, using a real run.

Nothing here is drawn from imagination. The bar is an actual reasoning trace at
its actual length, the cut points are the character offsets that were really
used, the excerpts are the text immediately before each cut, and the numbers
are the ones recorded in analysis/continuations/results.json for that run.

Reads analysis/continuations/selection.json, analysis/continuations/results.json and the trace itself. No API calls.

  uv run python analysis/continuations/plot_method.py
"""

import json
import textwrap
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "target_visibility"))
import palette as PAL  # noqa: E402
from select_traces import source_rows  # noqa: E402
from value_leakage.plot import INK, INK_MUTED  # noqa: E402

CONTINUATION_DIR = HERE

# One run, chosen because the stated and the continued numbers differ visibly
# at all three cuts, including one where they swap sides of the threshold.
# Runs where the two agree everywhere are the more common case, but they make a
# worse illustration: the two markers sit on top of each other and the reader
# has to hunt for the second one.
EXAMPLE = {"arm": "bet", "version": "bet_above", "i": 2}

# Which prompt the run came from, spelled out for the title.
FRAMING = {
    "bet_above": "bet, above the threshold pays",
    "bet_below": "bet, below the threshold pays",
    "instructed_above": "instructed to aim above the threshold",
    "instructed_below": "instructed to aim below the threshold",
}


def tidy(s: str) -> str:
    """The trailing words of the trace, on one line, for showing verbatim.

    Only the markdown bullet at the head of a line is dropped. Stripping every
    asterisk would also eat the multiplication signs, and these excerpts are
    presented as the model's own words.
    """
    lines = [ln.strip().lstrip("*").strip() for ln in s.split("\n")]
    return " ".join(" ".join(lines).split())


def main(model: str = "qwen3.5-122b-a10b", tail: int = 78):
    sel = json.loads((CONTINUATION_DIR / "selection.json").read_text())
    res = json.loads((CONTINUATION_DIR / "results.json").read_text())
    thr = res["threshold"]

    run = [r for r in sel["arms"][EXAMPLE["arm"]]["runs"]
           if r["version"] == EXAMPLE["version"] and r["i"] == EXAMPLE["i"]][0]
    rows = sorted([r for r in res["rows"]
                   if (r["arm"], r["version"], r["i"])
                   == (EXAMPLE["arm"], EXAMPLE["version"], EXAMPLE["i"])],
                  key=lambda r: r["cut_frac"])
    src_rows, _ = source_rows(model, EXAMPLE["version"])
    text = src_rows[EXAMPLE["i"]].get("reasoning") or ""
    n_chars = run["chars"]
    cuts = run["cut_chars"]

    fig, (tx, bx) = plt.subplots(
        2, 1, figsize=(13.6, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.5], "hspace": 0.1})

    # ---- top: the trace itself, at its real length ------------------------
    tx.barh(0, n_chars, height=0.34, color=PAL.BET_LIGHT,
            edgecolor=PAL.BET, linewidth=1.4, zorder=2)
    for c in cuts:
        tx.plot([c, c], [-0.17, 0.17], color=INK, linewidth=2.2, zorder=3)
    for c, r in zip(cuts, rows):
        # Wrapped by hand: unwrapped, the three excerpts are single long lines
        # and they run straight through each other.
        excerpt = "\n".join(textwrap.wrap(tidy(text[max(0, c - tail):c]), 34))
        tx.annotate(f"…{excerpt}",
                    xy=(c, 0.22), xytext=(c, 0.5), fontsize=7.4,
                    family="monospace", color=INK_MUTED,
                    ha="center", va="bottom", linespacing=1.45,
                    arrowprops=dict(arrowstyle="-", color=INK_MUTED,
                                    linewidth=0.8, shrinkA=0, shrinkB=2))
        tx.text(c, -0.3, f"cut at {r['cut_frac']:.0%}\n{c:,} chars",
                ha="center", va="top", fontsize=9, color=INK)
    tx.text(n_chars, 0.0, "  trace ends", va="center", ha="left",
            fontsize=9, color=INK_MUTED)
    tx.set_ylim(-0.85, 1.55)
    tx.axis("off")

    # ---- bottom: what it stated there, and where it went ------------------
    bx.axhline(thr / 1e6, color=INK_MUTED, linewidth=1.2, linestyle="--",
               zorder=1)
    bx.annotate(f"threshold {thr / 1e6:.0f}M",
                xy=(-n_chars * 0.045, thr / 1e6), xytext=(0, 7),
                textcoords="offset points", va="bottom", ha="left",
                fontsize=9, color=INK_MUTED)

    for c, r in zip(cuts, rows):
        st, ld = r["stated"] / 1e6, r["landing"] / 1e6
        bx.plot([c, c], [st, ld], color=PAL.BET, linewidth=1.6,
                alpha=0.55, zorder=2)
        bx.scatter([c], [st], s=150, facecolor="white",
                   edgecolor=PAL.BET, linewidth=2.0, zorder=3)
        bx.scatter([c], [ld], s=150, color=PAL.BET, edgecolor="white",
                   linewidth=1.4, zorder=3)
        # A stated number recovered as the midpoint of a range is flagged, so
        # the figure never presents a derived value as something the model
        # wrote out.
        note = ("\n(midpoint of a range)"
                if r["stated_kind"] == "range_midpoint" else "")
        # Each label pushed away from the other marker, so the pair never
        # collides when the two numbers are close together.
        up = 1 if st >= ld else -1
        # A label can still land on the threshold line; a plain white backing
        # keeps it readable without drawing a box around it.
        back = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
        bx.annotate(f"stated {st:,.1f}M{note}", xy=(c, st),
                    xytext=(-12, 10 * up), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=INK, bbox=back)
        bx.annotate(f"landed {ld:,.1f}M", xy=(c, ld),
                    xytext=(12, -10 * up), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=INK, bbox=back)

    # The answer the trace reached on its own, never interrupted.
    final = run["final"] / 1e6
    bx.scatter([n_chars], [final], s=170, marker="D", color=PAL.BET,
               edgecolor="white", linewidth=1.4, zorder=3)
    bx.annotate(f"final answer\n{final:,.1f}M", xy=(n_chars, final),
                xytext=(0, 16), textcoords="offset points", ha="center",
                fontsize=9, color=INK)

    vals = [v for r in rows for v in (r["stated"], r["landing"])] \
        + [thr, run["final"]]
    lo, hi = min(vals) / 1e6, max(vals) / 1e6
    pad = (hi - lo) * 0.28
    bx.set_ylim(lo - pad, hi + pad)
    bx.set_xlim(-n_chars * 0.055, n_chars * 1.13)
    bx.set_xticks(cuts + [n_chars])
    bx.set_xticklabels([f"{c:,}" for c in cuts] + [f"{n_chars:,}"],
                       fontsize=9)
    bx.set_xlabel("Position in the reasoning (characters)", fontsize=11,
                  color=INK)
    bx.set_ylabel("Estimate, millions of spots", fontsize=11, color=INK)
    bx.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    bx.set_axisbelow(True)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    bx.tick_params(colors=INK_MUTED)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=11,
                   markerfacecolor="white", markeredgecolor=PAL.BET,
                   markeredgewidth=2.0,
                   label="estimate when stopped at the cut"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=11,
                   markerfacecolor=PAL.BET, markeredgecolor="white",
                   label="estimate after continuation"),
        plt.Line2D([], [], marker="D", linestyle="none", markersize=10,
                   markerfacecolor=PAL.BET, markeredgecolor="white",
                   label="answer the uninterrupted trace reached"),
    ]
    bx.legend(handles=handles, fontsize=9.5, frameon=False, loc="upper right")

    # One title carrying the whole caption, rather than a title plus a subtitle
    # on each panel. The method text that used to sit in a footnote lives in
    # the report caption instead.
    fig.suptitle(f"Example: how a single reasoning trace was interrupted and "
                 f"continued\n({model}, {FRAMING[EXAMPLE['version']]}, "
                 f"{n_chars:,} characters)",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985,
                 linespacing=1.5)
    fig.subplots_adjust(top=0.855, bottom=0.105, left=0.075, right=0.985)
    out = CONTINUATION_DIR / "method.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight",
                pad_inches=0.3)
    fig.savefig(CONTINUATION_DIR / "method.svg", facecolor="white", bbox_inches="tight",
                pad_inches=0.3)
    plt.close(fig)
    print(f"saved {out.relative_to(CONTINUATION_DIR.parent)}")
    for c, r in zip(cuts, rows):
        print(f"  cut {r['cut_frac']:.0%} at {c:,} chars — stated "
              f"{r['stated']:,.0f}, landed {r['landing']:,.0f}")


if __name__ == "__main__":
    main()
