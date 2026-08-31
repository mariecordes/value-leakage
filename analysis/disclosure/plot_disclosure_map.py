"""DISCLOSURE_DIR step A4 — the disclosure map. No API calls.

Left panel  : one dot per model. x = share of its judged runs whose thinking
              DENIES being influenced, y = how far the bet pushed its answers
              onto the favoured side (bias_size).
Right panel : the full four-label breakdown per model, as stacked bars.

Writes analysis/disclosure/map.png and analysis/disclosure/results.json. Needs analysis/disclosure/bias.json (from
prepare_sample.py) and analysis/disclosure/disclosure_<model>.json (from judge_disclosure.py).

  uv run python analysis/disclosure/plot_disclosure_map.py
"""

import json
import sys

import fire
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

sys.path.insert(0, str(Path(__file__).parent))
from common import DISCLOSURE_DIR  # noqa: E402

# The starter dataset repo's ink (plot.py) so this figure sits beside the starter dataset figures.
INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"

# Measured judge instability: re-running the SAME prompt on a random 40 of the
# judged runs reproduced 62% of labels, and moved the DENIES share by 10 points
# (27.5% -> 17.5%). Treated as a 95%-level slop term on every share.
JUDGE_SHARE_SLOP = 0.10

sys.path.insert(0, str(Path(__file__).parent.parent))
import palette as PAL  # noqa: E402

# Four labels form one ordered scale: open about it -> silent -> denies it.
# Diverging by design (blue pole / neutral middle / orange pole). Adjacent-pair
# CVD and normal-vision separation checked with the dataviz validator; the two
# poles reuse the repo's own condition colours.
LABELS = ("ADMITS", "MENTIONS", "NO_MENTION", "DENIES")
LABEL_COLORS = PAL.DISCLOSURE
LABEL_TEXT = {"ADMITS": "admits influence", "MENTIONS": "mentions it",
              "NO_MENTION": "never raises it", "DENIES": "denies influence"}


def load(rejudge: bool = False):
    bias = json.loads((DISCLOSURE_DIR / "bias.json").read_text())
    # Optional overlay: corrections from the NO_MENTION audit. Applied on read,
    # never written back into disclosure_<model>.json.
    fixes = {}
    if rejudge:
        path = DISCLOSURE_DIR / "rejudge_nomention.json"
        if not path.exists():
            print("no analysis/disclosure/rejudge_nomention.json — run analysis/disclosure/validate_no_mention.py")
        else:
            audit = json.loads(path.read_text())
            for r in audit["rows"]:
                if r["group"] == "suspect" and r["label_rejudged"]:
                    fixes[(r["model"], r["condition"], r["idx"])] = \
                        r["label_rejudged"]
            print(f"applying {sum(1 for r in audit['rows'] if r['group']=='suspect' and r['changed'])} "
                  f"NO_MENTION corrections from the audit")

    rows = {}
    missing = []
    for model, b in bias.items():
        path = DISCLOSURE_DIR / f"disclosure_{model}.json"
        if not path.exists():
            missing.append(model)
            continue
        d = json.loads(path.read_text())
        labels = [fixes.get((model, r["condition"], r["idx"]), r["label"])
                  for r in d["rows"]]
        counts = {lab: sum(1 for x in labels if x == lab) for lab in LABELS} \
            if fixes else {lab: d["counts"].get(lab, 0) for lab in LABELS}
        n_labelled = sum(counts.values())
        rows[model] = {
            "bias_size": b["bias_size"],
            "bias_size_ties_half": b["bias_size_ties_half"],
            "separation": b["separation"],
            "separation_strict": b["separation_strict"],
            "n_runs_used": b["n_runs_used"],
            "backend": b["backend"],
            "model_id": b["model_id"],
            "threshold": b["threshold"],
            "n_judged": d["n"],
            "n_labelled": n_labelled,
            "n_unparsed": d["counts"].get("UNPARSED", 0),
            "counts": counts,
            "shares": {lab: (counts[lab] / n_labelled if n_labelled else None)
                       for lab in LABELS},
            "judge_model": d["judge_model"],
        }
    return rows, missing


def draw(rows):
    # Both panels sorted by denial rate so they tell the same story left to right.
    order = sorted(rows, key=lambda m: rows[m]["shares"]["ADMITS"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 7.2),
                                 gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ---- left: bias against denial rate ---------------------------------
    # Admits, not denies. It is the same four-label judgement, but admits is
    # the reliable end of it: re-running the judge reproduced 74% of ADMITS
    # labels against 36% of DENIES, and the denial share moved 10 points
    # between two identical passes. It also matches the right-hand panel's
    # ordering, so both halves of the figure key off the same variable.
    xs = np.array([rows[m]["shares"]["ADMITS"] for m in order])
    ys = np.array([rows[m]["separation"] for m in order])
    ns = np.array([rows[m]["n_labelled"] for m in order], dtype=float)
    # Descriptive uncertainty bars on each share. Two sources of slop are added
    # in quadrature: sampling (only 40 runs judged per model) and the observed
    # ten-point movement when the same judge prompt was rerun on the same traces.
    se_sampling = np.sqrt(np.clip(xs * (1 - xs), 1e-9, None) / ns)
    se_judge = JUDGE_SHARE_SLOP / 1.96
    err = 1.96 * np.sqrt(se_sampling ** 2 + se_judge ** 2)

    ax.axhline(0, color=INK_MUTED, linewidth=0.8, linestyle="--", zorder=0)
    # Bars recede, dots carry the reading.
    ax.errorbar(xs, ys, xerr=err, fmt="none", ecolor=INK_MUTED, elinewidth=1.1,
                capsize=4, capthick=1.1, alpha=0.42, zorder=2)
    ax.scatter(xs, ys, s=150, color=PAL.BELOW, edgecolor="white", linewidth=1.8,
               zorder=3)

    # Labels go above the dot, so they clear the horizontal error bar. Models
    # too close vertically get nudged below instead of colliding.
    texts = [ax.text(x, y, m, fontsize=9, color=INK, ha="center", va="bottom")
             for m, x, y in zip(order, xs, ys)]
    adjust_text(texts, x=list(xs), y=list(ys), ax=ax,
                expand=(1.3, 1.9), force_text=(0.4, 0.9),
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8,
                                shrinkA=2, shrinkB=6))

    ax.set_xlabel("Share of runs whose reasoning admits influence",
                  fontsize=11, color=INK)
    ax.set_ylabel("Condition-balanced answer shift", fontsize=11, color=INK)
    ax.set_title("Answer shifts and how often the reasoning reports influence",
                 fontsize=12, color=INK, loc="left")
    ax.set_xlim(xs.min() - 0.14, xs.max() + 0.12)
    ax.set_ylim(min(0, ys.min()) - 0.06, ys.max() + 0.10)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_MUTED)

    # ---- right: label breakdown -----------------------------------------
    # Ordered by how often the reasoning admits influence, most at the top, so
    # the model at the top here is also the top point on the left.
    bar_order = sorted(rows, key=lambda m: rows[m]["shares"]["ADMITS"])
    ypos = np.arange(len(bar_order))
    left = np.zeros(len(bar_order))
    for lab in LABELS:
        w = np.array([rows[m]["shares"][lab] for m in bar_order])
        bx.barh(ypos, w, left=left, height=0.68, color=LABEL_COLORS[lab],
                edgecolor="white", linewidth=2, label=LABEL_TEXT[lab])
        for y, wi, li in zip(ypos, w, left):
            if wi >= 0.08:                      # visible label, never colour-alone
                bx.text(li + wi / 2, y, f"{wi*100:.0f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if lab in ("ADMITS", "DENIES") else INK)
        left = left + w
    bx.set_yticks(ypos)
    bx.set_yticklabels(bar_order, fontsize=9, color=INK)
    bx.set_xlim(0, 1)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    bx.set_xlabel("Share of judged runs (n=40 per model)", fontsize=11, color=INK)
    bx.set_title("What each model's reasoning says about the bet",
                 fontsize=12, color=INK, loc="left")
    for spine in ("top", "right", "left"):
        bx.spines[spine].set_visible(False)
    bx.tick_params(colors=INK_MUTED, length=0)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4,
              frameon=False, fontsize=9)

    fig.suptitle("Answer shifts and how models describe the Donation Bet",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.98)
    # Short enough not to set the figure width; the rest lives in the caption.
    fig.text(0.012, 0.005,
             "Shift: difference in above-threshold landing rates; ties count "
             "as half in each condition.", fontsize=9, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    out = DISCLOSURE_DIR / "map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(DISCLOSURE_DIR / "map.svg", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"saved {out}")


def main(rejudge: bool = False):
    rows, missing = load(rejudge)
    if missing:
        print(f"no disclosure labels yet for: {', '.join(missing)}")
    if not rows:
        print("nothing to plot — run analysis/disclosure/judge_disclosure.py first")
        return
    (DISCLOSURE_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"{'model':24s} {'bias':>7s} {'denies':>7s} {'admits':>7s} "
          f"{'mentions':>8s} {'no_mention':>10s}")
    for m in sorted(rows, key=lambda m: -rows[m]["separation"]):
        r = rows[m]
        print(f"{m:24s} {r['separation']:+7.3f} {r['shares']['DENIES']:7.2f} "
              f"{r['shares']['ADMITS']:7.2f} {r['shares']['MENTIONS']:8.2f} "
              f"{r['shares']['NO_MENTION']:10.2f}")
    draw(rows)
    print(f"saved {DISCLOSURE_DIR/'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)
