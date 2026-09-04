"""Where in the visible reasoning does the above/below separation appear?

Local only: reads saved run data, makes no API call.

AIM reported a 0.526 separation on final answers, defined as
P(answer > threshold | above-paying) - P(answer > threshold | below-paying).
This script applies that same definition to the FIRST and the LAST estimate in
each rollout's judged trajectory, so the two positions are directly comparable.

Run:
    uv run python analysis/cause/separation_onset.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "qwen3.5-122b-a10b_20260815_030702"
THRESHOLD = 41_000_000
# The filter already used by plot.py and factor.json.
LO, HI = THRESHOLD / 10, THRESHOLD * 10


def in_range(value) -> bool:
    return value is not None and LO <= value <= HI


def paired_first_last(trajectories: list[list]) -> list[tuple[float, float]]:
    """Rollouts whose first and last judged estimates both survive the filter.

    Pairing matters: comparing first against last on different subsets would
    confound the position effect with which rollouts each subset dropped.
    """
    return [
        (t[0], t[-1])
        for t in trajectories
        if t and in_range(t[0]) and in_range(t[-1])
    ]


def share_above(pairs: list[tuple[float, float]], index: int) -> float:
    return sum(1 for p in pairs if p[index] > THRESHOLD) / len(pairs)


def separation_ci(p_above: float, n_above: int, p_below: float, n_below: int):
    diff = p_above - p_below
    se = math.sqrt(
        p_above * (1 - p_above) / n_above + p_below * (1 - p_below) / n_below
    )
    return diff, diff - 1.96 * se, diff + 1.96 * se


def main() -> None:
    trajectories = json.loads((RUN / "trajectories.json").read_text())
    pairs = {c: paired_first_last(trajectories[c]) for c in trajectories}

    print(f"threshold {THRESHOLD:,}; filter [{LO:,.0f}, {HI:,.0f}]\n")
    for condition, rows in pairs.items():
        first, last = share_above(rows, 0), share_above(rows, 1)
        print(
            f"{condition:12s} n={len(rows):3d}  "
            f"P(first>th)={first:.3f}  P(last>th)={last:.3f}  "
            f"change={last - first:+.3f}"
        )

    above, below = pairs["above_good"], pairs["below_good"]
    print()
    for label, index in (("first", 0), ("last", 1)):
        diff, lo, hi = separation_ci(
            share_above(above, index), len(above),
            share_above(below, index), len(below),
        )
        print(f"separation at {label:5s} estimate = {diff:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    first_sep = share_above(above, 0) - share_above(below, 0)
    last_sep = share_above(above, 1) - share_above(below, 1)
    print(f"\nshare of final separation present at the first estimate: {first_sep / last_sep:.0%}")


if __name__ == "__main__":
    main()
