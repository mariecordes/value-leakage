"""Browse rollouts for Phase 1 trace reading.

List mode — one line per rollout (trajectory shape, final estimate, side,
quick verbalization flags):

  uv run python view_rollout.py runs/kimi-k3_20260815_030702 above_good

Detail mode — one rollout in full (trajectory vs threshold, raw reasoning,
visible answer):

  uv run python view_rollout.py runs/kimi-k3_20260815_030702 above_good 7
"""

import json
import sys
import textwrap
from pathlib import Path

# crude keyword flags for a first-pass verbalization scan; the real
# classification later uses the paper's disclosure taxonomy via a judge
FLAGS = {"T": ("threshold",), "D": ("donat", "cause", "charit"), "B": ("bet",),
         "H": ("honest", "unbiased", "accurate estimate")}


def load(run_dir: Path, condition: str):
    thr = json.loads((run_dir / "threshold.json").read_text())["threshold"]
    traj = json.loads((run_dir / "trajectories.json").read_text())[condition]
    rows = json.loads((run_dir / f"{condition}.json").read_text())["rows"]
    # shipped runs only carry judged final estimates for baseline (threshold
    # computation); elsewhere fall back to the trajectory's last point
    est_all = json.loads((run_dir / "estimates.json").read_text())
    est = est_all.get(condition)
    if est is None:
        est = [(t[-1] if isinstance(t, list) and t else None) for t in traj]
    return thr, est, traj, rows


def rel(v, thr):
    return f"{(v - thr) / thr:+.2f}" if v is not None else "  —  "


def flags_for(text: str) -> str:
    low = (text or "").lower()
    return "".join(k if any(w in low for w in ws) else "·"
                   for k, ws in FLAGS.items())


def list_mode(run_dir: Path, condition: str):
    thr, est, traj, rows = load(run_dir, condition)
    favoured_above = condition == "above_good"
    print(f"{run_dir.name} | {condition} | threshold {thr:,} "
          f"| favoured side: {'ABOVE' if favoured_above else 'BELOW'}"
          f"{'' if condition != 'baseline' else ' (n/a)'}")
    print(f"{'i':>3} {'pts':>4} {'first':>6} {'last':>6} {'final':>6} "
          f"{'side':>5} {'fav?':>4}  flags(T=threshold D=donation B=bet H=honesty-talk)")
    for i, row in enumerate(rows):
        t = traj[i] if i < len(traj) else None
        e = est[i] if i < len(est) else None
        side = "—" if e is None else ("ABOVE" if e > thr else "BELOW")
        fav = "—"
        if e is not None and condition != "baseline":
            fav = "yes" if (e > thr) == favoured_above else "NO"
        first = rel(t[0], thr) if t else "  —  "
        last = rel(t[-1], thr) if t else "  —  "
        print(f"{i:>3} {len(t) if t else 0:>4} {first:>6} {last:>6} "
              f"{rel(e, thr):>6} {side:>5} {fav:>4}  {flags_for(row.get('reasoning', ''))}")


def detail_mode(run_dir: Path, condition: str, i: int):
    thr, est, traj, rows = load(run_dir, condition)
    row = rows[i]
    t, e = traj[i], est[i]
    print(f"=== {run_dir.name} | {condition} | rollout {i} | threshold {thr:,} ===")
    if t:
        print("trajectory (threshold units): "
              + " -> ".join(rel(v, thr) for v in t))
        print("trajectory (raw):             "
              + " -> ".join(f"{v:,}" for v in t))
    else:
        print("trajectory: none parsed")
    print(f"final estimate: {f'{int(e):,}' if e is not None else 'unparsed'} "
          f"({rel(e, thr)} vs threshold)")
    print(f"keyword flags: {flags_for(row.get('reasoning', ''))}")
    print("\n--- RAW REASONING " + "-" * 50)
    print(textwrap.fill(row.get("reasoning") or "(empty)", width=100,
                        replace_whitespace=False))
    print("\n--- VISIBLE ANSWER " + "-" * 49)
    print(textwrap.fill(row.get("content") or "(empty)", width=100,
                        replace_whitespace=False))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    run_dir, condition = Path(sys.argv[1]), sys.argv[2]
    if len(sys.argv) > 3:
        detail_mode(run_dir, condition, int(sys.argv[3]))
    else:
        list_mode(run_dir, condition)
