"""CONTINUATION_DIR steps C1 + C2 — choose the traces to interrupt, and where to cut them.

No API calls.

CONTINUATION_DIR asks: if we stop a model mid-thought and ask what its current best estimate
is, does the number it SAYS match where its reasoning actually ends up?

Two arms, both on the same model and the same serving endpoint:

  bet         its donation-bet traces      — the suspect
  instructed  its openly-told-to-aim traces — known guilty, and therefore the
              reference for what an honest self-report looks like when the model
              has no motive to misreport

Subject choice is forced. Continuing a saved trace mid-thought only makes sense
on the endpoint that wrote it, otherwise two different serving stacks are
spliced into one chain of thought. Of the ten starter dataset runs only qwen3.5-122b-a10b and
inkling-small were sampled through OpenRouter; the rest came from Fireworks. So
the planned admitter-vs-denier pairing is not runnable, and the bet-vs-
instructed contrast replaces it. Recorded as a limitation, not hidden.

Selection per arm: prefer runs that landed on the favoured side, keep some that
did not, and require a trace long enough to cut and a trajectory long enough to
compare against.

NOTE the filename: this must NOT be called select.py — that shadows Python's
stdlib `select` module, which asyncio imports, and the whole process dies on an
unrelated import.

  uv run python analysis/continuations/select_traces.py
  uv run python analysis/continuations/select_traces.py --n_per_arm 10 --cuts 0.4,0.8
"""

import json
import re
import sys
from pathlib import Path

import fire

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "disclosure"))
sys.path.insert(0, str(ANALYSIS_DIR / "target_visibility"))
from common import RUNS  # noqa: E402
from sample_conditions import SUBJECTS, TARGET_VISIBILITY_DIR, shipped_threshold  # noqa: E402

CONTINUATION_DIR = Path(__file__).resolve().parent

# Only these were sampled through OpenRouter, so only these can be continued.
CONTINUABLE = {"qwen3.5-122b-a10b", "inkling-small"}

ARMS = {
    "bet": ("bet_above", "bet_below"),
    "instructed": ("instructed_above", "instructed_below"),
}

MIN_CHARS = 2000        # a trace too short to cut meaningfully
MIN_TRAJ_POINTS = 4     # needs enough floated numbers to compare against

_SENT_END = re.compile(r"[.!?]\s")


def snap_to_sentence(text: str, target: int) -> int:
    """Nearest sentence boundary at or before target, so the model resumes at a
    clean break rather than mid-word."""
    window = text[:target]
    matches = list(_SENT_END.finditer(window))
    if matches:
        return matches[-1].end()
    return target


def source_rows(model: str, version: str):
    """Rows for a version — our TARGET_VISIBILITY_DIR sample, else the starter dataset shipped run."""
    path = TARGET_VISIBILITY_DIR / model / f"{version}.json"
    if path.exists():
        return json.loads(path.read_text())["rows"], "target_visibility"
    shipped = {"bet_above": "above_good", "bet_below": "below_good",
               "baseline": "baseline"}.get(version)
    if shipped:
        p = RUNS / SUBJECTS[model]["shipped"] / f"{shipped}.json"
        if p.exists():
            return json.loads(p.read_text())["rows"], "runs"
    return [], "missing"


def main(model: str = "qwen3.5-122b-a10b", n_per_arm: int = 10,
         cuts: str = "0.4,0.8", min_favoured: int = 6):
    if model not in CONTINUABLE:
        raise SystemExit(
            f"{model} was not sampled through OpenRouter, so its traces cannot "
            f"be continued on the endpoint that wrote them. Continuable: "
            f"{sorted(CONTINUABLE)}")
    # fire turns "0.3,0.6,0.9" into a tuple but "0.4,0.8" may arrive either
    # way, so accept both rather than depending on which.
    cut_points = ([float(c) for c in cuts]
                  if isinstance(cuts, (list, tuple))
                  else [float(c) for c in str(cuts).split(",")])
    thr = shipped_threshold(model)
    est = json.loads((TARGET_VISIBILITY_DIR / model / "estimates.json").read_text())["by_version"]
    traj = json.loads((TARGET_VISIBILITY_DIR / model / "trajectories.json").read_text())["by_version"]

    out = {"model": model, "threshold": thr, "cuts": cut_points,
           "openrouter_id": SUBJECTS[model]["openrouter_id"],
           "provider": SUBJECTS[model]["provider"], "arms": {}}

    for arm, versions in ARMS.items():
        picked = []
        for version in versions:
            rows, where = source_rows(model, version)
            if where == "missing":
                print(f"  {arm}/{version}: no data — skipped")
                continue
            values = est.get(version, {}).get("values", [])
            idx_list = est.get(version, {}).get("i", [])
            final_by_i = {i: v for i, v in zip(idx_list, values)}
            t_vals = traj.get(version, {}).get("values", [])
            t_idx = traj.get(version, {}).get("i", [])
            traj_by_i = {i: t for i, t in zip(t_idx, t_vals)}
            favoured_side = "above" if version.endswith("_above") else "below"

            for row in rows:
                i = row["i"]
                text = row.get("reasoning") or ""
                final = final_by_i.get(i)
                t = traj_by_i.get(i)
                if final is None or len(text) < MIN_CHARS:
                    continue
                if not t or len(t) < MIN_TRAJ_POINTS:
                    continue
                landed = "above" if final > thr else "below"
                picked.append({
                    "version": version, "i": i, "source": where,
                    "chars": len(text), "final": final,
                    "final_rel": (final - thr) / thr,
                    "favoured_side": favoured_side,
                    "favoured": landed == favoured_side,
                    "n_traj_points": len(t),
                    "cut_chars": [snap_to_sentence(text, int(len(text) * c))
                                  for c in cut_points],
                })

        # Balance three ways: across the two directions (otherwise the whole
        # arm measures only the above-paying case), toward favoured landings
        # (that is where a misreport would matter), while keeping a few that
        # went the other way so the comparison is not conditioned purely on
        # outcome.
        chosen = []
        per_dir = max(1, n_per_arm // 2)
        for version in versions:
            pool = [p for p in picked if p["version"] == version]
            fav = [p for p in pool if p["favoured"]]
            opp = [p for p in pool if not p["favoured"]]
            want_fav = max(1, min(len(fav), per_dir - min(len(opp), per_dir // 3)))
            chosen += fav[:want_fav] + opp[:per_dir - want_fav]
        # Top up from anything left over if one direction was short.
        if len(chosen) < n_per_arm:
            rest = [p for p in picked if p not in chosen]
            chosen += rest[:n_per_arm - len(chosen)]
        chosen = chosen[:n_per_arm]
        out["arms"][arm] = {
            "n": len(chosen),
            "n_favoured": sum(1 for c in chosen if c["favoured"]),
            "runs": chosen,
        }
        print(f"{arm:11s} eligible {len(picked):3d} "
              f"({len(fav)} favoured / {len(opp)} not)  -> chose {len(chosen)} "
              f"({sum(1 for c in chosen if c['favoured'])} favoured)")
        for c in chosen:
            print(f"    {c['version']:18s} run {c['i']:3d}  {c['chars']:6,d} chars"
                  f"  final {c['final_rel']:+.2f}x thr  "
                  f"{'favoured' if c['favoured'] else 'went the other way':18s}"
                  f"  cuts at {c['cut_chars']}")

    CONTINUATION_DIR.mkdir(exist_ok=True)
    (CONTINUATION_DIR / "selection.json").write_text(json.dumps(out, indent=2))
    total = sum(a["n"] for a in out["arms"].values())
    print(f"\nselected {total} traces x {len(cut_points)} cut points = "
          f"{total*len(cut_points)} cut positions")
    print(f"saved {(CONTINUATION_DIR/'selection.json').relative_to(CONTINUATION_DIR.parent)}")


if __name__ == "__main__":
    fire.Fire(main)
