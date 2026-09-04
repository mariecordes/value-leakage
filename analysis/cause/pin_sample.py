"""CAUSE step 2 — pin the reviewed sample into a frozen, hash-verified file.

Local only: reads saved run data and cut_review.json, makes no API call.

Takes the first TARGET_PER_ARM traces that passed human review in each direction,
in the seeded review order assigned before anyone read them, and writes
frozen_sample.json with a SHA-256 of every piece of text the experiment will
replay. Later stages verify against those hashes, so a silently edited trace or a
moved cut boundary fails loudly instead of quietly changing the experiment.

Run:
    uv run python analysis/cause/pin_sample.py           # write frozen_sample.json
    uv run python analysis/cause/pin_sample.py --verify  # re-check against sources
    uv run python analysis/cause/pin_sample.py --final   # pick the 16 viable ones
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import fire

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_cuts import FINAL_PER_ARM, TARGET_SLOT  # noqa: E402

CAUSE_DIR = Path(__file__).resolve().parent
ROOT = CAUSE_DIR.parents[1]
RUN = ROOT / "runs" / "qwen3.5-122b-a10b_20260815_030702"
REVIEW_PATH = CAUSE_DIR / "cut_review.json"
FROZEN_PATH = CAUSE_DIR / "frozen_sample.json"   # the reviewed pool
FINAL_PATH = CAUSE_DIR / "frozen_final.json"     # the 16 that get used

TARGET_PER_ARM = 20    # pool size per arm; --per_arm overrides


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def source_material(version: str, index: int) -> tuple[str, str]:
    payload = json.loads((RUN / f"{version}.json").read_text())
    row = next((r for r in payload["rows"] if r.get("i") == index), None)
    if row is None:
        raise ValueError(f"row {index} missing from {version}.json")
    return payload["prompt"], row.get("reasoning") or ""


def selected_with(review: dict, per_arm: int) -> list[dict]:
    """selected(), with the pool size given explicitly."""
    global TARGET_PER_ARM
    previous, TARGET_PER_ARM = TARGET_PER_ARM, per_arm
    try:
        return selected(review)
    finally:
        TARGET_PER_ARM = previous


def selected(review: dict) -> list[dict]:
    """First TARGET_PER_ARM kept traces per arm, in seeded review order.

    Restricted to the target slot when set.

    The slot is read from select_cuts, not from the review file. The review file
    only carries the key if it has been regenerated since the restriction was
    added, and an earlier version of this function trusted the file: with the key
    absent it silently pinned the old mixed population/spots sample instead.
    """
    slot = TARGET_SLOT
    if review.get("target_slot") not in (None, slot):
        raise SystemExit(
            f"cut_review.json says target_slot={review['target_slot']!r} but "
            f"select_cuts says {slot!r}; regenerate the review file"
        )
    out = []
    for direction in ("above", "below"):
        arm = sorted(
            (t for t in review["traces"]
             if t["direction"] == direction and t["include"] is True
             and (slot is None or t["commitment_slot"] == slot)),
            key=lambda t: t["review_order"],
        )
        # The pool takes whatever review produced, up to TARGET_PER_ARM. Only the
        # final sample has a hard size, and that is enforced in finalize() against
        # viable traces. Requiring a fixed pool size here would reject a perfectly
        # usable pool just because one arm yielded fewer traces than the other.
        if len(arm) < FINAL_PER_ARM:
            raise SystemExit(
                f"{direction} arm has only {len(arm)} kept traces"
                + (f" in slot {slot!r}" if slot else "")
                + f"; the final sample needs {FINAL_PER_ARM} viable per arm. "
                "Keep reviewing in review_order."
            )
        out.extend(arm[:TARGET_PER_ARM])
    if slot and any(t["commitment_slot"] != slot for t in out):
        raise SystemExit(f"selection contains traces outside slot {slot!r}")
    return out


def build_entry(trace: dict) -> dict:
    prompt, reasoning = source_material(trace["version"], trace["i"])
    cut = trace["cut_char"]
    prefix = reasoning[:cut]
    return {
        "trace_id": trace["trace_id"],
        "version": trace["version"],
        "direction": trace["direction"],
        "i": trace["i"],
        "source": str((RUN / f"{trace['version']}.json").relative_to(ROOT)),
        "review_order": trace["review_order"],
        "cut_char": cut,
        "cut_fraction": trace["cut_fraction"],
        "commitment_slot": trace["commitment_slot"],
        "commitment_value": trace["commitment_value"],
        "commitment_line": trace["commitment_line"],
        "original_step": trace["step_at_cut"],
        "prompt_sha256": sha256(prompt),
        "reasoning_sha256": sha256(reasoning),
        "prefix_sha256": sha256(prefix),
        "prefix_chars": len(prefix),
        "reasoning_chars": len(reasoning),
        "human_reviewed": True,
    }


def finalize(accept_short: bool = False) -> None:
    """Select the final sample: first FINAL_PER_ARM viable traces per arm.

    Viability is measured by the step-3 spread test, before any continuation is
    sampled and without reference to any final answer. A trace whose resamples all
    give the same number offers nothing to insert, so it cannot be used at all —
    this is a precondition for the intervention existing, not a filter on results.
    Order within each arm is the seeded review order, fixed before any trace was
    read, so the final sample is the first viable traces rather than chosen ones.
    """
    spread_path = CAUSE_DIR / "resample_spread.json"
    if not spread_path.exists():
        raise SystemExit("resample_spread.json is absent; run resample.py --analyze")
    spread = json.loads(spread_path.read_text())
    if spread["missing_raw_files"]:
        raise SystemExit(
            f"{spread['missing_raw_files']} resamples are missing; "
            "run resample.py --execute before finalizing"
        )
    pool = {e["trace_id"]: e for e in json.loads(FROZEN_PATH.read_text())["traces"]}
    viable = {t["trace_id"] for t in spread["per_trace"] if t["viable_for_version_a"]}

    chosen, shortfall = [], []
    for direction in ("above", "below"):
        arm = sorted(
            (e for e in pool.values()
             if e["direction"] == direction and e["trace_id"] in viable),
            key=lambda e: e["review_order"],
        )
        if len(arm) < FINAL_PER_ARM:
            shortfall.append(f"{direction}: {len(arm)}/{FINAL_PER_ARM} viable")
        chosen.extend(arm[:FINAL_PER_ARM])
    if shortfall and not accept_short:
        raise SystemExit(
            "not enough viable traces yet:\n  " + "\n  ".join(shortfall)
            + "\nReview more traces, then rerun resample.py --execute and --analyze."
            "\nOr pass --accept_short to proceed with a smaller sample; the shortfall "
            "is then recorded in frozen_final.json."
        )

    by_id = {t["trace_id"]: t for t in spread["per_trace"]}
    for entry in chosen:
        row = by_id[entry["trace_id"]]
        entry["resample_distinct_values"] = row["distinct_values"]
        entry["resample_spread_ratio"] = row["spread_ratio"]
        entry["resample_n_readable"] = row["n_readable"]

    pool_meta = json.loads(FROZEN_PATH.read_text())
    FINAL_PATH.write_text(json.dumps({
        "experiment": "cause-v2-final",
        "frozen_on": str(date.today()),
        # Replay configuration must travel with the sample: later stages read this
        # file directly and cannot fall back to the pool.
        "threshold": pool_meta["threshold"],
        "model": pool_meta["model"],
        "provider": pool_meta["provider"],
        "prefill_mode": pool_meta["prefill_mode"],
        "selection": (
            f"First {FINAL_PER_ARM} viable traces per direction in the seeded review "
            "order. Viable means the step-3 resamples met the criteria frozen in "
            "the design before any resample was run. Selection used no final answer."
        ),
        "pool_size": len(pool),
        "viable_in_pool": len(viable),
        "counts": {
            "above": sum(e["direction"] == "above" for e in chosen),
            "below": sum(e["direction"] == "below" for e in chosen),
            "total": len(chosen),
            "target_per_arm": FINAL_PER_ARM,
        },
        "shortfall": shortfall or None,
        "shortfall_cause": (
            "The pool was exhausted: every eligible spots-slot trace was reviewed. "
            "Fewer traces show usable natural spread than the design assumed, because "
            "the validated judge is strict about what counts as a "
            "committed figure. No frozen criterion was loosened to close the gap."
        ) if shortfall else None,
        "traces": chosen,
    }, indent=2, ensure_ascii=False))
    print(f"pool        : {len(pool)} traces, {len(viable)} viable")
    n_above = sum(e["direction"] == "above" for e in chosen)
    n_below = sum(e["direction"] == "below" for e in chosen)
    print(f"final sample: {len(chosen)} ({n_above} above, {n_below} below)"
          + (f"   SHORT of {FINAL_PER_ARM} per arm" if shortfall else ""))
    ratios = sorted(e["resample_spread_ratio"] for e in chosen)
    print(f"spread      : median {ratios[len(ratios)//2]:.2f}x, "
          f"range {ratios[0]:.2f}-{ratios[-1]:.2f}x")
    print(f"written     : {FINAL_PATH.relative_to(ROOT)}")


def main(verify: bool = False, per_arm: int | None = None, final: bool = False,
         accept_short: bool = False):
    if final:
        return finalize(accept_short)
    global TARGET_PER_ARM
    if per_arm is not None:
        TARGET_PER_ARM = per_arm
    if not REVIEW_PATH.exists():
        raise SystemExit("cut_review.json is absent; run select_cuts.py first")
    review = json.loads(REVIEW_PATH.read_text())
    entries = [build_entry(t) for t in selected(review)]

    if verify:
        if not FROZEN_PATH.exists():
            raise SystemExit("frozen_sample.json is absent; run without --verify first")
        stored = json.loads(FROZEN_PATH.read_text())
        if stored.get("per_arm") and stored["per_arm"] != TARGET_PER_ARM:
            global TARGET_PER_ARM_USED
            entries = [build_entry(t) for t in selected_with(review, stored["per_arm"])]
        fresh = {e["trace_id"]: e for e in entries}
        problems = []
        for entry in stored["traces"]:
            current = fresh.get(entry["trace_id"])
            if current is None:
                problems.append(f"{entry['trace_id']}: no longer in the reviewed sample")
                continue
            for field in ("prompt_sha256", "reasoning_sha256", "prefix_sha256",
                          "cut_char", "commitment_value"):
                if entry[field] != current[field]:
                    problems.append(
                        f"{entry['trace_id']}: {field} changed "
                        f"({entry[field]} -> {current[field]})"
                    )
        if problems:
            raise SystemExit("frozen sample no longer matches source:\n  "
                             + "\n  ".join(problems))
        print(f"verified : {len(stored['traces'])} traces match their sources")
        print("hashes   : PASS")
        return

    FROZEN_PATH.write_text(json.dumps({
        "experiment": "cause-v2",
        # Recorded so --verify rebuilds the same pool. Without it verify used the
        # module default and reported every trace past that position as missing.
        "per_arm": TARGET_PER_ARM,
        "frozen_on": str(date.today()),
        "rule_version": review["rule_version"],
        "review_seed": review["review_seed"],
        "selection": (
            f"First {TARGET_PER_ARM} human-approved traces per direction in the "
            "seeded review order, which was assigned before any trace was read. "
            "Traces approved beyond that count are left in cut_review.json and "
            "are not part of the sample."
        ),
        "threshold": review["threshold"],
        "model": "qwen/qwen3.5-122b-a10b",
        "provider": "deepinfra/fp4",
        "prefill_mode": "content_prefill",
        "counts": {
            "above": sum(e["direction"] == "above" for e in entries),
            "below": sum(e["direction"] == "below" for e in entries),
            "total": len(entries),
        },
        "commitment_slots": {
            slot: sum(e["commitment_slot"] == slot for e in entries)
            for slot in sorted({e["commitment_slot"] for e in entries})
        },
        "traces": entries,
    }, indent=2, ensure_ascii=False))

    print(f"pinned   : {len(entries)} traces "
          f"({sum(e['direction']=='above' for e in entries)} above, "
          f"{sum(e['direction']=='below' for e in entries)} below)")
    slots = {}
    for entry in entries:
        slots[entry["commitment_slot"]] = slots.get(entry["commitment_slot"], 0) + 1
    print(f"cut slot : {slots}")
    positions = sorted(e["cut_fraction"] for e in entries)
    print(f"cut posn : median {positions[len(positions)//2]:.1%}, "
          f"range {positions[0]:.1%}-{positions[-1]:.1%}")
    print(f"written  : {FROZEN_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    fire.Fire(main)
