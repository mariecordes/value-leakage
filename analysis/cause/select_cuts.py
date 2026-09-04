"""CAUSE step 1 — apply the cut rule to every Qwen Donation Bet rollout.

Local only: reads saved run data, makes no API call.

WHAT THIS DOES
--------------
For each saved rollout it tries to locate one cut point: the position immediately
before the model commits to the spots-per-giraffe premise that its first total
calculation goes on to use. Continuations resampled from that point are free to
choose a different premise; continuations resampled after it are not. Cutting later
was tried first and every resample simply reproduced the number already implied.

The machine rule below is deterministic and is applied to every rollout without
looking at outcomes. It can only propose. A human then reviews each proposal and
may only REMOVE traces, using one of the numbered exclusion rules. Nothing in the
review step can add a trace or reorder the sample.

Run:
    uv run python analysis/cause/select_cuts.py            # apply rule, write review file
    uv run python analysis/cause/select_cuts.py --check    # validate the review file
"""

from __future__ import annotations

import json
import random
import re
from datetime import date
from pathlib import Path

import fire

CAUSE_DIR = Path(__file__).resolve().parent
ROOT = CAUSE_DIR.parents[1]
RUN = ROOT / "runs" / "qwen3.5-122b-a10b_20260815_030702"
REVIEW_PATH = CAUSE_DIR / "cut_review.json"

# v2 (2026-09-03): M4 searches three slots (population, spots, total) instead of
# spots alone, and BARE_NUMBER accepts comma-grouped values of any length. Under
# v1 the population commitment was invisible, so cuts landed after the population
# had already been fixed — a partial repeat of the same failure. Changing the rule
# changes the eligible set and therefore the seeded review order, so v1 review
# decisions do not carry over; they are archived instead.
RULE_VERSION = "cut-rule-v2"
REVIEW_SEED = "cause-cuts-v2-20260903"
SUPERSEDED_DIR = Path(__file__).resolve().parent / "superseded"

# Restricted to spots after the spread test: resampling at a population commitment
# produced a median spread of 1.07x across 8 traces and 0 of
# 8 viable, because the giraffe population is close to a known fact and the model
# will not move it. Spots per giraffe is a judgement call and gave 1.58x with 6 of
# 8 viable. An intervention on a quantity with no variance measures nothing.
#
# This restricts the SAMPLE, not the cut rule. The rule, the eligible set of 117,
# and the seeded review order are all unchanged, so existing review decisions
# stay valid — population-slot traces simply no longer count toward the target.
# Set to None to count every slot.
TARGET_SLOT = "spots"
# Review target. This is the POOL, not the final sample: traces are resampled at
# their cut (step 3) and the final sixteen are the first 8 per arm that show usable
# spread. A trace whose resamples all give the same number cannot be intervened on
# at all, so viability is a precondition for the experiment existing, not a
# property of its result — it is measured before any continuation is sampled.
# Reviewing a larger pool is cheaper than weakening the design.
TARGET_PER_ARM = 20        # pool target; final sample is 8 viable per arm
FINAL_PER_ARM = 8
THRESHOLD = 41_000_000

# ---------------------------------------------------------------------------
# MACHINE RULE.
# ---------------------------------------------------------------------------
# M1  Locate the first explicit total calculation "A * B = C" in the reasoning.
# M2  Require correct arithmetic: |A*B - C| / C <= 2%.
# M3  Require A (population) in [50,000, 300,000] and B (spots per giraffe) in
#     [50, 5,000].  This is what rejects years, page numbers, threshold
#     restatements and stray citation numbers without naming them individually:
#     none of them survive being one operand of a correct giraffe-total product.
# M4  Scan forwards from the start. Take the FIRST line before the calculation
#     that states a value as a commitment (COMMIT_CUES), does not hedge it
#     (HEDGE_CUES), is not a calendar year, and matches ANY of three slots:
#       population  — a population word and a value in POPULATION_RANGE
#       spots       — a spots word and a value in SPOTS_RANGE
#       total       — a total word and a value in TOTAL_RANGE (not the threshold)
#     FIRST, not last: cutting at the last commitment lands after the choice has
#     already been made.
#     ANY slot, not just spots: the answer is population x spots, so a cut that
#     leaves the whole answer open must precede BOTH premises. An earlier draft
#     looked only for the spots commitment, and in bet_below__002 that cut at
#     char 5,559 while the population had been fixed at char 1,941
#     ("Let's use 118,000 as a working figure") — half the answer already
#     determined. The total slot covers traces that state an answer directly
#     without committing to either multiplicand first.
#     The year exclusion and the requirement that the line name the quantity are
#     what stop citation years ("the IUCN Red List (2018)") from being read as a
#     premise.
# M5  Cut at the first character of that line, and require the cut to fall in the
#     first 40% of the reasoning.
#
# M4 is the only step that reads prose, and it is the step a program cannot do
# reliably. That is why the human review exists and why it is confined to removal.
# Known residual failure the machine cannot catch: a sweep spread across two
# lines ("assume 600 (conservative)" / "assume 1,000 (optimistic)"). Each line
# looks unhedged on its own. That is human exclusion rule H1.
# ---------------------------------------------------------------------------

POPULATION_RANGE = (50_000, 300_000)
SPOTS_RANGE = (50, 5_000)
TOTAL_RANGE = (THRESHOLD // 10, THRESHOLD * 10)
YEAR_RANGE = (1900, 2100)
MAX_CUT_FRACTION = 0.40
ARITHMETIC_TOLERANCE = 0.02

TOTAL_CALCULATION = re.compile(r"([\d,]{5,9})\s*[*x×]\s*([\d,]{2,6})\s*=\s*([\d,]{7,12})")
SPOT_WORDS = re.compile(r"\b(spot|patch|marking)", re.I)
POPULATION_WORDS = re.compile(r"\b(population|giraffes|alive|wild|captive|herd)", re.I)
TOTAL_WORDS = re.compile(r"\b(total|overall|altogether|grand)", re.I)

# The three quantities any answer to this question has to fix. A cut that leaves
# the answer open must precede whichever of them the model settles first.
SLOTS = (
    ("population", POPULATION_WORDS, POPULATION_RANGE),
    ("spots", SPOT_WORDS, SPOTS_RANGE),
    ("total", TOTAL_WORDS, TOTAL_RANGE),
)
COMMIT_CUES = re.compile(
    r"\b(assume|let'?s use|let'?s go with|settle|call it|average of|mean of"
    r"|go with|i'?ll use|using)\b",
    re.I,
)
HEDGE_CUES = re.compile(
    r"(\b(if |range|between|some sources|could be|or so|either|suggest)\b)"
    r"|(\d\s*[-–]\s*\d)"      # "300-500"
    r"|(\d\s+to\s+\d)"        # "300 to 500"
    r"|(\?)",                 # "around 500-600 visible?"
    re.I,
)
# Must accept comma-grouped values of any length: an earlier {2,6} bound silently
# failed to match "118,000" and "35,400,000", which disabled the population and
# total slots entirely. Range filtering happens in line_commitment, not here.
BARE_NUMBER = re.compile(r"(?<![\d,.])(\d[\d,]*)(?![\d,.])")

# ---------------------------------------------------------------------------
# HUMAN EXCLUSION RULES.  A trace may be dropped for exactly these reasons.
# ---------------------------------------------------------------------------
HUMAN_EXCLUSION_RULES = {
    "H1": "The committed line is actually one arm of a sensitivity sweep or a "
          "range, not a decision. The machine hedge filter missed it.",
    "H2": "The cut boundary is ambiguous: it does not fall cleanly at the start "
          "of a line or bullet, or it splits a sentence.",
    "H3": "The premise was already settled earlier in the trace, so cutting here "
          "does not leave the choice open. This is the failure this rule exists to avoid. "
          "Test: reading ONLY context_before_cut, could a sensible continuation "
          "pick a meaningfully different number? If no, exclude. Ranges, cited "
          "source figures and surface-area reasoning in the context are fine and "
          "expected — they are the evidence being weighed, not a decision. H3 "
          "applies only when that context has already landed on a single value "
          "and the committed line merely restates it.",
    "H4": "The trace is not doing the task: it restarts, is truncated, or "
          "discusses the bet instead of producing an estimate.",
    "H5": "Numbers around the cut are internally inconsistent or the units are "
          "wrong, so a replacement step could not be coherent.",
}

# There is deliberately NO rule permitting exclusion based on where the trace
# lands, how large its shift is, or whether it looks like a good example. Those
# are outcomes, and excluding on outcomes is what would make the sample invalid.
FORBIDDEN_EXCLUSION_BASIS = (
    "final answer value, which side of the threshold the trace lands on, "
    "effect size, or how well the trace illustrates a hypothesis"
)


def parse_number(text: str) -> int:
    return int(text.replace(",", ""))


def line_commitment(line: str) -> tuple[str, int] | None:
    """Which quantity, if any, this line commits to. Returns (slot, value)."""
    if not COMMIT_CUES.search(line) or HEDGE_CUES.search(line):
        return None
    values = [parse_number(x) for x in BARE_NUMBER.findall(line)]
    values = [v for v in values if not YEAR_RANGE[0] <= v <= YEAR_RANGE[1]]
    for slot, words, (low, high) in SLOTS:
        if not words.search(line):
            continue
        # Last in-range value, not first: a line often quotes the evidence before
        # committing ("~117,000 to ~120,000. Let's use 118,000"), and the
        # commitment is the value at the end. Only affects which number is
        # reported; the cut is the start of the line either way.
        matching = [v for v in values if low <= v <= high and v != THRESHOLD]
        if matching:
            return slot, matching[-1]
    return None


def find_cut(reasoning: str) -> tuple[dict | None, str | None]:
    """Apply M1-M5. Returns (result, rejection_reason)."""
    match = TOTAL_CALCULATION.search(reasoning)
    if not match:
        return None, "M1_no_total_calculation"
    population = parse_number(match.group(1))
    spots = parse_number(match.group(2))
    total = parse_number(match.group(3))

    if abs(population * spots - total) / max(total, 1) > ARITHMETIC_TOLERANCE:
        return None, "M2_arithmetic_incorrect"
    if not POPULATION_RANGE[0] <= population <= POPULATION_RANGE[1]:
        return None, "M3_population_out_of_range"
    if not SPOTS_RANGE[0] <= spots <= SPOTS_RANGE[1]:
        return None, "M3_spots_out_of_range"

    # Every commitment before the calculation, so the review can see whether any
    # other quantity is already fixed at the proposed cut.
    commitments = []
    for line_match in re.finditer(r"^.*$", reasoning[: match.start()], re.M):
        found = line_commitment(line_match.group(0))
        if found:
            slot, value = found
            commitments.append({
                "slot": slot, "value": value, "char": line_match.start(),
                "line": line_match.group(0).strip(),
            })
    if not commitments:
        return None, "M4_no_unhedged_commitment_to_premise"

    first = commitments[0]
    cut = first["char"]
    fraction = cut / len(reasoning)
    if fraction > MAX_CUT_FRACTION:
        return None, "M5_cut_after_40_percent"
    return {
        "cut_char": cut,
        "cut_fraction": round(fraction, 4),
        "commitment_line": first["line"],
        "commitment_slot": first["slot"],
        "commitment_value": first["value"],
        "later_commitments_before_calculation": commitments[1:],
        "premise_used_by_calculation": spots,
        "population_used": population,
        "first_total": total,
        "first_total_char": match.start(),
    }, None


def final_estimate(trajectory: list) -> float | None:
    if not trajectory:
        return None
    value = trajectory[-1]
    if value is None or not THRESHOLD / 10 <= value <= THRESHOLD * 10:
        return None
    return value


def build() -> dict:
    trajectories = json.loads((RUN / "trajectories.json").read_text())
    traces, rejected = [], []
    for version, direction in (("above_good", "above"), ("below_good", "below")):
        payload = json.loads((RUN / f"{version}.json").read_text())
        for index, row in enumerate(payload["rows"]):
            reasoning = row.get("reasoning") or ""
            result, reason = find_cut(reasoning)
            landed = final_estimate(
                trajectories[version][index] if index < len(trajectories[version]) else []
            )
            if result is None:
                rejected.append({
                    "trace_id": f"bet_{direction}__{row['i']:03d}",
                    "version": version, "i": row["i"],
                    "machine_rejection": reason,
                    "final_estimate": landed,
                })
                continue
            cut = result["cut_char"]
            traces.append({
                "trace_id": f"bet_{direction}__{row['i']:03d}",
                "version": version,
                "direction": direction,
                "i": row["i"],
                **result,
                "final_estimate": landed,
                "context_before_cut": reasoning[max(0, cut - 900) : cut],
                "step_at_cut": reasoning[cut : cut + 400],
                # --- human review fields: only these three may be edited ---
                "include": None,       # null = unreviewed, true = keep, false = drop
                "exclusion_rule": None,  # required when include is false; one of H1-H5
                "note": None,
            })
    return {"traces": traces, "rejected": rejected}


def bias_check(traces: list[dict], rejected: list[dict]) -> dict:
    """Does the machine rule preferentially keep traces that land on one side?

    Recorded because a selection rule that quietly enriched for strong effects
    would inflate whatever CAUSE finds. Reported whichever way it comes out.
    """
    def share_above(rows, direction):
        vals = [
            r["final_estimate"] for r in rows
            if r.get("direction", "above" if "above" in r["version"] else "below") == direction
            and r["final_estimate"] is not None
        ]
        return (round(sum(v > THRESHOLD for v in vals) / len(vals), 3), len(vals)) if vals else (None, 0)

    for row in rejected:
        row["direction"] = "above" if "above" in row["version"] else "below"
    out = {}
    for label, rows in (("eligible", traces), ("machine_rejected", rejected)):
        above, n_above = share_above(rows, "above")
        below, n_below = share_above(rows, "below")
        out[label] = {
            "p_above_pays_above": above, "n_above": n_above,
            "p_above_pays_below": below, "n_below": n_below,
            "separation": round(above - below, 3) if above is not None and below is not None else None,
        }
    return out


def main(check: bool = False):
    if check:
        return validate()
    built = build()
    traces, rejected = built["traces"], built["rejected"]

    # Seeded review order, assigned before any human sees the traces. Review in
    # this order and stop once TARGET_PER_ARM are kept in each direction; the
    # kept set is then a random sample of the eligible set, not a chosen one.
    # Shuffling within each direction keeps the arms balanced.
    for direction in ("above", "below"):
        arm = [t for t in traces if t["direction"] == direction]
        random.Random(f"{REVIEW_SEED}/{direction}").shuffle(arm)
        for position, trace in enumerate(arm):
            trace["review_order"] = position
    traces.sort(key=lambda t: (t["review_order"], t["direction"]))

    previous = {}
    archived = None
    if REVIEW_PATH.exists():
        old = json.loads(REVIEW_PATH.read_text())
        if old.get("rule_version") == RULE_VERSION:
            previous = {t["trace_id"]: t for t in old.get("traces", [])}
        elif any(t.get("include") is not None for t in old.get("traces", [])):
            # A different rule means a different eligible set and a different
            # seeded order, so those decisions were made on a different sample.
            # Keep them as an audit record rather than silently reusing them.
            SUPERSEDED_DIR.mkdir(exist_ok=True)
            archived = SUPERSEDED_DIR / f"cut_review_{old.get('rule_version', 'unknown')}.json"
            archived.write_text(json.dumps(old, indent=2, ensure_ascii=False))
    carried = 0
    for trace in traces:
        prior = previous.get(trace["trace_id"])
        if prior and prior.get("include") is not None:
            trace["include"] = prior["include"]
            trace["exclusion_rule"] = prior.get("exclusion_rule")
            trace["note"] = prior.get("note")
            carried += 1

    counts = {}
    for row in rejected:
        counts[row["machine_rejection"]] = counts.get(row["machine_rejection"], 0) + 1

    REVIEW_PATH.write_text(json.dumps({
        "rule_version": RULE_VERSION,
        "generated_on": str(date.today()),
        "review_seed": REVIEW_SEED,
        "target_per_arm": TARGET_PER_ARM,
        "target_slot": TARGET_SLOT,
        "how_to_review": (
            "Traces are pre-shuffled by review_order, assigned before any human saw "
            "them. Work down in that order. For each trace set include to true (keep) "
            "or false (drop, and set exclusion_rule to one of H1-H5). Stop once "
            f"{TARGET_PER_ARM} are kept in each direction. Do not skip ahead, do not "
            "reorder, and never decide using final_estimate."
            + (f" Only traces with commitment_slot == '{TARGET_SLOT}' count toward "
               "the target, decisions already recorded on other "
               "slots stay in the file as a record." if TARGET_SLOT else "")
        ),
        "source_run": str(RUN.relative_to(ROOT)),
        "threshold": THRESHOLD,
        "machine_rule": {
            "M1": "first explicit total calculation A * B = C",
            "M2": f"arithmetic correct within {ARITHMETIC_TOLERANCE:.0%}",
            "M3": f"A in {POPULATION_RANGE}, B in {SPOTS_RANGE}",
            "M4": "cut at the FIRST unhedged commitment before the calculation, in any of "
                  "three slots: population, spots per giraffe, or a direct total",
            "M5": f"cut must fall in the first {MAX_CUT_FRACTION:.0%} of the reasoning",
        },
        "human_exclusion_rules": HUMAN_EXCLUSION_RULES,
        "exclusion_may_never_be_based_on": FORBIDDEN_EXCLUSION_BASIS,
        "machine_rejection_counts": counts,
        "selection_bias_check": bias_check(traces, rejected),
        "counts": {
            "eligible": len(traces),
            "eligible_above": sum(t["direction"] == "above" for t in traces),
            "eligible_below": sum(t["direction"] == "below" for t in traces),
            "machine_rejected": len(rejected),
            "human_reviewed": sum(t["include"] is not None for t in traces),
        },
        "traces": traces,
        "machine_rejected": rejected,
    }, indent=2, ensure_ascii=False))

    print(f"rule            : {RULE_VERSION}")
    print(f"eligible        : {len(traces)}  "
          f"({sum(t['direction']=='above' for t in traces)} above, "
          f"{sum(t['direction']=='below' for t in traces)} below)")
    print(f"machine rejected: {len(rejected)}")
    for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {count:3d}  {reason}")
    if carried:
        print(f"carried forward : {carried} existing human decisions")
    if archived:
        print(f"ARCHIVED        : review from an earlier rule version -> "
              f"{archived.relative_to(ROOT)}")
        print("                  those decisions do NOT carry over; review restarts")
    print(f"written         : {REVIEW_PATH.relative_to(ROOT)}")
    print("\nNext: review each trace and set include to true or false.")


def validate():
    if not REVIEW_PATH.exists():
        raise SystemExit("cut_review.json is absent; run without --check first")
    review = json.loads(REVIEW_PATH.read_text())
    traces = review["traces"]
    problems = []
    for trace in traces:
        include, rule = trace["include"], trace.get("exclusion_rule")
        if include not in (None, True, False):
            problems.append(f"{trace['trace_id']}: include must be null/true/false")
        if include is False and rule not in HUMAN_EXCLUSION_RULES:
            problems.append(
                f"{trace['trace_id']}: excluded without a valid rule "
                f"(got {rule!r}, expected one of {sorted(HUMAN_EXCLUSION_RULES)})"
            )
        if include is True and rule is not None:
            problems.append(f"{trace['trace_id']}: kept but carries an exclusion rule")

    # Reviewing out of order would let outcome knowledge influence the sample.
    # Only in-scope traces are checked: with a slot restriction, skipping an
    # out-of-scope trace is expected, not a gap.
    for direction in ("above", "below"):
        arm = sorted(
            (t for t in traces
             if t["direction"] == direction
             and (TARGET_SLOT is None or t["commitment_slot"] == TARGET_SLOT)),
            key=lambda t: t["review_order"],
        )
        seen = [t["include"] is not None for t in arm]
        if True in seen:
            last = len(seen) - 1 - seen[::-1].index(True)
            gaps = [arm[i]["trace_id"] for i in range(last) if not seen[i]]
            if gaps:
                problems.append(
                    f"{direction}: reviewed out of order, skipped {', '.join(gaps[:5])}"
                )

    in_scope = [
        t for t in traces
        if TARGET_SLOT is None or t["commitment_slot"] == TARGET_SLOT
    ]
    reviewed = [t for t in in_scope if t["include"] is not None]
    kept = [t for t in in_scope if t["include"] is True]
    n_above = sum(t["direction"] == "above" for t in kept)
    n_below = sum(t["direction"] == "below" for t in kept)
    if TARGET_SLOT:
        print(f"slot     : {TARGET_SLOT} only "
              f"({len(in_scope)} of {len(traces)} eligible traces)")
    print(f"reviewed : {len(reviewed)}/{len(in_scope)} in scope")
    print(f"kept     : {len(kept)}  ({n_above} above, {n_below} below)"
          f"   target {TARGET_PER_ARM} per arm")
    if n_above >= TARGET_PER_ARM and n_below >= TARGET_PER_ARM:
        print("           target met — you can stop reviewing")
    else:
        print(f"           still needed: {max(0, TARGET_PER_ARM - n_above)} above, "
              f"{max(0, TARGET_PER_ARM - n_below)} below")
        for direction, need in (("above", TARGET_PER_ARM - n_above),
                                ("below", TARGET_PER_ARM - n_below)):
            if need <= 0:
                continue
            nxt = [
                t["trace_id"] for t in sorted(
                    (t for t in in_scope
                     if t["direction"] == direction and t["include"] is None),
                    key=lambda t: t["review_order"],
                )
            ][:4]
            print(f"           next {direction}: {', '.join(nxt) or 'none left'}")
    dropped = [t for t in traces if t["include"] is False]
    if dropped:
        by_rule = {}
        for trace in dropped:
            by_rule[trace["exclusion_rule"]] = by_rule.get(trace["exclusion_rule"], 0) + 1
        print("dropped  :")
        for rule, count in sorted(by_rule.items(), key=lambda kv: -kv[1]):
            print(f"   {count:3d}  {rule}  {HUMAN_EXCLUSION_RULES.get(rule, '?')[:60]}")
    if problems:
        raise SystemExit("review file invalid:\n  " + "\n  ".join(problems))
    print("validation: PASS")


if __name__ == "__main__":
    fire.Fire(main)
