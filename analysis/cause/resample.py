"""CAUSE step 3 — resample the committed step at each frozen cut, measure spread.

Makes NO API call unless --execute is supplied. Every paid stage is run manually.

WHAT THIS ANSWERS
-----------------
At each of the sixteen frozen cuts, replay the exact prompt and prefix and let the
model write its own next step 12 times. Two questions, both pre-declared below:

1. Is there natural variation at this cut? If all 12 resamples commit to the same
   number, the cut is still downstream of the decision, exactly as at the old
   section-10a cut where 12 of 12 returned 48,000,000. Those traces are dropped.

2. Version A or Version B? Version A uses the resamples as they come, with no
   picking and no editing. Version B keeps only the lowest and highest. A is
   preferred because it involves no selection; B is the fallback if the natural
   spread is too narrow to measure anything.

The committed value in each resample is read with the SAME line_commitment rule
that located the cut, so the resamples are scored by the rule that selected them.

Run:
    uv run python analysis/cause/resample.py --dry_run    # show the plan, no calls
    uv run python analysis/cause/resample.py --execute    # PAID: 192 calls
    uv run python analysis/cause/resample.py --analyze    # local, from cached raw
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import fire

CAUSE_DIR = Path(__file__).resolve().parent
ROOT = CAUSE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CAUSE_DIR))

from pin_sample import FROZEN_PATH, sha256, source_material  # noqa: E402
from select_cuts import (  # noqa: E402
    BARE_NUMBER,
    COMMIT_CUES,
    SLOTS,
    THRESHOLD,
    YEAR_RANGE,
    line_commitment,
)
from value_leakage.api.openrouter.chat_completions import (  # noqa: E402
    get_openrouter_client,
    process_batch,
)

RAW_DIR = CAUSE_DIR / "raw" / "resample_v2"
SPREAD_PATH = CAUSE_DIR / "resample_spread.json"

# ---------------------------------------------------------------------------
# FROZEN before the first call. Do not edit after outputs are seen.
# ---------------------------------------------------------------------------
# Round 1 was 12. A second round of 12 was added on 2026-09-04, pre-declared as the
# ONLY additional round, because some resampled values could not be used as inserts:
# they appear only in a resample that had already computed a total by that point, and
# inserting that block would hand the model the answer. Applied to ALL final traces,
# not only the short ones, so no trace reaches viability through extra effort. The
# viability criteria are unchanged; DEGENERATE_AGREEMENT scales with the sample size
# so it keeps its meaning.
SAMPLES_PER_TRACE = 24
MAX_TOKENS = 600          # 256 truncated every one of the 36 v1 candidate calls
MAX_SPEND_USD = 1.00

# A trace has usable natural variation if its resamples commit to at least this
# many distinct values AND the highest is at least this multiple of the lowest.
# 1.5x on one premise moves the product by 1.5x, which at a 41M threshold is
# about 0.5 threshold units — comfortably more than the 0.2 the design needs.
MIN_DISTINCT_VALUES = 3
MIN_SPREAD_RATIO = 1.5
# A trace is degenerate — cut still too late — if this fraction agree exactly.
# Was 10 of 12; kept proportional when the sample size doubled.
DEGENERATE_AGREEMENT = round(SAMPLES_PER_TRACE * 10 / 12)
# Version A is used if at least this many of the 16 traces are viable.
MIN_VIABLE_TRACES_FOR_VERSION_A = 12


def raw_path(trace_id: str, k: int) -> Path:
    return RAW_DIR / f"{trace_id}__{k:02d}.json"


def load_raw(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def first_block(text: str) -> str:
    """The proposed step: everything up to the first blank line.

    A review aid and the unit the value is read from, not a scientific claim
    about where the step ends. The full text is in the raw file.
    """
    stripped = (text or "").lstrip("\n")
    return re.split(r"\n\s*\n", stripped, maxsplit=1)[0].rstrip()


def committed_value(block: str) -> tuple[str, int] | None:
    """First commitment in the block, read with the cut rule's own logic.

    Kept for reference. Superseded as the analysis reader by read_slot_value;
    see the RESAMPLE READER note below.
    """
    for line in block.split("\n"):
        found = line_commitment(line)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# RESAMPLE READER — repaired 2026-09-03 after the first analysis pass.
#
# Selecting a cut and reading a resample are different jobs, and the cut rule
# does the second one badly. Three defects, all found by inspecting the 192 saved
# responses, all repaired here without re-calling the model:
#
#   1. SLOT MIXING (a bug).  The first pass compared every commitment in a trace
#      regardless of quantity, so a resample committing to 110,000 giraffes and
#      one committing to 350 spots produced a "spread ratio" of 314. Spread is
#      now measured only among resamples committing to the SAME slot as the cut.
#
#   2. FIRST BLOCK ONLY.  The reader looked at the first paragraph and ignored
#      the rest of a 600-token response, discarding text already paid for. It now
#      scans the whole continuation.
#
#   3. HEDGE VETO TOO STRICT.  Rejecting any line containing a range is right
#      when picking a cut from an original trace — being conservative there just
#      costs eligible traces, and 117 was plenty. It is wrong when reading a
#      resample, because it discards real commitments that mention a range in the
#      same breath: "Let's assume an average of 400 spots per giraffe... between
#      the lower end (150) and higher end (800)" is a commitment to 400. The
#      reader now takes the first in-slot value AFTER a commitment cue, which
#      resolves that sentence to 400 without needing the veto.
#
# Readable resamples: 101 -> 154 of 192.
#
# The frozen criteria in the constants above are NOT changed by this repair.
# Thresholds stay at 3 distinct values, 1.5x spread, 10/12 degenerate, 12/16 for
# Version A. This repairs the instrument, not the standard it is judged against.
# ---------------------------------------------------------------------------

SLOT_WORDS = {slot: words for slot, words, _ in SLOTS}
SLOT_RANGES = {slot: bounds for slot, _, bounds in SLOTS}

# ---------------------------------------------------------------------------
# SECOND READER REPAIR, 2026-09-03. Dropping the hedge veto entirely (repair 1)
# went too far and admitted three kinds of non-commitment, all of which landed in
# the tails and so drove the spread ratio:
#
#   "If average spot is 50cm x 50cm?"      -> read 50 as a spot COUNT (it is a size)
#   "Even if we assume 100 spots: ... If we assume 500: ... If we assume 1000:"
#                                          -> read 100 as a commitment (sweep arm)
#   "Let's assume a higher number. 1,000 spots?"
#                                          -> read 1000 as a commitment (floated)
#
# The veto is now narrow and targets the commitment itself rather than any mention
# of a range: a conditional clause, a question, or a value carrying a unit. A line
# that commits and then cites a range still reads correctly, which was the point of
# repair 1: "Let's assume an average of 400 spots... between 150 and 800" -> 400.
# ---------------------------------------------------------------------------

CONDITIONAL_LINE = re.compile(r"^\s*[\*\-\s]*(even\s+)?if\b", re.I)
# Reporting what other people say is not the model committing to anything.
ATTRIBUTION = re.compile(
    r"\b(sources?\s+(say|claim|suggest|cite|state|put)|according to"
    r"|often cited|commonly cited|literature (says|suggests))\b", re.I
)
# The value is one end of a range: "100-200 spots", "300 to 500". Repair 1 dropped
# the range veto entirely to stop discarding real commitments that also mention a
# range; that went too far and let range endpoints through. These two check whether
# THIS value is part of a range, rather than whether the line mentions one anywhere.
RANGE_LEFT = re.compile(r"^\s*(?:[-–—]|to\b)\s*\d")
RANGE_RIGHT = re.compile(r"\d\s*(?:[-–—]|to)\s*$")
# "%" must not be followed by \b: "%" and the space after it are both non-word
# characters, so there is no word boundary between them and the alternative never
# matched. That let "the dark pigment covers about 50% of the skin" be read as a
# commitment to 50 spots per giraffe.
UNIT_AFTER_VALUE = re.compile(
    r"^\s*(?:%|percent\b|cm\b|mm\b|km\b|m\b|sq\b|square\b|inch\b|ft\b)", re.I
)


def read_slot_value(text: str, slot: str) -> int | None:
    """First value in `slot`'s range genuinely committed to on its line."""
    low, high = SLOT_RANGES[slot]
    for line in text.split("\n"):
        cue = COMMIT_CUES.search(line)
        if not cue or not SLOT_WORDS[slot].search(line):
            continue
        if (CONDITIONAL_LINE.match(line) or line.rstrip().endswith("?")
                or ATTRIBUTION.search(line)):
            continue
        tail = line[cue.end():]
        for match in BARE_NUMBER.finditer(tail):
            value = int(match.group(1).replace(",", ""))
            if YEAR_RANGE[0] <= value <= YEAR_RANGE[1] or value == THRESHOLD:
                continue
            after = tail[match.end():]
            if (UNIT_AFTER_VALUE.match(after)
                    or RANGE_LEFT.match(after)
                    or RANGE_RIGHT.search(tail[: match.start()])):
                continue
            if low <= value <= high:
                return value
    return None


def read_any_slot(text: str) -> tuple[str, int] | None:
    """Which quantity this resample fixes first, whatever it is."""
    best = None
    for slot in SLOT_RANGES:
        low, high = SLOT_RANGES[slot]
        for index, line in enumerate(text.split("\n")):
            cue = COMMIT_CUES.search(line)
            if not cue or not SLOT_WORDS[slot].search(line):
                continue
            for match in BARE_NUMBER.finditer(line[cue.end():]):
                value = int(match.group(1).replace(",", ""))
                if YEAR_RANGE[0] <= value <= YEAR_RANGE[1] or value == THRESHOLD:
                    continue
                if low <= value <= high and (best is None or index < best[0]):
                    best = (index, slot, value)
                break
    return (best[1], best[2]) if best else None


FINAL_PATH = CAUSE_DIR / "frozen_final.json"


def load_frozen(from_final: bool = False) -> dict:
    """The pool by default; the final sample when topping up round 2."""
    path = FINAL_PATH if from_final else FROZEN_PATH
    if not path.exists():
        raise SystemExit(f"{path.name} is absent; run pin_sample.py first")
    return json.loads(path.read_text())


def replay_material(trace: dict) -> tuple[str, str]:
    """Prompt and prefix, verified against the frozen hashes."""
    prompt, reasoning = source_material(trace["version"], trace["i"])
    prefix = reasoning[: trace["cut_char"]]
    if sha256(prompt) != trace["prompt_sha256"]:
        raise SystemExit(f"{trace['trace_id']}: prompt hash mismatch")
    if sha256(prefix) != trace["prefix_sha256"]:
        raise SystemExit(f"{trace['trace_id']}: prefix hash mismatch")
    return prompt, prefix


async def run_batch(frozen: dict, todo: list[tuple[dict, int]],
                    max_concurrent: int = 8, pause: float = 0.0) -> float:
    missing_config = [k for k in ("model", "provider") if k not in frozen]
    if missing_config:
        raise SystemExit(
            f"the sample file is missing {missing_config}; regenerate it with "
            "pin_sample.py so the replay configuration travels with the sample"
        )
    client = get_openrouter_client()
    body = {
        "usage": {"include": True},
        "reasoning": {"effort": "high"},
        "provider": {"order": [frozen["provider"]], "allow_fallbacks": False},
    }
    spent = 0.0
    for start in range(0, len(todo), 24):
        if pause and start:
            await asyncio.sleep(pause)
        chunk = todo[start : start + 24]
        messages = []
        for trace, _ in chunk:
            prompt, prefix = replay_material(trace)
            messages.append([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix},
            ])
        responses = await process_batch(
            client=client, model=frozen["model"], messages_list=messages,
            temperature=1.0, top_p=1.0, max_tokens=MAX_TOKENS,
            max_concurrent=max_concurrent, extra_body=body,
            return_exceptions=True,
        )
        # Write every response before checking the cap: these are already paid for.
        for (trace, k), response in zip(chunk, responses):
            path = raw_path(trace["trace_id"], k)
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(response, Exception):
                path.write_text(json.dumps({
                    "trace_id": trace["trace_id"], "sample_index": k,
                    "error": f"{type(response).__name__}: {response}",
                }, indent=2))
                continue
            message = response.choices[0].message
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )
            usage = response.usage.model_dump() if response.usage else {}
            spent += float(usage.get("cost") or 0.0)
            path.write_text(json.dumps({
                "trace_id": trace["trace_id"],
                "sample_index": k,
                "prefix_sha256": trace["prefix_sha256"],
                "continuation": reasoning,
                "content": message.content or "",
                "proposed_step": first_block(reasoning),
                "finish_reason": response.choices[0].finish_reason,
                "usage": usage,
            }, indent=2, ensure_ascii=False))
        if spent > MAX_SPEND_USD:
            raise RuntimeError(
                f"spend ${spent:.3f} passed cap ${MAX_SPEND_USD:.2f}; "
                "all completed responses are cached"
            )
    return spent


def analyse(frozen: dict) -> dict:
    # Values come from the validated judge, not from a parser.
    judged_path = CAUSE_DIR / "resample_judged.json"
    if not judged_path.exists():
        raise SystemExit(
            "resample_judged.json is absent; run judge_resamples.py --execute first"
        )
    payload = json.loads(judged_path.read_text())
    judged = {
        (r["trace_id"], r["sample_index"]): r["spots_value"] for r in payload["rows"]
    }
    unjudged = sum(
        1 for trace in frozen["traces"] for k in range(SAMPLES_PER_TRACE)
        if (trace["trace_id"], k) not in judged
        and (row := load_raw(raw_path(trace["trace_id"], k))) is not None
        and not row.get("error")
    )
    if unjudged:
        raise SystemExit(
            f"{unjudged} resamples have no judgement; run judge_resamples.py --execute"
        )

    per_trace, missing = [], 0
    for trace in frozen["traces"]:
        slot = trace["commitment_slot"]
        values, unreadable, truncated, off_slot = [], 0, 0, []
        failed = 0
        for k in range(SAMPLES_PER_TRACE):
            row = load_raw(raw_path(trace["trace_id"], k))
            if row is None:
                missing += 1
                continue
            if row.get("error"):
                # A refused or failed call is not evidence about the trace. Counted
                # separately so an unsampled trace can never be read as one whose
                # resamples agreed: a 402 out-of-credit run once produced 13 traces
                # reporting "0 readable", which looks identical to no variation.
                failed += 1
                continue
            if row.get("finish_reason") == "length":
                truncated += 1
            value = judged.get((trace["trace_id"], k))
            if value is None:
                # The judge found no committed figure. That is a property of the
                # resample, not a parsing failure.
                unreadable += 1
                continue
            values.append({"slot": slot, "value": value, "sample_index": k})

        numbers = [v["value"] for v in values]
        distinct = sorted(set(numbers))
        ratio = max(numbers) / min(numbers) if numbers else None
        top_agreement = max(
            (numbers.count(x) for x in distinct), default=0
        )
        degenerate = top_agreement >= DEGENERATE_AGREEMENT
        sampled = failed < SAMPLES_PER_TRACE
        viable = (
            sampled
            and not degenerate
            and len(distinct) >= MIN_DISTINCT_VALUES
            and ratio is not None
            and ratio >= MIN_SPREAD_RATIO
        )
        per_trace.append({
            "trace_id": trace["trace_id"],
            "direction": trace["direction"],
            "cut_slot": slot,
            "original_value": trace["commitment_value"],
            "n_readable": len(values),
            "n_failed_calls": failed,
            "fully_sampled": sampled,
            "n_off_slot": len(off_slot),
            "off_slot_commitments": off_slot,
            "n_unreadable": unreadable,
            "n_truncated": truncated,
            "distinct_values": distinct,
            "spread_ratio": round(ratio, 3) if ratio else None,
            "max_agreement": top_agreement,
            "degenerate": degenerate,
            "viable_for_version_a": viable,
            "commitments": values,
        })

    viable = sum(t["viable_for_version_a"] for t in per_trace)
    degenerate = [t["trace_id"] for t in per_trace if t["degenerate"]]

    # Kept as stored numbers so the report and any figure can be
    # rebuilt from this file without rerunning the analysis.
    by_slot = {}
    for slot in sorted({t["cut_slot"] for t in per_trace}):
        rows = [t for t in per_trace if t["cut_slot"] == slot]
        ratios = sorted(t["spread_ratio"] for t in rows if t["spread_ratio"])
        all_values = sorted(
            v["value"] for t in rows for v in t["commitments"]
        )
        by_slot[slot] = {
            "n_traces": len(rows),
            "n_viable": sum(t["viable_for_version_a"] for t in rows),
            "spread_ratios": ratios,
            "median_spread_ratio": ratios[len(ratios) // 2] if ratios else None,
            "n_resample_commitments": len(all_values),
            "value_min": all_values[0] if all_values else None,
            "value_max": all_values[-1] if all_values else None,
            "distinct_values_per_trace": {
                t["trace_id"]: t["distinct_values"] for t in rows
            },
        }

    return {
        "experiment": frozen["experiment"],
        "samples_per_trace": SAMPLES_PER_TRACE,
        "criteria": {
            "min_distinct_values": MIN_DISTINCT_VALUES,
            "min_spread_ratio": MIN_SPREAD_RATIO,
            "degenerate_agreement": DEGENERATE_AGREEMENT,
            "min_viable_traces_for_version_a": MIN_VIABLE_TRACES_FOR_VERSION_A,
        },
        "missing_raw_files": missing,
        "viable_traces": viable,
        "degenerate_traces": degenerate,
        "by_slot": by_slot,
        "decision": (
            "pending — raw responses missing" if missing else
            "version_a" if viable >= MIN_VIABLE_TRACES_FOR_VERSION_A
            else "version_b"
        ),
        "per_trace": per_trace,
    }


def main(dry_run: bool = False, execute: bool = False, analyze: bool = False,
         max_concurrent: int = 8, pause: float = 0.0, from_final: bool = False):
    """max_concurrent/pause throttle the provider. deepinfra/fp4 rate-limited a
    retry batch at the default 8, so use e.g. --max_concurrent 2 --pause 3 to
    recover stragglers slowly rather than burning them on repeated 429s."""
    frozen = load_frozen(from_final)
    traces = frozen["traces"]

    if analyze:
        result = analyse(frozen)
        SPREAD_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"traces analysed : {len(result['per_trace'])}")
        print(f"missing raw     : {result['missing_raw_files']}")
        broken = [t for t in result["per_trace"] if not t["fully_sampled"]]
        calls_failed = sum(t["n_failed_calls"] for t in result["per_trace"])
        if calls_failed:
            print(f"FAILED CALLS    : {calls_failed} across "
                  f"{sum(t['n_failed_calls'] > 0 for t in result['per_trace'])} traces"
                  + (f"; {len(broken)} never sampled at all" if broken else ""))
            print("                  rerun --execute after fixing the cause; "
                  "unsampled traces are excluded, not judged")
        for row in result["per_trace"]:
            flag = ("NOT SAMPLED" if not row["fully_sampled"]
                    else "DEGENERATE" if row["degenerate"]
                    else "ok" if row["viable_for_version_a"] else "narrow")
            print(f"  {row['trace_id']:18s} {row['n_readable']:2d} readable  "
                  f"{len(row['distinct_values']):2d} distinct  "
                  f"ratio {row['spread_ratio'] or 0:5.2f}  {flag}")
        print("\nby cut slot:")
        for slot, row in result["by_slot"].items():
            print(f"  {slot:11s} {row['n_viable']}/{row['n_traces']} viable   "
                  f"median spread {row['median_spread_ratio']:.2f}x   "
                  f"values {row['value_min']:,}-{row['value_max']:,}")
        # The 12-of-16 threshold was set for a fixed 16-trace sample. With a larger
        # pool the final sixteen are chosen by pin_sample.py --final as the first 8
        # viable per arm, so Version A holds by construction and that threshold no
        # longer decides anything. Report the count, not a verdict.
        print(f"\nviable traces : {result['viable_traces']}/{len(traces)}")
        if result["degenerate_traces"]:
            print(f"degenerate    : {', '.join(result['degenerate_traces'])}")
        print(f"written              : {SPREAD_PATH.relative_to(ROOT)}")
        return

    todo = [
        (trace, k)
        for trace in traces
        for k in range(SAMPLES_PER_TRACE)
        if (row := load_raw(raw_path(trace["trace_id"], k))) is None or row.get("error")
    ]
    for trace in traces:            # fails fast if a source trace changed
        replay_material(trace)

    print(f"traces      : {len(traces)}")
    print(f"per trace   : {SAMPLES_PER_TRACE} resamples, {MAX_TOKENS} tokens each")
    print(f"cached      : {len(traces) * SAMPLES_PER_TRACE - len(todo)}"
          f"/{len(traces) * SAMPLES_PER_TRACE}")
    print(f"calls needed: {len(todo)}")
    print(f"spend cap   : ${MAX_SPEND_USD:.2f}")
    print("hashes      : PASS")

    if dry_run or not execute:
        print("API calls   : NONE")
        if not dry_run:
            print("To authorize this paid stage, rerun with --execute.")
        return

    spent = asyncio.run(run_batch(frozen, todo, max_concurrent, pause))
    print(f"stage cost  : ${spent:.4f}")
    print(f"raw saved   : {RAW_DIR.relative_to(ROOT)}")
    print("Next: uv run python analysis/cause/resample.py --analyze")


if __name__ == "__main__":
    fire.Fire(main)
