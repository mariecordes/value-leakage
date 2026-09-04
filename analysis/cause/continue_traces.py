"""CAUSE step 4 — the experiment. Insert each step, let the trace run to an answer.

Makes NO API call unless --execute is supplied. Every paid stage is run manually.

WHAT THIS DOES
--------------
For each trace in frozen_final.json, and for each spots-per-giraffe figure its own
resamples committed to, replay:

    user prompt + reasoning up to the cut + the resampled step

and let the model run to a final answer, six times. The original step is included
unchanged as the baseline; the spread among its six continuations is the noise floor
that any effect has to beat.

The question: does a higher inserted figure produce a higher final answer, or does the
trace pull back toward the side the bet rewards?

WHAT IS AND IS NOT SELECTED
---------------------------
Every insert is a step the model itself wrote at this exact cut, used verbatim. No
picking of extremes, no editing. Two mechanical rules apply, both structural rather
than about outcomes:

  * the insert is the SINGLE line on which the resample commits to the figure. The
    prefix ends at a line boundary, so one line flows from it cleanly and states
    exactly one thing;
  * that line must not carry a total-scale number, which would hand the model the
    answer, nor a second plausible spots figure, which would state two things at
    once. Figures that only ever appear on such lines are dropped and recorded.

Where several resamples committed to the same figure, the lowest sample index is used.
That is arbitrary rather than chosen, and is recorded.

Run:
    uv run python analysis/cause/continue_traces.py --build     # write inserts.json
    uv run python analysis/cause/continue_traces.py --dry_run   # plan and cost
    uv run python analysis/cause/continue_traces.py --execute   # PAID
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

from pin_sample import FINAL_PATH, sha256, source_material  # noqa: E402
from resample import load_raw, raw_path  # noqa: E402
from value_leakage.api.openrouter.chat_completions import (  # noqa: E402
    get_openrouter_client,
    process_batch,
)

INSERTS_PATH = CAUSE_DIR / "inserts.json"
FINISH_DIR = CAUSE_DIR / "raw" / "finishes"
JUDGED_PATH = CAUSE_DIR / "resample_judged.json"

# Frozen before the first continuation call.
CONTINUATIONS_PER_VARIANT = 6
# 8,000 truncated 4 of the 12 pilot continuations before they produced any visible
# answer: these traces write 17,000-24,000 characters of reasoning. A truncated row
# still carries reasoning, so reading a number from its tail would return a mid-
# reasoning value rather than a final answer — they are retried instead.
CONTINUATION_MAX_TOKENS = 16000
MAX_SPEND_USD = 8.00
MIN_INSERTABLE_VALUES = 3
MIN_INSERTABLE_SPREAD = 1.5

BIG_NUMBER = re.compile(r"(?<![\d,.])(\d[\d,]{6,})(?![\d,])")


def value_pattern(value: int) -> re.Pattern:
    """Matches the figure written plainly or comma-grouped.

    The trailing guard excludes digits and commas, and a period only when it is a
    decimal point. An earlier version excluded any following period, which made a
    figure at the end of a sentence invisible ("around 350.") and silently fell
    through to a later line that already contained the total.
    """
    return re.compile(
        r"(?<![\d,.])(?:" + f"{value:,}|{value}" + r")(?![\d,])(?!\.\d)"
    )


def extract_step(trace_id: str, sample_index: int, value: int,
                 threshold: float) -> str | None:
    """The single line on which this resample commits to `value`.

    ONE LINE, not the block leading up to it. An earlier version took every line
    from the start of the resample up to the committed figure, which meant an
    insert for 400 could open with "Let's assume 300 spots" and only reach 400
    further down — four supposedly different variants all committing to 300 first.
    The prefix ends at a line boundary, so a single line flows from it cleanly and
    states exactly one figure.
    """
    row = load_raw(raw_path(trace_id, sample_index))
    if row is None or row.get("error"):
        return None
    pattern = value_pattern(value)
    for line in (row["continuation"] or "").split("\n"):
        if not pattern.search(line):
            continue
        if not line.strip():
            return None
        # A line that already carries a total hands the model the answer.
        for match in BIG_NUMBER.finditer(line):
            if int(match.group(1).replace(",", "")) >= threshold / 10:
                return None
        # A line naming another plausible spots figure states more than one thing.
        others = {
            int(m.group(1).replace(",", ""))
            for m in re.finditer(r"(?<![\d,.])(\d[\d,]*)(?![\d,])(?!\.\d)", line)
        }
        if any(50 <= o <= 5000 and o != value for o in others):
            return None
        return line.rstrip()
    return None


def build() -> dict:
    final = json.loads(FINAL_PATH.read_text())
    threshold = final["threshold"]
    judged = [
        (r["trace_id"], r["sample_index"], r["spots_value"])
        for r in json.loads(JUDGED_PATH.read_text())["rows"]
        if r["spots_value"] is not None
    ]

    traces = []
    for entry in final["traces"]:
        trace_id = entry["trace_id"]
        values = sorted({v for t, _, v in judged if t == trace_id})
        variants, dropped = [], []
        for value in values:
            indices = sorted(k for t, k, v in judged if t == trace_id and v == value)
            for k in indices:
                step = extract_step(trace_id, k, value, threshold)
                if step:
                    variants.append({
                        "variant": f"v{value}",
                        "inserted_value": value,
                        "source_sample_index": k,
                        "step": step,
                        "step_sha256": sha256(step),
                    })
                    break
            else:
                dropped.append({"value": value, "candidates": len(indices),
                                "reason": "every resample had computed a total first"})

        inserted = [v["inserted_value"] for v in variants]
        spread = max(inserted) / min(inserted) if len(inserted) > 1 else 1.0
        passes = (len(inserted) >= MIN_INSERTABLE_VALUES
                  and spread >= MIN_INSERTABLE_SPREAD)
        variants.insert(0, {
            "variant": "original",
            "inserted_value": entry["commitment_value"],
            "source_sample_index": None,
            # The stored original_step is a fixed 400-character slice and can end
            # mid-number. Use the commitment line the cut rule identified instead,
            # so the baseline is the same kind of object as every other variant.
            "step": entry["commitment_line"].rstrip(),
            "step_sha256": sha256(entry["commitment_line"].rstrip()),
        })
        traces.append({
            "trace_id": trace_id,
            "direction": entry["direction"],
            "version": entry["version"],
            "i": entry["i"],
            "cut_char": entry["cut_char"],
            "prompt_sha256": entry["prompt_sha256"],
            "prefix_sha256": entry["prefix_sha256"],
            "insertable_values": inserted,
            "insertable_spread": round(spread, 3),
            "dropped_values": dropped,
            "meets_insertable_criteria": passes,
            "variants": variants,
        })

    primary = [t for t in traces if t["meets_insertable_criteria"]]
    return {
        "experiment": "cause-v2-continuations",
        "threshold": threshold,
        "model": final["model"],
        "provider": final["provider"],
        "prefill_mode": final["prefill_mode"],
        "continuations_per_variant": CONTINUATIONS_PER_VARIANT,
        "criteria_on_insertable_values": {
            "min_values": MIN_INSERTABLE_VALUES,
            "min_spread": MIN_INSERTABLE_SPREAD,
            "note": (
                "Applied to the figures that can actually be inserted, not to every "
                "figure the resamples produced. A trace can pass on resampled values "
                "and fail here if the extra figures appear only in resamples that had "
                "already computed a total."
            ),
        },
        "primary_traces": [t["trace_id"] for t in primary],
        "secondary_traces": [
            t["trace_id"] for t in traces if not t["meets_insertable_criteria"]
        ],
        "analysis_plan": (
            "Headline result uses the primary traces. The secondary traces are "
            "sampled too and reported as a robustness check; their inserted range is "
            "narrow, so they are expected to add noise rather than signal. Both sets "
            "were declared before any continuation was sampled."
        ),
        "traces": traces,
    }


def observed_cost_per_call() -> float | None:
    """Mean cost of continuations already on disk.

    A fixed 0.0093 estimate carried over from an experiment whose continuations
    averaged 3,400 tokens under-projected these by about half: cutting at 10-20%
    of the trace leaves far more reasoning to generate, and they average 7,100.
    Once any rows exist, project from them rather than from a constant.
    """
    costs = []
    for path in FINISH_DIR.glob("*.json"):
        row = load_raw(path)
        if row and not row.get("error"):
            cost = (row.get("usage") or {}).get("cost")
            if cost:
                costs.append(float(cost))
    return sum(costs) / len(costs) if len(costs) >= 20 else None


def finish_path(trace_id: str, variant: str, k: int) -> Path:
    return FINISH_DIR / f"{trace_id}__{variant}__{k:02d}.json"


def replay(trace: dict) -> tuple[str, str]:
    prompt, reasoning = source_material(trace["version"], trace["i"])
    prefix = reasoning[: trace["cut_char"]]
    if sha256(prompt) != trace["prompt_sha256"]:
        raise SystemExit(f"{trace['trace_id']}: prompt hash mismatch")
    if sha256(prefix) != trace["prefix_sha256"]:
        raise SystemExit(f"{trace['trace_id']}: prefix hash mismatch")
    return prompt, prefix


def truncated_without_answer(row: dict) -> bool:
    """Ran out of tokens mid-reasoning, so there is no final answer to read."""
    return (row.get("finish_reason") == "length"
            and len((row.get("content") or "").strip()) < 40)


def pending(spec: dict) -> list[tuple[dict, dict, int]]:
    out = []
    for trace in spec["traces"]:
        for variant in trace["variants"]:
            for k in range(spec["continuations_per_variant"]):
                path = finish_path(trace["trace_id"], variant["variant"], k)
                row = load_raw(path)
                if row is None or row.get("error") or truncated_without_answer(row):
                    out.append((trace, variant, k))
    return out


async def run(spec: dict, todo: list, max_concurrent: int, pause: float,
              cap: float = MAX_SPEND_USD) -> float:
    client = get_openrouter_client()
    body = {
        "usage": {"include": True},
        "reasoning": {"effort": "high"},
        "provider": {"order": [spec["provider"]], "allow_fallbacks": False},
    }
    spent = 0.0
    for start in range(0, len(todo), 24):
        if pause and start:
            await asyncio.sleep(pause)
        chunk = todo[start : start + 24]
        messages = []
        for trace, variant, _ in chunk:
            prompt, prefix = replay(trace)
            text = prefix + variant["step"]
            if not text.endswith("\n"):
                text += "\n"
            messages.append([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
            ])
        responses = await process_batch(
            client=client, model=spec["model"], messages_list=messages,
            temperature=1.0, top_p=1.0, max_tokens=CONTINUATION_MAX_TOKENS,
            max_concurrent=max_concurrent, extra_body=body, return_exceptions=True,
        )
        # Write every response before checking spend: these are already paid for.
        for (trace, variant, k), response in zip(chunk, responses):
            path = finish_path(trace["trace_id"], variant["variant"], k)
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(response, Exception):
                path.write_text(json.dumps({
                    "trace_id": trace["trace_id"], "variant": variant["variant"],
                    "sample_index": k,
                    "error": f"{type(response).__name__}: {response}",
                }, indent=2))
                continue
            message = response.choices[0].message
            usage = response.usage.model_dump() if response.usage else {}
            spent += float(usage.get("cost") or 0.0)
            path.write_text(json.dumps({
                "trace_id": trace["trace_id"],
                "direction": trace["direction"],
                "variant": variant["variant"],
                "inserted_value": variant["inserted_value"],
                "sample_index": k,
                "step_sha256": variant["step_sha256"],
                "reasoning": (getattr(message, "reasoning_content", None)
                              or getattr(message, "reasoning", None) or ""),
                "content": message.content or "",
                "finish_reason": response.choices[0].finish_reason,
                "usage": usage,
            }, indent=2, ensure_ascii=False))
        if spent > cap:
            raise RuntimeError(
                f"spend ${spent:.2f} passed cap ${cap:.2f}; "
                "all completed responses are cached"
            )
    return spent


def main(build_only: bool = False, dry_run: bool = False, execute: bool = False,
         max_concurrent: int = 4, pause: float = 2.0, limit: int | None = None,
         max_spend: float | None = None):
    """--limit N runs only the first N pending continuations.

    Use it to verify the mechanism before the full batch: finding M6 in NOTES.md
    records that one prefill mode silently ignored the saved trace and began the
    task afresh, returning a perfectly normal-looking response that measured
    nothing. That was checked for full partial traces; these cuts are earlier and
    the insert is a single line, so it is worth one more cent to confirm.
    """
    if build_only or not INSERTS_PATH.exists():
        spec = build()
        INSERTS_PATH.write_text(json.dumps(spec, indent=2, ensure_ascii=False))
        print(f"built        : {INSERTS_PATH.relative_to(ROOT)}")
        for trace in spec["traces"]:
            mark = "primary  " if trace["meets_insertable_criteria"] else "secondary"
            print(f"  {mark} {trace['trace_id']:18s} "
                  f"{len(trace['variants'])} variants  "
                  f"spread {trace['insertable_spread']:4.2f}x  "
                  f"{trace['insertable_values']}")
        print(f"primary  : {len(spec['primary_traces'])} traces")
        print(f"secondary: {len(spec['secondary_traces'])} traces")
        if build_only:
            return
    else:
        spec = json.loads(INSERTS_PATH.read_text())

    outstanding = pending(spec)
    todo = outstanding[:limit] if limit else outstanding
    total = sum(len(t["variants"]) for t in spec["traces"]) * \
        spec["continuations_per_variant"]
    print(f"\nvariants     : {sum(len(t['variants']) for t in spec['traces'])}")
    print(f"continuations: {total} total, {total - len(outstanding)} cached, "
          f"{len(outstanding)} outstanding")
    if limit:
        print(f"limit        : running {len(todo)} of them this pass")
    observed = observed_cost_per_call()
    rate = observed or 0.0175
    print(f"projected    : ${len(todo) * rate:.2f} "
          f"(at ${rate:.4f}/call, {'observed' if observed else 'estimated'}; "
          f"cap ${max_spend or MAX_SPEND_USD:.2f})")
    for trace in spec["traces"]:
        replay(trace)
    print("hashes       : PASS")

    if dry_run or not execute:
        print("API calls    : NONE")
        if not dry_run:
            print("To authorize this paid stage, rerun with --execute.")
        return

    cap = MAX_SPEND_USD if max_spend is None else float(max_spend)
    spent = asyncio.run(run(spec, todo, max_concurrent, pause, cap))
    print(f"stage cost   : ${spent:.4f}")
    print(f"raw saved    : {FINISH_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    fire.Fire(main)
