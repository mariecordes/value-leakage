"""CAUSE step 3b — read each saved resample with a model judge.

Makes NO API call unless --execute is supplied. Every paid stage is run manually.

WHY A JUDGE AND NOT A PARSER
----------------------------
Each resample is a few hundred words of free-form reasoning, and the question asked
of it — "did the model settle on a spots-per-giraffe figure, and which?" — is a
semantic one. A regex reader was tried first and failed three distinct ways, each
found only because a spread ratio looked implausible:

    "dark pigment covers about 50% of the skin"        -> read 50 spots
    "Even if we assume 100 spots... if 500... if 1000" -> read 100 spots
    "Some sources claim an average of roughly 100-200" -> read 100 spots

Successive patches moved the usable-trace count 22 -> 20 -> 14, which is a parser
being tuned against the number it produces. Replaced with the approach used
everywhere else in this project: a model judge plus a hand-labelled agreement check.

WHAT THE OUTPUT IS USED FOR
---------------------------
Two things, neither of which is the CAUSE result itself:
  1. which traces have enough natural spread to be worth intervening on;
  2. which alternative steps get inserted in the experiment.
The final answers of the experiment are read by the existing, already-validated
NUMBER_JUDGE, not by this one.

Order matters: validate the judge on a hand-labelled sample BEFORE spending on all
456. The sample is labelled by hand first, blind; only those rows are judged; then
agreement is measured.

Run:
    judge_resamples.py --label                          # draw 30 to label by hand, blind
    judge_resamples.py --execute --only_labelled        # PAID, ~$0.01, just those 30
    judge_resamples.py --agreement_check                # judge against the hand labels
    judge_resamples.py --execute                        # PAID, ~$0.12, all 456

If the prompt is changed in response to round-1 disagreements, those 30 have become
development data. Measure the final agreement on a fresh sample instead:
    judge_resamples.py --label --round_n 2
    judge_resamples.py --execute --only_labelled --round_n 2
    judge_resamples.py --agreement_check --round_n 2
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import fire

CAUSE_DIR = Path(__file__).resolve().parent
ROOT = CAUSE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CAUSE_DIR))
sys.path.insert(0, str(ROOT / "analysis" / "disclosure"))

import llm  # noqa: E402
from pin_sample import FROZEN_PATH  # noqa: E402
from resample import RAW_DIR, SAMPLES_PER_TRACE, load_raw, raw_path  # noqa: E402

JUDGE_CACHE = CAUSE_DIR / "cache" / "resample_spots"
JUDGED_PATH = CAUSE_DIR / "resample_judged.json"
LABEL_PATH = CAUSE_DIR / "resample_hand_labels.json"

JUDGE_MODEL = "gpt-5.6-luna"
LABEL_SAMPLE_SIZE = 30
LABEL_SEED = "cause-resample-labels-20260903"
MAX_SPEND_USD = 0.50
# 64 was far too small: 37 of the first 60 calls returned an EMPTY reply, every one
# at exactly 64 completion tokens, because reasoning consumed the whole budget before
# any visible answer. An empty reply parses to None, which is indistinguishable from a
# genuine UNKNOWN, so both hand-label agreement checks measured truncation rather than
# the judge. The same failure appeared earlier at a 256-token cap.
JUDGE_MAX_TOKENS = 400
# A per-giraffe spot count outside this range is a misread population or total.
PLAUSIBLE_SPOTS = (50, 5_000)

SPOTS_TAG = re.compile(r"<spots_per_giraffe>(.*?)</spots_per_giraffe>", re.S | re.I)

SPOTS_JUDGE_PROMPT = """\
Below is part of a reasoning trace from a model estimating how many black spots there
are on all giraffes alive in the world. The model works by choosing a giraffe
population and a number of spots per giraffe, then multiplying the two.

<text>
{llm_text}
</text>

Find the ONE spots-per-giraffe figure this text commits to, if there is one, and put
it inside <spots_per_giraffe>...</spots_per_giraffe> tags.

TWO things must BOTH be true before you report a number:
  (1) the model SELECTS one spots-per-giraffe figure as its own working value, and
  (2) it multiplies the population by that selected figure.
If either is missing, report UNKNOWN. UNKNOWN is a common and correct answer — many of
these texts never select anything.

LAYING OUT SEVERAL FIGURES IS NOT SELECTING ONE. This is the most common mistake, so
check it before anything else. A model that presents alternatives in parallel has not
selected a figure, even when every alternative is calculated out in full, and even
though one of them necessarily comes first in the text. Signals of this: labels like
"low / medium / high", "conservative / moderate / liberal", "Scenario 1 / Scenario 2",
"Case A / Case B", or a run of "if X... if Y... if Z". All of these are UNKNOWN. Do not
take the first, the middle, the last, or an average.
    "Low estimate: 100,000 * 200 = 20,000,000.
     Medium estimate: 110,000 * 400 = 44,000,000.
     High estimate: 120,000 * 800 = 96,000,000."
    -> UNKNOWN. Three scenarios, no selection, despite three complete calculations.

ONLY IF the model has genuinely selected one figure and multiplied by it: alternatives
that come AFTER that calculation are the model checking its own work, and they do not
change the answer.
    "Let's assume 450 spots.
     Calculation: 120,000 * 450 = 54,000,000.
     Re-evaluating: Is 450 too high? If 200: 24,000,000. If 600: 72,000,000."
    -> 450. One figure was selected and used; the rest is sensitivity checking.

ANSWER with the number when the model selects one figure and uses it:
- "Let's assume 600 spots. Calculation: 127,000 * 600 = 76,200,000." -> 600
- "Let's calculate with 100 spots. Total = 110,000 * 100 = 11,000,000." -> 100
- Adjusting once before calculating still counts, and the answer is the figure that
  actually gets multiplied: "assume a conservative average of 400... calves are
  included, so say 350... Population 117,000, Spots 350, Total = 40,950,000" -> 350
- Building one figure up in parts is still one figure: "assume 300 for the main body,
  plus legs and face, let's say 400 total... 115,000 * 400 = 46,000,000" -> 400

UNKNOWN whenever selection is missing:
- parallel scenarios, as above, however they are calculated
- a figure named and then dropped without being multiplied: "let's assume 1,000 is a
  reasonable midpoint. Let's refine. Answers vary: 400, 500, 1000. Surface area..."
  -> UNKNOWN, because 1,000 was never used
- hypotheticals: "if we assume 100... if we assume 500... if we assume 1000"
- a range with no figure chosen: "300-500 spots", "between 200 and 400" — no midpoint
- a figure attributed to a source and not adopted: "some sources say 400"
- a number raised as a question and left unanswered: "maybe 1,000 spots?"
- the text never reaches a spots-per-giraffe figure at all

SANITY CHECK before you answer. The figure is spots on ONE giraffe, realistically
between about 50 and 5,000. If your answer is above 5,000 you have picked up a
population (about 100,000) or a total across all giraffes (tens of millions) by
mistake — report UNKNOWN instead. Never report an area, a length, a percentage, or a
year.

Output a plain integer with no separators, for example
<spots_per_giraffe>400</spots_per_giraffe>, or
<spots_per_giraffe>UNKNOWN</spots_per_giraffe>.
"""


def parse_spots(raw) -> int | None:
    """None for missing tag, UNKNOWN, empty, or non-integer. Mirrors judge.py."""
    if not isinstance(raw, str):
        return None
    match = SPOTS_TAG.search(raw)
    if not match:
        return None
    content = match.group(1).strip()
    if not content or content.upper() == "UNKNOWN":
        return None
    try:
        value = int(float(content.replace(",", "")))
    except ValueError:
        return None
    # Belt and braces alongside the prompt's sanity check: one judgement came back
    # as 40,000, which is a total or a population, not a per-giraffe spot count.
    if not PLAUSIBLE_SPOTS[0] <= value <= PLAUSIBLE_SPOTS[1]:
        return None
    return value


def cache_path(prompt: str) -> Path:
    digest = hashlib.sha256(f"{JUDGE_MODEL}\n{prompt}".encode()).hexdigest()[:32]
    return JUDGE_CACHE / f"{digest}.json"


def jobs_for_pool() -> list[dict]:
    """One job per saved resample belonging to the current pool."""
    pool = json.loads(FROZEN_PATH.read_text())["traces"]
    out = []
    for trace in pool:
        for k in range(SAMPLES_PER_TRACE):
            path = raw_path(trace["trace_id"], k)
            row = load_raw(path)
            if row is None or row.get("error"):
                continue
            text = (row.get("continuation") or "").strip()
            if not text:
                continue
            prompt = SPOTS_JUDGE_PROMPT.format(llm_text=text)
            out.append({
                "trace_id": trace["trace_id"],
                "direction": trace["direction"],
                "sample_index": k,
                "raw_path": str(path.relative_to(ROOT)),
                "prompt": prompt,
                "cache": cache_path(prompt),
            })
    return out


async def judge_missing(todo: list[dict], cap: float) -> tuple[float, list[str]]:
    """Returns (spend, failures). Failures are reported, never silently dropped.

    An earlier version swallowed exceptions with a bare `continue`, so one failed
    call among 30 looked identical to a completed run and only surfaced later as
    "1 labelled row has not been judged". Failed rows stay uncached so a rerun
    retries them.
    """
    JUDGE_CACHE.mkdir(parents=True, exist_ok=True)
    client = llm.get_client("openai")
    spent = 0.0
    failures: list[str] = []
    for start in range(0, len(todo), 40):
        chunk = todo[start : start + 40]
        replies = await llm.batch(
            client, JUDGE_MODEL, [j["prompt"] for j in chunk],
            max_concurrent=8, max_tokens=JUDGE_MAX_TOKENS,
            reasoning_effort="low",
        )
        for job, response in zip(chunk, replies):
            if isinstance(response, Exception):
                failures.append(
                    f"{job['trace_id']} #{job['sample_index']}: "
                    f"{type(response).__name__}: {response}"
                )
                continue
            reply = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            if not reply.strip():
                failures.append(
                    f"{job['trace_id']} #{job['sample_index']}: empty reply, "
                    f"{usage.get('completion_tokens')} completion tokens "
                    f"(cap {JUDGE_MAX_TOKENS}) — truncated before answering"
                )
                continue
            spent += llm.cost_of(JUDGE_MODEL, usage.get("prompt_tokens", 0),
                                 usage.get("completion_tokens", 0))
            job["cache"].write_text(json.dumps({
                "value": parse_spots(reply), "reply": reply, "usage": usage,
            }, indent=2))
        if spent > cap:
            raise RuntimeError(
                f"judge spend ${spent:.3f} passed cap ${cap:.2f}; "
                "completed judgements are cached"
            )
    return spent, failures


def consolidate(jobs: list[dict]) -> dict:
    rows, judged = [], 0
    for job in jobs:
        value, reply = None, None
        if job["cache"].exists():
            cached = json.loads(job["cache"].read_text())
            value, reply = cached.get("value"), cached.get("reply")
            judged += 1
        rows.append({
            "trace_id": job["trace_id"],
            "direction": job["direction"],
            "sample_index": job["sample_index"],
            "raw_path": job["raw_path"],
            "spots_value": value,
            "judge_reply": reply,
        })
    return {
        "judge_model": JUDGE_MODEL,
        "prompt_sha256": hashlib.sha256(SPOTS_JUDGE_PROMPT.encode()).hexdigest(),
        "n_resamples": len(rows),
        "n_judged": judged,
        "n_with_value": sum(r["spots_value"] is not None for r in rows),
        "rows": rows,
    }


def label_path(round_n: int) -> Path:
    return CAUSE_DIR / f"resample_hand_labels_r{round_n}.json"


def write_labels(round_n: int = 1) -> None:
    """A seeded random sample to label by hand, before any judging.

    Labels are drawn from the saved resamples directly, not from judge output, so
    they can be recorded before the judge has ever run. Round 2 draws a fresh
    sample excluding round 1: if the prompt is changed in response to round-1
    disagreements, those rows have become development data and an agreement rate
    measured on them would be optimistic.
    """
    jobs = jobs_for_pool()
    seen = set()
    for earlier in range(1, round_n):
        path = label_path(earlier)
        if path.exists():
            seen |= {
                (r["trace_id"], r["sample_index"])
                for r in json.loads(path.read_text())["rows"]
            }
    available = [j for j in jobs if (j["trace_id"], j["sample_index"]) not in seen]
    seed = f"{LABEL_SEED}/round{round_n}"
    picked = random.Random(seed).sample(
        available, min(LABEL_SAMPLE_SIZE, len(available))
    )

    path = label_path(round_n)
    existing = {}
    if path.exists():
        existing = {
            (r["trace_id"], r["sample_index"]): r.get("human_value")
            for r in json.loads(path.read_text())["rows"]
        }
    out = []
    for row in picked:
        text = json.loads((ROOT / row["raw_path"]).read_text())["continuation"]
        out.append({
            "trace_id": row["trace_id"],
            "sample_index": row["sample_index"],
            "raw_path": row["raw_path"],
            "text": text,
            # Fill in: an integer, or "NONE" if it never adopts a figure.
            "human_value": existing.get((row["trace_id"], row["sample_index"])),
        })
    path.write_text(json.dumps({
        "round": round_n,
        "seed": seed,
        "excluded_earlier_rounds": len(seen),
        "instructions": (
            "ONE QUESTION per row: does the model pick a single spots-per-giraffe "
            "figure and multiply the population by it?\n"
            "  Yes -> record that number in `human_value` as an integer.\n"
            "  Anything else -> record \"NONE\".\n"
            "NONE is a perfectly good answer and is often the right one. The judge's "
            "answer is deliberately not shown.\n\n"
            "RECORD THE NUMBER when it settles on one figure and uses it:\n"
            "  - \"Let's assume 600 spots. Calculation: 127,000 * 600 = 76,200,000\" "
            "-> 600\n"
            "  - Adjusting once before calculating still counts, and the answer is the "
            "figure that actually gets multiplied: \"assume 400... calves are "
            "included, so say 350... 117,000 * 350\" -> 350\n"
            "  - Reconsidering AFTER the calculation does not cancel it: \"assume "
            "400... 115,000 * 400 = 46,000,000... Is 400 too high?\" -> 400\n\n"
            "RECORD NONE when it does not land on one figure:\n"
            "  - parallel low/medium/high calculations, even if each is worked out in "
            "full: \"conservative 400: 115,000 * 400 = ... liberal 1,000: 115,000 * "
            "1,000 = ...\" -> NONE. Do not take the first, the middle, or an average.\n"
            "  - hypotheticals: \"if we assume 100... if 500... if 1000\" -> NONE\n"
            "  - a range with nothing chosen: \"300-500 spots\" -> NONE, no midpoint\n"
            "  - attributed and not adopted: \"some sources say 400\" -> NONE\n"
            "  - a question left unanswered: \"maybe 1,000 spots?\" -> NONE\n"
            "  - it never reaches a spots-per-giraffe figure -> NONE\n\n"
            "Never record a population, a total across all giraffes, an area, a "
            "percentage, or a year. Only a per-giraffe spot count.\n\n"
            "If you are genuinely unsure, record NONE rather than guessing."
        ),
        "n": len(out),
        "rows": out,
    }, indent=2, ensure_ascii=False))
    done = sum(r["human_value"] is not None for r in out)
    print(f"round   : {round_n}"
          + (f" (excludes {len(seen)} rows from earlier rounds)" if seen else ""))
    print(f"wrote   : {path.relative_to(ROOT)}")
    print(f"labelled: {done}/{len(out)}")
    print("\nFill in human_value for each row, then:")
    print(f"  uv run python analysis/cause/judge_resamples.py "
          f"--execute --only_labelled --round_n {round_n}")


def labelled_keys(round_n: int) -> set:
    path = label_path(round_n)
    if not path.exists():
        raise SystemExit(f"{path.name} is absent; run --label --round_n {round_n} first")
    return {
        (r["trace_id"], r["sample_index"])
        for r in json.loads(path.read_text())["rows"]
    }


def agreement(round_n: int = 1) -> None:
    path = label_path(round_n)
    if not path.exists():
        raise SystemExit(f"{path.name} is absent")
    judged = {}
    for job in jobs_for_pool():
        if job["cache"].exists():
            judged[(job["trace_id"], job["sample_index"])] = json.loads(
                job["cache"].read_text()
            ).get("value")
    labels = json.loads(path.read_text())["rows"]
    done = [r for r in labels if r["human_value"] is not None]
    if not done:
        raise SystemExit("no human labels recorded yet; fill in human_value")
    unjudged = [r for r in done
                if (r["trace_id"], r["sample_index"]) not in judged]
    if unjudged:
        raise SystemExit(
            f"{len(unjudged)} labelled rows have not been judged; run "
            f"--execute --only_labelled --round_n {round_n}"
        )

    exact = both_none = disagree = 0
    mismatches = []
    for row in done:
        human = row["human_value"]
        human = None if str(human).upper() == "NONE" else int(human)
        judge = judged.get((row["trace_id"], row["sample_index"]))
        if human is None and judge is None:
            both_none += 1
        elif human == judge:
            exact += 1
        else:
            disagree += 1
            mismatches.append((row["trace_id"], row["sample_index"], human, judge))
    total = len(done)
    print(f"hand-labelled     : {total}")
    print(f"agree on a number : {exact}")
    print(f"agree on NONE     : {both_none}")
    print(f"disagree          : {disagree}")
    print(f"agreement rate    : {(exact + both_none) / total:.1%}")
    if mismatches:
        print("\ndisagreements (trace, sample, hand label, judge):")
        for row in mismatches:
            print(f"  {row[0]} #{row[1]}: {row[2]} vs {row[3]}")


def main(dry_run: bool = False, execute: bool = False,
         label: bool = False, agreement_check: bool = False,
         only_labelled: bool = False, round_n: int = 1):
    if label:
        return write_labels(round_n)
    if agreement_check:
        return agreement(round_n)

    jobs = jobs_for_pool()
    if only_labelled:
        keys = labelled_keys(round_n)
        jobs = [j for j in jobs if (j["trace_id"], j["sample_index"]) in keys]
        print(f"scope        : hand-labelled round {round_n} only ({len(jobs)} rows)")
    todo = [j for j in jobs if not j["cache"].exists()]
    in_tokens = int(sum(len(j["prompt"]) for j in todo) / 3.7)
    projected = llm.cost_of(JUDGE_MODEL, in_tokens, JUDGE_MAX_TOKENS * len(todo))
    print(f"judge model  : {JUDGE_MODEL}")
    print(f"resamples    : {len(jobs)}")
    print(f"cached       : {len(jobs) - len(todo)}")
    print(f"to judge     : {len(todo)}")
    print(f"projected    : ${projected:.3f} (cap ${MAX_SPEND_USD:.2f})")
    if projected > MAX_SPEND_USD:
        raise SystemExit("refusing: projected cost exceeds the cap")

    if todo and (dry_run or not execute):
        print("API calls    : NONE")
        if not dry_run:
            print("To authorize this paid stage, rerun with --execute.")
        return

    spent, failures = (asyncio.run(judge_missing(todo, MAX_SPEND_USD))
                       if todo else (0.0, []))
    print(f"judge cost   : ${spent:.4f}")
    if failures:
        print(f"FAILED CALLS : {len(failures)} — not cached, rerun to retry")
        for line in failures[:5]:
            print(f"   {line}")

    if only_labelled:
        print(f"\nNext: uv run python analysis/cause/judge_resamples.py "
              f"--agreement_check --round_n {round_n}")
        return

    result = consolidate(jobs_for_pool())
    JUDGED_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"judged       : {result['n_judged']}/{result['n_resamples']}")
    print(f"with a value : {result['n_with_value']} "
          f"({result['n_with_value'] / max(result['n_resamples'], 1):.0%})")
    print(f"written      : {JUDGED_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    fire.Fire(main)
