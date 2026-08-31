"""TARGET_VISIBILITY_DIR pre-check — does our cheap judge extract the same numbers as starter dataset's?

starter dataset judged its shipped runs with claude-opus-5 and left the results in
runs/<run>/estimates.json — one final number per baseline answer. Those exact
answers are still on disk, so we can put the SAME text through the SAME verbatim
NUMBER_JUDGE prompt with gpt-5-mini and compare the two extractions row by row.

That turns "is our judge comparable to starter dataset's?" from an assumption into a
measurement, for about 3 cents instead of the ~$2 it would cost to re-judge
everything with opus-5.

What matters is not just the raw match rate but WHERE it disagrees: a judge that
returns UNKNOWN slightly more often is harmless (both sides drop the row); a
judge that returns a different NUMBER for the same answer is not.

Only baseline is checkable — starter dataset's estimates.json holds baseline alone.

  uv run python analysis/target_visibility/judge_agreement.py --dry_run
  uv run python analysis/target_visibility/judge_agreement.py
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "disclosure"))
from common import RUNS  # noqa: E402
import llm  # noqa: E402
from sample_conditions import SUBJECTS, TARGET_VISIBILITY_DIR  # noqa: E402

from value_leakage.judge import (  # noqa: E402  VERBATIM — never edited
    NUMBER_JUDGE_PROMPT, parse_tagged_estimate)

CACHE = TARGET_VISIBILITY_DIR / "cache" / "agreement"


def build_jobs(judge_model: str, models: list[str]) -> list[dict]:
    jobs = []
    for model in models:
        run_dir = RUNS / SUBJECTS[model]["shipped"]
        v1_cfg = json.loads((run_dir / "config.json").read_text())
        v1 = json.loads((run_dir / "estimates.json").read_text()).get("baseline", [])
        rows = json.loads((run_dir / "baseline.json").read_text())["rows"]
        for row in rows:
            i = row["i"]
            if i >= len(v1):
                continue
            text = (row.get("content") or "").strip()
            if not text:
                continue
            prompt = NUMBER_JUDGE_PROMPT.format(llm_text=text)
            key = hashlib.sha256(f"{judge_model}\n{prompt}".encode()).hexdigest()[:32]
            jobs.append({
                "model": model, "i": i, "prompt": prompt,
                "v1_value": v1[i], "v1_judge": v1_cfg.get("judge_model"),
                "path": CACHE / f"{key}.json",
            })
    for j in jobs:
        j["cached"] = json.loads(j["path"].read_text()) if j["path"].exists() else None
    return jobs


async def run_jobs(jobs, backend, judge_model, max_concurrent, max_out,
                   reasoning_effort):
    todo = [j for j in jobs if j["cached"] is None]
    if not todo:
        print("everything already cached — no API calls made")
        return 0.0
    CACHE.mkdir(parents=True, exist_ok=True)
    client = llm.get_client(backend)
    spent = 0.0
    for start in range(0, len(todo), 40):
        chunk = todo[start:start + 40]
        responses = await llm.batch(client, judge_model,
                                    [j["prompt"] for j in chunk],
                                    max_concurrent=max_concurrent,
                                    max_tokens=max_out,
                                    reasoning_effort=reasoning_effort)
        for j, r in zip(chunk, responses):
            if isinstance(r, Exception):
                print(f"  {j['model']}[{j['i']}]: {type(r).__name__}: {r}")
                continue
            text = r.choices[0].message.content or ""
            usage = r.usage.model_dump() if r.usage else {}
            cost = llm.cost_of(judge_model, usage.get("prompt_tokens", 0),
                               usage.get("completion_tokens", 0))
            spent += cost
            rec = {"reply": text, "value": parse_tagged_estimate(text),
                   "finish_reason": r.choices[0].finish_reason, "cost": cost}
            j["path"].write_text(json.dumps(rec, indent=2))
            j["cached"] = rec
        print(f"  judged {min(start+40, len(todo))}/{len(todo)} — ${spent:.3f}")
    return spent


def report(jobs, models, judge_model):
    out = {}
    for model in models:
        rows = [j for j in jobs if j["model"] == model and j["cached"]]
        if not rows:
            continue
        both = [j for j in rows if j["v1_value"] is not None
                and j["cached"]["value"] is not None]
        same = [j for j in both if float(j["v1_value"]) == float(j["cached"]["value"])]
        only_v1 = [j for j in rows if j["v1_value"] is not None
                   and j["cached"]["value"] is None]
        only_us = [j for j in rows if j["v1_value"] is None
                   and j["cached"]["value"] is not None]
        v1_judge = rows[0]["v1_judge"]
        print(f"\n{model}   starter dataset judge {v1_judge}  vs  ours {judge_model}")
        print(f"  both returned a number : {len(both):3d} of {len(rows)}")
        print(f"    identical            : {len(same):3d} "
              + (f"({len(same)/len(both):.1%})" if both else ""))
        print(f"  only starter dataset got a number   : {len(only_v1):3d}  "
              f"(we said UNKNOWN — row is dropped either way)")
        print(f"  only we got a number   : {len(only_us):3d}")
        diffs = [j for j in both if float(j["v1_value"]) != float(j["cached"]["value"])]
        for j in diffs[:5]:
            print(f"    differs: run {j['i']:3d}  starter dataset {j['v1_value']:,.0f}  "
                  f"ours {j['cached']['value']:,.0f}")
        if len(diffs) > 5:
            print(f"    ... and {len(diffs)-5} more")
        out[model] = {
            "v1_judge": v1_judge, "our_judge": judge_model, "n_rows": len(rows),
            "n_both": len(both), "n_identical": len(same),
            "identical_rate": (len(same) / len(both)) if both else None,
            "only_v1": len(only_v1), "only_ours": len(only_us),
            "disagreements": [{"i": j["i"], "v1": j["v1_value"],
                               "ours": j["cached"]["value"]} for j in diffs],
        }
    (TARGET_VISIBILITY_DIR / "judge_agreement.json").write_text(json.dumps(out, indent=2))
    print("\nsaved analysis/target_visibility/judge_agreement.json")
    rates = [v["identical_rate"] for v in out.values() if v["identical_rate"]]
    if rates:
        lo = min(rates)
        print(f"\nreading: {lo:.0%} of answers get the identical number from both "
              f"judges." + ("  Comparable — the cheap judge is fine, and the "
                            "write-up can say so with a number."
                            if lo >= 0.95 else
                            "  NOT comparable enough — reconsider using "
                            "claude-opus-5 for the number extraction."))


def main(backend: str = "openai", judge_model: str = "gpt-5-mini",
         model: str | None = None, max_concurrent: int = 6, max_out: int = 256,
         reasoning_effort: str = "low", max_spend: float = 0.20,
         dry_run: bool = False):
    models = [model] if model else list(SUBJECTS)
    jobs = build_jobs(judge_model, models)
    todo = [j for j in jobs if j["cached"] is None]
    in_tok = int(sum(len(j["prompt"]) for j in todo) / 3.7)
    worst = llm.cost_of(judge_model, in_tok, max_out * len(todo))
    print(f"comparing {judge_model} against the starter dataset judge on the SAME baseline "
          f"answers,\nusing the same verbatim NUMBER_JUDGE prompt.")
    print(f"jobs      : {len(jobs)} rows, {len(todo)} to call")
    print(f"input     : ~{in_tok:,} tokens")
    print(f"projected : ${worst:.3f} worst case   (--max_spend ${max_spend:.2f})")
    if dry_run:
        print("dry run — no API calls made")
        return
    if worst > max_spend:
        print(f"REFUSING TO RUN: ${worst:.3f} > --max_spend ${max_spend:.2f}")
        return
    spent = asyncio.run(run_jobs(jobs, backend, judge_model, max_concurrent,
                                 max_out, reasoning_effort))
    report(jobs, models, judge_model)
    print(f"cost ${spent:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
