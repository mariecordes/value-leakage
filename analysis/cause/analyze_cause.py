"""CAUSE step 5 — judge the final answers and test the anchoring hypothesis.

No judge call is made unless --execute is supplied. Cached judgements summarise
locally without it.

THE TEST
--------
Within each trace, every variant inserts a different spots-per-giraffe figure at the
same cut, in the model's own words. If that figure anchors the answer, a higher
insert should give a higher final total.

Primary result: a sign test across traces on the within-trace Spearman correlation
between inserted figure and median final answer. Each trace counts once, so traces
with more variants do not dominate.

Secondary: the retention ratio, how much of the inserted change survives to the
answer, and whether the below-paying arm pulls back toward its rewarded side more
than the above-paying arm does.

The noise floor is measured, not assumed: the six continuations from the unchanged
original step show how far the endpoint moves when nothing was changed.

Final answers are read by the repository's existing NUMBER_JUDGE, unchanged, with the
published [threshold/10, threshold*10] filter. That judge was validated in the earlier
work; the resample-reading judge is not used here.

Run:
    uv run python analysis/cause/analyze_cause.py --dry_run
    uv run python analysis/cause/analyze_cause.py --execute
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import fire

import numpy as np  # noqa: E402

CAUSE_DIR = Path(__file__).resolve().parent
ROOT = CAUSE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CAUSE_DIR))
sys.path.insert(0, str(ROOT / "analysis" / "disclosure"))

import llm  # noqa: E402
from continue_traces import INSERTS_PATH, finish_path  # noqa: E402
from plot_cause import main_figure  # noqa: E402
from resample import load_raw  # noqa: E402
from value_leakage.judge import NUMBER_JUDGE_PROMPT, parse_tagged_estimate  # noqa: E402

JUDGE_CACHE = CAUSE_DIR / "cache" / "cause_finals"
RESULTS_PATH = CAUSE_DIR / "cause_results.json"
JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_MAX_TOKENS = 400
MAX_SPEND_USD = 0.60


def cache_path(prompt: str) -> Path:
    digest = hashlib.sha256(f"{JUDGE_MODEL}\n{prompt}".encode()).hexdigest()[:32]
    return JUDGE_CACHE / f"{digest}.json"


def jobs(spec: dict) -> list[dict]:
    out = []
    for trace in spec["traces"]:
        for variant in trace["variants"]:
            for k in range(spec["continuations_per_variant"]):
                row = load_raw(finish_path(trace["trace_id"], variant["variant"], k))
                # A missing or failed continuation must still occupy a row. Dropping
                # it here would shrink the denominator and inflate the usable rate,
                # so "437 of 438" could silently become "437 of 437".
                text = "" if row is None or row.get("error") else \
                    (row.get("content") or "").strip()
                missing_reason = None
                if row is None:
                    missing_reason = "no_response_file"
                elif row.get("error"):
                    missing_reason = "call_failed"
                job = {
                    "trace_id": trace["trace_id"],
                    "direction": trace["direction"],
                    "variant": variant["variant"],
                    "inserted_value": variant["inserted_value"],
                    "sample_index": k,
                    "prompt": None,
                    "cache": None,
                    "no_visible_answer": not text,
                    "missing_reason": missing_reason,
                }
                if text:
                    prompt = NUMBER_JUDGE_PROMPT.format(llm_text=text)
                    job["prompt"] = prompt
                    job["cache"] = cache_path(prompt)
                out.append(job)
    return out


async def judge(todo: list[dict], cap: float) -> tuple[float, list[str]]:
    JUDGE_CACHE.mkdir(parents=True, exist_ok=True)
    client = llm.get_client("openai")
    spent, failures = 0.0, []
    for start in range(0, len(todo), 40):
        chunk = todo[start : start + 40]
        replies = await llm.batch(
            client, JUDGE_MODEL, [j["prompt"] for j in chunk],
            max_concurrent=8, max_tokens=JUDGE_MAX_TOKENS, reasoning_effort="low",
        )
        for job, response in zip(chunk, replies):
            if isinstance(response, Exception):
                failures.append(f"{job['trace_id']}/{job['variant']} #{job['sample_index']}: {response}")
                continue
            reply = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            if not reply.strip():
                failures.append(
                    f"{job['trace_id']}/{job['variant']} #{job['sample_index']}: "
                    f"empty reply at {usage.get('completion_tokens')} tokens"
                )
                continue
            spent += llm.cost_of(JUDGE_MODEL, usage.get("prompt_tokens", 0),
                                 usage.get("completion_tokens", 0))
            job["cache"].write_text(json.dumps({
                "value": parse_tagged_estimate(reply), "reply": reply, "usage": usage,
            }, indent=2))
        if spent > cap:
            raise RuntimeError(f"judge spend ${spent:.3f} passed cap ${cap:.2f}")
    return spent, failures


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation. None when either side has no variation to rank."""
    if len(xs) < 3:
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def sign_test(successes: int, n: int) -> float:
    """Two-sided exact binomial p at p=0.5."""
    from math import comb
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(successes, n - successes) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def collect(spec: dict, all_jobs: list[dict]) -> list[dict]:
    threshold = spec["threshold"]
    lo, hi = threshold * 0.1, threshold * 10
    rows = []
    for job in all_jobs:
        value = None
        if job["cache"] and job["cache"].exists():
            value = json.loads(job["cache"].read_text()).get("value")
        exclusion = None
        if job.get("missing_reason"):
            exclusion = job["missing_reason"]
        elif job["no_visible_answer"]:
            exclusion = "no_visible_answer"
        elif value is None:
            exclusion = "unparseable"
        elif not lo <= float(value) <= hi:
            exclusion = "outside_threshold_filter"
        rows.append({
            **{k: job[k] for k in
               ("trace_id", "direction", "variant", "inserted_value", "sample_index")},
            "final_value": float(value) if value is not None else None,
            "included": exclusion is None,
            "exclusion": exclusion,
        })
    return rows


def summarise(spec: dict, rows: list[dict]) -> dict:
    threshold = spec["threshold"]
    per_trace = []
    for trace in spec["traces"]:
        tid = trace["trace_id"]
        variants = []
        for variant in trace["variants"]:
            vals = [r["final_value"] for r in rows
                    if r["trace_id"] == tid and r["variant"] == variant["variant"]
                    and r["included"]]
            if not vals:
                continue
            variants.append({
                "variant": variant["variant"],
                "inserted_value": variant["inserted_value"],
                "n": len(vals),
                "median_final": float(np.median(vals)),
                "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
                "share_above_threshold": sum(v > threshold for v in vals) / len(vals),
            })
        alternatives = [v for v in variants if v["variant"] != "original"]
        baseline = next((v for v in variants if v["variant"] == "original"), None)

        rho = slope = retention = None
        if len(alternatives) >= 3:
            xs = [v["inserted_value"] for v in alternatives]
            ys = [v["median_final"] for v in alternatives]
            rho = spearman(xs, ys)
            if max(xs) > min(xs):
                lowest = min(alternatives, key=lambda v: v["inserted_value"])
                highest = max(alternatives, key=lambda v: v["inserted_value"])
                slope = (highest["median_final"] - lowest["median_final"]) / threshold
                # If the answer simply followed the premise, the total would move by
                # population x delta-spots. Retention is the observed move over that.
                population = lowest["median_final"] / max(lowest["inserted_value"], 1)
                expected = population * (highest["inserted_value"]
                                         - lowest["inserted_value"])
                retention = ((highest["median_final"] - lowest["median_final"])
                             / expected) if expected else None

        noise = None
        if baseline:
            vals = [r["final_value"] for r in rows
                    if r["trace_id"] == tid and r["variant"] == "original"
                    and r["included"]]
            if len(vals) > 1:
                noise = float(np.percentile(vals, 75) - np.percentile(vals, 25)) / threshold

        per_trace.append({
            "trace_id": tid,
            "direction": trace["direction"],
            "n_variants": len(variants),
            "inserted_range": [min((v["inserted_value"] for v in alternatives),
                                   default=None),
                               max((v["inserted_value"] for v in alternatives),
                                   default=None)],
            "spearman_rho": rho,
            "high_minus_low_rel": slope,
            "retention_ratio": retention,
            "baseline_iqr_rel": noise,
            "variants": variants,
        })

    scored = [t for t in per_trace if t["spearman_rho"] is not None]
    positive = sum(1 for t in scored if t["spearman_rho"] > 0)
    result = {
        "n_traces_scored": len(scored),
        "n_positive_rho": positive,
        "sign_test_p": sign_test(positive, len(scored)),
        "median_rho": float(np.median([t["spearman_rho"] for t in scored]))
        if scored else None,
        "median_retention": float(np.median(
            [t["retention_ratio"] for t in scored if t["retention_ratio"] is not None]
        )) if scored else None,
        "median_baseline_iqr_rel": float(np.median(
            [t["baseline_iqr_rel"] for t in per_trace if t["baseline_iqr_rel"] is not None]
        )),
    }
    for arm in ("above", "below"):
        subset = [t for t in scored if t["direction"] == arm]
        result[f"{arm}_n"] = len(subset)
        result[f"{arm}_positive"] = sum(1 for t in subset if t["spearman_rho"] > 0)
        result[f"{arm}_median_retention"] = float(np.median(
            [t["retention_ratio"] for t in subset if t["retention_ratio"] is not None]
        )) if subset else None
    # Report the alternatives separately: they are the denominator the test uses.
    # The pooled figure includes the 90 baseline continuations and is not the one
    # the sign test rests on.
    alternatives = [r for r in rows if r["variant"] != "original"]
    baseline = [r for r in rows if r["variant"] == "original"]
    result["continuations"] = {
        "total": len(rows),
        "usable": sum(r["included"] for r in rows),
        "alternatives_total": len(alternatives),
        "alternatives_usable": sum(r["included"] for r in alternatives),
        "baseline_total": len(baseline),
        "baseline_usable": sum(r["included"] for r in baseline),
    }
    return {"headline": result, "per_trace": per_trace}


def main(dry_run: bool = False, execute: bool = False, max_spend: float | None = None):
    spec = json.loads(INSERTS_PATH.read_text())
    all_jobs = jobs(spec)
    todo = [j for j in all_jobs if j["prompt"] and not j["cache"].exists()]
    cap = MAX_SPEND_USD if max_spend is None else float(max_spend)
    print(f"continuations : {len(all_jobs)}")
    print(f"judged        : {len(all_jobs) - len(todo)}")
    print(f"to judge      : {len(todo)}")
    print(f"cap           : ${cap:.2f}")

    if todo and (dry_run or not execute):
        print("API calls     : NONE")
        if not dry_run:
            print("To authorize this paid stage, rerun with --execute.")
        return

    spent = 0.0
    if todo:
        spent, failures = asyncio.run(judge(todo, cap))
        if failures:
            print(f"FAILED        : {len(failures)} — not cached, rerun to retry")
            for line in failures[:5]:
                print(f"   {line}")
        all_jobs = jobs(spec)

    rows = collect(spec, all_jobs)
    summary = summarise(spec, rows)
    RESULTS_PATH.write_text(json.dumps({
        "experiment": spec["experiment"],
        "threshold": spec["threshold"],
        "judge_model": JUDGE_MODEL,
        "judge_prompt": "value_leakage.judge.NUMBER_JUDGE_PROMPT, unchanged",
        **summary,
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    main_figure(json.loads(RESULTS_PATH.read_text()))

    h = summary["headline"]
    print(f"\njudge cost    : ${spent:.4f}")
    c = h["continuations"]
    print(f"usable answers: {c['alternatives_usable']}/{c['alternatives_total']} "
          f"alternatives (the test denominator), "
          f"{c['baseline_usable']}/{c['baseline_total']} baseline")
    print(f"\nPRIMARY: traces where a higher inserted figure gave a higher answer")
    print(f"  {h['n_positive_rho']}/{h['n_traces_scored']}   "
          f"sign test p = {h['sign_test_p']:.3g}")
    print(f"  above-paying: {h['above_positive']}/{h['above_n']}"
          f"   below-paying: {h['below_positive']}/{h['below_n']}")
    print(f"\nmedian rank correlation : {h['median_rho']:.3f}")
    print(f"median retention        : {h['median_retention']:.3f}"
          if h["median_retention"] is not None else "median retention        : n/a")
    print(f"  above arm {h['above_median_retention']:.3f}"
          f" | below arm {h['below_median_retention']:.3f}"
          if h["above_median_retention"] is not None else "")
    print(f"baseline noise (IQR)    : {h['median_baseline_iqr_rel']:.3f} threshold units")
    print(f"\nsaved         : {RESULTS_PATH.relative_to(ROOT)}, cause.png")


if __name__ == "__main__":
    fire.Fire(main)
