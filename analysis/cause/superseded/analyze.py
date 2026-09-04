"""Judge and summarize the CAUSE pilot.

No judge call is made unless --execute is supplied. Cached judge outputs can be
summarized locally without that flag.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import fire

os.environ.setdefault("MPLCONFIGDIR", "/tmp/value-leakage-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CAUSE_DIR = Path(__file__).resolve().parent
ROOT = CAUSE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CAUSE_DIR))
sys.path.insert(0, str(ROOT / "analysis" / "disclosure"))

import llm  # noqa: E402
from prepare import REVIEW_PATH, load_design  # noqa: E402
from sample import VARIANTS, continuation_path, validate_review  # noqa: E402
from value_leakage.judge import NUMBER_JUDGE_PROMPT, parse_tagged_estimate  # noqa: E402

JUDGE_CACHE = CAUSE_DIR / "cache" / "finish_numbers"
RESULTS_PATH = CAUSE_DIR / "results.json"


def finish_text(row: dict) -> tuple[str, str]:
    content = (row.get("content") or "").strip()
    # A valid visible answer may be only a number such as "41,600,000". Always
    # judge nonempty visible content; use reasoning only when no visible answer
    # was produced. The number judge, not a character-count heuristic, decides
    # whether the content contains an unambiguous final estimate.
    if content:
        return content, "content"
    reasoning = (row.get("reasoning") or "").strip()
    if len(reasoning) >= 40:
        return reasoning[-3000:], "reasoning_tail"
    return "", "missing"


def judge_key(model: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:32]
    return f"{digest}.json"


def build_jobs(design: dict, judge_model: str) -> list[dict]:
    jobs = []
    specs = (
        ("candidate", VARIANTS, design["continuations_per_variant"]),
        (
            "control",
            ("original", "alt1", "alt2"),
            design["control_continuations_per_variant"],
        ),
    )
    for trace in design["traces"]:
        for family, variants, count in specs:
            for variant in variants:
                for k in range(count):
                    raw = continuation_path(trace["trace_id"], family, variant, k)
                    if not raw.exists():
                        continue
                    row = json.loads(raw.read_text())
                    text, source = finish_text(row)
                    if not text:
                        jobs.append({
                            "trace_id": trace["trace_id"], "family": family,
                            "variant": variant, "k": k, "raw_path": raw,
                            "text_source": source, "prompt": None, "cache": None,
                        })
                        continue
                    prompt = NUMBER_JUDGE_PROMPT.format(llm_text=text)
                    cache = JUDGE_CACHE / judge_key(judge_model, prompt)
                    jobs.append({
                        "trace_id": trace["trace_id"], "family": family,
                        "variant": variant, "k": k, "raw_path": raw,
                        "text_source": source, "prompt": prompt, "cache": cache,
                    })
    return jobs


async def judge_missing(
    jobs: list[dict], judge_model: str, max_concurrent: int, max_out: int,
    max_spend: float,
) -> float:
    todo = [j for j in jobs if j["prompt"] and not j["cache"].exists()]
    if not todo:
        return 0.0
    JUDGE_CACHE.mkdir(parents=True, exist_ok=True)
    client = llm.get_client("openai")
    spent = 0.0
    for start in range(0, len(todo), 40):
        chunk = todo[start : start + 40]
        responses = await llm.batch(
            client, judge_model, [j["prompt"] for j in chunk],
            max_concurrent=max_concurrent, max_tokens=max_out,
            reasoning_effort="low",
        )
        for job, response in zip(chunk, responses):
            if isinstance(response, Exception):
                continue
            reply = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            cost = llm.cost_of(
                judge_model,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
            spent += cost
            job["cache"].write_text(json.dumps({
                "value": parse_tagged_estimate(reply),
                "reply": reply,
                "usage": usage,
            }, indent=2))
            if spent > max_spend:
                raise RuntimeError(
                    f"judge spend ${spent:.3f} passed cap ${max_spend:.2f}; "
                    "completed judgments are cached"
                )
    return spent


def collect(design: dict, review: dict, jobs: list[dict]) -> list[dict]:
    selected = {t["trace_id"]: t for t in review["traces"]}
    threshold = design["threshold"]
    lo = threshold * design["endpoint_min_threshold_ratio"]
    hi = threshold * design["endpoint_max_threshold_ratio"]
    rows = []
    for job in jobs:
        value = None
        judge_reply = None
        if job["cache"] and job["cache"].exists():
            cached = json.loads(job["cache"].read_text())
            value = cached.get("value")
            judge_reply = cached.get("reply")
        exclusion = None
        if value is None:
            exclusion = "unparseable_or_missing"
        elif not lo <= float(value) <= hi:
            exclusion = "outside_existing_threshold_filter"
        candidate_value = None
        if job["family"] == "candidate":
            candidate_value = selected[job["trace_id"]]["selected"][job["variant"]][
                "value"
            ]
        rows.append({
            "trace_id": job["trace_id"],
            "family": job["family"],
            "variant": job["variant"],
            "sample_index": job["k"],
            "raw_path": str(job["raw_path"].relative_to(ROOT)),
            "text_source": job["text_source"],
            "candidate_value": candidate_value,
            "final_value": float(value) if value is not None else None,
            "final_rel": ((float(value) - threshold) / threshold)
            if value is not None else None,
            "included": exclusion is None,
            "exclusion": exclusion,
            "judge_reply": judge_reply,
        })
    return rows


def median_for(rows: list[dict], trace_id: str, family: str, variant: str):
    values = [
        r["final_value"] for r in rows
        if r["trace_id"] == trace_id and r["family"] == family
        and r["variant"] == variant and r["included"]
    ]
    return float(np.median(values)) if values else None


def summarize(design: dict, review: dict, rows: list[dict]) -> tuple[list[dict], dict]:
    threshold = design["threshold"]
    approved = {t["trace_id"]: t for t in review["traces"]}
    per_trace = []
    for trace in design["traces"]:
        trace_id = trace["trace_id"]
        medians = {
            variant: median_for(rows, trace_id, "candidate", variant)
            for variant in VARIANTS
        }
        control_medians = {
            variant: median_for(rows, trace_id, "control", variant)
            for variant in ("original", "alt1", "alt2")
        }
        low_x = approved[trace_id]["selected"]["low"]["value"]
        high_x = approved[trace_id]["selected"]["high"]["value"]
        candidate_effect = None
        retention = None
        surface_effect = None
        control_effect = None
        if medians["low"] is not None and medians["high"] is not None:
            candidate_effect = (medians["high"] - medians["low"]) / threshold
            retention = (medians["high"] - medians["low"]) / (high_x - low_x)
        if medians["surface"] is not None and medians["original"] is not None:
            surface_effect = abs(medians["surface"] - medians["original"]) / threshold
        if all(v is not None for v in control_medians.values()):
            control_effect = float(np.mean([
                abs(control_medians["alt1"] - control_medians["original"]),
                abs(control_medians["alt2"] - control_medians["original"]),
            ])) / threshold
        per_trace.append({
            "trace_id": trace_id,
            "version": trace["version"],
            "candidate_values": {
                v: approved[trace_id]["selected"][v]["value"] for v in VARIANTS
            },
            "endpoint_medians": medians,
            "control_endpoint_medians": control_medians,
            "candidate_effect_rel": candidate_effect,
            "retention_ratio": retention,
            "surface_effect_rel": surface_effect,
            "specificity_rel": candidate_effect - surface_effect
            if candidate_effect is not None and surface_effect is not None else None,
            "nonnumeric_control_effect_rel": control_effect,
        })
    primary = [r for r in rows if r["family"] == "candidate"]
    usable = sum(r["included"] for r in primary)
    success_rate = usable / len(primary) if primary else 0.0
    automated = {
        "primary_expected": len(design["traces"]) * 4
        * design["continuations_per_variant"],
        "primary_observed": len(primary),
        "primary_usable": usable,
        "primary_success_rate": success_rate,
        "passes_minimum_success_rate": success_rate
        >= design["minimum_endpoint_success_rate"],
        "candidate_spread_passes": all(
            t["candidate_values"]["high"] - t["candidate_values"]["low"]
            >= design["minimum_candidate_spread_threshold_ratio"] * threshold
            for t in per_trace
        ),
        "human_completion_review": "pending",
        "pilot_decision": "pending human completion review",
    }
    return per_trace, automated


def figure(design: dict, per_trace: list[dict]):
    threshold = design["threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = {"bet_above": "#D97706", "bet_below": "#4F46E5"}
    for trace in per_trace:
        xs, ys = [], []
        for variant in VARIANTS:
            y = trace["endpoint_medians"].get(variant)
            if y is None:
                continue
            xs.append((trace["candidate_values"][variant] - threshold) / threshold)
            ys.append((y - threshold) / threshold)
        order = np.argsort(xs)
        axes[0].plot(
            np.asarray(xs)[order], np.asarray(ys)[order], marker="o",
            color=colors[trace["version"]], label=trace["trace_id"], alpha=0.85,
        )
    axes[0].axhline(0, color="#6B7280", lw=0.8)
    axes[0].axvline(0, color="#6B7280", lw=0.8)
    axes[0].set_xlabel("Inserted candidate, relative to threshold")
    axes[0].set_ylabel("Median final estimate, relative to threshold")
    axes[0].set_title("Pilot: candidate step and continuation endpoint")
    axes[0].legend(frameon=False, fontsize=8)
    labels = [t["trace_id"] for t in per_trace]
    x = np.arange(len(labels))
    width = 0.25
    for offset, key, label in (
        (-width, "candidate_effect_rel", "high minus low"),
        (0, "surface_effect_rel", "same-number paraphrase"),
        (width, "nonnumeric_control_effect_rel", "nonnumeric step"),
    ):
        vals = [t[key] if t[key] is not None else np.nan for t in per_trace]
        axes[1].bar(x + offset, vals, width, label=label)
    axes[1].axhline(0, color="#6B7280", lw=0.8)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Endpoint change / threshold")
    axes[1].set_title("Candidate effect against edit controls")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CAUSE_DIR / "pilot.png", dpi=180, facecolor="white")
    fig.savefig(CAUSE_DIR / "pilot.svg", facecolor="white")
    plt.close(fig)


def main(
    judge_model: str = "gpt-5.6-luna",
    execute: bool = False,
    dry_run: bool = False,
    max_concurrent: int = 6,
    max_out: int = 256,
    max_spend: float | None = None,
):
    design = load_design()
    try:
        review = validate_review(design)
    except ValueError as exc:
        if execute and not dry_run:
            raise
        print(f"human gate          : BLOCKED — {exc}")
        print("judge API calls     : NONE")
        return
    jobs = build_jobs(design, judge_model)
    expected = len(design["traces"]) * (
        4 * design["continuations_per_variant"]
        + 3 * design["control_continuations_per_variant"]
    )
    todo = [j for j in jobs if j["prompt"] and not j["cache"].exists()]
    in_tokens = int(sum(len(j["prompt"]) for j in todo) / 3.7)
    projected = llm.cost_of(judge_model, in_tokens, max_out * len(todo))
    cap = float(max_spend if max_spend is not None
                else design["pilot_judge_stage_max_spend_usd"])
    print(f"continuations found : {len(jobs)}/{expected}")
    print(f"judge outputs cached: {len(jobs) - len(todo)}/{len(jobs)}")
    print(f"judge calls missing : {len(todo)}")
    print(f"projected worst cost: ${projected:.3f} (cap ${cap:.2f})")
    if projected > cap:
        raise SystemExit("refusing: projected judge cost exceeds frozen pilot cap")
    if todo and (dry_run or not execute):
        print("API calls           : NONE")
        print("Marie must rerun with --execute to authorize paid judging.")
        return
    spent = 0.0
    if todo:
        spent = asyncio.run(judge_missing(
            jobs, judge_model, max_concurrent, max_out, cap
        ))
        jobs = build_jobs(design, judge_model)
    rows = collect(design, review, jobs)
    per_trace, automated = summarize(design, review, rows)
    RESULTS_PATH.write_text(json.dumps({
        "experiment": design["experiment"],
        "threshold": design["threshold"],
        "judge_model": judge_model,
        "automated_pilot_checks": automated,
        "per_trace": per_trace,
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    figure(design, per_trace)
    print(f"judge cost          : ${spent:.3f}")
    print(f"saved               : {RESULTS_PATH.relative_to(ROOT)}")
    print("pilot decision      : pending Marie's raw-completion review")


if __name__ == "__main__":
    fire.Fire(main)
