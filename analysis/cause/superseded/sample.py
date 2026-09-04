"""Generate CAUSE pilot candidates and continuations.

Safety property: this script makes no API call unless --execute is supplied.
Marie runs every paid command herself.
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

from prepare import (  # noqa: E402
    RAW_DIR,
    REVIEW_PATH,
    load_design,
    sha256,
    trace_material,
    validate_trace,
)
from value_leakage.api.openrouter.chat_completions import (  # noqa: E402
    get_openrouter_client,
    process_batch,
)

NUMBER_MENTION = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:billion|million|thousand|bn|b|m|k)?",
    re.I,
)
VARIANTS = ("original", "surface", "low", "high")
PREFIX_CONTEXT_MAX_CHARS = 1200


def build_prefill(prompt: str, prefix: str) -> list[dict]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prefix},
    ]


def message_text(response, prefer_reasoning: bool) -> tuple[str, str, str]:
    msg = response.choices[0].message
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
        or ""
    )
    content = msg.content or ""
    if prefer_reasoning:
        text = reasoning or content
        source = "reasoning" if reasoning else "content"
    else:
        text = content or reasoning
        source = "content" if content else "reasoning"
    return text, source, reasoning


def first_chunk(text: str) -> str:
    """A review aid, not an automatic scientific label.

    Candidate steps often contain a heading followed by several bullet lines, so
    the first paragraph is more useful than the first punctuation-delimited sentence.
    Marie still approves and may adjust the exact retained text.
    """
    stripped = (text or "").lstrip("\n")
    parts = re.split(r"\n\s*\n", stripped, maxsplit=1)
    return parts[0].rstrip() + ("\n" if parts and parts[0] else "")


def prefix_tail(text: str, max_chars: int = PREFIX_CONTEXT_MAX_CHARS) -> str:
    """Return readable context immediately before an intervention boundary."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    newline = tail.find("\n")
    return tail[newline + 1 :] if newline >= 0 else tail


def raw_path(kind: str, trace_id: str, sample_id: str) -> Path:
    return RAW_DIR / kind / f"{trace_id}__{sample_id}.json"


def load_raw(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def usage_cost(response) -> tuple[dict, float]:
    usage = response.usage.model_dump() if response.usage else {}
    return usage, float(usage.get("cost") or 0.0)


async def call_batch(design: dict, messages: list[list[dict]], max_tokens: int):
    body = {
        "usage": {"include": True},
        "reasoning": {"effort": design["reasoning_effort"]},
        "provider": {"order": [design["provider"]], "allow_fallbacks": False},
    }
    return await process_batch(
        client=get_openrouter_client(),
        model=design["model"],
        messages_list=messages,
        temperature=design["temperature"],
        top_p=design["top_p"],
        max_tokens=max_tokens,
        max_concurrent=min(8, len(messages)),
        extra_body=body,
        return_exceptions=True,
    )


def surface_prompt(step: str) -> str:
    return (
        "Rewrite the reasoning step below while preserving every number, equation, "
        "factual claim, and degree of uncertainty exactly. Change wording only. "
        "Keep the same bullet/block structure where practical. Output only the "
        "rewritten step, with no commentary.\n\nREASONING STEP:\n" + step
    )


async def generate_group(
    design: dict,
    trace: dict,
    kind: str,
    count: int,
    messages: list[dict],
    prefer_reasoning: bool,
    max_tokens: int,
    spent: float,
    max_spend: float,
) -> float:
    missing = []
    for k in range(count):
        path = raw_path(kind, trace["trace_id"], f"{k:02d}")
        existing = load_raw(path)
        if existing is None or existing.get("error"):
            missing.append((k, path))
    if not missing:
        return spent
    if spent >= max_spend:
        raise RuntimeError(f"stage spend ${spent:.3f} reached cap ${max_spend:.2f}")
    responses = await call_batch(
        design, [messages] * len(missing), max_tokens=max_tokens
    )
    for (k, path), response in zip(missing, responses):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(response, Exception):
            path.write_text(json.dumps({
                "trace_id": trace["trace_id"],
                "kind": kind,
                "sample_index": k,
                "error": f"{type(response).__name__}: {response}",
            }, indent=2))
            continue
        text, source, reasoning = message_text(response, prefer_reasoning)
        usage, cost = usage_cost(response)
        spent += cost
        msg = response.choices[0].message
        path.write_text(json.dumps({
            "trace_id": trace["trace_id"],
            "kind": kind,
            "sample_index": k,
            "text": text,
            "text_source": source,
            "reasoning": reasoning,
            "content": msg.content or "",
            "suggested_first_chunk": first_chunk(text),
            "number_mentions": NUMBER_MENTION.findall(first_chunk(text)),
            "finish_reason": response.choices[0].finish_reason,
            "usage": usage,
        }, indent=2, ensure_ascii=False))
        if spent > max_spend:
            raise RuntimeError(
                f"stage spend ${spent:.3f} passed cap ${max_spend:.2f}; "
                "successful responses are cached"
            )
    return spent


def candidate_review_entry(design: dict, trace: dict) -> dict:
    material = trace_material(trace)

    def records(kind: str, count: int) -> list[dict]:
        out = []
        for k in range(count):
            path = raw_path(kind, trace["trace_id"], f"{k:02d}")
            row = load_raw(path)
            out.append({
                "sample_index": k,
                "raw_path": str(path.relative_to(ROOT)),
                "error": (row or {}).get("error"),
                "suggested_text": (row or {}).get("suggested_first_chunk"),
                "number_mentions": (row or {}).get("number_mentions", []),
                "valid": None,
                "candidate_value": None,
                "rejection_reason": None,
            })
        return out

    return {
        "trace_id": trace["trace_id"],
        "version": trace["version"],
        "i": trace["i"],
        "candidate_prefix_sha256": sha256(material["candidate_prefix"]),
        "candidate_cut": {
            "start_char": trace["candidate"]["start"],
            "prefix_total_chars": len(material["candidate_prefix"]),
            "prefix_tail": prefix_tail(material["candidate_prefix"]),
            "note": (
                "prefix_tail is original reasoning immediately before the cut; "
                "new candidate sampling starts after its final character"
            ),
        },
        "original_candidate": {
            "text": material["candidate_step"],
            "value": trace["candidate"]["value"],
            "approved": True,
        },
        "natural_attempts": records("candidate", design["candidate_attempts"]),
        "surface_attempts": records("surface", design["surface_attempts"]),
        "control": {
            "prefix_sha256": sha256(material["control_prefix"]),
            "original_text": material["control_step"],
            "attempts": records("control", design["control_attempts"]),
            "selected": {
                "original": {"text": material["control_step"], "approved": True},
                "alt1": None,
                "alt2": None,
            },
        },
        "selected": {
            "original": {
                "text": material["candidate_step"],
                "value": trace["candidate"]["value"],
                "approved": True,
            },
            "surface": None,
            "low": None,
            "high": None,
        },
        "trace_human_approved": False,
    }


def write_review(design: dict):
    old = json.loads(REVIEW_PATH.read_text()) if REVIEW_PATH.exists() else {}
    old_by_id = {t["trace_id"]: t for t in old.get("traces", [])}
    traces = []
    for trace in design["traces"]:
        fresh = candidate_review_entry(design, trace)
        previous = old_by_id.get(trace["trace_id"])
        if previous:
            # Preserve human decisions while refreshing the list of raw attempts.
            fresh["selected"] = previous.get("selected", fresh["selected"])
            fresh["control"]["selected"] = previous.get("control", {}).get(
                "selected", fresh["control"]["selected"]
            )
            fresh["trace_human_approved"] = previous.get(
                "trace_human_approved", False
            )
        traces.append(fresh)
    REVIEW_PATH.write_text(json.dumps({
        "experiment": design["experiment"],
        "selection_rules": {
            "candidate_range": [
                design["candidate_min_threshold_ratio"] * design["threshold"],
                design["candidate_max_threshold_ratio"] * design["threshold"],
            ],
            "minimum_low_high_spread": (
                design["minimum_candidate_spread_threshold_ratio"]
                * design["threshold"]
            ),
        },
        "human_approved": old.get("human_approved", False),
        "traces": traces,
    }, indent=2, ensure_ascii=False))


async def run_candidates(design: dict, max_spend: float) -> float:
    spent = 0.0
    for trace in design["traces"]:
        material = trace_material(trace)
        spent = await generate_group(
            design, trace, "candidate", design["candidate_attempts"],
            build_prefill(material["prompt"], material["candidate_prefix"]),
            True, design["candidate_max_tokens"], spent, max_spend,
        )
        spent = await generate_group(
            design, trace, "surface", design["surface_attempts"],
            [{"role": "user", "content": surface_prompt(material["candidate_step"])}],
            False, design.get("surface_repair_max_tokens", design["candidate_max_tokens"]),
            spent, max_spend,
        )
        spent = await generate_group(
            design, trace, "control", design["control_attempts"],
            build_prefill(material["prompt"], material["control_prefix"]),
            True, design["candidate_max_tokens"], spent, max_spend,
        )
    write_review(design)
    return spent


def validate_review(design: dict) -> dict:
    if not REVIEW_PATH.exists():
        raise ValueError("candidate_review.json is absent; run candidate generation first")
    review = json.loads(REVIEW_PATH.read_text())
    if review.get("human_approved") is not True:
        raise ValueError("candidate_review.json has not received final human approval")
    expected = {t["trace_id"] for t in design["traces"]}
    actual = {t["trace_id"] for t in review.get("traces", [])}
    if actual != expected:
        raise ValueError(f"review trace IDs differ: expected {expected}, got {actual}")
    min_value = design["candidate_min_threshold_ratio"] * design["threshold"]
    max_value = design["candidate_max_threshold_ratio"] * design["threshold"]
    min_spread = (
        design["minimum_candidate_spread_threshold_ratio"] * design["threshold"]
    )
    for trace in review["traces"]:
        if trace.get("trace_human_approved") is not True:
            raise ValueError(f"{trace['trace_id']} lacks trace-level human approval")
        selected = trace.get("selected", {})
        for variant in VARIANTS:
            item = selected.get(variant)
            if not item or item.get("approved") is not True or not item.get("text"):
                raise ValueError(f"{trace['trace_id']}/{variant} is not approved")
            if item.get("value") is None:
                raise ValueError(f"{trace['trace_id']}/{variant} lacks candidate value")
        low, high = selected["low"]["value"], selected["high"]["value"]
        if not (min_value <= low < high <= max_value):
            raise ValueError(f"{trace['trace_id']} low/high are outside frozen range")
        if high - low < min_spread:
            raise ValueError(f"{trace['trace_id']} low/high spread is below {min_spread}")
        control = trace.get("control", {}).get("selected", {})
        for variant in ("original", "alt1", "alt2"):
            item = control.get(variant)
            if not item or item.get("approved") is not True or not item.get("text"):
                raise ValueError(f"{trace['trace_id']}/control/{variant} not approved")
    return review


def continuation_path(trace_id: str, family: str, variant: str, k: int) -> Path:
    return RAW_DIR / "finishes" / f"{trace_id}__{family}_{variant}__{k:02d}.json"


async def finish_variant(
    design: dict,
    trace: dict,
    family: str,
    variant: str,
    prompt: str,
    prefix: str,
    step: str,
    count: int,
    spent: float,
    max_spend: float,
) -> float:
    missing = [
        (k, continuation_path(trace["trace_id"], family, variant, k))
        for k in range(count)
        if (
            not continuation_path(trace["trace_id"], family, variant, k).exists()
            or json.loads(
                continuation_path(trace["trace_id"], family, variant, k).read_text()
            ).get("error")
        )
    ]
    if not missing:
        return spent
    text = prefix + step
    if text and not text.endswith(("\n", " ")):
        text += "\n"
    responses = await call_batch(
        design,
        [build_prefill(prompt, text)] * len(missing),
        max_tokens=design["continuation_max_tokens"],
    )
    for (k, path), response in zip(missing, responses):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(response, Exception):
            path.write_text(json.dumps({
                "trace_id": trace["trace_id"], "family": family,
                "variant": variant, "sample_index": k,
                "error": f"{type(response).__name__}: {response}",
            }, indent=2))
            continue
        msg = response.choices[0].message
        usage, cost = usage_cost(response)
        spent += cost
        path.write_text(json.dumps({
            "trace_id": trace["trace_id"],
            "family": family,
            "variant": variant,
            "sample_index": k,
            "prompt_sha256": sha256(prompt),
            "prefill_sha256": sha256(text),
            "inserted_step": step,
            "reasoning": (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            ),
            "content": msg.content or "",
            "finish_reason": response.choices[0].finish_reason,
            "usage": usage,
        }, indent=2, ensure_ascii=False))
        if spent > max_spend:
            raise RuntimeError(
                f"stage spend ${spent:.3f} passed cap ${max_spend:.2f}; "
                "successful responses are cached"
            )
    return spent


async def run_continuations(design: dict, review: dict, max_spend: float) -> float:
    by_id = {t["trace_id"]: t for t in review["traces"]}
    spent = 0.0
    for trace in design["traces"]:
        material = trace_material(trace)
        approved = by_id[trace["trace_id"]]
        for variant in VARIANTS:
            spent = await finish_variant(
                design, trace, "candidate", variant, material["prompt"],
                material["candidate_prefix"], approved["selected"][variant]["text"],
                design["continuations_per_variant"], spent, max_spend,
            )
        for variant in ("original", "alt1", "alt2"):
            spent = await finish_variant(
                design, trace, "control", variant, material["prompt"],
                material["control_prefix"],
                approved["control"]["selected"][variant]["text"],
                design["control_continuations_per_variant"], spent, max_spend,
            )
    return spent


def count_cached(design: dict, stage: str) -> tuple[int, int]:
    if stage == "candidates":
        specs = (
            ("candidate", design["candidate_attempts"]),
            ("surface", design["surface_attempts"]),
            ("control", design["control_attempts"]),
        )
        total = len(design["traces"]) * sum(n for _, n in specs)
        have = sum(
            (row := load_raw(raw_path(kind, t["trace_id"], f"{k:02d}")))
            is not None and not row.get("error")
            for t in design["traces"] for kind, n in specs for k in range(n)
        )
        return have, total
    total = len(design["traces"]) * (
        4 * design["continuations_per_variant"]
        + 3 * design["control_continuations_per_variant"]
    )
    have = 0
    for trace in design["traces"]:
        for family, variants, n in (
            ("candidate", VARIANTS, design["continuations_per_variant"]),
            ("control", ("original", "alt1", "alt2"),
             design["control_continuations_per_variant"]),
        ):
            for v in variants:
                for k in range(n):
                    path = continuation_path(trace["trace_id"], family, v, k)
                    if path.exists() and not json.loads(path.read_text()).get("error"):
                        have += 1
    return have, total


def main(
    stage: str = "candidates",
    execute: bool = False,
    dry_run: bool = False,
    max_spend: float | None = None,
):
    if stage not in {"candidates", "continuations"}:
        raise ValueError("stage must be 'candidates' or 'continuations'")
    design = load_design()
    failures = [
        f"{trace['trace_id']}: {failure}"
        for trace in design["traces"]
        for failure in validate_trace(trace, design["threshold"])
    ]
    if failures:
        raise SystemExit("frozen-design validation failed:\n  " + "\n  ".join(failures))
    have, total = count_cached(design, stage)
    cap_key = (
        "pilot_candidate_stage_max_spend_usd"
        if stage == "candidates"
        else "pilot_continuation_stage_max_spend_usd"
    )
    cap = float(max_spend if max_spend is not None else design[cap_key])
    print(f"stage       : {stage}")
    print(f"cached      : {have}/{total}")
    print(f"missing     : {total - have}")
    print(f"stage cap   : ${cap:.2f}")
    if stage == "continuations":
        try:
            validate_review(design)
            print("human gate  : PASS")
        except ValueError as exc:
            if execute and not dry_run:
                raise
            print(f"human gate  : BLOCKED — {exc}")
            print("API calls   : NONE")
            return
    if dry_run or not execute:
        print("API calls   : NONE")
        if not execute and not dry_run:
            print("To authorize this paid stage, Marie must rerun with --execute.")
        return
    spent = (
        asyncio.run(run_candidates(design, cap))
        if stage == "candidates"
        else asyncio.run(run_continuations(design, validate_review(design), cap))
    )
    print(f"stage cost  : ${spent:.3f}")
    if stage == "candidates":
        print(f"review next : {REVIEW_PATH.relative_to(ROOT)}")
        print("Do not run continuations until Marie has approved that file.")


if __name__ == "__main__":
    fire.Fire(main)
