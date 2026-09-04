"""Validate the frozen CAUSE pilot against the unchanged source traces.

This file performs local checks only. It never calls an API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fire

CAUSE_DIR = Path(__file__).resolve().parent
ROOT = CAUSE_DIR.parents[1]
DESIGN_PATH = CAUSE_DIR / "frozen_design.json"
REVIEW_PATH = CAUSE_DIR / "candidate_review.json"
RAW_DIR = CAUSE_DIR / "raw"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_design() -> dict:
    return json.loads(DESIGN_PATH.read_text())


def load_trace(trace: dict) -> tuple[dict, dict]:
    source = ROOT / trace["source"]
    payload = json.loads(source.read_text())
    row = next((r for r in payload["rows"] if r.get("i") == trace["i"]), None)
    if row is None:
        raise ValueError(f"missing row {trace['trace_id']} in {source}")
    return payload, row


def trace_material(trace: dict) -> dict:
    payload, row = load_trace(trace)
    reasoning = row.get("reasoning") or ""
    candidate = trace["candidate"]
    control = trace["nonnumeric_control"]
    return {
        "prompt": payload["prompt"],
        "reasoning": reasoning,
        "candidate_prefix": reasoning[: candidate["start"]],
        "candidate_step": reasoning[candidate["start"] : candidate["end"]],
        "control_prefix": reasoning[: control["start"]],
        "control_step": reasoning[control["start"] : control["end"]],
    }


def validate_trace(trace: dict, threshold: float) -> list[str]:
    material = trace_material(trace)
    candidate = trace["candidate"]
    control = trace["nonnumeric_control"]
    checks = {
        "prompt hash": sha256(material["prompt"]) == trace["prompt_sha256"],
        "reasoning hash": sha256(material["reasoning"]) == trace["reasoning_sha256"],
        "candidate prefix hash": sha256(material["candidate_prefix"])
        == candidate["prefix_sha256"],
        "candidate step hash": sha256(material["candidate_step"])
        == candidate["step_sha256"],
        "control prefix hash": sha256(material["control_prefix"])
        == control["prefix_sha256"],
        "control step hash": sha256(material["control_step"])
        == control["step_sha256"],
        "candidate before 40%": candidate["start"] / len(material["reasoning"]) <= 0.40,
        "candidate near threshold": 0.75
        <= candidate["value"] / threshold
        <= 1.25,
        "candidate human checked": candidate.get("human_checked") is True,
        "control human checked": control.get("human_checked") is True,
    }
    return [name for name, passed in checks.items() if not passed]


def main(dry_run: bool = True):
    """Print and validate the frozen pilot. There is intentionally no write mode."""
    design = load_design()
    failures = []
    print(f"experiment : {design['experiment']}")
    print(f"frozen     : {design['frozen_at']}")
    print(f"model      : {design['model']} via {design['provider']}")
    print(f"threshold  : {design['threshold']:,}")
    print(f"traces     : {len(design['traces'])}")
    for trace in design["traces"]:
        material = trace_material(trace)
        bad = validate_trace(trace, design["threshold"])
        failures.extend(f"{trace['trace_id']}: {x}" for x in bad)
        frac = trace["candidate"]["start"] / len(material["reasoning"])
        print(
            f"  {trace['trace_id']}: candidate {trace['candidate']['value']:,.0f} "
            f"at {frac:.1%}; step chars "
            f"[{trace['candidate']['start']}, {trace['candidate']['end']})"
        )
    natural = len(design["traces"]) * design["candidate_attempts"]
    surface = len(design["traces"]) * design["surface_attempts"]
    controls = len(design["traces"]) * design["control_attempts"]
    primary = (
        len(design["traces"])
        * 4
        * design["continuations_per_variant"]
    )
    diagnostic = (
        len(design["traces"])
        * (1 + design["control_alternatives"])
        * design["control_continuations_per_variant"]
    )
    print(f"candidate calls: {natural} natural + {surface} surface + {controls} control")
    print(f"continuations : {primary} primary + {diagnostic} diagnostic")
    subject_calls = natural + surface + controls + primary + diagnostic
    print(f"Qwen calls    : {subject_calls}")
    print(f"judge calls   : up to {primary + diagnostic} after continuations")
    print(f"paid requests : up to {subject_calls + primary + diagnostic}")
    print("dry run only  : yes" if dry_run else "local validation only: yes")
    if failures:
        raise SystemExit("frozen-design validation failed:\n  " + "\n  ".join(failures))
    print("validation  : PASS")


if __name__ == "__main__":
    fire.Fire(main)
