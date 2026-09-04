"""Independent local recomputation of the CAUSE pilot headline metrics."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results.json"


def main():
    if not RESULTS.exists():
        raise SystemExit("results.json is absent; run analyze.py after paid judging")
    payload = json.loads(RESULTS.read_text())
    threshold = payload["threshold"]
    rows = [r for r in payload["rows"] if r["included"]]
    stored = {r["trace_id"]: r for r in payload["per_trace"]}
    failures = []
    for trace_id, expected in stored.items():
        medians = {}
        for variant in ("original", "surface", "low", "high"):
            values = [
                r["final_value"] for r in rows
                if r["trace_id"] == trace_id
                and r["family"] == "candidate"
                and r["variant"] == variant
            ]
            medians[variant] = statistics.median(values) if values else None
        effect = (
            (medians["high"] - medians["low"]) / threshold
            if medians["high"] is not None and medians["low"] is not None
            else None
        )
        surface = (
            abs(medians["surface"] - medians["original"]) / threshold
            if medians["surface"] is not None and medians["original"] is not None
            else None
        )
        for name, actual, target in (
            ("candidate_effect_rel", effect, expected["candidate_effect_rel"]),
            ("surface_effect_rel", surface, expected["surface_effect_rel"]),
        ):
            if actual is None or target is None:
                if actual != target:
                    failures.append(f"{trace_id}/{name}: {actual} != {target}")
            elif abs(actual - target) > 1e-12:
                failures.append(f"{trace_id}/{name}: {actual} != {target}")
        print(
            f"{trace_id}: high-low={effect if effect is not None else 'NA'}; "
            f"surface={surface if surface is not None else 'NA'}"
        )
    if failures:
        raise SystemExit("independent verification FAILED:\n  " + "\n  ".join(failures))
    print("independent verification: PASS")


if __name__ == "__main__":
    main()
