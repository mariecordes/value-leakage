"""Summarize the hand audit of finished continuations. Local only, no API calls.

Reads finish_audit.json once YOUR_VERDICT and FAILED_CHECKS have been filled in,
and prints the counts the report needs: how many were checked, how they were
selected, what disagreed, and which checks failed.

Run:
    uv run python analysis/cause/summarize_audit.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

CAUSE_DIR = Path(__file__).resolve().parent
AUDIT_PATH = CAUSE_DIR / "finish_audit.json"
VALID = {"ok", "fail", "unsure"}


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text())
    rows = audit["rows"]
    done = [r for r in rows if r["YOUR_VERDICT"] is not None]

    problems = []
    for row in done:
        label = f"{row['trace_id']}/{row['variant']} #{row['sample_index']}"
        if row["YOUR_VERDICT"] not in VALID:
            problems.append(f"{label}: verdict must be one of {sorted(VALID)}")
        if row["YOUR_VERDICT"] == "fail" and not row["FAILED_CHECKS"]:
            problems.append(f"{label}: marked fail but no check listed")
        if row["YOUR_VERDICT"] == "ok" and row["FAILED_CHECKS"]:
            problems.append(f"{label}: marked ok but lists failed checks")
        for check in row["FAILED_CHECKS"]:
            if str(check) not in {k[0] for k in audit["checks"]}:
                problems.append(f"{label}: unknown check {check!r}")

    print(f"sample        : {len(rows)} continuations, seed {audit['seed']}")
    print(f"               drawn across all traces and variants before any were read")
    print(f"reviewed      : {len(done)}/{len(rows)}")
    if not done:
        print("\nNothing recorded yet. Fill in YOUR_VERDICT for each row.")
        return

    counts = Counter(r["YOUR_VERDICT"] for r in done)
    for verdict in ("ok", "fail", "unsure"):
        if counts.get(verdict):
            print(f"  {verdict:8s}: {counts[verdict]}")

    failed = Counter(str(c) for r in done for c in r["FAILED_CHECKS"])
    if failed:
        print("\nfailed checks:")
        for key, description in audit["checks"].items():
            n = failed.get(key[0], 0)
            if n:
                print(f"  {n} x check {key[0]} — {description.split('. Severity')[0]}")
    else:
        print("\nfailed checks: none")

    notes = [(r["trace_id"], r["variant"], r["sample_index"], r["YOUR_NOTES"])
             for r in done if r["YOUR_NOTES"]]
    if notes:
        print("\nnotes:")
        for trace, variant, k, note in notes:
            print(f"  {trace}/{variant} #{k}: {note}")

    if problems:
        raise SystemExit("audit file invalid:\n  " + "\n  ".join(problems))
    print("\nvalidation    : PASS")


if __name__ == "__main__":
    main()
