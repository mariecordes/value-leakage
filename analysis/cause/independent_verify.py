#!/usr/bin/env python3
"""Independent check of the inserted-value/answer association.

This intentionally does not import or inspect any cause-analysis implementation.
It reconstructs judge-cache keys from raw continuation content, applies the stated
range filter, calculates variant medians, and computes Spearman correlations.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN_PATH = HERE / "inserts.json"
FINISH_DIR = HERE / "raw" / "finishes"
CACHE_DIR = HERE / "cache" / "cause_finals"
JUDGE_PATH = ROOT / "src" / "value_leakage" / "judge.py"
OUTPUT_PATH = HERE / "independent_verification_cache_normalized.json"
JUDGE_MODEL = "gpt-5.6-luna"


def load_number_prompt() -> str:
    """Read the literal constant through Python's AST without importing judge.py."""
    module = ast.parse(JUDGE_PATH.read_text())
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "NUMBER_JUDGE_PROMPT"
                   for t in statement.targets):
            continue
        value = ast.literal_eval(statement.value)
        if not isinstance(value, str):
            raise TypeError("NUMBER_JUDGE_PROMPT is not a string")
        return value
    raise RuntimeError("NUMBER_JUDGE_PROMPT was not found")


def average_ranks(values: list[float]) -> list[float]:
    """Return 1-based ranks, assigning tied values their average rank."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        average = ((start + 1) + stop) / 2
        for index in ordered[start:stop]:
            ranks[index] = average
        start = stop
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [v - mean_x for v in x]
    centered_y = [v - mean_y for v in y]
    denominator = math.sqrt(
        sum(v * v for v in centered_x) * sum(v * v for v in centered_y)
    )
    if denominator == 0:
        raise ValueError("correlation is undefined for a constant input")
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def exact_two_sided_sign_test(positive: int, negative: int) -> float:
    """Two-sided exact binomial test under P(positive) = P(negative) = 1/2."""
    n = positive + negative
    smaller_tail = min(positive, negative)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(smaller_tail + 1)) / 2**n)


def main() -> None:
    design = json.loads(DESIGN_PATH.read_text())
    prompt_template = load_number_prompt()
    lower = design["threshold"] / 10
    upper = design["threshold"] * 10
    repetitions = design["continuations_per_variant"]

    traces_out = []
    exclusions = []
    expected_count = 0
    included_count = 0
    missing_finish_count = 0
    missing_cache_count = 0
    null_judgment_count = 0
    exact_cache_key_count = 0
    stripped_content_cache_key_count = 0

    for trace in design["traces"]:
        variant_rows = []
        for variant in trace["variants"]:
            if variant["variant"] == "original":
                continue
            expected_count += repetitions
            kept_values = []
            for sample_index in range(repetitions):
                filename = (
                    f'{trace["trace_id"]}__{variant["variant"]}__{sample_index:02d}.json'
                )
                finish_path = FINISH_DIR / filename
                if not finish_path.exists():
                    missing_finish_count += 1
                    exclusions.append({"continuation": filename, "reason": "missing finish"})
                    continue
                finish = json.loads(finish_path.read_text())
                expected_metadata = {
                    "trace_id": trace["trace_id"],
                    "variant": variant["variant"],
                    "inserted_value": variant["inserted_value"],
                    "sample_index": sample_index,
                }
                actual_metadata = {key: finish.get(key) for key in expected_metadata}
                if actual_metadata != expected_metadata:
                    raise AssertionError(
                        f"metadata mismatch in {filename}: {actual_metadata} != {expected_metadata}"
                    )
                content = finish.get("content")
                if not isinstance(content, str):
                    raise TypeError(f"content is not text in {filename}")
                prompt = prompt_template.format(llm_text=content)
                stem = hashlib.sha256(f"{JUDGE_MODEL}\n{prompt}".encode()).hexdigest()[:32]
                cache_path = CACHE_DIR / f"{stem}.json"
                if not cache_path.exists():
                    stripped_prompt = prompt_template.format(llm_text=content.strip())
                    stripped_stem = hashlib.sha256(
                        f"{JUDGE_MODEL}\n{stripped_prompt}".encode()
                    ).hexdigest()[:32]
                    stripped_cache_path = CACHE_DIR / f"{stripped_stem}.json"
                    if not stripped_cache_path.exists():
                        missing_cache_count += 1
                        exclusions.append({
                            "continuation": filename,
                            "reason": "missing cached judgment",
                            "expected_cache_stem": stem,
                            "stripped_content_cache_stem": stripped_stem,
                        })
                        continue
                    stem = stripped_stem
                    cache_path = stripped_cache_path
                    stripped_content_cache_key_count += 1
                else:
                    exact_cache_key_count += 1
                judged = json.loads(cache_path.read_text())
                value = judged.get("value")
                if value is None:
                    null_judgment_count += 1
                    exclusions.append({
                        "continuation": filename,
                        "reason": "judge returned no final number",
                        "cache_stem": stem,
                    })
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"non-numeric judgment in {cache_path.name}: {value!r}")
                if not lower <= value <= upper:
                    exclusions.append({
                        "continuation": filename,
                        "reason": "outside allowed range",
                        "value": value,
                        "lower": lower,
                        "upper": upper,
                        "cache_stem": stem,
                    })
                    continue
                kept_values.append(float(value))
                included_count += 1

            if not kept_values:
                raise RuntimeError(
                    f'no usable continuations for {trace["trace_id"]}/{variant["variant"]}'
                )
            variant_rows.append({
                "variant": variant["variant"],
                "inserted_value": variant["inserted_value"],
                "included_continuations": len(kept_values),
                "median_final_answer": statistics.median(kept_values),
            })

        rho = spearman(
            [row["inserted_value"] for row in variant_rows],
            [row["median_final_answer"] for row in variant_rows],
        )
        traces_out.append({
            "trace_id": trace["trace_id"],
            "direction": trace["direction"],
            "variants": variant_rows,
            "spearman_rho": rho,
        })

    correlations = [row["spearman_rho"] for row in traces_out]
    tolerance = 1e-15
    positive = sum(rho > tolerance for rho in correlations)
    negative = sum(rho < -tolerance for rho in correlations)
    zero = len(correlations) - positive - negative
    outside_range_count = sum(e["reason"] == "outside allowed range" for e in exclusions)

    result = {
        "method": {
            "variant_filter": "variant != 'original'",
            "variant_summary": "median of retained continuation judgments",
            "trace_statistic": "Spearman rank correlation with average ranks for ties",
            "range_filter_inclusive": [lower, upper],
            "cache_key": (
                f"sha256('{JUDGE_MODEL}\\n' + formatted NUMBER_JUDGE_PROMPT)[:32]; "
                "if absent, retry after stripping leading/trailing whitespace from content"
            ),
            "sign_test": "two-sided exact Binomial(n_nonzero, 0.5); zero correlations omitted",
        },
        "audit": {
            "expected_inserted_alternative_continuations": expected_count,
            "included_continuations": included_count,
            "excluded_continuations": len(exclusions),
            "outside_range": outside_range_count,
            "missing_finish": missing_finish_count,
            "missing_cache": missing_cache_count,
            "null_judgment": null_judgment_count,
            "exact_cache_key_matches": exact_cache_key_count,
            "stripped_content_cache_key_matches": stripped_content_cache_key_count,
            "exclusions": exclusions,
        },
        "traces": traces_out,
        "across_traces": {
            "positive_correlations": positive,
            "negative_correlations": negative,
            "zero_correlations": zero,
            "traces_total": len(correlations),
            "sign_test_n_nonzero": positive + negative,
            "two_sided_exact_sign_test_p": exact_two_sided_sign_test(positive, negative),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "audit": result["audit"] | {"exclusions": f"{len(exclusions)} records in output"},
        "correlations": [
            {"trace_id": row["trace_id"], "spearman_rho": row["spearman_rho"]}
            for row in traces_out
        ],
        "across_traces": result["across_traces"],
    }, indent=2))


if __name__ == "__main__":
    main()
