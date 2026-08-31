"""TARGET_VISIBILITY_DIR step B3 — extract numbers from the new runs. Needs OPENAI_API_KEY.

Two extractions, both using the paper's VERBATIM prompts, imported from
src/value_leakage/judge.py and never edited:

  estimates     NUMBER_JUDGE on each run's short visible answer (the `content`
                field) -> one final number per run. This is what the landing
                measurement rests on, so it runs on every row. Cheap: the
                answers are a few hundred characters.

  trajectories  TRAJECTORY_JUDGE on the full reasoning trace -> the ordered
                candidate estimates. Expensive (whole traces), so it runs
                on only --traj_n rows per version, and only for the lineup
                figure.

Judge consistency rule: every cell in a figure must come from ONE
judge. qwen3.5's control conditions live in the starter dataset shipped runs and were judged
by claude-opus-5, so they are re-extracted here with the same gpt-5-mini used
for the new versions. The starter dataset numbers stay untouched in runs/ and may still be
quoted in the text as the original replication.

A DISCLOSURE_DIR caveat carries over: gpt-5-mini cannot be pinned to temperature 0, and on
the four-way disclosure rubric it only reproduced 62% of its own labels.
Pulling one number out of a short answer is a far more mechanical task, but do
not assume — `--recheck N` re-extracts N rows through a separate cache and
reports how often the two passes agree.

  uv run python analysis/target_visibility/judge_conditions.py --dry_run
  uv run python analysis/target_visibility/judge_conditions.py --kind estimates
  uv run python analysis/target_visibility/judge_conditions.py --kind trajectories --traj_n 20
  uv run python analysis/target_visibility/judge_conditions.py --kind estimates --recheck 30
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "disclosure"))
import prompts as P  # noqa: E402
from common import RUNS  # noqa: E402
import llm  # noqa: E402
from sample_conditions import SUBJECTS, TARGET_VISIBILITY_DIR, out_path  # noqa: E402

from value_leakage.judge import (  # noqa: E402  VERBATIM — never edited
    NUMBER_JUDGE_PROMPT, TRAJECTORY_JUDGE_PROMPT,
    parse_tagged_estimate, parse_trajectory)

CACHE = TARGET_VISIBILITY_DIR / "cache"

# qwen3.5's controls come from the shipped run; these are the file names there.
SHIPPED_CONDITION = {"baseline": "baseline", "bet_above": "above_good",
                     "bet_below": "below_good"}


def source_rows(model: str, version: str) -> tuple[list[dict], str]:
    """(rows, where) for one model+version — our sample, or the shipped run."""
    path = out_path(model, version)
    if path.exists():
        return json.loads(path.read_text())["rows"], "target_visibility"
    cond = SHIPPED_CONDITION.get(version)
    if cond:
        shipped = RUNS / SUBJECTS[model]["shipped"] / f"{cond}.json"
        if shipped.exists():
            return json.loads(shipped.read_text())["rows"], "runs"
    return [], "missing"


def cache_path(kind: str, judge_model: str, text: str, ns: str = "") -> Path:
    key = hashlib.sha256(f"{kind}\n{judge_model}\n{ns}\n{text}".encode()).hexdigest()[:32]
    return CACHE / kind / f"{key}.json"


def build_jobs(kind: str, judge_model: str, models: list[str], traj_n: int,
               ns: str = "") -> list[dict]:
    template = NUMBER_JUDGE_PROMPT if kind == "estimates" else TRAJECTORY_JUDGE_PROMPT
    field = "content" if kind == "estimates" else "reasoning"
    jobs = []
    for model in models:
        for version in P.VERSIONS:
            rows, where = source_rows(model, version)
            if where == "missing":
                continue
            if kind == "trajectories":
                rows = rows[:traj_n]
            for row in rows:
                text = (row.get(field) or "").strip()
                if not text:
                    continue
                prompt = template.format(llm_text=text)
                path = cache_path(kind, judge_model, prompt, ns)
                jobs.append({
                    "model": model, "version": version, "source": where,
                    "i": row["i"], "prompt": prompt, "path": path,
                    "cached": json.loads(path.read_text()) if path.exists() else None,
                })
    return jobs


async def run_jobs(jobs, kind, backend, judge_model, max_concurrent, max_out,
                   max_spend, reasoning_effort):
    todo = [j for j in jobs if j["cached"] is None]
    if not todo:
        print("everything already cached — no API calls made")
        return 0.0
    parse = parse_tagged_estimate if kind == "estimates" else parse_trajectory
    client = llm.get_client(backend)
    spent, n_bad, CHUNK = 0.0, 0, 40
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        responses = await llm.batch(client, judge_model,
                                    [j["prompt"] for j in chunk],
                                    max_concurrent=max_concurrent,
                                    max_tokens=max_out,
                                    reasoning_effort=reasoning_effort)
        for j, r in zip(chunk, responses):
            if isinstance(r, Exception):
                n_bad += 1
                print(f"  {j['model']}/{j['version']}[{j['i']}]: "
                      f"{type(r).__name__}: {r}")
                continue
            text = r.choices[0].message.content or ""
            usage = r.usage.model_dump() if r.usage else {}
            cost = llm.cost_of(judge_model, usage.get("prompt_tokens", 0),
                               usage.get("completion_tokens", 0))
            spent += cost
            value = parse(text)
            if value is None:
                # Genuinely unparseable answers exist (the paper's judge returns
                # UNKNOWN for ranges), so record them — but a truncated reply is
                # our bug, not the data's. Flag that case loudly and don't cache.
                if r.choices[0].finish_reason == "length":
                    n_bad += 1
                    print(f"  TRUNCATED: {j['model']}/{j['version']}[{j['i']}] "
                          f"— raise --max_out")
                    continue
            rec = {"judge_model": judge_model, "reply": text, "value": value,
                   "finish_reason": r.choices[0].finish_reason, "cost": cost}
            j["path"].parent.mkdir(parents=True, exist_ok=True)
            j["path"].write_text(json.dumps(rec, indent=2))
            j["cached"] = rec
        done = min(start + CHUNK, len(todo))
        print(f"  judged {done}/{len(todo)} — spent ${spent:.3f}"
              + (f" — {n_bad} failed" if n_bad else ""))
        if spent > max_spend:
            print(f"STOPPING: ${spent:.3f} passed --max_spend ${max_spend:.2f}")
            break
    return spent


def consolidate(jobs, kind: str, models: list[str]) -> None:
    for model in models:
        out, meta = {}, {}
        for version in P.VERSIONS:
            rows = [j for j in jobs if j["model"] == model
                    and j["version"] == version]
            if not rows:
                continue
            vals = [(j["cached"] or {}).get("value") for j in rows]
            out[version] = {"i": [j["i"] for j in rows], "values": vals}
            ok = sum(1 for v in vals if v is not None)
            meta[version] = {"n": len(vals), "parsed": ok,
                             "source": rows[0]["source"]}
            print(f"  {model:22s} {version:18s} {ok:3d}/{len(vals):3d} parsed "
                  f"(from {rows[0]['source']})")
        if not out:
            continue
        path = TARGET_VISIBILITY_DIR / model / f"{kind}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": kind, "meta": meta, "by_version": out},
                                   indent=2))
        print(f"  saved {path.relative_to(TARGET_VISIBILITY_DIR.parent)}")


def main(kind: str = "estimates", model: str | None = None,
         backend: str = "openai", judge_model: str = "gpt-5-mini",
         traj_n: int = 20, max_concurrent: int = 6,
         max_out: int | None = None, reasoning_effort: str = "low",
         max_spend: float = 1.00, recheck: int = 0, dry_run: bool = False):
    if kind not in ("estimates", "trajectories"):
        raise ValueError("kind must be 'estimates' or 'trajectories'")
    models = [model] if model else list(SUBJECTS)
    # A final-number reply is ~20 tokens; a trajectory list can be long. The
    # cap includes the judge's hidden reasoning, so it cannot be tiny (DISCLOSURE_DIR's
    # first pilot lost every token to thinking at a 24-token cap).
    if max_out is None:
        max_out = 256 if kind == "estimates" else 2048
    ns = "recheck" if recheck else ""
    jobs = build_jobs(kind, judge_model, models, traj_n, ns)
    if recheck:
        jobs = jobs[::max(1, len(jobs) // recheck)][:recheck]

    todo = [j for j in jobs if j["cached"] is None]
    in_tok = int(sum(len(j["prompt"]) for j in todo) / 3.7)
    worst = llm.cost_of(judge_model, in_tok, max_out * len(todo))
    likely = llm.cost_of(judge_model, in_tok,
                         (40 if kind == "estimates" else 300) * len(todo))
    print(f"kind        : {kind}" + ("  (RECHECK pass)" if recheck else ""))
    print(f"judge model : {judge_model} (verbatim paper prompt)")
    print(f"jobs        : {len(jobs)} rows, {len(todo)} to call")
    print(f"input size  : ~{in_tok:,} tokens")
    print(f"projected   : ${likely:.3f} likely, ${worst:.3f} worst case   "
          f"(--max_spend ${max_spend:.2f})")
    if dry_run:
        print("dry run — no API calls made")
        return
    if not jobs:
        print("nothing to judge — sample with analysis/target_visibility/sample_conditions.py first")
        return
    if worst > max_spend:
        print(f"REFUSING TO RUN: ${worst:.3f} > --max_spend ${max_spend:.2f}")
        return

    spent = asyncio.run(run_jobs(jobs, kind, backend, judge_model,
                                 max_concurrent, max_out, max_spend,
                                 reasoning_effort))

    if recheck:
        first = build_jobs(kind, judge_model, models, traj_n, "")
        by_key = {(j["model"], j["version"], j["i"]): j for j in first}
        agree = n = 0
        for j in jobs:
            other = by_key.get((j["model"], j["version"], j["i"]))
            if not (other and other["cached"] and j["cached"]):
                continue
            n += 1
            agree += other["cached"]["value"] == j["cached"]["value"]
        print(f"\nrecheck: {agree}/{n} rows identical across two passes "
              + (f"({agree/n:.0%})" if n else ""))
        print(f"cost ${spent:.3f}. Nothing consolidated (recheck cache is separate).")
        return

    print()
    consolidate(jobs, kind, models)
    print(f"\nTARGET_VISIBILITY_DIR {kind} judging cost ${spent:.3f}")


if __name__ == "__main__":
    fire.Fire(main)
