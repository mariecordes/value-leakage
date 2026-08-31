"""TARGET_VISIBILITY_DIR step B2 — sample the six prompt versions. Needs OPENROUTER_API_KEY.

Writes analysis/target_visibility/<model>/<version>.json in the same shape as the repo's condition
files (model, backend, provider, condition, threshold, prompt, rows[]), so the
existing judge and plot machinery can read them unchanged.

Two subjects, picked in DISCLOSURE_DIR:
  qwen3.5-122b-a10b   the admitter (82% of its favoured runs admit influence)
  glm-5p2             the denier   (40% deny)

qwen3.5's shipped runs were already sampled through OpenRouter on a pinned
provider, so its baseline/bet controls come free from runs/ and only the four
new versions are sampled. glm-5p2's shipped runs came from Fireworks, so its
controls are re-sampled here too — otherwise a difference between "bet" and
"hidden" could just be a difference between two serving stacks.

Resumable: a version whose file already exists is skipped, so an interrupted
run never pays twice.

  uv run python analysis/target_visibility/sample_conditions.py --dry_run
  uv run python analysis/target_visibility/sample_conditions.py --pilot 2
  uv run python analysis/target_visibility/sample_conditions.py --model glm-5p2
"""

import asyncio
import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "disclosure"))
import prompts as P  # noqa: E402
from common import RUNS  # noqa: E402  (analysis/disclosure/common.py)

from value_leakage.api.openrouter.chat_completions import (  # noqa: E402
    get_openrouter_client, process_batch)

TARGET_VISIBILITY_DIR = Path(__file__).resolve().parent
# Every rollout is written here the moment it comes back, one file per rollout,
# named (model, version, index). The per-version file below is assembled from
# these. So a crash — or a Ctrl-C — loses at most the handful still in flight:
# re-running pays only for the indices that are missing.
SAMPLE_CACHE = TARGET_VISIBILITY_DIR / "cache" / "samples"

# OpenRouter list prices, USD per million tokens (checked 2026-08-29).
PRICES = {
    "qwen/qwen3.5-122b-a10b": (0.29, 2.40),
    "z-ai/glm-5.2": (1.19, 3.74),
}

SUBJECTS = {
    "qwen3.5-122b-a10b": {
        "role": "admitter",
        "openrouter_id": "qwen/qwen3.5-122b-a10b",
        # The starter dataset pin. Same id AND same upstream => the 100 shipped runs
        # per control condition are a valid comparison, so we don't re-buy them.
        "provider": "deepinfra/fp4",
        "shipped": "qwen3.5-122b-a10b_20260815_030702",
        "sample_versions": P.NEW_VERSIONS,
        "max_tokens": 64000,
        "reasoning_effort": "high",
        # 30 is enough here: the 100 free starter dataset control runs already tighten the
        # bet side, so the bet-vs-hidden comparison lands at +/-0.289.
        "n": 30,
    },
    "glm-5p2": {
        "role": "denier",
        "openrouter_id": "z-ai/glm-5.2",
        "provider": None,
        "shipped": "glm-5p2_20260815_030703",
        # Shipped runs are Fireworks; re-sample the controls here so every
        # version in the comparison comes off the same endpoint.
        "sample_versions": P.NEW_VERSIONS + P.CONTROL_VERSIONS,
        "max_tokens": 64000,
        "reasoning_effort": "high",
        # 50, not 30: every version here is bought fresh, so this side has no
        # free head start. 50 brings it to +/-0.277 — matched to the other
        # subject's precision, which is what needs to be equal, not the counts.
        "n": 50,
    },
}


def shipped_threshold(model: str) -> int:
    path = RUNS / SUBJECTS[model]["shipped"] / "threshold.json"
    return int(json.loads(path.read_text())["threshold"])


def cost_per_rollout(model: str) -> float:
    """Measured from the shipped runs of this same model, not guessed.

    qwen3.5's shipped rows carry OpenRouter's own per-call cost. glm's came
    from Fireworks, so its token counts are priced at OpenRouter's rates.
    """
    import numpy as np
    run_dir = RUNS / SUBJECTS[model]["shipped"]
    costs, comp, prompt_tok = [], [], []
    for cond in ("baseline", "above_good", "below_good"):
        path = run_dir / f"{cond}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["rows"]:
            u = row.get("usage") or {}
            if u.get("cost") is not None:
                costs.append(float(u["cost"]))
            if u.get("completion_tokens"):
                comp.append(u["completion_tokens"])
                prompt_tok.append(u.get("prompt_tokens", 180))
    if costs:
        return float(np.mean(costs))
    p_in, p_out = PRICES[SUBJECTS[model]["openrouter_id"]]
    return float(np.mean(prompt_tok)) / 1e6 * p_in + float(np.mean(comp)) / 1e6 * p_out


def out_path(model: str, version: str) -> Path:
    return TARGET_VISIBILITY_DIR / model / f"{version}.json"


def plan(models: list[str], n: int | None, pilot: int) -> list[dict]:
    jobs = []
    for model in models:
        cfg = SUBJECTS[model]
        thr = shipped_threshold(model)
        per = cost_per_rollout(model)
        versions = cfg["sample_versions"]
        if pilot:
            versions = ("hidden_above", "instructed_above")
        for v in versions:
            path = out_path(model, version=v)
            count = pilot or n or cfg["n"]
            jobs.append({
                "model": model, "version": v, "threshold": thr,
                "count": count, "path": path,
                "done": path.exists() and not pilot,
                "cost": per * count, "per_rollout": per,
                "prompt": P.build(v, thr),
            })
    return jobs


def rollout_cache_path(model: str, version: str, i: int) -> Path:
    return SAMPLE_CACHE / f"{model}__{version}__{i:04d}.json"


def cached_rollouts(model: str, version: str, count: int) -> dict[int, dict]:
    """Successful rollouts already on disk for this version, by index."""
    out = {}
    for i in range(count):
        path = rollout_cache_path(model, version, i)
        if path.exists():
            row = json.loads(path.read_text())
            if not row.get("error"):
                out[i] = row
    return out


async def sample_one(job, cfg, max_concurrent):
    """Fill in whichever rollout indices are missing, caching each as it lands."""
    model, version, count = job["model"], job["version"], job["count"]
    SAMPLE_CACHE.mkdir(parents=True, exist_ok=True)
    have = cached_rollouts(model, version, count)
    missing = [i for i in range(count) if i not in have]
    if have:
        print(f"    {len(have)}/{count} already cached — requesting {len(missing)}")

    body = {"usage": {"include": True}}
    if cfg["reasoning_effort"]:
        body["reasoning"] = {"effort": cfg["reasoning_effort"]}
    if cfg["provider"]:
        body["provider"] = {"order": [cfg["provider"]], "allow_fallbacks": False}

    spent = 0.0
    # One wave at a time so cache files land every ~30s rather than only at the
    # very end — that is what makes a mid-version interruption cheap.
    for start in range(0, len(missing), max_concurrent):
        wave = missing[start:start + max_concurrent]
        responses = await process_batch(
            client=get_openrouter_client(),
            model=cfg["openrouter_id"],
            messages_list=[[{"role": "user", "content": job["prompt"]}]] * len(wave),
            max_tokens=cfg["max_tokens"],
            max_concurrent=max_concurrent,
            extra_body=body,
            return_exceptions=True,
        )
        for i, r in zip(wave, responses):
            if isinstance(r, Exception):
                # Errors are cached too, so the console can show them, but
                # cached_rollouts() ignores them and a re-run retries.
                rollout_cache_path(model, version, i).write_text(json.dumps(
                    {"i": i, "error": f"{type(r).__name__}: {r}"}, indent=2))
                continue
            msg = r.choices[0].message
            usage = r.usage.model_dump() if r.usage else None
            if usage and usage.get("cost") is not None:
                spent += float(usage["cost"])
            row = {
                "i": i,
                "reasoning": getattr(msg, "reasoning_content", None)
                             or getattr(msg, "reasoning", None) or "",
                "content": msg.content or "",
                "finish_reason": r.choices[0].finish_reason,
                "usage": usage,
            }
            rollout_cache_path(model, version, i).write_text(
                json.dumps(row, indent=2, ensure_ascii=False))
            have[i] = row

    rows = []
    for i in range(count):
        if i in have:
            rows.append(have[i])
            continue
        path = rollout_cache_path(model, version, i)
        if path.exists():                      # cached error row
            rows.append(json.loads(path.read_text()))
        else:
            rows.append({"i": i, "error": "no response"})
    return rows, spent, len(have)


async def run(jobs, max_concurrent, max_spend, pilot):
    spent = 0.0
    for job in jobs:
        if job["done"]:
            print(f"  {job['model']}/{job['version']}: already sampled — skipped")
            continue
        cfg = SUBJECTS[job["model"]]
        print(f"  {job['model']}/{job['version']}: {job['count']} rollouts via "
              f"{cfg['openrouter_id']}"
              + (f" (provider pinned {cfg['provider']})" if cfg["provider"] else ""))
        rows, cost, n_ok = await sample_one(job, cfg, max_concurrent)
        spent += cost
        print(f"    {n_ok}/{job['count']} succeeded — cost ${cost:.3f}, "
              f"running total ${spent:.3f}")
        if n_ok == 0:
            print("    STOPPING: every call failed. First error:")
            print(f"    {rows[0].get('error')}")
            break
        if pilot:
            job["_rows"] = rows
            continue
        job["path"].parent.mkdir(parents=True, exist_ok=True)
        job["path"].write_text(json.dumps({
            "model": cfg["openrouter_id"], "backend": "openrouter",
            "provider": cfg["provider"], "condition": job["version"],
            "threshold": job["threshold"], "prompt": job["prompt"],
            "max_tokens": cfg["max_tokens"],
            "reasoning_effort": cfg["reasoning_effort"],
            "measured_cost_usd": cost, "rows": rows,
        }, indent=2, ensure_ascii=False))
        print(f"    saved {job['path'].relative_to(TARGET_VISIBILITY_DIR.parent)}")
        if spent > max_spend:
            print(f"STOPPING: ${spent:.2f} passed --max_spend ${max_spend:.2f}. "
                  f"Finished versions are saved; re-run to continue.")
            break
    return spent


def main(model: str | None = None, n: int | None = None, max_concurrent: int = 10,
         max_spend: float = 6.00, pilot: int = 0, dry_run: bool = False):
    """n overrides the per-subject default (30 / 50, set for matched precision).

    pilot: sample this many rollouts of two versions and print one trace,
    without saving — proves the model id, the provider pin and the prompts work
    before the full batch."""
    models = [model] if model else list(SUBJECTS)
    for m in models:
        if m not in SUBJECTS:
            raise ValueError(f"unknown model {m!r}; known: {list(SUBJECTS)}")
    jobs = plan(models, n, pilot)
    todo = [j for j in jobs if not j["done"]]
    projected = sum(j["cost"] for j in todo)

    print(f"{'model':22s} {'version':18s} {'n':>4s} {'$/rollout':>10s} "
          f"{'$ version':>10s}  status")
    for j in jobs:
        print(f"{j['model']:22s} {j['version']:18s} {j['count']:4d} "
              f"{j['per_rollout']:10.4f} {j['cost']:10.2f}  "
              f"{'cached' if j['done'] else 'to sample'}")
    print(f"\nthreshold per model: "
          + ", ".join(f"{m} {shipped_threshold(m):,}" for m in models))
    print(f"projected: ${projected:.2f}   (--max_spend ${max_spend:.2f})")
    if dry_run:
        print("dry run — no API calls made")
        return
    if projected > max_spend:
        print(f"REFUSING TO RUN: ${projected:.2f} > --max_spend ${max_spend:.2f}")
        return

    spent = asyncio.run(run(jobs, max_concurrent, max_spend, pilot))

    if pilot:
        print("\n--- pilot samples (nothing saved) ---")
        for j in jobs:
            rows = [r for r in j.get("_rows", []) if not r.get("error")]
            if not rows:
                continue
            r = rows[0]
            print("=" * 78)
            print(f"{j['model']} / {j['version']}   threshold {j['threshold']:,}")
            print(f"  reasoning {len(r['reasoning']):,} chars, "
                  f"answer {len(r['content']):,} chars, "
                  f"finish={r['finish_reason']}")
            print(f"  thinking starts: {r['reasoning'][:300]!r}")
            print(f"  answer ends    : {r['content'][-300:]!r}")
        print(f"\npilot cost ${spent:.3f}. Re-run without --pilot for the batch.")
        return
    print(f"\nTARGET_VISIBILITY_DIR sampling cost ${spent:.3f}")


if __name__ == "__main__":
    fire.Fire(main)
