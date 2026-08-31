"""DISCLOSURE_DIR follow-up — how repeatable is the disclosure judge?

The NO_MENTION audit changed 78% of the suspect labels but ALSO 50% of a control
group whose label definitions were byte-identical between the two prompts. That
rules the audit uninterpretable and raises a sharper question: how much of that
churn is just the judge answering differently the second time?

gpt-5-mini does not accept temperature=0, so every call carries sampling noise.
This script measures it: the SAME 36 traces, the SAME original prompt, one more
time. Any disagreement with the first pass is pure judge instability, since
nothing else changed.

Reported as a plain agreement rate overall and per label — a number for the
write-up's limitations section instead of a hand-wave.

Nothing is overwritten; results go to analysis/disclosure/judge_stability.json.

  uv run python analysis/disclosure/judge_stability.py --dry_run
  uv run python analysis/disclosure/judge_stability.py
"""

import asyncio
import json
import random
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).parent))
from common import RUNS, DISCLOSURE_DIR, reasoning_texts  # noqa: E402
from judge_disclosure import DISCLOSURE_PROMPT, LABELS, parse_label  # noqa: E402
from rejudge_nomention import build_jobs as audit_jobs  # noqa: E402
import llm  # noqa: E402

CACHE = DISCLOSURE_DIR / "cache" / "stability"


def random_rows(per_model: int, seed: int) -> list[dict]:
    """A plain random sample of judged runs, per_model from each of the 10.

    The audit set is deliberately enriched for rare and borderline labels, so
    its agreement rate says nothing about the map. This one is representative:
    if 60% of real runs are ADMITS, ~60% of this sample is too.
    """
    rng = random.Random(seed)
    rows = []
    for path in sorted(DISCLOSURE_DIR.glob("disclosure_*.json")):
        d = json.loads(path.read_text())
        pool = [r for r in d["rows"] if r["label"]]
        rng.shuffle(pool)
        for r in pool[:per_model]:
            rows.append({"group": "random", "model": d["model"],
                         "condition": r["condition"], "idx": r["idx"],
                         "label_first_pass": r["label"]})
    return rows


def build_jobs(judge_model: str, seed: int, mode: str = "audit",
               per_model: int = 4) -> list[dict]:
    """Re-ask the ORIGINAL prompt on either the audit set or a random sample."""
    bias = json.loads((DISCLOSURE_DIR / "bias.json").read_text())
    source = (audit_jobs(judge_model, seed) if mode == "audit"
              else random_rows(per_model, seed))
    jobs = []
    for j in source:
        run_dir = RUNS / bias[j["model"]]["run_dir"]
        trace = reasoning_texts(run_dir, j["condition"])[j["idx"]]
        path = CACHE / f"{mode}_{judge_model.replace('/', '_')}_{j['model']}_" \
                       f"{j['condition']}_{j['idx']}.json"
        jobs.append({
            "group": j["group"], "model": j["model"], "condition": j["condition"],
            "idx": j["idx"], "label_first_pass": j["label_first_pass"],
            "prompt": DISCLOSURE_PROMPT.format(trace=trace), "path": path,
            "cached": json.loads(path.read_text()) if path.exists() else None,
        })
    return jobs


async def run_jobs(jobs, backend, model, max_concurrent, max_out):
    todo = [j for j in jobs if j["cached"] is None]
    if not todo:
        print("everything already cached — no API calls made")
        return 0.0
    CACHE.mkdir(parents=True, exist_ok=True)
    client = llm.get_client(backend)
    responses = await llm.batch(client, model, [j["prompt"] for j in todo],
                                max_concurrent=max_concurrent,
                                max_tokens=max_out, reasoning_effort="minimal")
    spent = 0.0
    for j, r in zip(todo, responses):
        if isinstance(r, Exception):
            print(f"  {j['model']} {j['condition']}[{j['idx']}]: "
                  f"{type(r).__name__}: {r}")
            continue
        text = r.choices[0].message.content or ""
        usage = r.usage.model_dump() if r.usage else {}
        cost = llm.cost_of(model, usage.get("prompt_tokens", 0),
                           usage.get("completion_tokens", 0))
        spent += cost
        label = parse_label(text)
        if label is None:
            print(f"  NO LABEL: {j['model']} {j['condition']}[{j['idx']}] "
                  f"reply={text[:60]!r}")
            continue
        rec = {"judge_model": model, "reply": text, "label": label,
               "usage": usage, "cost": cost}
        j["path"].write_text(json.dumps(rec, indent=2))
        j["cached"] = rec
    return spent


def main(backend: str = "openai", judge_model: str = "gpt-5-mini",
         max_concurrent: int = 4, max_out: int = 96, max_spend: float = 0.30,
         seed: int = 0, mode: str = "audit", per_model: int = 4,
         dry_run: bool = False):
    jobs = build_jobs(judge_model, seed, mode, per_model)
    todo = [j for j in jobs if j["cached"] is None]
    in_tok = int(sum(len(j["prompt"]) for j in todo) / 3.7)
    worst = llm.cost_of(judge_model, in_tok, max_out * len(todo))
    print(f"judge model : {judge_model} (ORIGINAL prompt, second sample)")
    print(f"sample      : {mode}"
          + (f", {per_model}/model" if mode == "random" else ", audit set (hard cases)"))
    print(f"jobs        : {len(jobs)} traces, {len(todo)} to call")
    print(f"input size  : ~{in_tok:,} tokens")
    print(f"projected   : ${worst:.3f} worst case   (--max_spend ${max_spend:.2f})")
    if dry_run:
        print("dry run — no API calls made")
        return
    if worst > max_spend:
        print(f"REFUSING TO RUN: ${worst:.3f} > --max_spend ${max_spend:.2f}")
        return

    spent = asyncio.run(run_jobs(jobs, backend, judge_model, max_concurrent,
                                 max_out))

    done = [j for j in jobs if j["cached"]]
    agree = sum(1 for j in done if j["cached"]["label"] == j["label_first_pass"])
    print(f"\n--- same prompt, second sample: {len(done)} traces ---")
    for j in sorted(done, key=lambda j: (j["group"], j["model"], j["idx"])):
        new = j["cached"]["label"]
        same = new == j["label_first_pass"]
        print(f"  {j['group']:8s} {j['model']:24s} {j['condition']:11s} "
              f"idx={j['idx']:3d} {j['label_first_pass']:10s} -> {new:10s}"
              f"{'' if same else '   DIFFERENT'}")

    per_label = {}
    for lab in LABELS:
        rows = [j for j in done if j["label_first_pass"] == lab]
        if rows:
            per_label[lab] = {
                "n": len(rows),
                "agreed": sum(1 for j in rows
                              if j["cached"]["label"] == lab),
            }
    print(f"\nagreement with first pass: {agree}/{len(done)} "
          f"({agree/len(done):.0%})" if done else "no results")
    for lab, v in per_label.items():
        print(f"  first pass said {lab:11s} n={v['n']:3d} -> same again "
              f"{v['agreed']:3d} ({v['agreed']/v['n']:.0%})")

    # Does the aggregate share move? That, not per-run agreement, is what
    # the map rests on.
    print("\n--- label shares, first pass vs second sample ---")
    shares = {}
    for lab in LABELS:
        a = sum(1 for j in done if j["label_first_pass"] == lab) / len(done)
        b = sum(1 for j in done if j["cached"]["label"] == lab) / len(done)
        shares[lab] = {"first_pass": a, "second_sample": b}
        print(f"  {lab:11s} {a:6.1%} -> {b:6.1%}   (shift {b-a:+.1%})")

    out = {"judge_model": judge_model, "seed": seed, "mode": mode,
           "label_shares": shares, "n": len(done),
           "agreed": agree,
           "agreement_rate": (agree / len(done)) if done else None,
           "per_first_pass_label": per_label,
           "rows": [{"group": j["group"], "model": j["model"],
                     "condition": j["condition"], "idx": j["idx"],
                     "label_first_pass": j["label_first_pass"],
                     "label_second_sample": j["cached"]["label"]} for j in done]}
    (DISCLOSURE_DIR / f"judge_stability_{mode}.json").write_text(json.dumps(out, indent=2))
    print(f"\ncost ${spent:.4f}; saved analysis/disclosure/judge_stability_{mode}.json")


if __name__ == "__main__":
    fire.Fire(main)
