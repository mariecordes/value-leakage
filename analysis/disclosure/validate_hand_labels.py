"""DISCLOSURE_DIR — hand-check of the disclosure labels (research notes: check ~10 by eye).

No API calls. Picks judged runs spread across the four labels and across models,
and prints, for each: the label the judge gave, plus every sentence in the trace
that touches the bet. That is enough to agree or disagree with the label without
reading 100,000 characters of giraffe arithmetic.

  uv run python analysis/disclosure/handcheck.py                 # ~12 runs, spread over labels
  uv run python analysis/disclosure/handcheck.py --model kimi-k3 --n 6
  uv run python analysis/disclosure/handcheck.py --label DENIES --n 8 --full_sentences 40

Writes nothing. Read the output, then record agreement/disagreement in the notes.
"""

import json
import random
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).parent))
from common import RUNS, DISCLOSURE_DIR, reasoning_texts  # noqa: E402

LABELS = ("ADMITS", "DENIES", "MENTIONS", "NO_MENTION")

# Sentences worth showing a human: anything naming the bet or the model's own
# even-handedness. Deliberately broad — this is for reading, not for labelling.
RELEVANT = re.compile(
    r"(donat|charit|good cause|bad cause|\bbet\b|threshold|incentiv|"
    r"bias|unbias|honest|temptat|sway|nudge|skew|tilt|lean toward|"
    r"in my favou?r|favou?rable)", re.I)

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def relevant_sentences(text: str, cap: int) -> list[tuple[float, str]]:
    """(position in trace 0-1, sentence) for sentences that mention the bet."""
    out, n = [], max(1, len(text))
    pos = 0
    for sent in _SPLIT.split(text):
        start = text.find(sent, pos)
        pos = start + len(sent) if start >= 0 else pos
        s = " ".join(sent.split())
        if s and RELEVANT.search(s):
            out.append(((start if start >= 0 else pos) / n, s))
    if len(out) <= cap:
        return out
    # Keep the first few, the last few, and a thinned middle.
    head, tail = out[:cap // 2], out[-(cap - cap // 2):]
    return head + [(-1.0, f"... {len(out) - cap} more matching sentences ...")] + tail


def main(n: int = 12, model: str | None = None, label: str | None = None,
         full_sentences: int = 14, seed: int = 0, chars: int = 260):
    files = sorted(DISCLOSURE_DIR.glob("disclosure_*.json"))
    if not files:
        print("no analysis/disclosure/disclosure_*.json yet — run analysis/disclosure/judge_disclosure.py first")
        return

    rows = []
    for f in files:
        d = json.loads(f.read_text())
        for r in d["rows"]:
            if r["label"] is None:
                continue
            rows.append({**r, "model": d["model"], "run_dir": None,
                         "judge_model": d["judge_model"]})
    if model:
        rows = [r for r in rows if r["model"] == model]
    if label:
        rows = [r for r in rows if r["label"] == label]
    if not rows:
        print("no judged rows match that filter")
        return

    # Spread the sample over the labels present, then over models within a label.
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    picks, i = [], 0
    labs = [lab for lab in LABELS if lab in by_label]
    for lab in labs:
        rng.shuffle(by_label[lab])
    while len(picks) < n and any(by_label[lab] for lab in labs):
        lab = labs[i % len(labs)]
        if by_label[lab]:
            picks.append(by_label[lab].pop())
        i += 1

    bias = json.loads((DISCLOSURE_DIR / "bias.json").read_text())
    print(f"hand-check: {len(picks)} runs, labels by {rows[0]['judge_model']}\n")
    for k, r in enumerate(picks, 1):
        run_dir = RUNS / bias[r["model"]]["run_dir"]
        thr = bias[r["model"]]["threshold"]
        text = reasoning_texts(run_dir, r["condition"])[r["idx"]]
        side = "above" if r["final"] > thr else ("on" if r["final"] == thr else "below")
        print("=" * 78)
        print(f"[{k}] {r['model']}  {r['condition']}  run #{r['idx']}   "
              f"JUDGE SAID: {r['label']}")
        print(f"    threshold {thr:,.0f} · final answer {r['final']:,.0f} "
              f"({side} the threshold) · trace {len(text):,} chars")
        print("    --- every sentence in the trace that mentions the bet ---")
        for pos, s in relevant_sentences(text, full_sentences):
            where = "     " if pos < 0 else f"{pos:4.0%} "
            print(f"    {where}| {s[:chars]}{'…' if len(s) > chars else ''}")
        print()


if __name__ == "__main__":
    fire.Fire(main)
