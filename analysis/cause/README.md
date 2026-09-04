# CAUSE — does the first number the model writes cause its answer?

The earlier experiments established that a model's answer to the Donation Bet moves
toward the side the bet rewards, and that the difference is visible early in the
reasoning. They could not say whether the first number the model writes *causes* the
final answer or merely reports a direction settled elsewhere.

This intervenes on that number directly. For a saved trace, cut the reasoning just
before the model commits to a spots-per-giraffe figure, put a different figure in its
place — one the model itself wrote at that same cut on another sample — and let it run
to a final answer. If the figure anchors the answer, a higher one should give a higher
answer.

**Result.** All 15 traces show a positive relationship between the inserted figure and
the final estimate (two-sided exact sign test, p = 6.1e-05, median rank correlation
0.70). The magnitude is small: about 13% of a proportional change survives to the
answer, and the shift is comparable to the spread of continuations from an unchanged
step. The first written number is a real but leaky anchor.

## Pipeline

Every stage is local unless marked PAID, and no paid stage runs without `--execute`.

```bash
# 1. Find a cut in every rollout, then review the proposals by hand
uv run python analysis/cause/select_cuts.py            # writes cut_review.json
uv run python analysis/cause/select_cuts.py --check    # validates the review

# 2. Pin the reviewed pool, hash-verified against the source rollouts
uv run python analysis/cause/pin_sample.py --per_arm 40
uv run python analysis/cause/pin_sample.py --verify

# 3. Resample the committed step, then read each resample with a judge
uv run python analysis/cause/resample.py --from_final --execute        # PAID
uv run python analysis/cause/judge_resamples.py --label                # draw 30
uv run python analysis/cause/judge_resamples.py --execute --only_labelled  # PAID
uv run python analysis/cause/judge_resamples.py --agreement_check
uv run python analysis/cause/judge_resamples.py --execute              # PAID
uv run python analysis/cause/resample.py --analyze
uv run python analysis/cause/pin_sample.py --final --accept_short

# 4. The experiment: insert each step, run to an answer
uv run python analysis/cause/continue_traces.py --build_only           # inserts.json
uv run python analysis/cause/continue_traces.py --dry_run
uv run python analysis/cause/continue_traces.py --execute              # PAID

# 5. Judge the answers, summarise, plot
uv run python analysis/cause/analyze_cause.py --execute                # PAID
uv run python analysis/cause/plot_cause.py
uv run python analysis/cause/plot_cause_method.py
uv run python analysis/cause/independent_verify.py
uv run python analysis/cause/summarize_audit.py
```

Total cost was about $10.30. Responses are cached by content, so a rerun only pays for
what is missing.

## How the sample was chosen

A deterministic rule proposes one cut per rollout: the first line before the model's
first total calculation on which it commits to a population, a spots-per-giraffe
figure, or a total outright. The rule is applied to all 200 rollouts without looking
at any outcome, and it found a cut in 117.

A human then reviews the proposals and may only **remove** traces, using five numbered
reasons recorded in `select_cuts.py`. Nothing in the review can add a trace or reorder
the sample: proposals are pre-shuffled with a recorded seed, and `--check` fails if the
review skipped ahead in that order. No trace may be excluded on the basis of its final
answer.

Two restrictions were applied after measurement, both recorded in the code with their
cause:

- **Spots only.** Resampling at a population commitment produced almost no variation —
  the giraffe population is close to a checkable fact and the model reproduces it.
  Spots per giraffe is a judgement call and varies over a factor of four. An
  intervention on a quantity with no variance measures nothing.
- **Traces with usable spread only.** A trace whose resamples all give the same figure
  offers nothing to insert. This is measured before any continuation is sampled and
  without reference to any final answer.

Fifteen traces survived, eight above-paying and seven below-paying — one short of the
target in the below arm, with the eligible pool exhausted and no criterion loosened.

## What is checked

- Every replayed prompt, reasoning trace and prefix is verified by SHA-256 against the
  unchanged source rollout before any call is made.
- The judge that reads the resamples was validated against 60 hand labels across two
  blind rounds; the final agreement of 86.7% is a development-set figure, since the
  second round was used to diagnose prompt defects.
- Final answers are read by the repository's existing number judge, unchanged, with the
  published outlier filter.
- Ten finished continuations were audited end to end against five criteria — all
  passed. See `finish_audit.json` and `summarize_audit.py`.
- `independent_verify.py` recomputes the headline from the raw response files and the
  judge cache without importing the analysis code.

## Files

| | |
|---|---|
| `select_cuts.py` | the cut rule, and the exclusion-only review gate |
| `pin_sample.py` | pins the pool and the final sample, hash-verified |
| `resample.py` | resamples the committed step; measures spread |
| `judge_resamples.py` | reads each resample; hand-label validation |
| `continue_traces.py` | builds the inserts and runs the experiment |
| `analyze_cause.py` | judges final answers, computes the result |
| `plot_cause.py` | the three result figures |
| `plot_cause_method.py` | the procedure on one trace, quoted verbatim |
| `independent_verify.py` | independent recomputation of the headline |
| `separation_onset.py` | a side analysis, not used in the result |
| `superseded/` | an earlier design, retired; kept as a record |

Raw responses are under `raw/`; per-request judge caches under `cache/` are ignored by
git and can be regenerated.
