# Follow-up analyses

These analyses build on the raw rollouts in `runs/` and are organized by the
question each experiment tests. TELL asks whether the reasoning acknowledges the
donation incentive, AIM whether the answer shift depends on a target the model can act
on, KNOW whether an estimate stated mid-trace predicts where continuations land, and
CAUSE whether the first number the model commits to changes the answer it reaches.

## 1. TELL — disclosure

`disclosure/` measures the relationship between behavioral bias and what a
model says about the donation incentive in its reasoning trace. The four-way
rubric distinguishes traces that admit influence, mention the incentive without
claiming influence, do not mention it, or explicitly deny influence.

Recompute the sample and figures from saved data:

```bash
uv run python analysis/disclosure/prepare_sample.py
uv run python analysis/disclosure/plot_disclosure_map.py
uv run python analysis/disclosure/plot_judge_matrix.py
```

`judge_disclosure.py` and the `validate_*.py` scripts rerun model-based judging
and therefore require the relevant API key.

## 2. AIM — target visibility

`target_visibility/` tests whether the behavioral shift depends on a visible
numerical target. It compares the original donation bet with a hidden-threshold
variant and an explicit instruction to aim above or below the threshold.

Recompute summaries and figures from saved data:

```bash
uv run python analysis/target_visibility/summarize.py
uv run python analysis/target_visibility/plot_ladder.py
```

`sample_conditions.py` resamples the prompt variants, while
`judge_conditions.py` extracts final estimates and reasoning trajectories.
Both require the relevant API key.

## 3. KNOW — interrupted continuations

`continuations/` compares the estimate elicited at a cut point with the median
endpoint of six continuations resampled from the same partial trace. The
explicitly instructed condition provides a reference case in which aiming is
overt.

Recompute the selection and figures from saved data:

```bash
uv run python analysis/continuations/select_traces.py
uv run python analysis/continuations/plot_method.py
uv run python analysis/continuations/plot_gap_scatter.py
uv run python analysis/continuations/plot_gap_bars.py
```

`sample_continuations.py` creates new probes and continuations.
`summarize.py` reruns number extraction on those responses. Both require the
relevant API key; the repository already includes their consolidated
`results.json` output for inspection and figure generation.

`summarize.py` also emits a combined two-panel `honesty_<model>.png`. That figure is
superseded by the two standalone ones above and is not used in the write-up; it is
still generated so the earlier draft remains reproducible.

## 4. CAUSE — causal intervention on the first committed number

`cause/` tests whether the first spots-per-giraffe figure Qwen commits to causes its
final answer. Each saved trace is cut just before that commitment, a different figure
is put in its place — one the model itself wrote at the same cut on another sample —
and the trace runs to a final answer.

Across 15 traces and 528 continuations, a higher inserted figure gave a higher final
estimate in every trace (two-sided exact sign test, p = 6.1e-05), but only about 13%
of a proportional change survived, and the shift is comparable to the spread of
continuations from an unchanged step.

Recompute the figures and the headline from saved data:

```bash
uv run python analysis/cause/plot_cause.py
uv run python analysis/cause/plot_cause_method.py
uv run python analysis/cause/analyze_cause.py
uv run python analysis/cause/independent_verify.py
```

All commands are safe by default: paid model or judge calls require an explicit
`--execute`. See [`cause/README.md`](cause/README.md) for the full sequence, how the
sample was chosen, and what was checked.

## Data policy

The repository retains observations needed to audit the reported results:

- `runs/` contains the supplied replication rollouts and their judged outputs.
- `target_visibility/<model>/*.json` contains the newly sampled prompt
  conditions and consolidated judge outputs.
- `continuations/raw/` contains the probe and continuation responses underlying
  the interruption analysis.
- `cause/raw/` contains every resampled step and every continuation, including the
  resamples from population-slot cuts. Those are the evidence that the model will not
  move a premise it treats as factual, and are retained even though those traces are
  not in the final sample.
- `cause/superseded/` holds an earlier design and its review files, kept so the
  selection history is auditable.
- Experiment-level JSON files contain selections, aggregate results, and
  validation checks.

Per-request retry files and derived judge caches under `analysis/**/cache/` are
ignored. They duplicate the retained observations or can be regenerated and
are not needed to inspect the reported results.
