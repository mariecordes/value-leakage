# Follow-up analyses

These analyses build on the raw rollouts in `runs/` and are organized by the
question each experiment tests.

## 1. Disclosure

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

## 2. Target visibility

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

## 3. Interrupted continuations

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

## Data policy

The repository retains observations needed to audit the reported results:

- `runs/` contains the supplied replication rollouts and their judged outputs.
- `target_visibility/<model>/*.json` contains the newly sampled prompt
  conditions and consolidated judge outputs.
- `continuations/raw/` contains the probe and continuation responses underlying
  the interruption analysis.
- Experiment-level JSON files contain selections, aggregate results, and
  validation checks.

Per-request retry files and derived judge caches under `analysis/**/cache/` are
ignored. They duplicate the retained observations or can be regenerated and
are not needed to inspect the reported results.
