# Tracing motivated reasoning in the Donation Bet

*Independent research project extending an existing replication of the Donation Bet,
undertaken to build hands-on experience in chain-of-thought faithfulness research.*

When an irrelevant incentive shifts a reasoning model's numerical answer, how faithfully
does its visible reasoning reflect and explain that shift? This repository studies that
question in Betley et al.'s (2026) Donation Bet, where a donation goes to a good or bad
cause depending on which side of a threshold a model's Fermi estimate falls.

**Read the full report: [research_report.pdf](docs/research_report.pdf)**

## Key findings

Four complementary tests, named after the question each asks. TELL covers ten models;
the deeper tests focus on Qwen 3.5 122B A10B (Qwen).

- **TELL:** Does the reasoning *tell* us that the incentive influenced the estimate?
  Qwen usually discloses the influence (33 of 40 traces), but this differs across models:
  Claude Opus 4.7 denies influence in 25 of 40 despite a behavioral shift.
- **AIM:** What allows the model to *aim* at the rewarded outcome? An actionable
  numerical target. Qwen's shift is 0.526 in the ordinary bet, 0.167 when the threshold
  is withheld, and 1.000 when explicitly told to aim.
- **KNOW:** Does the model appear to *know* where its answer is heading? Estimates Qwen
  states at 25%, 50%, and 75% of its reasoning track the median of resampled
  continuations, with no systematic rewarded-direction gap.
- **CAUSE:** What *causes* the answer to shift toward the reward? Changing the first
  spots-per-giraffe estimate moves the final answer in the predicted direction in all
  15 traces (median ρ = 0.70), but only about 13% of the change survives.

Overall, Qwen's visible reasoning is **partially faithful: it reflects the
incentive-driven shift more faithfully than it explains it.**

### Selected figures

<p align="center">
  <img src="analysis/target_visibility/lineup_qwen3.5-122b-a10b_exec_summary.png" width="80%" alt="AIM: Qwen's trajectories under the four prompt versions">
</p>

**AIM.** When the threshold is visible or Qwen is told to aim, its above- and below-pays
estimates separate from the first numerical candidate and stay apart; withholding the
threshold weakens the separation. See Section 3.2 of the report.

<p align="center">
  <img src="analysis/continuations/stated_vs_resampled_scatter_qwen3.5-122b-a10b_exec_summary.png" width="60%" alt="KNOW: stated estimates versus continuation medians">
</p>

**KNOW.** The estimate Qwen states when interrupted closely tracks the median of six
continuations from the same point, with no systematic tilt toward the rewarded side.
See Section 3.3 of the report.

<p align="center">
  <img src="analysis/cause/cause_consistency_exec_summary.png" width="70%" alt="CAUSE: within-trace correlation and retention">
</p>

**CAUSE.** In every trace, a larger inserted spots-per-giraffe estimate leads to a larger
final answer, but only a small share of the inserted change reaches the final estimate.
See Section 3.4 of the report.

## Built on

This project extends [Aditya Singh's replication](https://github.com/adsingh-64/value-leakage)
of the Donation Bet from [Betley et al. (2026)](https://doi.org/10.48550/arXiv.2607.14345).
The replication supplied the prompts, sampling pipeline, saved rollouts for ten models,
and answer extraction; its original README is kept in [README_ORIGINAL.md](README_ORIGINAL.md).
Everything under `analysis/` and `docs/` is new.

## Repository layout

```
docs/
  research_report.pdf    the report (with its Word and Markdown sources)
analysis/                follow-up experiments (new), see analysis/README.md
  disclosure/            TELL: do traces acknowledge the incentive?
  target_visibility/     AIM: does the effect need a visible threshold?
  continuations/         KNOW: stated estimates vs. resampled continuations
  cause/                 CAUSE: intervening on the first spots-per-giraffe premise
  palette.py             shared figure palette
  view_rollout.py        inspect individual saved traces
  make_exec_summary_figures.py   compact versions of key figures (shown above)
src/value_leakage/       replication pipeline (from the original repository)
  sample.py              prompts + sampling (Fireworks / OpenRouter / Anthropic backends)
  judge.py               estimate + trajectory judges (Claude)
  run.py                 end-to-end pipeline: baseline -> threshold -> conditions -> judges -> plot
  plot.py                per-run trajectory figure + motivated_reasoning_factor (factor.json)
  panel.py               mega panel: all runs x {pooled, start-above, start-below}
  api/                   thin API clients (Anthropic, Fireworks, OpenRouter)
runs/<model>_<stamp>/    saved rollouts for ten models (from the original repository)
  config.json            model, backend, count, judge
  baseline.json          raw rollouts: reasoning + visible answer per sample
  below_good.json        same, below-favoured condition
  above_good.json        same, above-favoured condition
  estimates.json         judge: final number per rollout (null = unparseable)
  trajectories.json      judge: in-reasoning estimate sequence per rollout
  threshold.json         median baseline estimate
  factor.json            drift metrics
  fig.png                per-run figure
mega_panel.png           all-run panel from value_leakage.panel
```

The raw reasoning lives in `{baseline,below_good,above_good}.json` under
`rows[*].reasoning`. Anthropic-backend caveat: Claude returns a summarized trace, not
raw chain-of-thought.

## Setup and reproduction

Requires [uv](https://docs.astral.sh/uv/). From a fresh clone, this creates the virtual
environment, installs Python 3.12 if needed, and installs the locked dependencies:

```
uv sync
```

All analyses run from cached model responses; no API keys are needed to regenerate the
reported statistics and figures. See [analysis/README.md](analysis/README.md) for
per-experiment commands and the data-retention policy.

The original replication figures can be regenerated from the shipped data:

```
uv run python -m value_leakage.plot --run_dir runs/inkling_20260815_030703
uv run python -m value_leakage.panel
```

Sampling new rollouts or rerunning judges requires API keys (copy `.env.example` to
`.env`), for example:

```
uv run python -m value_leakage.run --target_model <id> --target_backend fireworks --count 100
```
