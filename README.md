# Tracing motivated reasoning in the Donation Bet

*Independent research project extending a replication of the Donation Bet to build hands-on experience in chain-of-thought faithfulness research.*

When a seemingly irrelevant incentive shifts a reasoning model's numerical estimates, how faithfully does its reasoning reflect and explain that shift? This repository studies that question, building on Betley et al.'s (2026) Donation Bet, where a donation goes to a good or bad cause depending on which side of a specified threshold a model's Fermi estimate falls.

## Research overview

### Key findings

Across four complementary experiments with the latter three focusing on Qwen 3.5 122B A10B, I find that the model's visible reasoning is **partially faithful as it reflects the incentive-driven shift more faithfully than it explains it.**

- Qwen usually **discloses** the incentive's influence (33 of 40 traces) although this differs across the ten investigated models (e.g., Claude Opus 4.7 denies influence in 25 of 40 traces despite a behavioural answer shift).
- The model needs an actionable numerical threshold to **target** the rewarded outcome (shift of 0.526 in the ordinary bet, 0.167 when the threshold is withheld and 1.000 when explicitly told to aim).
- When interrupted midway through its reasoning and asked for an early estimate, Qwen's stated estimates allow us to **anticipate** where the reasoning will land for its final estimate (with no systematic bias towards the rewarded side).
- Changing the first committed spots-per-giraffe estimate **moves** the final answer in the corresponding direction (median rank correlation ρ = 0.70), although only ~13% of that change reaches the final answer.

### Research summary

The [research summary](docs/research_summary.md) provides a concise overview of the experiment setup, results and key findings along with select figures. 

### Full write-up

The [full report](https://mariecordes.github.io/value-leakage/research_report.pdf) describes related literature, details the methodology and experimental setup, presents the results and observations, and discusses limitations and directions for future work. It was written with the help of LLMs and is still work in progress and thus subject to change.

## Built on

This project extends [Aditya Singh's replication](https://github.com/adsingh-64/value-leakage)
of the Donation Bet from [Betley et al. (2026)](https://doi.org/10.48550/arXiv.2607.14345).
The replication supplied the prompts, sampling pipeline, saved rollouts for ten models,
and answer extraction; its original README is kept in [README_ORIGINAL.md](README_ORIGINAL.md).
Everything under `analysis/` and `docs/` is new.

## Repository layout

```
docs/
  research_report.pdf    the report (with its Word source)
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
