# Tracing motivated reasoning in the Donation Bet

*Independent research project extending a replication of the Donation Bet to build hands-on experience in chain-of-thought faithfulness research.*

When a seemingly irrelevant incentive shifts a reasoning model's numerical estimates, how faithfully does its reasoning reflect and explain that shift? This repository studies that question, building on Betley et al.'s (2026) Donation Bet, where a donation goes to a good or bad cause depending on which side of a specified threshold a model's Fermi estimate falls.

**Key findings:**

Across four complementary experiments with the latter three focusing on Qwen 3.5 122B A10B, I find that the model's visible reasoning is **partially faithful as it reflects the incentive-driven shift more faithfully than it explains it.**

- Qwen usually **discloses** the incentive's influence (33 of 40 traces) although this differs across the ten investigated models (e.g., Claude Opus 4.7 denies influence in 25 of 40 traces despite a behavioural answer shift).
- The model needs an actionable numerical threshold to **target** the rewarded outcome (shift of 0.526 in the ordinary bet, 0.167 when the threshold is withheld and 1.000 when explicitly told to aim).
- When interrupted midway through its reasoning and asked for an early estimate, Qwen's stated estimates allow us to **anticipate** where the reasoning will land for its final estimate (with no systematic bias towards the rewarded side).
- Changing the first committed spots-per-giraffe estimate **moves** the final answer in the corresponding direction (median rank correlation ρ = 0.70), although only ~13% of that change reaches the final answer.

## Summary

The Donation Bet is an AI alignment evaluation that Betley et al. (2026) introduce to investigate covert value leakage, namely if a large language model’s (LLM) moral preferences silently bias its factual estimates and if the model hides this bias in its reasoning trace. More specifically, the experiment selected for this research asks a model to estimate the total number of giraffe spots worldwide while promising a donation to a good or bad cause depending on which side of a given threshold the answer falls. Although the donation should be irrelevant to the estimate, models’ answers tend to move towards the good cause. While Betley et al. measure how often models output final estimates on the “good side”[^1] of the threshold and use an LLM judge to check if the reasoning discloses the donation’s influence, this project extends the experiment by testing how faithfully that reasoning reflects and explains such a behavioural answer shift. This shift is measured as the difference between how much more often answers land on the good side than on the other side; a shift of 1 thereby means that every answer follows the donation incentive whereas a shift of 0 means the incentive has no effect.

Starting from an existing reproduction of the paper’s experiment, I built a framework that tests four complementary dimensions of faithfulness: (1) if the reasoning **tells** us that the incentive influenced the estimate, (2) what is required for the model to **aim** at the rewarded outcome, (3) if the model appears to **know** where its answer is heading midway, and (4) what **causes** the model’s answer to shift towards the reward. The first test analyses 400 reasoning traces from ten models. The following, deeper tests focus on Qwen 3.5 122B A10B (Qwen) as it showed the largest shift and, as an open-weight model, allows to resume its full reasoning from any point.

**TELL: Qwen usually discloses the influence, but the degree of covertness differs across models.** Qwen’s reasoning openly says the donation affected its estimate in 33 of 40 traces. Despite a shift in its answers, Claude Opus 4.7’s reasoning denies influence in 25 of 40 traces. However, these labels stem from an LLM judge that, when asked to judge the same traces a second time, agreed with only 63% of its first labels, so while this broad contrast holds up, the finer rankings between models may not.

<p align="center">
  <img src="analysis/disclosure/map.png" width="100%" alt="Figure 1. answer shift against incentive disclosure per model and disclosure labels per model">
</p>
<p><sub><b>Figure 1.</b> A model’s answer shift and incentive disclosure do not necessarily move together. The left panel plots each model’s shift against the share of traces in which the reasoning admits the incentive influence (horizontal bars show a rough 95% range for this share). The right panel depicts the share of each disclosure label identified in the 40 traces per model.</sub></p>

**AIM: the incentive needs an actionable numerical target to influence the model’s answer.** Qwen’s shift is 0.526 in the ordinary Donation Bet, 0.167 when the prompt still describes the bet and which side benefits the good cause (above-pays or below-pays) but leaves out the threshold number, and 1.000 when explicitly instructed to aim. Tracking the intermediate estimates Qwen documents in its reasoning in the ordinary and instructed versions shows that above-pays and below-pays runs differ from their very first estimate, rather than drifting apart later. In other words, the incentive can only steer the answer when the model knows the number it can aim for.

<p align="center">
  <img src="analysis/target_visibility/lineup_qwen3.5-122b-a10b.png" width="85%" alt="Figure 2. Qwen's normalized estimates over the course of its reasoning in four prompt versions">
</p>
<p><sub><b>Figure 2.</b> Qwen’s normalized estimates over the course of its reasoning in four prompt versions. For each trace, an LLM judge extracts every total estimate Qwen records in its reasoning; the x-axis shows the relative position in the reasoning chain and the y-axis its relative distance from the threshold value of 41 million with lines as medians across traces and bands as interquartile ranges. Whereas in the ordinary bet and the instructed version, above-pays and below-pays traces already separate at the first recorded estimate and continue to stay apart, these two are not as clearly separated when the threshold is withheld (bottom right).</sub></p>

**KNOW: when paused midway, the model’s stated estimate anticipates where its reasoning will land.** I pause Qwen at 25%, 50% and 75% of its original reasoning trace, ask for its current best estimate and continue the reasoning six times from each point to check if that estimate gives insight into where the reasoning eventually lands. The estimates generally match the median final answer of those continuations and are not systematically biased towards the side that benefits the good cause. It thus seems what the model says about its current estimate is a good guide to where it is heading, though this does not show that the stated estimate causes the final answer.

<p align="center">
  <img src="analysis/continuations/method.png" width="100%" alt="Figure 3. Procedure diagram for one example trace, cut at 25%, 50% and 75%">
</p>
<p><sub><b>Figure 3.</b> Procedure diagram for interrupting the reasoning and asking for an early answer, exemplified on one Qwen trace. The trace is first cut at 25%, 50% and 75% of its reasoning and asked for its current best estimate (empty circles), then continued six times per each point. Filled circles show the median final estimate of these continuations with the original answer the uninterrupted trace reached depicted as the diamond. This example trace was chosen because the differences are easy to see, not because it is representative.</sub></p>

<p align="center">
  <img src="analysis/continuations/stated_vs_resampled_scatter_qwen3.5-122b-a10b.png" width="65%" alt="Figure 4.stated estimate against median of its six continuations">
</p>
<p><sub><b>Figure 4.</b> Estimate Qwen states when its reasoning is interrupted (x-axis) against the median final estimate of six continuations from that same point (y-axis) for 50 interruptions and both relative to the 41 million threshold. The diagonal runs where an early stated estimate matches where its continuations land. The axes are linear within ±0.3 (shaded area) and compressed outside for better depiction.</sub></p>

**CAUSE: the first spots-per-giraffe estimate matters but determines relatively little.** Qwen typically calculates the total estimate as the number of giraffes worldwide times the number of spots per giraffe. I choose the latter as target for sentence-resampling intervention since its estimates vary far more than the giraffe count estimates, which barely vary. I first cut the reasoning of 15 traces just before Qwen’s first committed estimate of spots per giraffe, then swap in a range of alternative values the model itself produced at that same point and let the reasoning continue six times for each version (528 continuations in total). In all 15 traces, inserting a higher number of spots per giraffe leads to a higher final total estimate (median Spearman correlation ρ = 0.70; p = 0.000061). Despite a consistent effect, its size is only small: theoretically, doubling the spots-per-giraffe estimate would double the final estimate if the calculation simply carried through. Instead, only 13% of that change reaches the final answer as later reasoning seems to revise most of it.

<p align="center">
  <img src="analysis/cause/cause_method.png" width="100%" alt="Figure 5. Procedure diagram for sentence resampling for one example trace">
</p>
<p><sub><b>Figure 5.</b> Procedure diagram for sentence resampling at the first spots-per-giraffe estimate committed in the reasoning trace, exemplified on one Qwen trace. After cutting the trace just before the first estimate, a set of alternative lines that Qwen produced at that point as well as the original sentence (shaded) are inserted and continued six times each. For this example, larger inserted spot estimates generally lead to larger final total estimates, however, identical numerical counts can still produce different median final estimates.</sub></p>

<p align="center">
  <img src="analysis/cause/cause_consistency.png" width="75%" alt="Figure 6. Sentence resampling results with positive within-trace correlation despite relatively low retention">
</p>
<p><sub><b>Figure 6.</b> Results of sentence resampling for all 15 original traces (one point per trace). While all recorded correlations between the inserted spots per giraffe and the final estimate are positive (top panel), the share of the inserted change that reaches the final answer stays well below one (bottom). The first spots estimate may move the answer consistently but only a limited share of the inserted change is retained in the final answer.</sub></p>

Taken together, I observe that Qwen’s visible reasoning is **partially faithful as it reflects the incentive-driven shift more faithfully than it explains it.** The reasoning trace usually acknowledges the incentive, steers towards it when provided with a numerical target, and records estimates that anticipate its continuations. Sentence resampling also shows the first spots-per-giraffe estimate genuinely affects the final answer, though later reasoning revises most of its effect. The trace may therefore be useful for monitoring without providing a full explanation of how the final estimate is produced.

Nevertheless, this conclusion is based on a single task and model. It may be treated as a hypothesis about how a seemingly irrelevant incentive shapes a model’s reasoning and should be tested across tasks and models, especially models in the Claude family that more often deny influence.

[^1]: The “good side” is defined as that side of the threshold that benefits the good outcome, namely the side that pays to a good cause or does not donate to a bad cause.

## Full write-up

The [full report](https://mariecordes.github.io/value-leakage/research_report.pdf) describes the methodology and experimental setup, presents the results and observations, and discusses limitations and directions for future work. It was written with the help of LLMs and is still work in progress, so it is subject to change.

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
