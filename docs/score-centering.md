# Optional PPO/TIS score centering

Score centering adds a loss term to regular PPO with tokenwise truncated importance sampling (TIS).
SC and its sampled diagnostics are disabled by default. Keep TIS enabled when enabling SC.
The [design](https://echo.oa.dev/wiki/608) describes the correction derived from the
[score-centering paper](https://arxiv.org/html/2609.20807v1).

## Configuration

Use local vLLM and unpacked Megatron with tensor, context and sequence parallel size one.
Data and pipeline parallelism use the existing whole-step loss normalization. The head contains the
sampler's natural top K token IDs and their full-vocabulary probabilities. The trainer scores those same IDs
at the same causal positions, once before the update and again in the training forward.

```yaml
trainer:
  strategy: megatron
  use_sample_packing: false
  algorithm:
    policy_loss_type: regular
    off_policy_correction: tis
    score_centering_topk: 128
    score_centering_enabled: true
    score_centering_diagnostic_tokens: 0
  policy:
    sequence_parallel_size: 1
    megatron_config:
      tensor_model_parallel_size: 1
      context_parallel_size: 1
generator:
  backend: vllm
  run_engines_locally: true
  sampling_params:
    logprobs: 128
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    min_p: 0.0
```

The `tis` preset uses an upper cap of two. A custom correction must contain exactly one token truncation
rule. Distillation, auxiliary FTPO, other policy losses, remote inference and other trainer backends are
rejected. Sampling must preserve the full temperature-scaled distribution; nucleus sampling and other
probability-changing filters cannot supply these behavior probabilities.

Set `score_centering_enabled: false` with positive `score_centering_topk` to measure a matched control.
It retains head capture and old/current rescoring while skipping the correction kernel. Ordinary TIS uses
`score_centering_topk: 0`, `score_centering_enabled: false` and `generator.sampling_params.logprobs: 0`.
It collects only the chosen token's behavior probability and performs no head rescoring.

The sampled action's TIS weight applies to the PPO row. The additive SC row already integrates TIS and is
reduced separately with the same whole-step counts. Multiplying it by that action's TIS weight again changes
the gradient. Outside the captured head, old and behavior probabilities are modeled as proportional to the
current policy, preserving each distribution's measured omitted mass. Small omitted mass alone does not
bound parameter-gradient error or establish a learning benefit.

## Diagnostics

Ordinary `trainer.training_metrics` uses already available chosen-token probabilities and policy-step tags.
`policy/mismatch/*` compares the trainer's pre-update probabilities with the actual generating policy.
Its staleness-zero bucket measures residual trainer–inference mismatch at matching weights. Stale buckets
also contain weight drift. The worker's `policy/log_ratio_*` and PPO clipping metrics compare current
probabilities with the stored old trainer probabilities. An exactly unit PPO ratio can coexist with TIM.

`async/staleness_tokens_*` reports consumed token age in policy steps. Local vLLM stamps tokens at sampling;
tokens without a stamp use the rollout lease's step. These two provenance cases remain distinguishable in
the existing metrics. A lease alone does not prove which weights generated a token.

With an explicitly configured head, set `score_centering_diagnostic_tokens` to a positive integer to sample
at most that many valid tokens per microbatch. `score_centering/sampled_*` reports current/old/behavior
retained and omitted mass, the current-versus-behavior log ratio, and signed/absolute SC loss values.
The existing aggregation averages these microbatch samples across ranks; they are not pooled token means.
The sample takes the first valid tokens and does not estimate the response tail. These scalar correction
values measure a loss term, not parameter-gradient change. They require no additional model forward.
Leaving this count zero avoids the new diagnostics. Ordinary telemetry does not enable head capture.
