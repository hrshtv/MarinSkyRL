"""Trajectory assembly when the served context re-renders earlier assistant turns (native tool calls)."""

from skyrl_train.trajectory_runners.trajectory_processing import (
    AlignmentStats,
    ServedContext,
    TitoFullDeclineReason,
    _assemble_response_ids_tito_full,
)

GENERATION_PROMPT = [900, 901]
INITIAL_PROMPT = [1, 2, 3]
OBSERVATION_0 = [50, 51]
OBSERVATION_1 = [60]
EOS = 99


class _Tokenizer:
    eos_token_id = EOS


def _assemble(prompts, completions, served_context):
    stats = AlignmentStats()
    messages = [{"role": "assistant", "content": ""} for _ in completions]
    logprobs = [[-0.25 * (turn + 1)] * len(completion) for turn, completion in enumerate(completions)]
    result = _assemble_response_ids_tito_full(
        messages,
        _Tokenizer(),
        GENERATION_PROMPT,
        logprobs,
        completions,
        prompts,
        None,
        stats,
        None,
        None,
        served_context=served_context,
    )
    return result, stats


def _three_turns(rendered_turn_1):
    """Three turns; turn 1 was sampled as [20, 21, EOS] and reaches the next prompt as ``rendered_turn_1``."""
    completions = [[10, 11, EOS], [20, 21, EOS], [30, EOS]]
    prompt_0 = INITIAL_PROMPT + GENERATION_PROMPT
    prompt_1 = prompt_0 + completions[0] + OBSERVATION_0 + GENERATION_PROMPT
    prompt_2 = prompt_1 + rendered_turn_1 + OBSERVATION_1 + GENERATION_PROMPT
    return [prompt_0, prompt_1, prompt_2], completions


def test_rerendered_turn_stays_in_context_and_leaves_the_loss():
    rendered = [20, 777, 21, EOS]
    prompts, completions = _three_turns(rendered)

    result, stats = _assemble(prompts, completions, ServedContext.RERENDERED)

    assert result.decline_reason is None
    served = prompts[-1] + completions[-1]
    assert result.response_ids == served[len(INITIAL_PROMPT) :]
    in_loss = [token for token, mask in zip(result.response_ids, result.loss_mask) if mask]
    assert in_loss == completions[0] + completions[2]
    scored = [logprob for logprob, mask in zip(result.rollout_logprobs, result.loss_mask) if mask]
    assert scored == [-0.25] * 3 + [-0.75] * 2
    assert (stats.n_turns, stats.n_rerendered_turns, stats.n_rerendered_tokens) == (3, 1, 3)
    assert stats.n_unaligned == 0


def test_identical_rendering_trains_every_turn():
    prompts, completions = _three_turns([20, 21, EOS])

    result, stats = _assemble(prompts, completions, ServedContext.RERENDERED)

    in_loss = [token for token, mask in zip(result.response_ids, result.loss_mask) if mask]
    assert in_loss == completions[0] + completions[1] + completions[2]
    assert stats.n_rerendered_turns == 0


def test_sampled_context_declines_a_rerendered_turn():
    prompts, completions = _three_turns([20, 777, 21, EOS])

    result, _ = _assemble(prompts, completions, ServedContext.SAMPLED)

    assert result.decline_reason is TitoFullDeclineReason.PREFIX_MISMATCH


def test_rerendered_context_declines_prompts_that_are_not_nested():
    prompts, completions = _three_turns([20, 21, EOS])
    prompts[2] = [7] + prompts[2][1:]

    result, _ = _assemble(prompts, completions, ServedContext.RERENDERED)

    assert result.decline_reason is TitoFullDeclineReason.PREFIX_MISMATCH
