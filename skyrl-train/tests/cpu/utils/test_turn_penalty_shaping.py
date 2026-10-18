import pytest

from skyrl_train.utils.turn_penalty_shaping import (
    PARSE_ERROR_PREFIX,
    compute_turn_penalty_shaping,
    trainable_runs,
    turn_keystrokes,
)


def _reply(*keystrokes: str) -> dict:
    commands = ", ".join('{"keystrokes": "%s", "duration": 1.0}' % k for k in keystrokes)
    return {"role": "assistant", "content": '<think>plan</think>{"analysis": "x", "commands": [%s]}' % commands}


def _obs(text: str = "ok") -> dict:
    return {"role": "user", "content": text}


def test_trainable_runs_finds_each_assistant_span():
    assert trainable_runs([0, 1, 1, 0, 0, 1, 0, 1, 1]) == [(1, 3), (5, 6), (7, 9)]


def test_keystrokes_are_decoded_in_order():
    assert turn_keystrokes(_reply("ls -la\\n", "cat a.txt\\n")["content"]) == ("ls -la\n", "cat a.txt\n")


def test_invalid_and_repeated_turns_are_penalized_on_their_own_tokens():
    messages = [
        _reply("ls\\n"),
        _obs(),
        {"role": "assistant", "content": "not json at all"},
        _obs(f"{PARSE_ERROR_PREFIX}\nERROR: no valid JSON"),
        _reply("make\\n"),
        _obs(),
        _reply("make\\n"),
        _obs(),
    ]
    #            turn0     obs   turn1  obs  turn2  obs  turn3  obs
    loss_mask = [1, 1] + [0] + [1, 1, 1] + [0] + [1] + [0] + [1, 1] + [0]
    result = compute_turn_penalty_shaping(messages, loss_mask, invalid_turn_penalty=0.2, repeated_turn_penalty=0.1)
    assert result.aligned and (result.assistant_turns, result.invalid_turns, result.repeated_turns) == (4, 1, 1)
    assert result.shaping == [0, 0, 0, -0.2, -0.2, -0.2, 0, 0, 0, -0.1, -0.1, 0]


def test_same_command_after_a_different_one_is_not_a_repeat():
    messages = [_reply("pytest\\n"), _obs(), _reply("vim a.py\\n"), _obs(), _reply("pytest\\n"), _obs()]
    result = compute_turn_penalty_shaping(
        messages, [1, 0, 1, 0, 1, 0], invalid_turn_penalty=0.2, repeated_turn_penalty=0.1
    )
    assert result.repeated_turns == 0 and not any(result.shaping)


def test_blank_keystrokes_used_for_waiting_are_not_a_repeat():
    messages = [_reply(""), _obs(), _reply(""), _obs()]
    result = compute_turn_penalty_shaping(messages, [1, 0, 1, 0], invalid_turn_penalty=0.2, repeated_turn_penalty=0.1)
    assert result.repeated_turns == 0


def test_truncated_tail_still_penalizes_leading_turns():
    messages = [
        {"role": "assistant", "content": "garbage"},
        _obs(f"{PARSE_ERROR_PREFIX}\nERROR: x"),
        _reply("ls\\n"),
    ]
    result = compute_turn_penalty_shaping(messages, [1, 1, 0], invalid_turn_penalty=0.2, repeated_turn_penalty=0.1)
    assert result.aligned and result.shaping == [-0.2, -0.2, 0]


def test_more_runs_than_turns_leaves_the_vector_zero():
    result = compute_turn_penalty_shaping(
        [_reply("ls\\n"), _obs()], [1, 0, 1], invalid_turn_penalty=0.2, repeated_turn_penalty=0.1
    )
    assert not result.aligned and not any(result.shaping)


def test_negative_magnitudes_are_rejected():
    with pytest.raises(ValueError):
        compute_turn_penalty_shaping([], [], invalid_turn_penalty=-0.1, repeated_turn_penalty=0.0)
