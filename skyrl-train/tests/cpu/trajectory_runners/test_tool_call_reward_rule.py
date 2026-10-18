"""The multi-tool-call reward rule: a turn with more calls than allowed marks the trajectory."""

from skyrl_train.trajectory_runners.harbor.runner import has_excess_tool_calls


def _turn(n_calls):
    return {"role": "assistant", "tool_calls": [{"function": {"name": "bash"}} for _ in range(n_calls)]}


def test_one_call_per_turn_is_allowed():
    messages = [{"role": "user", "content": "task"}, _turn(1), {"role": "tool", "content": "ok"}, _turn(1)]
    assert not has_excess_tool_calls(messages, 1)


def test_a_turn_with_two_calls_marks_the_trajectory():
    messages = [{"role": "user", "content": "task"}, _turn(1), {"role": "tool", "content": "ok"}, _turn(2)]
    assert has_excess_tool_calls(messages, 1)


def test_turns_without_calls_and_non_assistant_messages_are_ignored():
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "prose"}, {"role": "tool", "content": "x"}]
    assert not has_excess_tool_calls(messages, 1)
