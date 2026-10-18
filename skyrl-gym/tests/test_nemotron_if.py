import json
import os

import pytest
from omegaconf import DictConfig
from transformers import AutoTokenizer

import skyrl_gym
from skyrl_gym.envs.nemotron_if.env import FormatViolation, followed_instructions, parse_response

MODEL_PATH = os.environ.get("NEMOTRON_IF_TOKENIZER_PATH")  # a Snowball tokenizer directory (cluster-specific)
pytestmark = pytest.mark.skipif(MODEL_PATH is None, reason="set NEMOTRON_IF_TOKENIZER_PATH to a Snowball tokenizer directory")
VARIED = " ".join(f"point{i} about requirement{i}" for i in range(40))


class _Evidence:
    def __init__(self, response_token_ids):
        self.response_token_ids = tuple(response_token_ids)


def _well_formed(answer, thinking=VARIED):
    return f"<|start_think|>\n{thinking}\n<|end_think|>\n\n{answer}<|eot_id|>"


def _step(prompt, ids, kwargs, marked_response, min_thinking_tokens=0):
    """Score a response the way a rollout does: special tokens in the ids, stripped from the text."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    spec = json.dumps({"prompt": prompt, "instruction_id_list": ids, "kwargs": kwargs})
    env = skyrl_gym.make(
        "nemotron_if",
        env_config=DictConfig({"tokenizer_path": MODEL_PATH, "min_thinking_tokens": min_thinking_tokens}),
        extras={"reward_spec": {"ground_truth": spec}},
    )
    token_ids = tokenizer.encode(marked_response, add_special_tokens=False)
    env.set_rollout_evidence(_Evidence(token_ids))
    return env.step(tokenizer.decode(token_ids, skip_special_tokens=True))


def _violation(marked_response, min_thinking_tokens=100):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return parse_response(tokenizer.encode(marked_response, add_special_tokens=False), MODEL_PATH, min_thinking_tokens).violation


@pytest.mark.parametrize(
    "ids, kwargs, good, bad",
    [
        (["punctuation:no_comma"], [{}], "A short reply without the mark.", "A reply, with a comma."),
        (["last_word:last_word_answer"], [{"last_word": "contest"}], "We all entered the contest", "The contest was long ago"),
        (["change_case:english_lowercase"], [{}], "everything here is written in lowercase letters for you.", "This Sentence Has Capitals in it."),
        (["detectable_format:title"], [{}], "<<A Title>>\nBody text follows here.", "No title in this reply."),
        (["length_constraints:number_words"], [{"num_words": 5, "relation": "less than"}], "Just four words here", "This reply has many more words than the limit allows"),
    ],
)
def test_single_instruction_pass_and_fail(ids, kwargs, good, bad):
    assert followed_instructions(good, "prompt", ids, kwargs) == [True]
    assert followed_instructions(bad, "prompt", ids, kwargs) == [False]


def test_increment_word_keywords_stored_as_lists_are_unwrapped():
    ids, kwargs = ["count:count_increment_word"], [{"keyword1": ["help"], "keyword2": ["dump"]}]
    assert followed_instructions("Please help me with the dump and then the second dump.", "p", ids, kwargs) == [True]
    assert followed_instructions("Please help, help, and dump.", "p", ids, kwargs) == [False]


def test_a_well_formed_response_is_parsed_into_its_answer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    parsed = parse_response(tokenizer.encode(_well_formed("The answer."), add_special_tokens=False), MODEL_PATH, 100)
    assert (parsed.violation, parsed.answer) == (None, "The answer.")
    assert parsed.thinking_tokens >= 100 and parsed.distinct_thinking_tokens >= 25


@pytest.mark.parametrize(
    "marked, violation",
    [
        (f"<|start_think|>\n{VARIED}\n<|end_think|>\n\nanswer", FormatViolation.NOT_ENDED),  # cut off, no <|eot_id|>
        (f"<|start_think|>\n{VARIED}\n<|end_think|>\n\nanswer<|end_of_text|>", FormatViolation.NOT_ENDED),
        (f"{VARIED}\n<|end_think|>\n\nanswer<|eot_id|>", FormatViolation.NO_LEADING_START),
        (f"Sure.<|start_think|>\n{VARIED}\n<|end_think|>\n\nanswer<|eot_id|>", FormatViolation.NO_LEADING_START),
        ("plain answer<|eot_id|>", FormatViolation.NO_LEADING_START),
        (f"<|start_think|>\n<|start_think|>\n{VARIED}\n<|end_think|>\n\nanswer<|eot_id|>", FormatViolation.MARKER_COUNT),
        (f"<|start_think|>\n{VARIED}<|eot_id|>", FormatViolation.MARKER_COUNT),  # never closed
        # the redrafting exploit: close, draft, close again
        (f"<|start_think|>\n{VARIED}\n<|end_think|>\n\ndraft\n<|end_think|>\n\nanswer<|eot_id|>", FormatViolation.MARKER_COUNT),
        (f"<|start_think|>\n{VARIED}\n<|end_think|>\n\nanswer<|start_think|>\nmore<|eot_id|>", FormatViolation.MARKER_COUNT),
        (f"<|start_think|>\n{VARIED}\n<|end_think|>\n\nanswer<|eot_id|>more<|eot_id|>", FormatViolation.SPECIAL_TOKEN),
        (f"<|start_think|>\n{VARIED}<|start_header_id|>\n<|end_think|>\n\nanswer<|eot_id|>", FormatViolation.SPECIAL_TOKEN),
        (f"<|start_think|>{VARIED}\n<|end_think|>\n\nanswer<|eot_id|>", FormatViolation.WHITESPACE),
        (f"<|start_think|>\n{VARIED}\n<|end_think|>answer<|eot_id|>", FormatViolation.WHITESPACE),
        (f"<|start_think|>\n{VARIED}\n<|end_think|>\nanswer<|eot_id|>", FormatViolation.WHITESPACE),
        ("<|start_think|>\nbrief\n<|end_think|>\n\nanswer<|eot_id|>", FormatViolation.TOO_LITTLE_THINKING),
        ("<|start_think|>\n" + "the " * 300 + "\n<|end_think|>\n\nanswer<|eot_id|>", FormatViolation.TOO_LITTLE_THINKING),
        ("<|start_think|>" + "\n" * 300 + "<|end_think|>\n\nanswer<|eot_id|>", FormatViolation.TOO_LITTLE_THINKING),
        (f"<|start_think|>\n{VARIED}\n<|end_think|>\n\n<|eot_id|>", FormatViolation.EMPTY_ANSWER),
        (f"<|start_think|>\n{VARIED}\n<|end_think|>\n\n \n<|eot_id|>", FormatViolation.EMPTY_ANSWER),
    ],
)
def test_each_departure_from_the_format_is_a_violation(marked, violation):
    assert _violation(marked) == violation


def test_reward_needs_the_format_and_every_instruction():
    ids, kwargs = ["punctuation:no_comma", "last_word:last_word_answer"], [{}, {"last_word": "done"}]
    both = _step("p", ids, kwargs, _well_formed("The work is done"), 100)
    one = _step("p", ids, kwargs, _well_formed("The work, however, is done"), 100)
    unformatted = _step("p", ids, kwargs, "The work is done<|eot_id|>", 100)
    redrafted = _step("p", ids, kwargs, _well_formed("First, a draft\n<|end_think|>\n\nThe work is done"), 100)
    assert (both["reward"], both["metadata"]["instructions_followed"], both["metadata"]["format_ok"]) == (1.0, 2, True)
    assert (one["reward"], one["metadata"]["instructions_followed"]) == (0.0, 1)
    assert (unformatted["reward"], unformatted["metadata"]["format_violation_no_leading_start"]) == (0.0, True)
    assert (redrafted["reward"], redrafted["metadata"]["format_violation_marker_count"]) == (0.0, True)


def test_the_engine_text_has_no_markers_so_commas_in_thinking_must_not_count():
    # The regression this guards: special tokens are stripped from the text passed to step().
    out = _step("p", ["punctuation:no_comma"], [{}], _well_formed("No commas here", thinking=VARIED + " first, second, third"), 100)
    assert out["reward"] == 1.0


def test_step_without_token_ids_fails_loudly():
    spec = json.dumps({"prompt": "p", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]})
    env = skyrl_gym.make(
        "nemotron_if",
        env_config=DictConfig({"tokenizer_path": MODEL_PATH, "min_thinking_tokens": 0}),
        extras={"reward_spec": {"ground_truth": spec}},
    )
    with pytest.raises(RuntimeError):
        env.step("anything")


def _json_step(answer, min_thinking_tokens=100):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    spec = json.dumps({"prompt": "p", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]})
    env = skyrl_gym.make(
        "nemotron_if_json",
        env_config=DictConfig({"tokenizer_path": MODEL_PATH, "min_thinking_tokens": min_thinking_tokens}),
        extras={"reward_spec": {"ground_truth": spec}},
    )
    token_ids = tokenizer.encode(_well_formed(answer), add_special_tokens=False)
    env.set_rollout_evidence(_Evidence(token_ids))
    return env.step(tokenizer.decode(token_ids, skip_special_tokens=True))


def test_json_variant_checks_the_instructions_on_the_response_field_only():
    # analysis and plan contain commas; only the response string is the deliverable
    good = json.dumps({"analysis": "short, simple", "plan": "write it, then stop", "response": "No commas here"})
    bad = json.dumps({"analysis": "short", "plan": "write", "response": "One, two"})
    assert _json_step(good)["reward"] == 1.0
    assert _json_step(bad)["reward"] == 0.0


@pytest.mark.parametrize(
    "answer",
    [
        "No commas here",  # not JSON
        '```json\n{"analysis": "a", "plan": "b", "response": "No commas here"}\n```',  # fenced
        '{"analysis": "a", "plan": "b", "response": "No commas here"} Done.',  # trailing text
        '{"analysis": "a", "response": "No commas here"}',  # missing field
        '{"analysis": "a", "plan": "b", "response": "No commas here", "task_complete": true}',  # extra field
        '{"analysis": "a", "plan": "b", "response": ["No commas here"]}',  # response is not a string
        '{"analysis": "a", "plan": "b", "response": ""}',  # empty deliverable
    ],
)
def test_json_variant_rejects_anything_but_the_exact_object(answer):
    out = _json_step(answer)
    assert out["reward"] == 0.0 and out["metadata"]["format_ok"]
