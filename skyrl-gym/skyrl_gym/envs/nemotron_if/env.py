"""Single-turn environment for nvidia/Nemotron-RL-instruction_following.

Each row pairs a user prompt with one to three verifiable instructions (Open-Instruct IFEvalG ids
plus kwargs). The reward is 1.0 when the response has the exact format below and its answer follows
every instruction (the strict prompt-level criterion of IFEval), and 0.0 otherwise.

Format. The chat template renders an assistant turn as `content<|eot_id|>`, and this model writes
thinking content as one block, so a response must be exactly

    <|start_think|>\n THINKING <|end_think|>\n\n ANSWER <|eot_id|>

with one start marker (the first token), one end marker, no other special token anywhere, enough
THINKING, and a non-empty ANSWER. Anything looser is exploitable: when the answer was "the text after
the last end marker", the policy learned within two updates to draft, close, redraft and close again
up to the generation limit, because every redraft was free.

The markers are SPECIAL tokens, and the inference engine strips special tokens from the response text
it hands to `step`, so the format is checked on the response token ids delivered through
`set_rollout_evidence`, and the answer tokens are decoded with the run's tokenizer.
"""

import functools
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import langdetect
from omegaconf import DictConfig
from transformers import AutoTokenizer

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.nemotron_if.ifevalg import instructions_registry

THINK_START = "<|start_think|>"
THINK_END = "<|end_think|>"
END_OF_TURN = "<|eot_id|>"
AFTER_THINK_START = "\n"
AFTER_THINK_END = "\n\n"
# A quarter of the required thinking tokens must be distinct, so repeated filler does not count.
DISTINCT_TOKEN_DIVISOR = 4

# langdetect samples internally; a fixed seed makes the language checks reproducible.
langdetect.DetectorFactory.seed = 0


class FormatViolation(StrEnum):
    NOT_ENDED = "not_ended"  # the last token is not <|eot_id|>: cut off at the limit, or another terminator
    NO_LEADING_START = "no_leading_start"
    MARKER_COUNT = "marker_count"  # anything but one start marker and one end marker
    SPECIAL_TOKEN = "special_token"  # a special token inside the thinking or the answer
    WHITESPACE = "whitespace"  # the markers are not followed by the newline / blank line the model writes
    TOO_LITTLE_THINKING = "too_little_thinking"
    EMPTY_ANSWER = "empty_answer"


@dataclass(frozen=True)
class SpecialTokens:
    start: int
    end: int
    end_of_turn: int
    all_ids: frozenset[int]


@functools.cache
def _tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = {}
    for token in (THINK_START, THINK_END, END_OF_TURN):
        ids[token] = tokenizer.convert_tokens_to_ids(token)
        if tokenizer.decode([ids[token]]) != token:
            raise ValueError(f"{path}: tokenizer has no single-token {token}")
    all_ids = frozenset(i for i, added in tokenizer.added_tokens_decoder.items() if added.special)
    return tokenizer, SpecialTokens(ids[THINK_START], ids[THINK_END], ids[END_OF_TURN], all_ids)


@dataclass(frozen=True)
class ParsedResponse:
    violation: FormatViolation | None
    answer: str
    thinking_tokens: int
    distinct_thinking_tokens: int


def parse_response(response_token_ids, tokenizer_path: str, min_thinking_tokens: int) -> ParsedResponse:
    """Check the exact response format on token ids and return the answer, or the first violation."""
    tokenizer, special = _tokenizer(tokenizer_path)
    ids = list(response_token_ids)

    def violated(violation: FormatViolation, thinking: list[int] = ()) -> ParsedResponse:
        return ParsedResponse(violation, "", len(thinking), len(set(thinking)))

    if not ids or ids[-1] != special.end_of_turn:
        return violated(FormatViolation.NOT_ENDED)
    if ids[0] != special.start:
        return violated(FormatViolation.NO_LEADING_START)
    if ids.count(special.start) != 1 or ids.count(special.end) != 1:
        return violated(FormatViolation.MARKER_COUNT)
    boundary = ids.index(special.end)
    thinking, answer_ids = ids[1:boundary], ids[boundary + 1 : -1]
    if any(token in special.all_ids for token in thinking + answer_ids):
        return violated(FormatViolation.SPECIAL_TOKEN, thinking)
    answer_text = tokenizer.decode(answer_ids)
    if not tokenizer.decode(thinking[:2]).startswith(AFTER_THINK_START) or not answer_text.startswith(AFTER_THINK_END):
        return violated(FormatViolation.WHITESPACE, thinking)
    if len(thinking) < min_thinking_tokens or len(set(thinking)) < min_thinking_tokens // DISTINCT_TOKEN_DIVISOR:
        return violated(FormatViolation.TOO_LITTLE_THINKING, thinking)
    if not answer_text.strip():
        return violated(FormatViolation.EMPTY_ANSWER, thinking)
    return ParsedResponse(None, answer_text.strip(), len(thinking), len(set(thinking)))


def _normalized_kwargs(instruction_id: str, kwargs: dict | None) -> dict:
    """Drop unset arguments and undo one serialization quirk of the Nemotron dataset.

    Rows of count:count_increment_word store keyword1 / keyword2 as one-element lists (the
    generator's random.sample output), while the checker expects the word itself.
    """
    cleaned = {key: value for key, value in (kwargs or {}).items() if value is not None}
    if instruction_id == "count:count_increment_word":
        for key in ("keyword1", "keyword2"):
            value = cleaned.get(key)
            if isinstance(value, list):
                if len(value) != 1:
                    raise ValueError(f"{instruction_id}: expected one word for {key}, got {value!r}")
                cleaned[key] = value[0]
    return cleaned


def followed_instructions(answer: str, prompt: str, instruction_ids: list[str], kwargs_list: list[dict]) -> list[bool]:
    """Check each instruction the way Open-Instruct's strict IFEval verifier does."""
    results = []
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        instruction.build_description(**_normalized_kwargs(instruction_id, kwargs))
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        results.append(bool(answer.strip()) and bool(instruction.check_following(answer)))
    return results


JSON_ANSWER_FIELDS = frozenset({"analysis", "plan", "response"})


def json_response_field(answer: str) -> str | None:
    """The `response` string of an answer that is exactly one JSON object with the three string fields."""
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != JSON_ANSWER_FIELDS:
        return None
    if not all(isinstance(value, str) for value in parsed.values()):
        return None
    return parsed["response"]


class NemotronIFEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}
        assert "reward_spec" in extras, "reward_spec field is required"
        spec = json.loads(extras["reward_spec"]["ground_truth"])
        self.prompt = spec["prompt"]
        self.instruction_ids = spec["instruction_id_list"]
        self.kwargs_list = spec["kwargs"]
        self.tokenizer_path = env_config.get("tokenizer_path")
        if not self.tokenizer_path:
            raise ValueError("environment.skyrl_gym.nemotron_if.tokenizer_path is required")
        # An answer-only reward taught the policy to drop its thinking within 20 steps, so the format
        # requires this many thinking tokens.
        self.min_thinking_tokens = int(env_config.get("min_thinking_tokens"))
        self.response_token_ids = None

    def _deliverable(self, answer: str) -> str | None:
        """The text the instructions are checked on; None when the answer does not carry one."""
        return answer

    def set_rollout_evidence(self, evidence) -> None:
        self.response_token_ids = evidence.response_token_ids

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.response_token_ids is None:
            raise RuntimeError("nemotron_if needs the response token ids (set_rollout_evidence) to find the answer")
        parsed = parse_response(self.response_token_ids, self.tokenizer_path, self.min_thinking_tokens)
        deliverable = None if parsed.violation is not None else self._deliverable(parsed.answer)
        followed = followed_instructions(deliverable or "", self.prompt, self.instruction_ids, self.kwargs_list)
        reward = 1.0 if deliverable is not None and all(followed) else 0.0
        metadata = {
            "instructions_followed": sum(followed),
            "instructions_total": len(followed),
            "thinking_tokens": parsed.thinking_tokens,
            "distinct_thinking_tokens": parsed.distinct_thinking_tokens,
            "format_ok": parsed.violation is None,
            "deliverable_ok": deliverable is not None,
            **{f"format_violation_{v.value}": parsed.violation is v for v in FormatViolation},
        }
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=metadata)


class NemotronIFJsonEnv(NemotronIFEnv):
    """The same task under a prompt shaped like the terminal harness's.

    This model writes one clean thinking block under the harness prompt (a role preamble, a JSON
    response schema, then the task) and stutters its markers under a bare chat request. Here the
    answer must be exactly one JSON object {"analysis", "plan", "response"} of strings, as the harness
    demands exactly one JSON object, and the instructions are checked on the `response` string.
    """

    def _deliverable(self, answer: str) -> str | None:
        return json_response_field(answer)
