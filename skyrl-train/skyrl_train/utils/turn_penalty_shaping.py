"""Per-turn behaviour penalties written onto the token-level shaping channel.

The outcome reward gives every token of a trajectory the same advantage. This module adds a
negative advantage to the tokens of two kinds of wasted turn, so the credit lands on the turn
that caused the waste instead of being spread over the whole episode:

* an *invalid* turn: the agent's reply did not parse, so the harness answered with its
  parse-error prompt and no command ran;
* a *repeated* turn: the reply sends exactly the same non-empty keystrokes as the previous turn.

Turns are located through ``loss_mask``: assistant-generated tokens are the only trainable
tokens, and observations always separate two assistant turns, so the k-th contiguous run of
ones is the k-th assistant message. No re-tokenization is involved.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

PARSE_ERROR_PREFIX = "Previous response had parsing errors:"
_KEYSTROKES_RE = re.compile(r'"keystrokes"\s*:\s*("(?:[^"\\]|\\.)*")')


@dataclass(frozen=True)
class TurnPenaltyResult:
    shaping: List[float]
    assistant_turns: int
    invalid_turns: int
    repeated_turns: int
    aligned: bool


def trainable_runs(loss_mask: Sequence[int]) -> List[tuple[int, int]]:
    """Return the ``[start, end)`` bounds of each contiguous run of ones."""
    runs: List[tuple[int, int]] = []
    start = None
    for index, value in enumerate(loss_mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(loss_mask)))
    return runs


def _content_text(message: Dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def turn_keystrokes(assistant_text: str) -> tuple[str, ...]:
    """Return the keystroke strings a terminus JSON reply sends, in order."""
    return tuple(json.loads(raw) for raw in _KEYSTROKES_RE.findall(assistant_text))


def compute_turn_penalty_shaping(
    response_messages: Sequence[Dict[str, Any]],
    loss_mask: Sequence[int],
    *,
    invalid_turn_penalty: float,
    repeated_turn_penalty: float,
) -> TurnPenaltyResult:
    """Build the per-token penalty vector for one trajectory.

    Args:
        response_messages: the conversation after the first prompt, as the trainer tokenizes it.
        loss_mask: one entry per response token, 1 on assistant-generated tokens.
        invalid_turn_penalty: magnitude subtracted from every token of an unparsed turn.
        repeated_turn_penalty: magnitude subtracted from every token of a repeated turn.

    Returns:
        The shaping vector (same length as ``loss_mask``) and the turn counts. When the mask
        has more trainable runs than there are assistant messages the mapping is unknown, so
        the vector stays zero and ``aligned`` is False. Fewer runs than messages means the tail
        was cut; the leading turns still line up and are penalized.
    """
    if invalid_turn_penalty < 0 or repeated_turn_penalty < 0:
        raise ValueError("turn penalties are magnitudes and must be non-negative")

    shaping = [0.0] * len(loss_mask)
    runs = trainable_runs(loss_mask)
    turns = [i for i, message in enumerate(response_messages) if message.get("role") == "assistant"]
    if len(runs) > len(turns):
        return TurnPenaltyResult(shaping, len(turns), 0, 0, aligned=False)

    invalid = repeated = 0
    previous_keystrokes: tuple[str, ...] = ()
    for (start, end), message_index in zip(runs, turns):
        following = response_messages[message_index + 1] if message_index + 1 < len(response_messages) else None
        is_invalid = (
            following is not None
            and following.get("role") == "user"
            and _content_text(following).lstrip().startswith(PARSE_ERROR_PREFIX)
        )
        penalty = 0.0
        if is_invalid:
            invalid += 1
            penalty = invalid_turn_penalty
            keystrokes: tuple[str, ...] = ()
        else:
            keystrokes = turn_keystrokes(_content_text(response_messages[message_index]))
            if keystrokes and any(k.strip() for k in keystrokes) and keystrokes == previous_keystrokes:
                repeated += 1
                penalty = repeated_turn_penalty
        previous_keystrokes = keystrokes
        if penalty:
            shaping[start:end] = [-penalty] * (end - start)
    return TurnPenaltyResult(shaping, len(turns), invalid, repeated, aligned=True)
