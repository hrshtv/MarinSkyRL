"""Grade PivotRL rows with the verifyit graders that measured their pass rates.

PivotRL selects pivots by how often a frozen policy passes them under one component of these
verdicts, so training rewards that same component. Every component is reported in the metrics.
"""

from collections.abc import Mapping
from typing import Any

from verifyit.adapters.nemotron_pivot import grade_terminus, grade_tool_call
from verifyit.grade import InvalidTask, Reward

SWE_PIVOT_AGENT = "swe_pivot_single_step_tool_use_with_argument_comparison_agent"
TERMINUS_AGENT = "terminus_judge_string_only_simple_agent"
PIVOT_COMPONENTS = {
    SWE_PIVOT_AGENT: frozenset({"tool_name", "nemo", "exact"}),
    TERMINUS_AGENT: frozenset({"schema_completion", "exact_commands", "string_90"}),
}


def pivot_component(setting: str | Mapping[str, str] | None, agent: str) -> str | None:
    """The component ``agent``'s pivot rows train on.

    ``setting`` is one component name, or one per agent for a blend with SWE and Terminus rows, whose
    components are disjoint. Rows of other agents ignore it.
    """
    if isinstance(setting, Mapping):
        unknown = set(setting) - set(PIVOT_COMPONENTS)
        if unknown:
            raise ValueError(f"pivot_reward names agents without pivot components: {sorted(unknown)}")
        setting = setting.get(agent)
    if agent in PIVOT_COMPONENTS and setting not in {None, *PIVOT_COMPONENTS[agent]}:
        raise ValueError(f"{agent} has no pivot component {setting!r}")
    return setting


def _pivot_verdict(agent: str, record: dict[str, Any], action: str, message: dict[str, Any]) -> Reward:
    if agent == SWE_PIVOT_AGENT:
        return grade_tool_call(record["expected_action"], message)
    return grade_terminus(record["expected_answer"], action)


def grade_pivot(
    agent: str, record: dict[str, Any], action: str, message: dict[str, Any], component: str
) -> tuple[float, dict[str, Any]]:
    """Reward one component of the row's pivot verdict and report every component."""
    try:
        verdict = _pivot_verdict(agent, record, action, message)
    except InvalidTask as error:
        raise ValueError(f"malformed {agent} pivot row: {error}") from error
    components = verdict.detail["components"]
    details = {f"pivot_{name}": score for name, score in components.items()}
    return components[component], {"pivot_reason": verdict.detail["reason"], **details}
