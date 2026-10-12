"""Storage fields for prepared TaskSpecs."""

from dataclasses import dataclass

from taskcompendium.models import TaskSpec

TASK_SPEC_COLUMN = "task_spec"


@dataclass(frozen=True)
class TaskDatasetRow:
    task: TaskSpec
    teacher_route: str | None = None
    metric_tags: tuple[str, ...] = ()
