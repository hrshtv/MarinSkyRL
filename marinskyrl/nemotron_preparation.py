"""Prepare mixed Nemotron rows as portable tasks before rollout execution."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from skyrl_gym.source_task import session_parameters
from taskcompendium.models import EnvironmentRequirements, TaskSpec

from marinskyrl.harbor_preparation import HARBOR_ID_PREFIX
from marinskyrl.task_preparation import TASK_SPEC_COLUMN, PARQUET_BATCH_SIZE, source_row_task


def terminal_task_index(path: Path) -> dict[str, TaskSpec]:
    """Index lowered Harbor tasks by their package and source instance identities."""
    result = {}
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=[TASK_SPEC_COLUMN]):
            for row in batch.to_pylist():
                record = TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])
                identifiers = {
                    record.id.casefold(),
                    *(tag.removeprefix(HARBOR_ID_PREFIX) for tag in record.tags if tag.startswith(HARBOR_ID_PREFIX)),
                }
                for identifier in identifiers:
                    if identifier in result and result[identifier] != record:
                        raise ValueError(f"Duplicate terminal-bench task ID {identifier!r}")
                    result[identifier] = record
    return result


def resolve_terminal_task(lowered: TaskSpec, terminals: Mapping[str, TaskSpec]) -> TaskSpec:
    """Resolve a terminal source row before rollout execution."""
    task = lowered
    parameters = session_parameters(task)
    extras = parameters.extras
    ultra = (extras.get("extra_info") or {}).get("nemotron_ultra")
    if ultra is None:
        return lowered
    if not all(isinstance(ultra.get(key), str) and ultra[key] for key in ("blend", "agent")):
        raise ValueError("Nemotron Ultra tasks require non-empty blend and agent fields")
    if ultra.get("route") != "terminal_bench":
        return lowered
    identifier = ultra.get("terminal_bench_instance_id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("A terminal-bench row must provide terminal_bench_instance_id")
    if identifier.casefold() not in terminals:
        raise ValueError(f"Terminal-bench task {identifier!r} is absent from the configured task data")
    terminal = terminals[identifier.casefold()]
    return terminal.model_copy(
        update={
            "id": task.id,
            "source": task.source,
        }
    )


def nemotron_tasks(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_name: str,
    config: Mapping[str, Any],
    environments: Mapping[str, EnvironmentRequirements],
    terminals: Mapping[str, TaskSpec],
) -> Iterable[TaskSpec]:
    """Resolve mixed source rows before the trainer receives task records."""
    for index, row in enumerate(rows):
        agent = row["extra_info"]["nemotron_ultra"]["agent"]
        task = source_row_task(row, index, source_name=source_name, config=config, environment=environments.get(agent))
        yield resolve_terminal_task(task, terminals)
