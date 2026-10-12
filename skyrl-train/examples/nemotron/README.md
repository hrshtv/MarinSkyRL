# Nemotron task preparation

Prepare source rows before training. From the repository root:

```bash
uv run python -m marinskyrl.nemotron_preparation \
  --input /path/to/source.parquet --output /path/to/tasks.parquet \
  --source my_blend --config /path/to/graders.json \
  --environments skyrl-train/examples/nemotron/environments.json \
  --terminal-tasks /path/to/harbor-tasks.parquet
```

The environment file declares the Python tool image and Lean workspace.
The grader file contains verifier parameters and configured judge identities.
Keep credentials outside that file.
Omit `--terminal-tasks` when the source contains no SWE rows.
The output contains bare TaskSpecs with separate training labels.

Prepare Harbor packages with `marinskyrl.harbor_preparation` first.
Merge its execution settings into the run config at `environment.task_sessions.sources`.
Nemotron SWE tasks retain the Harbor source used by those settings.
Other tasks use the selected `nemotron_ultra` session and the run's machine settings.
The supplied Iris configs are geometry templates. Their launch inputs must supply prepared TaskSpec files.
