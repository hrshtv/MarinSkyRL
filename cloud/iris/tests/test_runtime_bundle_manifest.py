"""Regression tests for the runtime bundle file manifest."""

from __future__ import annotations

import sys
import json
import os
import shutil
import subprocess
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from cloud.iris.runtime_bundle import BUNDLE_FILE_MANIFEST, read_manifest_paths  # noqa: E402


def test_every_runtime_module_is_in_the_runtime_bundle(tmp_path: Path) -> None:
    """Ship both Python packages imported by the task runtime."""
    manifest = set(read_manifest_paths(_REPOSITORY_ROOT))
    on_disk = {
        path.relative_to(_REPOSITORY_ROOT).as_posix()
        for package in ("cloud/iris", "marinskyrl")
        for path in (_REPOSITORY_ROOT / package).rglob("*.py")
        if "tests" not in path.relative_to(_REPOSITORY_ROOT / package).parts
    }
    missing = on_disk - manifest
    assert not missing, (
        f"runtime modules not shipped to Iris tasks: {sorted(missing)}. "
        f"Add them to {BUNDLE_FILE_MANIFEST} or add an explicit exclusion in this test."
    )
    for name in manifest:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPOSITORY_ROOT / name, destination)
    program = """
import json
from pathlib import Path
import marinskyrl.recipe_schema as schema
import marinskyrl.speculative_decoding
recipe = schema.SkyRLRecipe(context_budget=schema.ContextBudget(request_window_tokens=256, max_new_tokens_per_turn=64, max_turns=1))
print(json.dumps({"source": schema.__file__, "document": recipe.to_skyrl()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert Path(output["source"]).resolve().is_relative_to(tmp_path)
    assert output["document"]["context_budget"]["max_turns"] == 1
