# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""resolve_rl_train_data(kind='parquet') staging routing.

Local paths / HF ids pass through unchanged; an object-store URI is staged to node-local
disk (datasets.load_dataset refuses a remote URI under HF_HUB_OFFLINE=1 → OfflineModeIsEnabled).
"""

from pathlib import Path

import fsspec

from cloud.iris.rl_data import resolve_rl_train_data, resolve_rl_train_data_with_sources


def test_parquet_local_and_hf_pass_through(tmp_path):
    p = tmp_path / "train.parquet"
    p.write_bytes(b"local")
    out = resolve_rl_train_data([str(p), "allenai/RLVR-MATH"], kind="parquet", verbose=False)
    assert out == [str(p), "allenai/RLVR-MATH"]  # unchanged: no staging for local path / HF id


def test_structured_directory_source_resolves_to_materialized_parquet():
    source = {
        "kind": "directory",
        "uri": "s3://artifacts/gsm8k",
        "identity": "gsm8k@v1:abc123",
        "local_path": "/tmp/marinskyrl/data/gsm8k",
        "relative_path": "train.parquet",
    }

    resolved = resolve_rl_train_data_with_sources([source], kind="parquet", verbose=False)

    assert resolved.paths == ("/tmp/marinskyrl/data/gsm8k/train.parquet",)
    assert resolved.sources == (source,)


def test_parquet_remote_uri_staged_to_local(tmp_path):
    remote = "memory://rl-data/train.parquet"
    with fsspec.open(remote, "wb") as destination:
        destination.write(b"PARQUET-BYTES")
    out = resolve_rl_train_data(
        [remote],
        scratch_dir=str(tmp_path),
        kind="parquet",
        verbose=False,
    )
    staged = Path(out[0])
    assert staged.is_absolute() and staged.exists()
    assert staged.name == "train.parquet"
    assert "://" not in out[0]  # a local path, not the remote URI
    assert staged.read_bytes() == b"PARQUET-BYTES"


def test_remote_inputs_with_the_same_filename_keep_distinct_contents(tmp_path):
    uris = ["memory://first/train.parquet", "memory://second/train.parquet"]
    for uri, content in zip(uris, [b"FIRST", b"SECOND"], strict=True):
        with fsspec.open(uri, "wb") as destination:
            destination.write(content)
    paths = resolve_rl_train_data(uris, scratch_dir=str(tmp_path), verbose=False)
    assert paths[0] != paths[1]
    assert [Path(path).read_bytes() for path in paths] == [b"FIRST", b"SECOND"]
