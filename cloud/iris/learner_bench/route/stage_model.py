"""Copy a model export from object storage to node-local disk once per node, with bounded concurrency."""

import argparse
import shutil
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

MODEL_FILE_SUFFIXES = (".safetensors", ".json")
SKIP_PREFIX = ".export-progress"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--min-free-bytes", type=int, required=True)
    parser.add_argument("--parallel-files", type=int, default=8)
    parser.add_argument("--parts-per-file", type=int, default=4)
    args = parser.parse_args()
    node = socket.gethostname()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    print(f"STAGE_DISK node={node} dest={dest} free_bytes={free}", flush=True)
    if free < args.min_free_bytes:
        raise SystemExit(f"STAGE_ABORT node={node}: {free} free bytes < required {args.min_free_bytes}")

    parsed = urlparse(args.source)
    bucket, prefix = parsed.netloc, parsed.path.strip("/") + "/"
    client = boto3.client("s3", config=Config(s3={"addressing_style": "virtual"}, max_pool_connections=64))
    objects = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            name = item["Key"][len(prefix) :]
            if "/" in name or name.startswith(SKIP_PREFIX) or not name.endswith(MODEL_FILE_SUFFIXES):
                continue
            objects.append((name, item["Size"]))
    total = sum(size for _, size in objects)
    print(f"STAGE_START node={node} files={len(objects)} bytes={total}", flush=True)
    if total > free:
        raise SystemExit(f"STAGE_ABORT node={node}: model {total} bytes > free {free}")
    transfer = TransferConfig(
        max_concurrency=args.parts_per_file, multipart_chunksize=64 * 1024 * 1024, use_threads=True
    )

    def fetch(entry: tuple[str, int]) -> None:
        name, size = entry
        target = dest / name
        client.download_file(bucket, prefix + name, str(target), Config=transfer)
        if target.stat().st_size != size:
            raise RuntimeError(f"{name}: local size {target.stat().st_size} != {size}")

    started = time.perf_counter()
    with ThreadPoolExecutor(args.parallel_files) as pool:
        list(pool.map(fetch, sorted(objects, key=lambda entry: -entry[1])))
    seconds = time.perf_counter() - started
    print(
        f"STAGE_DONE node={node} seconds={seconds:.1f} bytes={total} gbps={total / seconds / 1e9:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
