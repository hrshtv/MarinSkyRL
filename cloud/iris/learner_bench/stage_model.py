"""Copy an object-store HF checkpoint (weights, config and tokenizer) to local disk."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

SKIPPED_PREFIXES = (".export-progress-",)
TRANSFER = TransferConfig(multipart_chunksize=64 * 2**20, max_concurrency=8)


def stage(source: str, destination: Path, workers: int) -> None:
    parsed = urlsplit(source)
    bucket, prefix = parsed.netloc, parsed.path.strip("/")
    client = boto3.client("s3", config=Config(s3={"addressing_style": "virtual"}, max_pool_connections=workers * 8))
    objects = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            relative = obj["Key"][len(prefix) + 1 :]
            if not relative.startswith(SKIPPED_PREFIXES):
                objects.append((relative, obj))
    destination.mkdir(parents=True, exist_ok=True)

    def copy(item) -> None:
        relative, obj = item
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        client.download_file(bucket, obj["Key"], str(path), Config=TRANSFER)
        if path.stat().st_size != obj["Size"]:
            raise ValueError(f"{relative}: staged {path.stat().st_size} bytes, object has {obj['Size']}")
        print(json.dumps({"staged": relative, "bytes": obj["Size"], "seconds": time.monotonic() - started}), flush=True)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(copy, objects))
    total = sum(obj["Size"] for _, obj in objects)
    print(json.dumps({"staged_bytes": total, "seconds": time.monotonic() - started}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    stage(args.source, args.destination, args.workers)


if __name__ == "__main__":
    main()
