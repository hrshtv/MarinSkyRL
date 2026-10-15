"""Run a custom driver command under the production Ray gang supervisor (after Romain's hero_task_runtime.py)."""

import argparse
from pathlib import Path

from cloud.iris import task_runtime as runtime
from cloud.iris.ray_storage import RaySpillBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendezvous-dir", required=True)
    parser.add_argument("--ray-log-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    args.rendezvous_timeout = 7200
    args.cluster_join_timeout = 7200
    args.driver_liveness_timeout = 3600
    args.ray_port = 6379
    args.ray_spill_dir = "/tmp/route-ray-spill"
    args.ray_spill_backend = RaySpillBackend.LOCAL
    runtime._pin_boto3_s3_addressing_style()
    interface = runtime.pin_socket_ifname()
    runtime.ensure_fr_dump_dir()
    runtime.training_driver_command = lambda _config_path: command
    if runtime._rank() == 0:
        return runtime.run_head(args, Path("/tmp/route-task-driver"), interface)
    return runtime.run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
