"""Run a learner-benchmark driver command on a one- or multi-node Iris GPU gang.

Every node runs this after its local setup. The production task supervisor (``cloud.iris.task_runtime``) starts one
Ray cluster across the gang; rank 0 runs the given command with ``RAY_ADDRESS`` set and the other ranks contribute
their GPUs until it finishes.
"""

import argparse
import os
from pathlib import Path

from cloud.iris import task_runtime as runtime
from cloud.iris.ray_storage import RaySpillBackend
from marinskyrl.environment_contract import TrainingType

RAY_PORT = 6379
RAY_SPILL_DIR = "/tmp/lb-ray-spill"
DRIVER_DIR = Path("/tmp/lb-task-driver")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rendezvous-dir", required=True)
    parser.add_argument("--ray-log-dir", required=True)
    parser.add_argument("--rendezvous-timeout", type=int, required=True)
    parser.add_argument("--cluster-join-timeout", type=int, required=True)
    parser.add_argument("--driver-liveness-timeout", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a driver command is required")
    args.ray_port = RAY_PORT
    args.ray_spill_dir = RAY_SPILL_DIR
    args.ray_spill_backend = RaySpillBackend.LOCAL
    runtime._pin_boto3_s3_addressing_style()
    interface = runtime.pin_socket_ifname()
    runtime.export_telemetry_environment(os.environ["LB_RUN_NAME"], TrainingType.SYNC)
    runtime.ensure_fr_dump_dir()
    # Replace only the driver command; gang lifecycle, log capture and cleanup stay the supervisor's.
    runtime.training_driver_command = lambda _config_path: command
    if runtime._rank() == 0:
        return runtime.run_head(args, DRIVER_DIR, interface)
    return runtime.run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
