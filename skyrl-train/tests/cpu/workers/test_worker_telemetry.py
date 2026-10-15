import json
from contextlib import suppress
import os
import select
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import ray
from omegaconf import OmegaConf

from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.workers.worker import PPORayActorGroup, Worker
from tests.process_gang import launch_process_gang, run_after_rank_readiness, signal_rank_ready_and_wait_for_start
from ray.util.placement_group import placement_group, remove_placement_group
from skyrl_train.utils.algorithm_registry import sync_registries

_WORKER_CONFIG = OmegaConf.create(
    {
        "trainer": {
            "progress": {
                "mode": "tqdm",
                "min_interval_seconds": 0.5,
                "heartbeat_seconds": 15,
                "percent_step": 5,
                "count_step": 1000,
            },
            "algorithm": {"batch_invariant": False},
        }
    }
)


@pytest.fixture(autouse=True)
def _isolated_rank_environment(monkeypatch):
    # Register absent variables for restoration after worker construction creates them.
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _build_worker() -> Worker:
    return Worker(
        cfg=_WORKER_CONFIG,
        world_size=1,
        rank=0,
        local_rank=0,
        sequence_parallel_size=1,
    )


@pytest.mark.parametrize("endpoint", [True, False])
def test_a_worker_reports_its_lifecycle_only_with_an_endpoint(telemetry_endpoint, monkeypatch, endpoint):
    if not endpoint:
        monkeypatch.delenv("SKYRL_TELEMETRY_ENDPOINT")
    worker = _build_worker()
    worker.close_telemetry()

    delivered = [(row["name"], row["attributes"]["role"], row["resource"]["run_id"]) for row in telemetry_endpoint.rows]
    assert delivered == (
        [("lifecycle", "worker", "telemetry-test"), ("terminal", "worker", "telemetry-test")] if endpoint else []
    )


def test_the_actor_group_drains_every_worker_before_killing_it(monkeypatch):
    """ray.kill runs no atexit handler in the actor, so the drain must be requested and awaited first."""
    order: list[str] = []
    drained = object()

    class Handle:
        def __init__(self, name):
            self.name = name
            self.close_telemetry = SimpleNamespace(remote=self._close)

        def _close(self):
            order.append(f"drain {self.name}")
            return drained

    handles = [Handle("rank0"), Handle("rank1")]
    monkeypatch.setattr(
        ray, "wait", lambda refs, *, num_returns, timeout: order.append(f"wait {num_returns} {timeout}")
    )
    monkeypatch.setattr(ray, "kill", lambda actor, *, no_restart: order.append(f"kill {actor.name} {no_restart}"))
    group = PPORayActorGroup.__new__(PPORayActorGroup)
    group._actor_handlers = handles

    group.kill_actors()

    assert order == ["drain rank0", "drain rank1", "wait 2 5.0", "kill rank0 True", "kill rank1 True"]


class _ConstructorBarrierWorker(Worker):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        directory = Path(cfg.constructor_records)
        (directory / f"rank-{self._rank}.json").write_text(json.dumps({"pid": os.getpid()}))
        barrier = os.open(directory / "barrier", os.O_RDWR | os.O_NONBLOCK)
        try:
            if self._rank == 0:
                ready, _, _ = select.select([barrier], [], [], 30)
                if not ready or os.read(barrier, 1) != b"r":
                    raise RuntimeError("rank 1 did not release the constructor barrier")
            else:
                os.write(barrier, b"r")
        finally:
            os.close(barrier)

    def init_worker_process_group(self, master_addr, master_port):
        self.mesh_rank = MeshRank(
            dp=self._rank, sp=0, tp=0, pp=0, world_size=self._world_size, dp_size=self._world_size, pp_size=1
        )


def _construct_policy_ranks(directory: Path) -> None:
    group = None
    pg = None
    try:
        # Policy bundles leave CPUs available for the algorithm registry actors.
        ray.init(num_cpus=4, num_gpus=2, include_dashboard=False)
        sync_registries()
        pg = placement_group([{"CPU": 1, "GPU": 1}, {"CPU": 1, "GPU": 1}])
        ray.get(pg.ready(), timeout=30)
        signal_rank_ready_and_wait_for_start(directory, 0)
        config = OmegaConf.merge(_WORKER_CONFIG, {"constructor_records": str(directory)})
        group = PPORayActorGroup(config, 1, 2, ray.remote(_ConstructorBarrierWorker), pg=pg)
        (directory / "result.json").write_text(json.dumps({"ranks": [info.rank.dp for info in group.actor_infos]}))
    finally:
        if group is not None:
            group.kill_actors()
        if pg is not None:
            remove_placement_group(pg)
        ray.shutdown()


@pytest.mark.slow
def test_all_policy_ranks_construct_before_rank_zero_rendezvous(tmp_path):
    """Rank 1 must release rank 0 while rank 0's constructor is running."""
    os.mkfifo(tmp_path / "barrier")
    # Preserve an early rank-1 write until rank 0 opens the FIFO.
    barrier_keepalive = os.open(tmp_path / "barrier", os.O_RDWR | os.O_NONBLOCK)
    try:
        with launch_process_gang(
            command=[sys.executable, "-m", "tests.cpu.workers.test_worker_telemetry", str(tmp_path)],
            working_directory=Path.cwd(),
            environment=os.environ.copy(),
            control_directory=tmp_path,
            log_path=tmp_path / "driver.log",
            reap_timeout_seconds=5,
            process_description="policy-construction driver",
        ) as gang:
            result = run_after_rank_readiness(gang, expected_ranks=1, setup_timeout_seconds=120, run_timeout_seconds=45)
            assert result.returncode == 0, result.output
            assert json.loads((tmp_path / "result.json").read_text()) == {"ranks": [0, 1]}
        for record in tmp_path.glob("rank-*.json"):
            pid = json.loads(record.read_text())["pid"]
            assert not psutil.pid_exists(pid), f"policy rank process {pid} survived cleanup"
    finally:
        os.close(barrier_keepalive)
        for record in tmp_path.glob("rank-*.json"):
            with suppress(psutil.NoSuchProcess):
                process = psutil.Process(json.loads(record.read_text())["pid"])
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    from tests.cpu.workers.test_worker_telemetry import _construct_policy_ranks as run_driver

    run_driver(Path(sys.argv[1]))
