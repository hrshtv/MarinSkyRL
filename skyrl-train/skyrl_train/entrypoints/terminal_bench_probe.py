"""Generation-only pass@k probe over a terminal-bench task set.

Builds exactly what the RL run builds for rollouts - the same inference engines, HTTP
inference endpoint, tokenizer, sampling params and Harbor trajectory runner, all from the
same Hydra config - and nothing else: no policy, reference model or trainer. Requests are
submitted the way `FullyAsyncRayPPOTrainer` submits them (one prompt group per
`trajectory_runner.run` call, phase "train"), with at most `RTS_PROBE_MAX_GROUPS` groups
in flight. Every finished group is appended to a JSONL file so an interrupted probe
resumes where it stopped.

Environment:
    RTS_PROBE_OUT         JSONL path for per-trajectory results (required).
    RTS_PROBE_MAX_GROUPS  prompt groups in flight (default: n_concurrent_trials // n_samples).
"""

import asyncio
import json
import os
from collections import Counter
from pathlib import Path

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import config_dir, run_ray_driver
from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request

OUT_ENV = "RTS_PROBE_OUT"
# Inference-side failures (a dead vLLM engine answers 500). The RL recipe scores unknown
# exceptions zero, which would record a false "unsolved"; such groups are not written, so a
# resumed probe runs them again.
INFRA_EXCEPTIONS = frozenset(
    {"InternalServerError", "APIConnectionError", "ServiceUnavailableError", "OSError", "DownloadVerifierDirError"}
)
MAX_GROUPS_ENV = "RTS_PROBE_MAX_GROUPS"


def _scalar_reward(reward) -> float:
    if isinstance(reward, (list, tuple)):
        return float(sum(reward))
    return float(reward)


def _completed_uids(out_path: Path, n_samples: int) -> set[str]:
    if not out_path.exists():
        return set()
    counts: Counter[str] = Counter()
    with out_path.open() as f:
        for line in f:
            counts[json.loads(line)["uid"]] += 1
    return {uid for uid, n in counts.items() if n >= n_samples}


class TerminalBenchProbeExp(TerminalBenchExp):
    async def _probe(self) -> None:
        out_path = Path(os.environ[OUT_ENV])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_samples = int(self.cfg.generator.n_samples_per_prompt)
        seats = int(self.cfg.terminal_bench_config.harbor.n_concurrent_trials)
        max_groups = int(os.environ.get(MAX_GROUPS_ENV, max(1, seats // n_samples)))
        sampling_params = get_sampling_params_for_backend(
            self.cfg.generator.backend, self.cfg.generator.sampling_params
        )

        done = _completed_uids(out_path, n_samples)
        prompts = [p for p in self.train_dataset if p["uid"] not in done]
        logger.info(
            f"[probe] {len(prompts)} prompts to run ({len(done)} already complete), "
            f"n_samples={n_samples}, seats={seats}, groups in flight={max_groups}, out={out_path}"
        )

        inference_engine_client = self.create_inference_engine_client()
        await inference_engine_client.wake_up()
        trajectory_runner = self.get_trajectory_runner(self.cfg, self.tokenizer, inference_engine_client)
        await trajectory_runner.startup()

        gate = asyncio.Semaphore(max_groups)
        write_lock = asyncio.Lock()
        finished = 0

        async def run_group(prompt: dict) -> None:
            nonlocal finished
            async with gate:
                request, _ = prepare_trajectory_request(
                    [prompt], n_samples, sampling_params, self.cfg.environment.env_class, "train", 0
                )
                batch = await trajectory_runner.run(request)
            ids = batch.get("trajectory_ids") or request["trajectory_ids"]
            exception_types = batch.get("exception_types") or [None] * len(ids)
            treatments = batch.get("error_treatments") or [None] * len(ids)
            stop_reasons = batch.get("stop_reasons") or [None] * len(ids)
            excluded = batch.get("exclude_from_baseline") or [False] * len(ids)
            rows = [
                {
                    "uid": tid.instance_id,
                    "rep": tid.repetition_id,
                    "reward": _scalar_reward(batch["rewards"][i]),
                    "exception_type": exception_types[i],
                    "error_treatment": treatments[i],
                    "stop_reason": stop_reasons[i],
                    "excluded": bool(excluded[i]),
                    "response_tokens": len(batch["response_ids"][i]),
                }
                for i, tid in enumerate(ids)
            ]
            if any(row["exception_type"] in INFRA_EXCEPTIONS for row in rows):
                logger.warning(f"[probe] group {prompt['uid']} hit an inference failure; not recorded, will rerun on resume")
                return
            async with write_lock:
                with out_path.open("a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                finished += 1
                if finished % 50 == 0:
                    logger.info(f"[probe] {finished}/{len(prompts)} groups finished")

        try:
            results = await asyncio.gather(*(run_group(p) for p in prompts), return_exceptions=True)
        finally:
            await trajectory_runner.shutdown()
        failures = [r for r in results if isinstance(r, BaseException)]
        logger.info(f"[probe] done: {finished} groups written, {len(failures)} groups raised")
        if failures:
            logger.error(f"[probe] first failure: {failures[0]!r}")

    def run(self):
        asyncio.run(self._probe())


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    TerminalBenchProbeExp(cfg).run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.HARBOR, failure_message="Probe failed")


if __name__ == "__main__":
    main()
