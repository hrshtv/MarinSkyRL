"""Generation-only pass@k probe over a single-turn SkyRL-Gym dataset.

Builds the inference engines, tokenizer, sampling params and SkyRL-Gym trajectory runner from the
same Hydra config as a training run, and nothing else: no policy, reference model or trainer.
Prompts are submitted in chunks the way the trainer submits a batch (phase "train"), with a bounded
number of chunks in flight. Every finished chunk is appended to a JSONL file, one row per sample,
so an interrupted probe resumes where it stopped. The response token ids of every sample go to a
sibling `.samples.jsonl`.

Environment:
    GYM_PROBE_OUT       JSONL path for per-sample results (required).
    GYM_PROBE_CHUNK     prompts per trajectory_runner.run call (default 64).
    GYM_PROBE_INFLIGHT  chunks in flight (default 8).
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
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request

OUT_ENV = "GYM_PROBE_OUT"
CHUNK_ENV = "GYM_PROBE_CHUNK"
INFLIGHT_ENV = "GYM_PROBE_INFLIGHT"


def _scalar_reward(reward) -> float:
    if isinstance(reward, (list, tuple)):
        return float(sum(reward))
    return float(reward)


def _completed_ids(out_path: Path, n_samples: int) -> set[int]:
    if not out_path.exists():
        return set()
    counts: Counter[int] = Counter()
    with out_path.open() as f:
        for line in f:
            counts[json.loads(line)["id"]] += 1
    return {row_id for row_id, n in counts.items() if n >= n_samples}


class SkyRLGymProbeExp(BasePPOExp):
    def get_eval_dataset(self):
        """The probe reads the training split only."""
        return None

    async def _probe(self) -> None:
        out_path = Path(os.environ[OUT_ENV])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_samples = int(self.cfg.generator.n_samples_per_prompt)
        chunk_size = int(os.environ.get(CHUNK_ENV, 64))
        inflight = int(os.environ.get(INFLIGHT_ENV, 8))
        sampling_params = get_sampling_params_for_backend(
            self.cfg.generator.backend, self.cfg.generator.sampling_params
        )

        done = _completed_ids(out_path, n_samples)
        prompts = []
        for index in range(len(self.train_dataset)):
            (item,) = self.train_dataset.collate_fn([self.train_dataset[index]])
            if int(item["env_extras"]["extra_info"]["id"]) not in done:
                prompts.append(item)
        chunks = [prompts[i : i + chunk_size] for i in range(0, len(prompts), chunk_size)]
        logger.info(
            f"[probe] {len(prompts)} prompts to run ({len(done)} already complete), n_samples={n_samples}, "
            f"chunk={chunk_size}, chunks in flight={inflight}, out={out_path}"
        )

        inference_engine_client = self.create_inference_engine_client()
        await inference_engine_client.wake_up()
        trajectory_runner = self.get_trajectory_runner(self.cfg, self.tokenizer, inference_engine_client)
        await trajectory_runner.startup()

        gate = asyncio.Semaphore(inflight)
        write_lock = asyncio.Lock()
        finished = 0

        async def run_chunk(chunk: list[dict]) -> None:
            nonlocal finished
            id_of_uid = {item["uid"]: int(item["env_extras"]["extra_info"]["id"]) for item in chunk}
            async with gate:
                request, _ = prepare_trajectory_request(
                    chunk, n_samples, sampling_params, self.cfg.environment.env_class, "train", 0
                )
                batch = await trajectory_runner.run(request, disable_tqdm=True)
            ids = batch.get("trajectory_ids") or request["trajectory_ids"]
            stop_reasons = batch.get("stop_reasons") or [None] * len(ids)
            rows = [
                {
                    "id": id_of_uid[tid.instance_id],
                    "rep": tid.repetition_id,
                    "reward": _scalar_reward(batch["rewards"][i]),
                    "stop_reason": stop_reasons[i],
                    "response_tokens": len(batch["response_ids"][i]),
                }
                for i, tid in enumerate(ids)
            ]
            async with write_lock:
                with out_path.open("a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                # Every response's token ids, so live scoring can be audited and a changed reward rule
                # re-scored offline instead of probing again.
                with out_path.with_suffix(".samples.jsonl").open("a") as f:
                    for i, row in enumerate(rows):
                        f.write(json.dumps({**row, "response_ids": list(batch["response_ids"][i])}) + "\n")
                finished += len(chunk)
                logger.info(f"[probe] {finished}/{len(prompts)} prompts finished")

        try:
            results = await asyncio.gather(*(run_chunk(c) for c in chunks), return_exceptions=True)
        finally:
            await trajectory_runner.shutdown()
        failures = [r for r in results if isinstance(r, BaseException)]
        logger.info(f"[probe] done: {finished} prompts written, {len(failures)} chunks raised")
        if failures:
            logger.error(f"[probe] first failure: {failures[0]!r}")

    def run(self):
        asyncio.run(self._probe())


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    SkyRLGymProbeExp(cfg).run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.SKYRL_GYM, failure_message="Probe failed")


if __name__ == "__main__":
    main()
