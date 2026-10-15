import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random

from megatron.core import tensor_parallel
import numpy as np
from omegaconf import OmegaConf
import ray
import torch

from cloud.iris.launch_config import load_launch_config
from skyrl_train.checkpoint_generation import resolve_checkpoint_payload
from skyrl_train.distributed.megatron.megatron_utils import materialize_megatron_params
from skyrl_train.entrypoints.main_base import run_ray_driver
from skyrl_train.io import io
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.tracking import Tracking
from skyrl_train.workers.megatron.megatron_worker import CriticWorker, MegatronPolicyWorkerBase, RefWorker
from tests.gpu.fault_injection.checkpoint_digest import digest_value


WORLD_SIZE = 32
GEOMETRY = (1, 2, 1, 8)
SEQUENCE_LENGTH = 128
RESPONSE_LENGTH = 16


class FixedDataset:
    def __len__(self):
        return 1024

    def uid(self, index):
        return str(index)

    def __getitem__(self, index):
        return {"uid": str(index), "prompt": [{"role": "user", "content": f"Parity prompt {index}"}]}

    def collate_fn(self, items):
        return items


class NoRollouts:
    def set_trajectory_sink(self, sink):
        pass

    async def run_task(self, task, writer):
        raise RuntimeError("Numerical checkpoint parity must not dispatch rollout work")

    async def shutdown(self):
        pass


class ParityWorker(MegatronPolicyWorkerBase):
    def parity_rng(self):
        return self._rank, digest_value(
            {
                "generic": self.strategy.get_rng_state(),
                "cuda_tracker": tensor_parallel.get_cuda_rng_tracker().get_states(),
            }
        )

    def parity_state(self):
        materialize_megatron_params(self.model.actor_module)
        model = self.model.actor_module[0].module
        shards = model.sharded_state_dict()
        optimizer = self.optimizer.sharded_state_dict(
            shards, metadata={"distrib_optim_sharding_type": "dp_reshardable"}
        )
        return self._rank, {
            "model": digest_value(
                {"parameters": dict(model.named_parameters()), "buffers": dict(model.named_buffers())}
            ),
            "optimizer": digest_value(optimizer),
            "scheduler": digest_value(self.scheduler.state_dict()),
            "rng": self.parity_rng()[1],
        }


def rank_values(trainer, method):
    results = ray.get(trainer.policy_model.async_run_ray_method("pass_through", method))
    assert sorted(rank for rank, _ in results) == list(range(WORLD_SIZE)), "Incomplete policy rank set"
    return {str(rank): value for rank, value in results}


def driver_rng():
    return digest_value({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()})


def fixed_batch(samples):
    count = 32 * samples
    sequences = (torch.arange(SEQUENCE_LENGTH)[None, :] + torch.arange(count)[:, None]) % 1000 + 100
    shape = (count, RESPONSE_LENGTH)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": torch.ones_like(sequences),
            "action_log_probs": torch.full(shape, 0.1),
            "base_action_log_probs": torch.full(shape, 0.2),
            "rollout_logprobs": torch.full(shape, 0.11),
            "values": torch.full(shape, 0.1),
            "returns": torch.full(shape, 0.1),
            "advantages": torch.full(shape, 0.5),
            "loss_mask": torch.ones(shape),
            "response_mask": torch.ones(shape),
        }
    )
    batch.metadata = {"response_length": RESPONSE_LENGTH, "global_step": 1}
    return batch


def update(trainer, batch):
    status = trainer.train_critic_and_policy(batch)
    assert status["policy_update_steps"] == 1, f"Expected one actual optimizer update: {status}"
    assert np.isfinite(status["raw_grad_norm"]) and status["raw_grad_norm"] > 0, "Expected finite, nonzero gradients"


def require_equal(expected, actual, stage):
    assert expected == actual, f"Checkpoint parity mismatch at {stage}: expected={expected}, actual={actual}"


async def execute(cfg):
    probe = cfg.checkpoint_parity
    root = str(probe.root).rstrip("/")
    assert root.startswith("s3://marin-us-east-02a/tmp/ttl=14d/skyrl/users/atqamar/"), "Use an owned east TTL root"
    assert probe.phase in ("reference", "resume"), "Unknown parity phase"
    geometry = cfg.trainer.policy.megatron_config
    assert (
        geometry.tensor_model_parallel_size,
        geometry.pipeline_model_parallel_size,
        geometry.context_parallel_size,
        geometry.expert_model_parallel_size,
    ) == GEOMETRY
    assert cfg.trainer.placement.policy_num_nodes == 4 and cfg.trainer.placement.policy_num_gpus_per_node == 8
    assert geometry.optimizer_checkpoint_sharding_type == "dp_reshardable"
    assert (
        not cfg.trainer.critic.model.path
        and not cfg.trainer.algorithm.use_kl_loss
        and not cfg.trainer.algorithm.use_kl_in_reward
    )
    assert not cfg.generator.run_engines_locally and cfg.trainer.rollout_buffer.max_staleness_steps == 0
    assert not cfg.generator.enable_http_endpoint and cfg.trainer.restore_dataloader_state
    assert str(cfg.trainer.ckpt_path) == root + "/checkpoints"
    contract = OmegaConf.to_container(cfg.trainer, resolve=True)
    for key in ("run_name", "entrypoint_node_ip", "resume_mode", "resume_path", "export_path"):
        contract.pop(key, None)
    model = contract["policy"]["model"]
    model["path"] = model["tokenizer_path"] = model["source_identity"]
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    checkpoint = root + "/checkpoints/global_step_1"
    record_path = root + "/parity-reference.json"
    batch = fixed_batch(cfg.generator.n_samples_per_prompt)
    batch_digest = digest_value({"batch": dict(batch), "metadata": batch.metadata})
    seed = 42 if probe.phase == "reference" else 999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = FixedDataset()
    runner = NoRollouts()
    context = TrainingContext.from_config(cfg, dataset, runner)
    with Tracking(cfg.trainer.project_name, cfg.trainer.run_name, backends="console", config=cfg) as tracker:
        trainer = RayPPOTrainer(cfg, tracker, tokenizer_from_config(cfg), dataset, None, runner, context)
        try:
            trainer.build_models(ray.remote(num_gpus=1)(ParityWorker), CriticWorker, RefWorker)
            if probe.phase == "reference":
                assert not io.exists(record_path) and not io.exists(checkpoint), "Reference needs a fresh prefix"
                prompts = [context.loader.next_prompt(set()) for _ in range(32)]
                context.loader.retry(prompts[-1])
                trainer._offload_policy_optimizer(trainer.all_timings, timer_label="parity_initial_offload")
                update(trainer, batch)
                trainer.global_step = 1
                trainer._offload_policy_optimizer(trainer.all_timings, timer_label="parity_save_offload")
                saved_rng = rank_values(trainer, "parity_rng")
                saved_driver_rng = driver_rng()
                await trainer.save_checkpoints()
                require_equal(saved_rng, rank_values(trainer, "parity_rng"), "save preserves worker RNG")
                require_equal(saved_driver_rng, driver_rng(), "save preserves driver RNG")
                pre = rank_values(trainer, "parity_state")
                assert all(state["optimizer"]["tensor_count"] > 0 for state in pre.values()), (
                    "Optimizer is uninitialized"
                )
                rollout = digest_value(await context.state_dict())
                next_prompt = context.loader.next_prompt(set())
                driver_pre = driver_rng()
                update(trainer, fixed_batch(cfg.generator.n_samples_per_prompt))
                post = rank_values(trainer, "parity_state")
                assert any(pre[rank]["model"] != post[rank]["model"] for rank in pre), (
                    "Next step did not change model values"
                )
                record = {
                    "revision": str(probe.revision),
                    "contract": fingerprint,
                    "batch": batch_digest,
                    "checkpoint": checkpoint,
                    "payload": resolve_checkpoint_payload(checkpoint, verify_files=True),
                    "world_size": WORLD_SIZE,
                    "geometry": GEOMETRY,
                    "pre": pre,
                    "post": post,
                    "driver_pre": driver_pre,
                    "driver_post": driver_rng(),
                    "rollout": rollout,
                    "next_prompt": next_prompt,
                }
                with io.open_file(record_path, "w") as output:
                    json.dump(record, output, sort_keys=True)
            else:
                with io.open_file(record_path) as source:
                    record = json.load(source)
                require_equal(record["revision"], str(probe.revision), "source revision")
                require_equal(record["contract"], fingerprint, "trainer contract")
                require_equal(record["batch"], batch_digest, "identical training input")
                require_equal(record["geometry"], list(GEOMETRY), "geometry")
                require_equal(record["world_size"], WORLD_SIZE, "world size")
                loaded_step, payload = trainer.load_checkpoints()
                require_equal((loaded_step, payload), (1, record["payload"]), "own committed checkpoint")
                trainer.global_step = loaded_step
                assert trainer._restored_rollout_state is not None, "Missing rollout continuation state"
                await context.load_state_dict(trainer._restored_rollout_state)
                trainer._restored_rollout_state = None
                require_equal(record["rollout"], digest_value(await context.state_dict()), "rollout continuation")
                require_equal(record["next_prompt"], context.loader.next_prompt(set()), "next retry prompt")
                trainer._restore_loaded_driver_rng()
                require_equal(record["driver_pre"], driver_rng(), "driver RNG before next step")
                require_equal(record["pre"], rank_values(trainer, "parity_state"), "all rank state after restore")
                update(trainer, batch)
                require_equal(record["driver_post"], driver_rng(), "driver RNG after next step")
                require_equal(record["post"], rank_values(trainer, "parity_state"), "all rank state after next step")
            print(
                "SNOWBALL_CHECKPOINT_PARITY_OK "
                + json.dumps(
                    {
                        "phase": str(probe.phase),
                        "ranks": WORLD_SIZE,
                        "record": record_path,
                        "revision": str(probe.revision),
                    }
                ),
                flush=True,
            )
        finally:
            await trainer.shutdown()


@ray.remote(num_cpus=1, max_retries=0)
def checkpoint_entrypoint(cfg):
    asyncio.run(execute(cfg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = load_launch_config(arguments.config)
    OmegaConf.update(
        config.skyrl,
        "checkpoint_parity",
        {
            "phase": str(config.run.checkpoint_parity_phase),
            "root": str(config.run.checkpoint_parity_root),
            "revision": str(config.runtime.launcher_commit),
        },
        force_add=True,
    )
    run_ray_driver(config.skyrl, checkpoint_entrypoint)


if __name__ == "__main__":
    main()
