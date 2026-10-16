"""Finite real-worker alignment and timing probe. Excluded from the merge candidate."""

import asyncio
import hashlib
import importlib.metadata
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import fsspec
import ray
import torch
from omegaconf import OmegaConf
from rigging.filesystem.s3_compat import configure_coreweave_s3

from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.objective.correction import compute_correction
from skyrl_train.config.objective_spec import off_policy_correction
from skyrl_train.objective.score_centering import collate_score_centering
from skyrl_train.trainer import RayPPOTrainer, convert_prompts_responses_to_batch_tensors, _token_staleness_metrics
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.timing_observability import StepWallTime


_device_memory_peak = 0
_device_memory_lock = threading.Lock()
_device_memory_thread = None


def sample_device_memory(device):
    global _device_memory_peak
    timer = threading.Event()
    while not timer.wait(0.1):
        used = torch.cuda.device_memory_used(device)
        with _device_memory_lock:
            _device_memory_peak = max(_device_memory_peak, used)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def memory_stats(reset=False):
    global _device_memory_peak, _device_memory_thread
    torch.cuda.synchronize()
    used = torch.cuda.device_memory_used()
    with _device_memory_lock:
        _device_memory_peak = used if reset else max(_device_memory_peak, used)
        peak = _device_memory_peak
    if _device_memory_thread is None:
        _device_memory_thread = threading.Thread(
            target=sample_device_memory, args=(torch.cuda.current_device(),), daemon=True
        )
        _device_memory_thread.start()
    if not _device_memory_thread.is_alive():
        raise RuntimeError("Device memory sampler stopped before measurement finished")
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return {
        "device": torch.cuda.get_device_name(),
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "device_used": used,
        "sampled_device_peak_used": peak,
        "device_sampling_interval": 0.1,
    }


def reference_forward(worker, data):
    original = worker.model._response_logprobs
    stats = {"positions": 0, "chosen_max_error": 0.0, "head_max_error": 0.0}

    def checked(logits, sequences, attention_mask, num_actions, packed_seq_params, candidate_ids=None):
        result = original(logits, sequences, attention_mask, num_actions, packed_seq_params, candidate_ids)
        for row in range(len(sequences)):
            valid = candidate_ids[row].ge(0).any(-1).nonzero().flatten().tolist()
            if not valid:
                continue
            prompt_length = int(attention_mask[row, :-num_actions].sum())
            for position in sorted({valid[0], valid[len(valid) // 2], valid[-1]}):
                # Independent causal-position calculation on the real compacted model logits.
                normalized = logits[row, prompt_length - 1 + position].float().log_softmax(-1)
                token = sequences[row, -num_actions + position]
                ids = candidate_ids[row, position].long()
                chosen_error = abs(float(normalized[token] - result.scores[row, position]))
                head_error = float((normalized[ids] - result.selected_logprobs[row, position]).abs().max())
                stats["positions"] += 1
                stats["chosen_max_error"] = max(stats["chosen_max_error"], chosen_error)
                stats["head_max_error"] = max(stats["head_max_error"], head_error)
        return result

    worker.model._response_logprobs = checked
    try:
        output = worker.forward(data)
    finally:
        worker.model._response_logprobs = original
    output.metadata = {**(output.metadata or {}), "readiness_reference": stats}
    return output


class ReadinessExp(BasePPOExp):
    def get_trainer(self, **kwargs):
        trainer = super().get_trainer(**kwargs)
        trainer.__class__ = ReadinessTrainer
        return trainer


class ReadinessTrainer(RayPPOTrainer):
    def save_probe(self):
        path = Path(self.cfg.trainer.export_path) / "readiness.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.probe, indent=2, allow_nan=False) + "\n")
        with fsspec.open(self.cfg.trainer.score_centering_readiness.output_uri, "wt") as output:
            json.dump(self.probe, output, allow_nan=False)
        print(
            "READINESS "
            + json.dumps(
                {
                    "complete": self.probe.get("complete", False),
                    "windows": len(self.probe["windows"]),
                    "qualification": list(self.probe["qualification"]),
                }
            ),
            flush=True,
        )

    async def settings(self, width, enabled, telemetry=True, diagnostic_tokens=0, temperature=1.0):
        self.cfg.trainer.algorithm.score_centering_topk = width
        self.cfg.trainer.algorithm.score_centering_enabled = enabled
        self.cfg.trainer.algorithm.score_centering_diagnostic_tokens = diagnostic_tokens
        self._training_metrics_enabled = telemetry
        return await asyncio.to_thread(
            ray.get,
            self.policy_model.async_run_ray_method(
                "pass_through", "readiness_settings", width, enabled, diagnostic_tokens, temperature
            ),
        )

    async def inference_memory(self, reset=False):
        return await asyncio.gather(
            *(
                engine.inference_engine_actor.readiness_memory.remote(reset)
                for engine in self.inference_engine_client.engines
            )
        )

    async def generate_workload(self, width, *, temperature=1.0, short=False):
        limits = [8, 24, 64, 128] if short else [64, 256, 1024, 4096]
        outputs = []
        for group, limit in enumerate(limits):
            prompts = self.prompts[group::4]
            result = await self.inference_engine_client.generate(
                {
                    "prompt_token_ids": prompts,
                    "sampling_params": {
                        "temperature": temperature,
                        "top_p": 1.0,
                        "top_k": -1,
                        "min_p": 0.0,
                        "max_tokens": limit,
                        "ignore_eos": True,
                        "seed": 611 + group,
                        "logprobs": width,
                    },
                }
            )
            assert all(len(row) == limit for row in result["response_ids"])
            assert all(set(row) == {1} for row in result["response_policy_steps"])
            outputs.extend((prompt, result, index) for index, prompt in enumerate(prompts))
        return outputs

    def batch(self, outputs, width):
        prompts = [prompt for prompt, _, _ in outputs]
        responses = [output["response_ids"][index] for _, output, index in outputs]
        chosen = [np.array(output["response_logprobs"][index]) for _, output, index in outputs]
        masks = [[int(position % 31 != 7) for position in range(len(row))] for row in responses]
        masks[0] = [0] * len(masks[0])
        rewards = [[0.0] * len(row) for row in responses]
        values = convert_prompts_responses_to_batch_tensors(
            self.tokenizer, prompts, responses, rewards, masks, chosen, None, None
        )
        sequences, attention, response_mask, _, loss_mask, sampled, _, _ = values
        policy_steps = torch.full_like(loss_mask, -1, dtype=torch.int32)
        for row, (_, output, index) in enumerate(outputs):
            stamps = output["response_policy_steps"][index]
            policy_steps[row, : len(stamps)] = torch.tensor(stamps, dtype=torch.int32)
        staleness = torch.where(response_mask.bool(), self.global_step - policy_steps, 0)
        payload = {
            "sequences": sequences,
            "attention_mask": attention,
            "response_mask": response_mask,
            "loss_mask": loss_mask,
            "rollout_logprobs": sampled,
            "rollout_staleness": staleness,
            "base_action_log_probs": None,
            "values": None,
            "returns": torch.zeros_like(loss_mask),
            # Fixed signed advantages qualify optimizer execution, without a learning-benefit experiment.
            "advantages": loss_mask * torch.tensor([1.0, -1.0] * (len(prompts) // 2)).unsqueeze(-1),
        }
        if width:
            trajectory = {
                "loss_masks": masks,
                "student_topk_indices": [np.asarray(output["student_topk_indices"][i]) for _, output, i in outputs],
                "behavior_topk_logprobs": [np.asarray(output["behavior_topk_logprobs"][i]) for _, output, i in outputs],
            }
            ids, behavior = collate_score_centering(
                trajectory, responses, response_mask, width, sampled_logprobs=sampled
            )
            payload.update(score_topk_indices=ids, score_behavior_logprobs=behavior)
        batch = TrainingInputBatch(payload)
        if self._training_metrics_enabled:
            self.all_metrics.update(_token_staleness_metrics(staleness, loss_mask, policy_steps))
        batch.metadata = {"response_length": response_mask.shape[1], "uids": list(map(str, range(len(prompts))))}
        return batch

    async def reference(self, batch):
        keys = ["sequences", "attention_mask", "response_mask", "score_topk_indices"]
        data = batch.select(keys, metadata_keys=["response_length"])
        rows = await asyncio.to_thread(
            ray.get, self.policy_model.async_run_ray_method("mesh", "readiness_reference", data)
        )
        stats = [
            row.metadata["readiness_reference"] for row in rows if row.metadata["readiness_reference"]["positions"]
        ]
        assert stats and max(row["chosen_max_error"] for row in stats) < 2e-5
        assert max(row["head_max_error"] for row in stats) < 2e-5
        return stats

    async def prefix_replay(self, outputs, *, temperature=1.0):
        prefixes, overrides, expected, samples = [], [], [], []
        for prompt, output, index in outputs[:: max(1, len(outputs) // 8)]:
            response = output["response_ids"][index]
            for position in sorted({0, len(response) // 2, len(response) - 1}):
                ids = output["student_topk_indices"][index][position]
                scores = output["behavior_topk_logprobs"][index][position]
                if response[position] not in ids:
                    ids = [*ids, response[position]]
                    scores = [*scores, output["response_logprobs"][index][position]]
                prefixes.append(prompt + response[:position])
                overrides.append({"logprob_token_ids": ids})
                expected.append(scores)
                samples.append(
                    {
                        "prompt_ids": prompt,
                        "response_prefix_ids": response[:position],
                        "chosen_token_id": response[position],
                        "position": position,
                        "captured_support_ids": ids,
                        "captured_support_logprobs": scores,
                        "captured_chosen_logprob": output["response_logprobs"][index][position],
                        "temperature": temperature,
                        "generating_policy_step": output["response_policy_steps"][index][position],
                    }
                )
        replay = await self.inference_engine_client.generate(
            {
                "prompt_token_ids": prefixes,
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 1,
                    "logprobs": 128,
                },
                "sampling_params_per_prompt": overrides,
            }
        )
        shifted = await self.inference_engine_client.generate(
            {
                "prompt_token_ids": [prefix[:-1] for prefix in prefixes],
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 1,
                    "logprobs": 128,
                },
                "sampling_params_per_prompt": overrides,
            }
        )
        gaps = []
        shifted_gaps = []
        for index, target in enumerate(expected):
            assert replay["student_topk_indices"][index][0] == overrides[index]["logprob_token_ids"]
            gaps.extend(np.asarray(replay["behavior_topk_logprobs"][index][0]) - np.asarray(target))
            shifted_gaps.extend(np.asarray(shifted["behavior_topk_logprobs"][index][0]) - np.asarray(target))
            samples[index]["prefix_replay_support_logprobs"] = replay["behavior_topk_logprobs"][index][0]
        assert np.abs(gaps).mean() < np.abs(shifted_gaps).mean()
        return {
            "prefixes": len(prefixes),
            "prefix_ids_sha256": digest(prefixes),
            "support_ids_sha256": digest(overrides),
            "temperature": temperature,
            "same_engine_decode_prefill_gap_abs_mean": float(np.abs(gaps).mean()),
            "same_engine_decode_prefill_gap_abs_max": float(np.abs(gaps).max()),
            "one_token_shift_control_gap_abs_mean": float(np.abs(shifted_gaps).mean()),
            "policy_steps": sorted({step for row in replay["response_policy_steps"] for step in row}),
            "samples": samples,
        }

    async def update(self, batch):
        started = time.perf_counter()
        batch = await asyncio.to_thread(self.fwd_logprobs_values_reward, batch)
        forward = time.perf_counter() - started
        correction = compute_correction(
            batch["action_log_probs"],
            batch["rollout_logprobs"],
            batch["loss_mask"],
            off_policy_correction(self.cfg.trainer.algorithm),
        )
        batch["correction_weights"] = correction.weights
        trained = time.perf_counter()
        status = await asyncio.to_thread(self.train_critic_and_policy, batch)
        update = time.perf_counter() - trained
        assert status["policy_update_steps"] == 1 and status["skipped_steps"] == 0
        assert math.isfinite(status["raw_grad_norm"]) and status["raw_grad_norm"] > 0
        self._readiness_applied_updates += int(status["policy_update_steps"])
        self._policy_version_step = self.global_step
        await asyncio.to_thread(self._offload_policy_optimizer, self.all_timings, timer_label="probe_offload")
        memory = await asyncio.to_thread(
            ray.get, self.policy_model.async_run_ray_method("pass_through", "readiness_memory")
        )
        return {
            "forward_seconds": forward,
            "update_seconds": update,
            "status": status,
            "learner_memory": memory,
            "mismatch": {k: v for k, v in self.all_metrics.items() if "mismatch" in k},
        }

    async def _train_loop(self):
        configure_coreweave_s3()
        short = self.cfg.trainer.score_centering_readiness.workload == "qwen"
        self._readiness_applied_updates = 0
        self.probe = {
            "schema": 1,
            "workload": "qwen" if short else "snowball",
            "candidate": self.cfg.trainer.score_centering_readiness.candidate,
            "windows": [],
            "qualification": {},
            "setup_timings": self.all_startup_timings,
        }
        self.probe["runtime"] = {
            "lock_sha256": hashlib.sha256((Path(__file__).resolve().parents[3] / "uv.lock").read_bytes()).hexdigest(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "vllm", "megatron-core", "transformers", "marin-iris")
            },
            "policy_parallelism": OmegaConf.to_container(self.cfg.trainer.policy.megatron_config),
            "sampling": OmegaConf.to_container(self.cfg.generator.sampling_params),
        }
        await self._init_weight_sync()
        await self._sync_policy_for_rollouts(reason="initial")
        self.probe["setup_timings"] = dict(self.all_startup_timings)
        count = 32 if short else 128
        self.prompts = []
        for index in range(count):
            messages, _, extra, _ = self.train_dataset[index // 4]
            if "reference_prompt_ids" in extra and not short:
                prompt = extra["reference_prompt_ids"]
            else:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]
            self.prompts.append(list(prompt))
        self.probe["prompt_sha256"] = digest(self.prompts)
        self.probe["prompt_lengths"] = list(map(len, self.prompts))
        await self.settings(128, False)
        captured = await self.generate_workload(128, short=short)
        self.probe["qualification"]["prefix_replay"] = await self.prefix_replay(captured)
        self.global_step = 1
        batch = self.batch(captured, 128)
        self.probe["qualification"]["reference"] = await self.reference(batch)
        batch = await asyncio.to_thread(self.fwd_logprobs_values_reward, batch)
        matching_logprobs = batch["action_log_probs"].clone()
        valid = batch["loss_mask"] > 0
        gap = batch["action_log_probs"][valid] - batch["rollout_logprobs"][valid]
        self.probe["qualification"]["matching_weights_tim"] = {
            "tokens": gap.numel(),
            "abs_mean": gap.abs().mean().item(),
            "abs_p99": torch.quantile(gap.abs(), 0.99).item(),
            "abs_max": gap.abs().max().item(),
        }
        head_gap = (batch["score_old_logprobs"] - batch["score_behavior_logprobs"])[valid]
        sampled_head_gap = head_gap.flatten()[:: max(1, head_gap.numel() // 65536)]
        self.probe["qualification"]["matching_weights_head_tim"] = {
            "entries": head_gap.numel(),
            "abs_mean": head_gap.abs().mean().item(),
            "abs_max": head_gap.abs().max().item(),
            "sampled_abs_p99": torch.quantile(sampled_head_gap.abs(), 0.99).item(),
        }
        self.save_probe()
        # This second temperature exercises the actual engine and trainer scaling before any updates.
        await self.settings(128, False, temperature=0.7)
        alternate = await self.generate_workload(128, temperature=0.7, short=True)
        self.probe["qualification"]["temperature_0_7_reference"] = await self.reference(self.batch(alternate, 128))
        self.probe["qualification"]["temperature_0_7_prefix_replay"] = await self.prefix_replay(
            alternate, temperature=0.7
        )
        modes = [
            ("ordinary", 0, False, False),
            ("telemetry", 0, False, True),
            ("capture", 128, False, True),
            ("sc", 128, True, True),
        ]
        repetitions = 2 if short else 4
        for repetition in range(repetitions):
            order = modes[repetition % 4 :] + modes[: repetition % 4]
            for name, width, enabled, telemetry in order:
                await self.settings(width, enabled, telemetry)
                await self.inference_memory(reset=True)
                self.all_metrics, self.all_timings = {}, {}
                started = time.perf_counter()
                outputs = await self.generate_workload(width, short=short)
                generated = time.perf_counter() - started
                self.global_step = self._readiness_applied_updates + 1
                prepared = time.perf_counter()
                batch = self.batch(outputs, width)
                preparation = time.perf_counter() - prepared
                result = await self.update(batch)
                result.update(
                    mode=name,
                    repetition=repetition,
                    warmup=repetition == 0,
                    generation_seconds=generated,
                    preparation_seconds=preparation,
                    total_seconds=time.perf_counter() - started,
                    tokens=int(batch["response_mask"].sum()),
                    loss_tokens=int(batch["loss_mask"].sum()),
                    sequence_shape=list(batch["sequences"].shape),
                    response_ids_sha256=digest([o["response_ids"][i] for _, o, i in outputs]),
                    inference_memory=await self.inference_memory(),
                    applied_updates=self._readiness_applied_updates,
                    generator_applied_updates=0,
                    generator_policy_step=1,
                )
                self.probe["windows"].append(result)
                self.save_probe()
        await self.settings(128, True, diagnostic_tokens=16)
        self.global_step = self._readiness_applied_updates + 1
        stale_batch = self.batch(captured, 128)
        result = await self.update(stale_batch)
        drift = (stale_batch["action_log_probs"] - matching_logprobs)[valid]
        assert drift.abs().max() > 0
        self.probe["qualification"]["trainer_weight_drift_on_identical_tokens"] = {
            "abs_mean": drift.abs().mean().item(),
            "abs_max": drift.abs().max().item(),
            "changed_fraction": (drift != 0).float().mean().item(),
        }
        self.probe["qualification"]["sampled_diagnostics"] = result
        self.probe["qualification"]["stale_prefix_replay"] = await self.prefix_replay(captured)
        assert self.probe["qualification"]["stale_prefix_replay"]["policy_steps"] == [1]
        if not short:
            # Qualify the actual task projection, admission and training path too.
            await self._sync_policy_for_rollouts(reason="native_qualification")
            self.context.start()
            await self._publish(self.global_step + 1)
            self.global_step += 1
            self.all_metrics, self.all_timings = {}, {}
            wall = StepWallTime(OmegaConf.to_container(self.cfg.trainer.step_phase_budgets))
            native_started = time.perf_counter()
            native = await self._train_step(0, wall)
            assert self.all_metrics["policy/policy_update_steps"] == 1
            assert self.all_metrics["policy/skipped_steps"] == 0
            assert (
                math.isfinite(self.all_metrics["policy/raw_grad_norm"]) and self.all_metrics["policy/raw_grad_norm"] > 0
            )
            assert self.all_metrics["async/staleness_tokens_max"] == 0
            self._readiness_applied_updates += 1
            self.probe["qualification"]["native_training"] = {
                "seconds": time.perf_counter() - native_started,
                "shape": list(native["sequences"].shape),
                "tokens": int(native["response_mask"].sum()),
                "loss_tokens": int(native["loss_mask"].sum()),
                "uids": native.metadata["uids"],
                "metrics": dict(self.all_metrics),
                "timings": dict(self.all_timings),
                "applied_updates": 1,
            }
        self.probe["complete"] = True
        self.probe["applied_updates"] = self._readiness_applied_updates
        self.save_probe()
