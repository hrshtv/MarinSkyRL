import asyncio

from loguru import logger

from skyrl_train.batch_assembly import forward_input
from skyrl_train.batch_digest import BatchInputPhase, digest
from skyrl_train.batch_source import DriverBatchSource, WorkerBatchSource
from skyrl_train.rollouts.buffer import AdmittedRollout
from skyrl_train.rollouts.context import RolloutBatchMetadata, TrainingContext
from skyrl_train.workers.worker import PPORayActorGroup


class VerifyBatchSource:
    """Compare retained worker forward inputs before executing the driver lifecycle."""

    def __init__(
        self,
        worker: WorkerBatchSource,
        *,
        driver: DriverBatchSource,
        context: TrainingContext,
        policy: PPORayActorGroup,
    ):
        self.worker, self.driver, self.context, self.policy = worker, driver, context, policy
        self.stall_timeout = None

    async def admit(self, *, stall_timeout: float) -> dict[str, float]:
        self.stall_timeout = stall_timeout
        return await self.worker.admit(stall_timeout=stall_timeout)

    async def prepare(self, *, global_step: int, step_wall) -> None:
        await self.worker.prepare(global_step=global_step, step_wall=step_wall)
        plan = self.worker.admitted_plan
        selected, offset = [], 0
        for index, facts in enumerate(plan.facts):
            selected.append(
                AdmittedRollout(
                    index,
                    plan.uids[offset],
                    global_step - plan.rollout_staleness[offset],
                    len(facts.response_len),
                    int(facts.response_len.sum()),
                    facts,
                )
            )
            offset += len(facts.response_len)
        metadata = RolloutBatchMetadata(plan.batch_id, tuple(selected), {})
        reader = self.context.reader(metadata, stall_timeout=self.stall_timeout)
        self.driver.groups = await self.context.wait(
            reader.read(plan.batch_id, tuple(group.index for group in selected))
        )
        await self.driver.prepare(global_step=global_step, step_wall=step_wall)

    async def forward(self, *, step_wall) -> None:
        plan = self.worker.admitted_plan
        expected_batch = forward_input(self.driver.batch)
        expected_batch.metadata["global_step"] = plan.global_step
        dp_size = self.policy.actor_infos[0].rank.dp_size
        expected = [digest(chunk) for chunk in expected_batch.chunk(expected_batch.batch_size // dp_size)]
        observed = await self.context.wait(
            asyncio.gather(
                *self.policy.async_run_ray_method(
                    "pass_through", "digest_loaded", plan.batch_id, BatchInputPhase.FORWARD
                )
            )
        )
        for actor, actual in zip(self.policy.actor_infos, observed, strict=True):
            if actual != expected[actor.rank.dp]:
                raise ValueError(f"worker forward input digest mismatch on rank {actor.rank}")
        logger.info("Worker forward input digests verified on {} ranks", len(observed))
        await self.driver.forward(step_wall=step_wall)

    async def finalize(self, *, step_wall) -> None:
        await self.driver.finalize(step_wall=step_wall)

    async def train(self, *, step_wall) -> dict:
        return await self.driver.train(step_wall=step_wall)

    def summary(self):
        return self.driver.summary()

    async def release(self, error: BaseException | None = None) -> None:
        try:
            await self.worker.release(error)
        finally:
            await self.driver.release(error)
