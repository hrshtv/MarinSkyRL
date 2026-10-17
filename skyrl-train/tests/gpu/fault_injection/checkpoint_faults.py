from skyrl_train.distributed.megatron import direct_checkpoint
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase


class FailingOnceMegatronPolicyWorker(MegatronPolicyWorkerBase):
    def fail_after_next_distributed_save(self) -> int:
        """Inject an error after DCP completes and before trainer publication."""
        real_save = direct_checkpoint.checkpoint.save
        self._checkpoint_failure_seen = False

        def save_then_fail(*args, **kwargs):
            real_save(*args, **kwargs)
            direct_checkpoint.checkpoint.save = real_save
            self._checkpoint_failure_seen = True
            raise OSError("injected post-DCP save failure")

        direct_checkpoint.checkpoint.save = save_then_fail
        return self._rank

    def checkpoint_failure_seen(self) -> tuple[int, bool]:
        return self._rank, self._checkpoint_failure_seen
