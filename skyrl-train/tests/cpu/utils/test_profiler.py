from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from skyrl_train.utils.profiler import Profiler


@pytest.mark.parametrize("captured_update", [1, 2])
def test_profiler_saves_only_first_forward_of_the_configured_update(tmp_path, monkeypatch, captured_update):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    profiler = Profiler(SimpleNamespace(ranks=[0], save_path=str(tmp_path), update_index=captured_update))

    for update_index in range(captured_update + 1):
        profiler.begin_update()
        for mini_batch_index in range(2):
            selected = profiler.for_mini_batch(mini_batch_index)
            for micro_batch_index in range(2):
                with selected.capture_forward() if selected is not None else nullcontext():
                    with torch.profiler.record_function(
                        f"update_{update_index}_mini_{mini_batch_index}_micro_{micro_batch_index}"
                    ):
                        torch.ones(4).sum()
        profiler.save()

    table = (tmp_path / "prof_rank_0.txt").read_text()
    assert f"update_{captured_update}_mini_0_micro_0" in table
    assert f"update_{captured_update - 1}_mini_0_micro_0" not in table
    assert f"update_{captured_update}_mini_0_micro_1" not in table
    assert f"update_{captured_update}_mini_1_micro_0" not in table
    assert list(tmp_path.iterdir()) == [tmp_path / "prof_rank_0.txt"]
