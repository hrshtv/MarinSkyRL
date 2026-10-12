import json

from examples.livecodebench.lcb_dataset import LIVECODEBENCH, process_example
from skyrl_gym.source_task import session_parameters
from taskcompendium.submission import conversation_messages

from marinskyrl.task_preparation import source_row_task


REVERSE_SOLUTION = """```python
print(input()[::-1])
```"""


def test_lcb_example_builder_preserves_executable_reference_tests():
    row = process_example(
        {
            "problem": "Read one line and print it in reverse.",
            "tests": [{"input": "abc\n", "output": "cba\n", "testtype": "stdin"}],
            "completion": REVERSE_SOLUTION,
        },
        idx=3,
        dataset_name=LIVECODEBENCH,
        split="test",
    )

    task = source_row_task(
        row,
        0,
        source_name=LIVECODEBENCH,
        config={},
    )
    public = conversation_messages(task.context.events)
    parameters = session_parameters(task)
    assert "```python" in public[0]["content"]
    assert "cba" not in public[0]["content"]
    assert json.loads(parameters.extras["reward_model"]["ground_truth"]) == [
        {"input": "abc\n", "output": "cba\n", "testtype": "stdin"}
    ]
