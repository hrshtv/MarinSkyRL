Checkpointing
=============

SkyRL provides checkpointing features to resume training from a previous state. Training state is saved at regular intervals and provides flexible configuration options for checkpoint management.

Checkpointed state can be stored in the local file system or uploaded to cloud storage (S3, GCP).

What State is Saved
-------------------

SkyRL saves several types of state to enable complete training resumption:

**Model States**
  - **Policy Model**: Model parameters, optimizer state, and learning rate scheduler state
  - **Reference Model**: Not checkpointed (recreated from policy model)

**Training State**
  - **Global Step**: Current training step counter
  - **Configuration**: Complete training configuration used
  - **Dataloader State**: Current position in dataset iteration (enables resuming from exact data position)

Directory Structure
-------------------

Megatron checkpoints use the following directory structure.

.. _megatron-checkpointing:

Megatron Checkpointing
~~~~~~~~~~~~~~~~~~~~~~

Megatron checkpoints are handled by the Megatron `dist_checkpointing <https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html>`_ library to perform checkpointing in parallel across ranks. 
This comes with support for reloading checkpoints in a different parallelism scheme than the one they were saved in.

.. code-block::

    {ckpt_path}/
    ├── latest_ckpt_global_step.txt          # Holds the global step of the latest checkpoint
    ├── global_step_10/                      # Checkpoint at training step 10
    │   ├── policy/                          # Policy model checkpoint directory
    │   │   ├── metadata.json                # Megatron checkpoint metadata
    │   │   ├── huggingface/                 # HuggingFace config and tokenizer
    │   │   ├── __0_0.distcp                 # Megatron checkpoint files
    │   │   ├── __0_1.distcp                 
    │   │   └── ...
    ├── global_step_20/                      # Checkpoint at training step 20
    │   └── ...
    └── global_step_30/                      # Checkpoint at training step 30
        └── ...

Key Configuration Parameters
----------------------------

Checkpointing behavior is controlled by several parameters in the YAML configuration (see :doc:`../configuration/config` for the full training config):

**Checkpoint Saving**

``ckpt_interval``
  - **Default**: ``10``
  - **Purpose**: Save checkpoints every N training steps

``ckpt_path``
  - **Default**: ``"${oc.env:HOME}/ckpts/"``
  - **Purpose**: Base directory where all checkpoints are stored
  - **Options**:
    - Local file system path (e.g., ``/path/to/ckpts/``)
    - Cloud storage path (S3, GCP) (e.g., ``s3://path/to/ckpts/``)

**Checkpoint Cleanup**

``max_ckpts_to_keep``
  - **Default**: ``-1`` (keep all checkpoints)
  - **Purpose**: Limit number of stored checkpoints to save disk space
  - **Options**:

    - ``-1``: Keep all checkpoints indefinitely
    - ``N`` (positive integer): Keep only the last N checkpoints, automatically delete older ones

**Training Resumption**

``resume_mode``
  - **Default**: ``"latest"``
  - **Purpose**: Controls how training resumption works
  - **Options**:
  
    - ``"none"`` or ``null``: Start training from scratch, ignore existing checkpoints
    - ``"latest"``: Automatically resume from the most recent checkpoint
    - ``"from_path"``: Resume from a specific checkpoint (requires ``resume_path``)

``resume_path``
  - **Default**: ``null``
  - **Purpose**: Specific checkpoint directory to resume from (only used when ``resume_mode: "from_path"``)
  - **Format**: Must point to a ``global_step_N`` directory

Fixed Noise and Frozen Sampler Policies
--------------------------------------

Keep the configured initial model and noise seed unchanged across restarts.
Fixed sampler noise is initialized from that original model before loading the
learner checkpoint. The resumed learner weights then receive the same fixed delta.

For a periodically refreshed sampler, checkpoint metadata also records which
policy generated rollouts and where its weights are stored. A restart restores
that older serving snapshot while keeping the current learner and optimizer state.
The initial sampler uses the configured original model. Checkpoints without sampler
metadata or a required older policy snapshot cannot resume this mode.

Resuming a frozen sampler requires disaggregated synchronous full batches and a
checkpoint interval that divides the sampler refresh period. This ensures every
refreshed sampler has a checkpoint. For example:

.. code-block:: yaml

    generator:
      sampler_refresh_period: 65
    trainer:
      resume_mode: latest
      ckpt_interval: 13
      max_ckpts_to_keep: 2

When an explicit callback list is configured, set the checkpoint callback's
``save_steps`` accordingly; it replaces the legacy ``ckpt_interval`` setting.
Cleanup preserves the recent committed checkpoints and any older sampler policies
they require. A newer partial upload does not evict those restore dependencies.

To preserve scheduled evaluation files when continuing a run, disable the
evaluation callback's before-training evaluation on resume. Fresh starts still
evaluate at step zero, and interval and train-end evaluations retain their schedule:

.. code-block:: yaml

    trainer:
      callbacks:
        - type: checkpoint
          save_steps: 13
        - type: evaluation
          eval_steps: 20
          eval_before_train: true
          eval_on_resume: false

``eval_on_resume`` defaults to true, preserving the existing evaluation behavior.
Checkpoints retain the last evaluated step at the time of the snapshot. When all
evaluation callbacks opt out of resume evaluation, that progress is restored, so a
completed run can recognize an evaluation already saved with its checkpoint.
Resetting the training step for a new stage clears this evaluation progress.

HuggingFace Model Export
------------------------

In addition to checkpointing, users can optionally save the policy model in HuggingFace safetensors format at regular intervals.

**Configuration Parameters:**

``hf_save_interval``
  - **Default**: ``-1`` (disabled)
  - **Purpose**: Save HuggingFace format policy models every N training steps

``export_path``
  - **Default**: ``"${oc.env:HOME}/exports/"``
  - **Purpose**: Base directory where HuggingFace models and other artifacts are saved
  - **Structure**: Models are saved to ``{export_path}/global_step_{N}/policy/``
