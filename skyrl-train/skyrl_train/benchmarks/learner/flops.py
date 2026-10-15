"""Analytic FLOP model for Grug MoE / Hero policy updates.

Counts follow the MFU convention: every matmul costs ``2 * m * n * k``, a training token costs one forward plus
a backward of twice the forward, and activation recomputation is excluded from *model* FLOPs. Attention is priced
per sequence with exact causal pair counts: sliding-window layers attend to at most ``sliding_window`` keys and
long layers (``grug_long_layer_flags``) attend to the whole causal prefix. Elementwise work (norm scaling, softmax,
XSA projection, RoPE) is ignored.

Two counts differ from what the GPU executes and are reported separately as *hardware* FLOPs:

* Megatron's fused QKV projection always materializes ``num_key_value_heads`` (the maximum of the local and global
  KV-head counts) and slices global layers down to ``global_kv_heads`` afterwards. Model FLOPs price the logical
  head count, matching Marin's Hero pretraining MFU; hardware FLOPs price the stored projection.
* Activation recomputation re-runs part of the forward during the backward pass.

The Marin Hero pretraining count (``experiments/grug/moe_hero_ep/train.py::_compute_flops``) uses Levanter's
``lm_flops_per_token``, which prices attention as non-causal ``seq_len * span`` maps at the configured
``max_seq_len`` and omits the gated norms, attention gate and ShortConv. This model counts what one RL batch's
actual sequence lengths cost instead, so the two agree on the linear layers but not on attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from skyrl_train.models.grug_moe import GRUG_GATED_NORM_RANK, GrugMoeConfig, grug_long_layer_flags

# Gated RMSNorms per decoder layer (pre-attention and pre-MLP) and outside the stack (embedding and final norm).
GATED_NORMS_PER_LAYER = 2
GATED_NORMS_OUTSIDE_LAYERS = 2
# Backward costs twice the forward: one matmul for input gradients and one for weight gradients.
BACKWARD_TO_FORWARD = 2


class Accelerator(StrEnum):
    H100 = "H100"
    GB200 = "GB200"


# Dense BF16 tensor-core peak per GPU, without structured sparsity.
PEAK_BF16_DENSE_FLOPS: dict[Accelerator, float] = {
    Accelerator.H100: 989e12,
    Accelerator.GB200: 2.5e15,
}


def accelerator_from_device_name(device_name: str) -> Accelerator:
    """Map ``torch.cuda.get_device_name()`` to a peak-table key, failing on unknown hardware."""
    matches = [accelerator for accelerator in Accelerator if accelerator.value in device_name.upper()]
    if len(matches) != 1:
        raise ValueError(
            f"no unique peak-FLOP entry for device {device_name!r}; known: {[a.value for a in Accelerator]}"
        )
    return matches[0]


class RecomputePolicy(StrEnum):
    """Activation recomputation that adds hardware FLOPs to a training step."""

    NONE = "none"
    # Megatron recompute_granularity=full, method=uniform, num_layers=1: every decoder layer reruns its forward.
    FULL_LAYERS = "full_layers"
    # Megatron recompute_granularity=selective with core_attn: attention scores and values rerun.
    CORE_ATTENTION = "core_attention"


def causal_attention_pairs(seq_len: int, window: int | None) -> int:
    """Return the number of (query, key) pairs a causal mask admits, optionally limited to ``window`` keys."""
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if window is None or seq_len <= window:
        return seq_len * (seq_len + 1) // 2
    return window * (window + 1) // 2 + (seq_len - window) * window


@dataclass(frozen=True)
class SequenceFlops:
    """FLOPs for a set of sequences; every field is additive across sequences."""

    tokens: int
    forward: float
    # Forward work done again by activation recomputation.
    recompute: float
    # Forward FLOPs the GPU executes beyond the logical model (stored-vs-logical KV heads).
    forward_hardware_excess: float

    @property
    def training_model(self) -> float:
        """Forward plus backward, excluding recomputation: the MFU numerator."""
        return (1 + BACKWARD_TO_FORWARD) * self.forward

    @property
    def training_hardware(self) -> float:
        """Executed training FLOPs including recomputation and stored-projection excess."""
        executed_forward = self.forward + self.forward_hardware_excess
        return (1 + BACKWARD_TO_FORWARD) * executed_forward + self.recompute

    def __add__(self, other: SequenceFlops) -> SequenceFlops:
        return SequenceFlops(
            tokens=self.tokens + other.tokens,
            forward=self.forward + other.forward,
            recompute=self.recompute + other.recompute,
            forward_hardware_excess=self.forward_hardware_excess + other.forward_hardware_excess,
        )


ZERO_FLOPS = SequenceFlops(tokens=0, forward=0.0, recompute=0.0, forward_hardware_excess=0.0)


@dataclass(frozen=True)
class GrugFlopsModel:
    """Dimensions of a Grug MoE / Hero model that determine its FLOPs."""

    hidden: int
    num_heads: int
    head_dim: int
    stored_kv_heads: int
    local_kv_heads: int
    global_kv_heads: int
    long_layers: tuple[bool, ...]
    sliding_window: int
    num_experts: int
    top_k: int
    expert_intermediate: int
    latent_dim: int | None
    shared_intermediate: int
    num_shared_experts: int
    vocab_size: int
    sconv_sites: tuple[str, ...]
    sconv_kernel: int

    @classmethod
    def from_config(cls, config: GrugMoeConfig) -> GrugFlopsModel:
        return cls(
            hidden=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            stored_kv_heads=config.num_key_value_heads,
            local_kv_heads=config.local_kv_heads or config.num_key_value_heads,
            global_kv_heads=config.global_kv_heads or config.num_key_value_heads,
            long_layers=grug_long_layer_flags(config.num_hidden_layers, config.global_every),
            sliding_window=config.sliding_window,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            expert_intermediate=config.intermediate_size,
            latent_dim=config.latent_dim,
            shared_intermediate=config.shared_expert_intermediate_size,
            num_shared_experts=config.num_shared_experts,
            vocab_size=config.vocab_size,
            sconv_sites=config.sconv_sites if config.sconv else (),
            sconv_kernel=config.sconv_kernel,
        )

    @property
    def num_layers(self) -> int:
        return len(self.long_layers)

    def _qkv_projection(self, kv_heads: int) -> float:
        return 2.0 * self.hidden * (self.num_heads + 2 * kv_heads) * self.head_dim

    def _sconv(self, kv_heads: int) -> float:
        channels = {"k": kv_heads * self.head_dim, "attn": self.hidden, "mlp": self.hidden}
        return sum(2.0 * self.sconv_kernel * channels[site] for site in self.sconv_sites)

    def layer_linear_flops_per_token(self, *, is_long: bool) -> float:
        """Forward matmul FLOPs per token of one decoder layer, excluding attention scores."""
        kv_heads = self.global_kv_heads if is_long else self.local_kv_heads
        query_width = self.num_heads * self.head_dim
        attention = (
            self._qkv_projection(kv_heads)
            + 2.0 * query_width * self.hidden  # output projection
            + 2.0 * self.hidden * self.num_heads  # per-head output gate
        )
        expert_width = self.latent_dim if self.latent_dim is not None else self.hidden
        routed = self.top_k * 3 * 2.0 * expert_width * self.expert_intermediate
        latent_projections = 0.0 if self.latent_dim is None else 2 * 2.0 * self.hidden * self.latent_dim
        shared = self.num_shared_experts * 3 * 2.0 * self.hidden * self.shared_intermediate
        router = 2.0 * self.hidden * self.num_experts
        gated_norms = GATED_NORMS_PER_LAYER * 2 * 2.0 * self.hidden * GRUG_GATED_NORM_RANK
        return attention + routed + latent_projections + shared + router + gated_norms + self._sconv(kv_heads)

    def stack_linear_flops_per_token(self) -> float:
        """Forward matmul FLOPs per token outside attention scores, including the LM head."""
        layers = sum(self.layer_linear_flops_per_token(is_long=is_long) for is_long in self.long_layers)
        outside = GATED_NORMS_OUTSIDE_LAYERS * 2 * 2.0 * self.hidden * GRUG_GATED_NORM_RANK
        return layers + outside + 2.0 * self.hidden * self.vocab_size

    def attention_score_flops(self, seq_len: int) -> float:
        """Forward QK^T and PV FLOPs for one causal sequence across all layers."""
        per_pair = 2 * 2.0 * self.num_heads * self.head_dim
        return sum(
            per_pair * causal_attention_pairs(seq_len, None if is_long else self.sliding_window)
            for is_long in self.long_layers
        )

    def sequence_flops(self, seq_len: int, recompute: RecomputePolicy) -> SequenceFlops:
        """FLOPs for one unpadded sequence of ``seq_len`` attended tokens."""
        scores = self.attention_score_flops(seq_len)
        forward = seq_len * self.stack_linear_flops_per_token() + scores
        if recompute is RecomputePolicy.NONE:
            recomputed = 0.0
        elif recompute is RecomputePolicy.CORE_ATTENTION:
            recomputed = scores
        else:
            lm_head = 2.0 * self.hidden * self.vocab_size
            outside_norms = GATED_NORMS_OUTSIDE_LAYERS * 2 * 2.0 * self.hidden * GRUG_GATED_NORM_RANK
            recomputed = forward - seq_len * (lm_head + outside_norms)
        excess_per_token = sum(
            self._qkv_projection(self.stored_kv_heads)
            - self._qkv_projection(kv)
            + self._sconv(self.stored_kv_heads)
            - self._sconv(kv)
            for kv in (self.global_kv_heads if is_long else self.local_kv_heads for is_long in self.long_layers)
        )
        return SequenceFlops(
            tokens=seq_len,
            forward=forward,
            recompute=recomputed,
            forward_hardware_excess=seq_len * excess_per_token,
        )

    def batch_flops(self, seq_lens: list[int], recompute: RecomputePolicy) -> SequenceFlops:
        """Sum ``sequence_flops`` over a batch's per-sequence attended lengths."""
        total = ZERO_FLOPS
        for seq_len in seq_lens:
            total = total + self.sequence_flops(seq_len, recompute)
        return total


def mfu(flops: float, seconds: float, num_gpus: int, accelerator: Accelerator) -> float:
    """Fraction of the cluster's dense BF16 peak that ``flops`` in ``seconds`` represents."""
    if seconds <= 0 or num_gpus <= 0:
        raise ValueError(f"seconds and num_gpus must be positive, got {seconds} and {num_gpus}")
    return flops / (seconds * num_gpus * PEAK_BF16_DENSE_FLOPS[accelerator])
