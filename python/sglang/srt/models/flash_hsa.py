"""
FlashHSA model definition for SGLang.

This provides a native SGLang model entry class named `HSAForCausalLM`, matching
FlashHSA config.json:
  - "architectures": ["HSAForCausalLM"]
  - "model_type": "flash_hsa"

Key FlashHSA semantic differences we must respect at load time:
  - Landmark token (LMK) is a *real* token id. FlashHSA uses `lmk_id == vocab_size`
    and allocates embeddings with `next_of_y(vocab_size + 1, 32)` rows.
  - We keep `config.vocab_size` as the *base* vocab size (so logits/sampling and
    stop conditions behave as usual), but allocate embed/lm_head weights using the
    padded vocab size so LMK has a valid embedding row and weights can be loaded.
  - Sliding-window size used for the SWA→HSA merged path is
    `sliding_window_merging_size` and gated by `use_sliding_window_merging`.

NOTE: The actual fused SWA/HSA kernels are implemented by attention backends.
This file focuses on model construction + weight loading compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.olmo2 import Olmo2MLP
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import make_layers
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def next_of_y(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _get_sliding_window_merging_size(config) -> Optional[int]:
    if hasattr(config, "sliding_window_merging_size") and getattr(
        config, "sliding_window_merging_size"
    ) is not None:
        return int(getattr(config, "sliding_window_merging_size"))
    return None


def _get_sliding_window_attention_size(config) -> Optional[int]:
    if hasattr(config, "sliding_window_attention_size") and getattr(
        config, "sliding_window_attention_size"
    ) is not None:
        return int(getattr(config, "sliding_window_attention_size"))
    return None


def _get_flashhsa_padded_vocab_size(config) -> int:
    base_vocab_size = int(getattr(config, "vocab_size"))
    # FlashHSA: lmk_id == base_vocab_size, so embeddings must have >= base_vocab_size + 1 rows.
    return int(next_of_y(base_vocab_size + 1, 32))


def _get_innerx_split_counts_total(config) -> Dict[str, int]:
    """Compute InnerX head splits (total heads, not TP-partitioned)."""
    h_q = int(getattr(config, "num_attention_heads"))
    h_kv = int(getattr(config, "num_key_value_heads"))
    hsa_heads = int(getattr(config, "hsa_heads"))
    hsa_qk_ratio = int(getattr(config, "hsa_qk_ratio"))
    if hsa_heads <= 0 or hsa_heads > h_q:
        raise ValueError(f"Invalid hsa_heads={hsa_heads} for num_attention_heads={h_q}")
    if h_q % hsa_heads != 0:
        raise ValueError("InnerX: num_attention_heads must be divisible by hsa_heads")
    hsa_denom = h_q // hsa_heads
    if h_kv % hsa_denom != 0:
        raise ValueError("InnerX: num_key_value_heads must be divisible by hsa_denom")
    # In ultra reference:
    # - SWA uses (denom-1)/denom of q/k/v heads
    # - HSA uses 1/denom of q heads and 1/denom of kv heads
    hq_swa = h_q * (hsa_denom - 1) // hsa_denom
    hq_hsa = h_q // hsa_denom
    hk_swa = h_kv * (hsa_denom - 1) // hsa_denom
    hk_hsa = h_kv // hsa_denom
    if hq_swa + hq_hsa != h_q:
        raise AssertionError("InnerX: q head split does not sum to total")
    if hk_swa + hk_hsa != h_kv:
        raise AssertionError("InnerX: kv head split does not sum to total")
    # Optional sanity: within the HSA branch, q/k ratio should match config expectation.
    if hk_hsa > 0 and (hq_hsa // hk_hsa) != hsa_qk_ratio:
        logger.warning(
            "InnerX: HSA branch q/k ratio mismatch: hq_hsa=%d hk_hsa=%d hsa_qk_ratio=%d",
            hq_hsa,
            hk_hsa,
            hsa_qk_ratio,
        )
    return dict(
        h_q=h_q,
        h_kv=h_kv,
        hsa_heads=hsa_heads,
        hsa_qk_ratio=hsa_qk_ratio,
        hsa_denom=hsa_denom,
        hq_swa=hq_swa,
        hq_hsa=hq_hsa,
        hk_swa=hk_swa,
        hk_hsa=hk_hsa,
    )


class FlashHSAInnerXAttention(nn.Module):
    """OLMo3-style SWA attention with full-tensor QK norm."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int,
        rope_theta: float,
        rope_scaling: Optional[Dict[str, Any]],
        head_dim: Optional[int],
        max_position_embeddings: int,
        quant_config=None,
        rms_norm_eps: float,
        attention_bias: bool,
        prefix: str,
        sliding_window_size: int = -1,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.total_num_heads = int(num_heads)
        self.total_num_kv_heads = int(num_kv_heads)

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size

        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)

        self.head_dim = int(head_dim or (self.hidden_size // self.total_num_heads))
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.layer_id = int(layer_id)
        self.alt_stream = alt_stream
        self.tp_size = attn_tp_size
        self.tp_rank = attn_tp_rank

        # OLMo3: RMSNorm on full projected tensor (all heads concatenated).
        # Must use TOTAL head count (not TP-local) so that the norm statistics
        # are identical regardless of TP degree.  In forward() we all-gather
        # before norm and split back afterwards, matching OLMo2 semantics.
        self.q_norm = RMSNorm(self.total_num_heads * self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.total_num_kv_heads * self.head_dim, eps=rms_norm_eps)

        # Explicit q/k/v projections (weight names match HF-style checkpoints).
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=self.layer_id,
            sliding_window_size=sliding_window_size,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # OLMo3 full-tensor QK norm: all-gather → norm → split (matches OLMo2).
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            q = split_tensor_along_last_dim(q, self.tp_size)[self.tp_rank]
            k = split_tensor_along_last_dim(k, self.tp_size)[self.tp_rank]
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class FlashHSAInnerXHierarchicalSparseAttention(nn.Module):
    """InnerX split-head HSA layer (ultra reference semantics for decode)."""

    def __init__(
        self,
        *,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.layer_id = int(layer_id)
        self.alt_stream = alt_stream

        rope_theta = float(getattr(config, "rope_theta", 1000000))
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        split = _get_innerx_split_counts_total(config)
        self.total_num_heads = int(split["h_q"])
        self.total_num_kv_heads = int(split["h_kv"])
        self.hq_swa_total = int(split["hq_swa"])
        self.hq_hsa_total = int(split["hq_hsa"])
        self.hk_swa_total = int(split["hk_swa"])
        self.hk_hsa_total = int(split["hk_hsa"])
        self.hsa_qk_ratio = int(split["hsa_qk_ratio"])

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        if self.total_num_heads % attn_tp_size != 0:
            raise ValueError("InnerX: num_attention_heads must be divisible by attention TP size")
        if self.total_num_kv_heads % attn_tp_size != 0:
            raise ValueError(
                "InnerX: num_key_value_heads must be divisible by attention TP size "
                "(kv head replication is not supported for InnerX split-head yet)"
            )

        self.num_heads = self.total_num_heads // attn_tp_size
        self.num_kv_heads = self.total_num_kv_heads // attn_tp_size
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.total_num_heads))
        self.scaling = self.head_dim**-0.5

        # Local (TP-partitioned) splits.
        self.hq_swa = self.hq_swa_total // attn_tp_size
        self.hq_hsa = self.hq_hsa_total // attn_tp_size
        self.hk_swa = self.hk_swa_total // attn_tp_size
        self.hk_hsa = self.hk_hsa_total // attn_tp_size
        self.unified_retrieval = bool(getattr(config, "unified_retrieval", False))
        self.retrieval_head_num = 1 if self.unified_retrieval else self.hk_hsa
        if self.hq_swa + self.hq_hsa != self.num_heads:
            raise AssertionError("InnerX: local q split mismatch")
        if self.hk_swa + self.hk_hsa != self.num_kv_heads:
            raise AssertionError("InnerX: local kv split mismatch")
        if self.hk_hsa <= 0 or self.hq_hsa <= 0:
            raise ValueError("InnerX: HSA partition must be non-empty")
        if (self.hq_hsa // self.hk_hsa) != self.hsa_qk_ratio:
            logger.warning(
                "InnerX: local HSA q/k ratio mismatch: hq_hsa=%d hk_hsa=%d ratio=%d",
                self.hq_hsa,
                self.hk_hsa,
                self.hsa_qk_ratio,
            )

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        # Bug 10 fix: respect layerwise_qk_norm config.
        # - layerwise_qk_norm=True  → per-layer norm on full projected tensor [total_heads * head_dim]
        #   (same as HSAFullAttention; weight shape = [hidden_size])
        # - layerwise_qk_norm=False → per-head norm [head_dim], applied via reshape trick
        #   SWA and HSA branches share the same norm weights in per-head mode.
        self._layerwise_qk = bool(getattr(config, "layerwise_qk_norm", False))
        if self._layerwise_qk:
            self.q_norm = RMSNorm(self.total_num_heads * self.head_dim, eps=rms_norm_eps, **norm_kwargs)
            self.k_norm = RMSNorm(self.total_num_kv_heads * self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        else:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        # lmk_q_norm is only needed when enable_lmk_q_proj=True.
        # Created conditionally below alongside lmk_q_proj.
        self.lmk_q_norm = None

        # Projections (explicit matrices, matching ultra reference naming).
        # When hsa_denom == 1 (all heads are HSA), there are no SWA heads and
        # q/k/v_proj are not needed.  Skip creation to avoid zero-sized linears.
        self.has_swa_branch = self.hq_swa_total > 0
        if self.has_swa_branch:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hq_swa_total * self.head_dim,
                bias=attention_bias,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("q_proj", prefix),
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hk_swa_total * self.head_dim,
                bias=attention_bias,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("k_proj", prefix),
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hk_swa_total * self.head_dim,
                bias=attention_bias,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("v_proj", prefix),
            )
        else:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None

        self.hsa_q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hq_hsa_total * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("hsa_q_proj", prefix),
        )
        self.hsa_k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hk_hsa_total * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("hsa_k_proj", prefix),
        )
        self.hsa_v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hk_hsa_total * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("hsa_v_proj", prefix),
        )
        self.enable_lmk_q_proj = bool(getattr(config, "enable_lmk_q_proj", False))
        retrieval_head_num_total = 1 if self.unified_retrieval else self.hk_hsa_total
        default_retrieval_dim = self.hk_hsa_total * self.head_dim
        self.retrieval_dim = int(getattr(config, "retrieval_dim", None) or default_retrieval_dim)
        # lmk_q_norm uses the FULL (non-TP-split) retrieval_dim so that norm
        # statistics are identical regardless of TP degree.  In forward() we
        # all-gather before norm and split back afterwards.
        retrieval_head_num_local = 1 if self.unified_retrieval else self.hk_hsa
        self.lmk_q_norm_dim = self.retrieval_dim // retrieval_head_num_total
        self.lmk_q_norm_dim_local = (self.retrieval_dim // attn_tp_size) // retrieval_head_num_local
        if self.enable_lmk_q_proj:
            self.lmk_q_norm = RMSNorm(self.lmk_q_norm_dim, eps=rms_norm_eps, **norm_kwargs)
            self.lmk_q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.retrieval_dim,
                bias=attention_bias,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("lmk_q_proj", prefix),
            )
        else:
            self.lmk_q_proj = None

        self.enable_gate_attn = bool(getattr(config, "enable_gate", False))
        if self.enable_gate_attn:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hq_hsa_total * self.head_dim,
                bias=attention_bias,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("gate_proj", prefix),
            )
        else:
            self.gate_proj = None

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        # RoPE is applied ONLY to the SWA branch in the ultra reference.
        # When apply_hsa_rope=True (e.g. OLMo3), RoPE is also applied to the HSA branch.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.apply_hsa_rope = bool(getattr(config, "apply_hsa_rope", False))

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=self.layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # --- HSA branch projections (always needed) ---
        hsa_q, _ = self.hsa_q_proj(hidden_states)
        hsa_k, _ = self.hsa_k_proj(hidden_states)
        hsa_v, _ = self.hsa_v_proj(hidden_states)

        # --- SWA branch projections ---
        if self.has_swa_branch:
            swa_q, _ = self.q_proj(hidden_states)
            swa_k, _ = self.k_proj(hidden_states)
            swa_v, _ = self.v_proj(hidden_states)

        # --- QK norm ---
        if self._layerwise_qk:
            # Per-layer norm: concat SWA + HSA, norm with full [hidden_size] weight, split back
            if self.has_swa_branch:
                swa_q_dim = swa_q.shape[-1]
                swa_k_dim = swa_k.shape[-1]
                full_q = torch.cat([swa_q, hsa_q], dim=-1)
                full_k = torch.cat([swa_k, hsa_k], dim=-1)
                full_q = self.q_norm(full_q)
                full_k = self.k_norm(full_k)
                swa_q = full_q[..., :swa_q_dim].contiguous()
                hsa_q = full_q[..., swa_q_dim:].contiguous()
                swa_k = full_k[..., :swa_k_dim].contiguous()
                hsa_k = full_k[..., swa_k_dim:].contiguous()
            else:
                hsa_q = self.q_norm(hsa_q)
                hsa_k = self.k_norm(hsa_k)
        else:
            # Per-head norm: reshape to [..., head_dim], norm, reshape back
            if self.has_swa_branch:
                swa_q = self.q_norm(swa_q.view(-1, self.head_dim)).view(swa_q.shape)
                swa_k = self.k_norm(swa_k.view(-1, self.head_dim)).view(swa_k.shape)
            hsa_q = self.q_norm(hsa_q.view(-1, self.head_dim)).view(hsa_q.shape)
            hsa_k = self.k_norm(hsa_k.view(-1, self.head_dim)).view(hsa_k.shape)

        # --- RoPE ---
        if self.has_swa_branch:
            swa_q, swa_k = self.rotary_emb(positions, swa_q, swa_k)
        if self.apply_hsa_rope:
            hsa_q, hsa_k = self.rotary_emb(positions, hsa_q, hsa_k)

        # --- Selection query ---
        if self.enable_lmk_q_proj and self.lmk_q_proj is not None:
            # Separate lmk_q_proj for page selection (NO RoPE).
            lmk_q, _ = self.lmk_q_proj(hidden_states)  # [T, retrieval_dim // tp_size]
            rhn = self.retrieval_head_num  # 1 for unified, hk_hsa for standard
            # All-gather → norm → split to match training semantics.
            attn_tp_size = get_attention_tp_size()
            if attn_tp_size > 1:
                lmk_q = tensor_model_parallel_all_gather(lmk_q.contiguous())
            norm_dim = self.lmk_q_norm_dim  # full (non-TP-split) norm dim
            rhn_for_norm = 1 if self.unified_retrieval else (self.hk_hsa * attn_tp_size if attn_tp_size > 1 else self.hk_hsa)
            lmk_q_3 = lmk_q.view(-1, rhn_for_norm, norm_dim)
            # RMSNorm expects 2D; apply over norm_dim with flatten/unflatten.
            lmk_q_norm = self.lmk_q_norm(lmk_q_3.reshape(-1, norm_dim)).reshape(
                lmk_q_3.shape[0], lmk_q_3.shape[1], lmk_q_3.shape[2]
            )  # [T, rhn_for_norm, norm_dim]
            if attn_tp_size > 1:
                # Split back to local heads.
                tp_rank = get_attention_tp_rank()
                lmk_q_flat = lmk_q_norm.reshape(lmk_q_norm.shape[0], -1)  # [T, retrieval_dim]
                lmk_q_flat = split_tensor_along_last_dim(lmk_q_flat, attn_tp_size)[tp_rank]
                local_norm_dim = self.lmk_q_norm_dim_local
                lmk_q_norm = lmk_q_flat.view(-1, rhn, local_norm_dim)  # [T, rhn, local_norm_dim]
            if self.unified_retrieval:
                # unified_retrieval: keep as single retrieval head [T, 1, retrieval_dim // tp].
                # The backend will all-gather + group+sum the KV cache to match this dim.
                sel_q = lmk_q_norm  # [T, 1, retrieval_dim // tp_size]
            else:
                # Standard: expand kv heads to q-head space for selector: [T, hq_hsa, D]
                assert self.hq_hsa % self.hk_hsa == 0
                G = self.hq_hsa // self.hk_hsa
                sel_q = (
                    lmk_q_norm[:, :, None, :]
                    .expand(lmk_q_norm.shape[0], self.hk_hsa, G, self.head_dim)
                    .reshape(lmk_q_norm.shape[0], self.hq_hsa, self.head_dim)
                    .contiguous()
                )
        else:
            # No separate lmk_q_proj: use hsa_q directly for selection
            # (matches official behavior when enable_lmk_q_proj=False).
            sel_q = hsa_q.view(-1, self.hq_hsa, self.head_dim)

        # Concatenate heads into the single KV cache layout: [SWA | HSA].
        if self.has_swa_branch:
            q_full = torch.cat([swa_q, hsa_q], dim=-1)
            k_full = torch.cat([swa_k, hsa_k], dim=-1)
            v_full = torch.cat([swa_v, hsa_v], dim=-1)
        else:
            q_full = hsa_q
            k_full = hsa_k
            v_full = hsa_v

        # Use hsa_sliding_window if configured, else fall back to merging window
        hsa_sw = getattr(self.config, "hsa_sliding_window", None)
        if hsa_sw is not None:
            window = int(hsa_sw)
        elif bool(getattr(self.config, "use_sliding_window_merging", False)):
            window = _get_sliding_window_merging_size(self.config)
        else:
            window = None

        # Upper SWA branch sliding window (config.sliding_window in official model).
        # In the official code, only FA2 actually enforces this window on the SWA
        # heads inside HSA layers.  We pass it to the triton extend kernel which
        # implements equivalent SWA masking.
        _upper_sw = getattr(self.config, "sliding_window_attention_size", None)
        if _upper_sw is None:
            # Fall back: official uses config.sliding_window for both merging + upper SWA
            _upper_sw = getattr(self.config, "hsa_sliding_window", None)
        upper_swa_window = int(_upper_sw) if _upper_sw is not None else None

        hsa_split_head_info = dict(
            hq_swa=int(self.hq_swa),
            hq_hsa=int(self.hq_hsa),
            h_swa=int(self.hk_swa),
            h_hsa=int(self.hk_hsa),
            unified_retrieval=self.unified_retrieval,
            retrieval_dim=self.retrieval_dim if self.unified_retrieval else None,
            swa_window_size=window,
            upper_swa_window_size=upper_swa_window,
            swa_exclude_lmk=False,
        )
        attn_output = self.attn(
            q_full,
            k_full,
            v_full,
            forward_batch,
            hsa_split_head_info=hsa_split_head_info,
            hsa_selection_q=sel_q,
        )

        if self.enable_gate_attn and self.gate_proj is not None:
            gate, _ = self.gate_proj(hidden_states)  # [T, hq_hsa*D]
            gate_3 = gate.view(-1, self.hq_hsa, self.head_dim)
            attn_3 = attn_output.view(-1, self.num_heads, self.head_dim)
            attn_3[:, self.hq_swa : self.hq_swa + self.hq_hsa, :] = attn_3[
                :, self.hq_swa : self.hq_swa + self.hq_hsa, :
            ] * torch.sigmoid(gate_3)
            attn_output = attn_3.reshape(attn_output.shape[0], -1)

        out, _ = self.o_proj(attn_output)
        return out


class FlashHSAInnerXDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config=None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.layer_id = int(layer_id)

        rope_theta = float(getattr(config, "rope_theta", 1000000))
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        head_dim = getattr(config, "head_dim", None)

        interleave = int(getattr(config, "full_attn_interleave", 0) or 0)
        num_swa_layers = int(getattr(config, "num_swa_layers", 0))
        if self.layer_id < num_swa_layers:
            is_hsa_layer = False
        else:
            adjusted_idx = self.layer_id - num_swa_layers
            is_hsa_layer = bool(interleave > 0 and (adjusted_idx % interleave) == (interleave - 1))

        if is_hsa_layer:
            self.self_attn = FlashHSAInnerXHierarchicalSparseAttention(
                config=config,
                layer_id=self.layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
                alt_stream=alt_stream,
            )
        else:
            swa_window = -1
            if bool(getattr(config, "use_sliding_window_attention", False)):
                swa_window = int(
                    getattr(config, "sliding_window_attention_size") or -1
                )
            self.self_attn = FlashHSAInnerXAttention(
                hidden_size=self.hidden_size,
                num_heads=int(getattr(config, "num_attention_heads")),
                num_kv_heads=int(getattr(config, "num_key_value_heads")),
                layer_id=self.layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
                attention_bias=bool(getattr(config, "attention_bias", False)),
                sliding_window_size=swa_window,
                prefix=add_prefix("self_attn", prefix),
                alt_stream=alt_stream,
            )

        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        # OLMo3 post-norm: norm AFTER attention/MLP output
        self.mlp = Olmo2MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # OLMo3 post-norm
        residual = hidden_states
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class FlashHSAInnerXModel(nn.Module):
    """OLMo3-based model with InnerX LHSA decoder layers."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        _is_cuda = torch.cuda.is_available()
        self.alt_stream = torch.cuda.Stream() if _is_cuda else None

        padded_vocab_size = _get_flashhsa_padded_vocab_size(config)
        self.embed_tokens = VocabParallelEmbedding(
            padded_vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            padding_size=32,
            prefix=add_prefix("embed_tokens", prefix),
        )
        result = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: FlashHSAInnerXDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )
        if isinstance(result, tuple):
            self.layers, self.start_layer, self.end_layer = result
        else:
            self.layers = result
            self.start_layer = 0
            self.end_layer = config.num_hidden_layers
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(positions, hidden_states, forward_batch)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class HSAForCausalLM(nn.Module):
    """
    OLMo3-based FlashHSA / LHSA entry class for SGLang.
    Standalone — no Qwen3 inheritance.
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        if getattr(config, "hsa_heads", None) is None or getattr(config, "hsa_qk_ratio", None) is None:
            raise ValueError(
                "InnerX ultra requires `hsa_heads` and `hsa_qk_ratio` in config."
            )
        self.model = FlashHSAInnerXModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        padded_vocab_size = _get_flashhsa_padded_vocab_size(config)

        # LM head
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    padded_vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    padding_size=32,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.pooler import Pooler, PoolingType

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    # ---- FlashHSA-specific helpers for engine ----

    def get_attention_sliding_window_size(self) -> Optional[int]:
        # Used by ModelRunner to configure standalone sliding-window attention layers.
        if not bool(getattr(self.config, "use_sliding_window_attention", False)):
            return None
        return _get_sliding_window_attention_size(self.config)

    def get_flashhsa_merging_sliding_window_size(self) -> Optional[int]:
        # Used by HSAAttnBackend selection (SWA→HSA exclusion) / future merged kernels.
        if not bool(getattr(self.config, "use_sliding_window_merging", False)):
            return None
        return _get_sliding_window_merging_size(self.config)

    def get_flashhsa_hsa_layer_ids(self) -> List[int]:
        """Default per-layer HSA pattern from FlashHSA: every `full_attn_interleave`-th layer."""
        interleave = int(getattr(self.config, "full_attn_interleave", 0) or 0)
        num_layers = int(getattr(self.config, "num_hidden_layers", 0) or 0)
        num_swa_layers = int(getattr(self.config, "num_swa_layers", 0))
        if interleave <= 0 or num_layers <= 0:
            return []
        result = []
        for i in range(num_layers):
            if i < num_swa_layers:
                continue
            adj = i - num_swa_layers
            if (adj % interleave) == (interleave - 1):
                result.append(i)
        return result

    def get_flashhsa_chunk_size(self) -> Optional[int]:
        if hasattr(self.config, "chunk_size") and getattr(self.config, "chunk_size") is not None:
            return int(getattr(self.config, "chunk_size"))
        return None

    # ---- Weight loading ----
    # We only support the InnerX ultra layout (explicit q_proj/k_proj/v_proj + hsa_* projections).
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # OLMo3 MLP: gate_proj + up_proj stacked into gate_up_proj
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "Embedding" in str(getattr(self.config, "name_or_path", "")):
                name = add_prefix(name, "model")

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            if getattr(self.config, "tie_word_embeddings", False) and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue

            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # Handle stacked params (e.g. OLMo3 gate_up_proj from gate_proj + up_proj)
            handled_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                stacked_name = name.replace(weight_name, param_name)
                if stacked_name in params_dict:
                    param = params_dict[stacked_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    handled_stacked = True
                    break
            if handled_stacked:
                continue

            # TP 切分 attention norm 权重（q_norm / k_norm / lmk_q_norm）
            # SWA 层的 OLMo3 风格 QK norm 和 HSA 层的 lmk_q_norm 在 TP 模式下
            # param 维度 = total / tp_size，需要从 checkpoint 的完整权重中切出
            # 当前 rank 对应的分片。HSA 层的 per-head q_norm/k_norm 维度为
            # head_dim，不受 TP 影响，通过 param.shape != loaded_weight.shape
            # 条件自动跳过切分。
            if (
                name in params_dict
                and ("q_norm" in name or "k_norm" in name)
            ):
                param = params_dict[name]
                if param.shape != loaded_weight.shape:
                    tp_rank = get_attention_tp_rank()
                    shard_size = param.shape[0]
                    start = tp_rank * shard_size
                    loaded_weight = loaded_weight[start : start + shard_size]
                param.data.copy_(loaded_weight)
                continue

            # Direct load
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.warning("Parameter %s not found in params_dict", name)

    # Weight loading is now explicitly implemented above.


# Model registry entrypoint
EntryClass = HSAForCausalLM
