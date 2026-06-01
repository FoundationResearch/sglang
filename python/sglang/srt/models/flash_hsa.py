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
import os
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
from sglang.srt.layers.elementwise import fused_post_norm_add
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

        # --- Fused QKV (3 separate matmuls → 1) ---
        # Same launch-overhead motivation as the HSA variant below.  Built lazily
        # on first forward (also called from load_weights() for safety).
        self._fused_qkv_w: Optional[torch.Tensor] = None
        self._fused_qkv_b: Optional[torch.Tensor] = None
        self._fused_split_sizes: Optional[List[int]] = None
        self._fusion_enabled: bool = os.environ.get("HSA_DISABLE_QKV_FUSION", "0") != "1"

    @torch.no_grad()
    def _fuse_projections(self) -> None:
        if not self._fusion_enabled or self._fused_qkv_w is not None:
            return
        projs = [self.q_proj, self.k_proj, self.v_proj]
        weights, biases = [], []
        for p in projs:
            w = getattr(p, "weight", None)
            if w is None or not isinstance(w, torch.Tensor):
                return
            weights.append(w)
            b = getattr(p, "bias", None)
            biases.append(b if (b is not None and isinstance(b, torch.Tensor)) else None)
        in_size = weights[0].shape[1]
        for w in weights:
            if w.shape[1] != in_size:
                return
        any_b, all_b = any(b is not None for b in biases), all(b is not None for b in biases)
        if any_b and not all_b:
            return
        self._fused_qkv_w = torch.cat([w.contiguous() for w in weights], dim=0).contiguous()
        self._fused_qkv_b = (
            torch.cat([b.contiguous() for b in biases], dim=0).contiguous() if all_b else None
        )
        self._fused_split_sizes = [w.shape[0] for w in weights]

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self._fusion_enabled and self._fused_qkv_w is None:
            self._fuse_projections()
        if self._fused_qkv_w is not None:
            fused = torch.nn.functional.linear(
                hidden_states, self._fused_qkv_w, self._fused_qkv_b
            )
            q, k, v = (c.contiguous() for c in fused.split(self._fused_split_sizes, dim=-1))
        else:
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
        # 2026-05-21 single-projection rewrite: new upstream lhsa_layer.py uses
        # nn.Linear(d_model, d_model) for lmk_q_proj (h_q heads × head_dim, one
        # per attention head), then per-head RMSNorm over head_dim. The old dual-
        # projection path here used hk_hsa-head lmk_q_proj of size retrieval_dim
        # and broadcast kv→q in forward. We switch to the new shape when the
        # config is in canonical "no SWA branch" mode (hsa_denom=1, i.e.
        # hsa_heads == num_attention_heads). For configs that still have an SWA
        # branch (hsa_denom > 1) we keep the old behaviour for back-compat.
        _hsa_heads_cfg = int(getattr(config, "hsa_heads", self.total_num_heads))
        _hsa_denom_cfg = self.total_num_heads // _hsa_heads_cfg if _hsa_heads_cfg > 0 else 1
        self.lmk_q_full_dim = (_hsa_denom_cfg == 1)

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
            if self.lmk_q_full_dim:
                # New-arch lmk_q_proj: one head_dim per q-head, no GQA broadcast in forward.
                _lmk_q_out_dim_total = self.hq_hsa_total * self.head_dim
                self.lmk_q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
            else:
                _lmk_q_out_dim_total = self.retrieval_dim
                self.lmk_q_norm = RMSNorm(self.lmk_q_norm_dim, eps=rms_norm_eps, **norm_kwargs)
            self.lmk_q_proj = ColumnParallelLinear(
                self.hidden_size,
                _lmk_q_out_dim_total,
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

        # Per-q-head lmk_k path (mirrors the official chunk_attn_pool default
        # MHA-style mode).  Active when (a) we have lmk_q_proj producing
        # h_q-headed query *and* (b) HSA branch uses GQA on the q-side
        # (hq_hsa > hk_hsa).  When active, prefill computes the chunk-aggregated
        # h_q-headed lmk_k once per chunk and stores it in the backend's
        # LandmarkLmkKPool; the selector reads it at decode time and passes
        # G = hq_hsa // hk_hsa to the topk kernel.
        self._per_qhead_lmk_k_active = bool(
            self.enable_lmk_q_proj
            and self.lmk_q_full_dim
            and (self.hq_hsa > self.hk_hsa)
        )
        self.chunk_size = int(getattr(config, "chunk_size", 64))

        # --- Fused QKV projection ---
        # At decode bs=1 each of the 6-7 ColumnParallelLinear matmuls
        # (q/k/v + hsa_q/hsa_k/hsa_v + lmk_q) is launch-overhead bound.
        # _fuse_projections() concatenates their weights into a single fused
        # weight; forward() then does ONE F.linear + split.  Set lazily by
        # the model after load_weights() so the loader is untouched.
        self._fused_qkv_w: Optional[torch.Tensor] = None
        self._fused_qkv_b: Optional[torch.Tensor] = None
        self._fused_split_sizes: Optional[List[int]] = None
        # Toggle (env var) to disable fusion for debugging.
        self._fusion_enabled: bool = os.environ.get("HSA_DISABLE_QKV_FUSION", "0") != "1"

    # ------------------------------------------------------------------
    # Fused-QKV helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _fuse_projections(self) -> None:
        """Concatenate q/k/v/hsa_q/hsa_k/hsa_v/lmk_q projection weights into a
        single fused weight in the layout [q_full | k_full | v_full | lmk_q],
        where q_full = [swa_q ; hsa_q] (k/v similarly).  This lets the forward
        slice q_full/k_full/v_full as contiguous chunks (no torch.cat) AND norm
        them in a single call per branch (instead of one per swa/hsa half).

        Skipped (returns silently) when:
          * fusion disabled by env var
          * any projection uses quantized weights
          * any projection has bias and the rest don't (mixed bias)
        """
        if not self._fusion_enabled or self._fused_qkv_w is not None:
            return

        # Group projections by role so q_full / k_full / v_full come out as
        # contiguous slices of the fused output.
        q_projs: List = []
        k_projs: List = []
        v_projs: List = []
        if self.has_swa_branch:
            q_projs.append(self.q_proj)
            k_projs.append(self.k_proj)
            v_projs.append(self.v_proj)
        q_projs.append(self.hsa_q_proj)
        k_projs.append(self.hsa_k_proj)
        v_projs.append(self.hsa_v_proj)

        projs_ordered = q_projs + k_projs + v_projs
        if self.enable_lmk_q_proj and self.lmk_q_proj is not None:
            projs_ordered.append(self.lmk_q_proj)

        weights = []
        biases = []
        for p in projs_ordered:
            w = getattr(p, "weight", None)
            if w is None or not isinstance(w, torch.Tensor):
                return  # quantized / non-standard, skip
            weights.append(w)
            b = getattr(p, "bias", None)
            biases.append(b if (b is not None and isinstance(b, torch.Tensor)) else None)

        # ColumnParallelLinear weight shape = [out_per_partition, in_size]
        in_size = weights[0].shape[1]
        for w in weights:
            if w.shape[1] != in_size:
                return  # shape mismatch, can't fuse

        fused_w = torch.cat([w.contiguous() for w in weights], dim=0).contiguous()
        any_bias = any(b is not None for b in biases)
        all_bias = all(b is not None for b in biases)
        fused_b: Optional[torch.Tensor] = None
        if any_bias and not all_bias:
            return  # mixed; skip fusion
        if all_bias:
            fused_b = torch.cat([b.contiguous() for b in biases], dim=0).contiguous()

        # Sum per-role rows = the contiguous slice for q_full / k_full / v_full.
        q_full_dim = sum(p.weight.shape[0] for p in q_projs)
        k_full_dim = sum(p.weight.shape[0] for p in k_projs)
        v_full_dim = sum(p.weight.shape[0] for p in v_projs)
        lmk_dim = (
            self.lmk_q_proj.weight.shape[0]
            if (self.enable_lmk_q_proj and self.lmk_q_proj is not None)
            else 0
        )

        self._fused_qkv_w = fused_w
        self._fused_qkv_b = fused_b
        # Coarse split = how forward() reads the fused output (no cat needed).
        if lmk_dim > 0:
            self._fused_split_sizes = [q_full_dim, k_full_dim, v_full_dim, lmk_dim]
        else:
            self._fused_split_sizes = [q_full_dim, k_full_dim, v_full_dim]
        # Per-branch sizes within q_full / k_full / v_full (used for the rare
        # !apply_hsa_rope split before RoPE).
        self._fused_q_swa_dim = (
            self.q_proj.weight.shape[0] if self.has_swa_branch else 0
        )
        self._fused_k_swa_dim = (
            self.k_proj.weight.shape[0] if self.has_swa_branch else 0
        )
        self._fused_v_swa_dim = (
            self.v_proj.weight.shape[0] if self.has_swa_branch else 0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # --- Projections ---
        # Fast path: a single fused GEMM laid out as
        #   [q_full | k_full | v_full | (lmk_q)]
        # where q_full = [swa_q ; hsa_q] (k/v similarly).  Slicing the fused
        # output already produces q_full / k_full / v_full as adjacent regions,
        # so we don't need an explicit torch.cat downstream.  We also norm and
        # RoPE the full q/k tensors in a single call each (when apply_hsa_rope
        # is on, which is the common config).
        if self._fusion_enabled and self._fused_qkv_w is None:
            # Lazy build on first forward (covers paths that bypass load_weights,
            # e.g. dev/align/bootstrap.py which does direct param.data.copy_).
            self._fuse_projections()
        # Local fast-path flag: per-head norm AND apply_hsa_rope (or no SWA
        # branch).  These are the configs where we can keep q_full/k_full/v_full
        # unified throughout — matches both 7B HSA (olmo) and 345M (qwen).
        unified_path = (
            self._fused_qkv_w is not None
            and not self._layerwise_qk
            and (self.apply_hsa_rope or not self.has_swa_branch)
        )
        swa_q = swa_k = swa_v = None
        hsa_q = hsa_k = hsa_v = None
        q_full = k_full = v_full = None
        lmk_q_fused = None

        if self._fused_qkv_w is not None:
            fused = torch.nn.functional.linear(
                hidden_states, self._fused_qkv_w, self._fused_qkv_b
            )
            # split() returns views that share strides with the parent — these
            # are NOT contiguous along dim=-1 unless T==1, so downstream .view()
            # ops would fail.  Calling .contiguous() once per chunk is a tiny
            # memcpy (bs=1 → ~1-3KB each) that's cheap vs the matmul savings.
            chunks = [
                c.contiguous() for c in fused.split(self._fused_split_sizes, dim=-1)
            ]
            q_full, k_full, v_full = chunks[0], chunks[1], chunks[2]
            if self.enable_lmk_q_proj and self.lmk_q_proj is not None:
                lmk_q_fused = chunks[3]
            if not unified_path:
                # Re-split q_full/k_full/v_full into swa/hsa halves so the
                # rare branches (_layerwise_qk, !apply_hsa_rope) match the
                # original semantics.
                if self.has_swa_branch:
                    swa_q, hsa_q = q_full.split(
                        [self._fused_q_swa_dim, q_full.shape[-1] - self._fused_q_swa_dim],
                        dim=-1,
                    )
                    swa_k, hsa_k = k_full.split(
                        [self._fused_k_swa_dim, k_full.shape[-1] - self._fused_k_swa_dim],
                        dim=-1,
                    )
                    swa_v, hsa_v = v_full.split(
                        [self._fused_v_swa_dim, v_full.shape[-1] - self._fused_v_swa_dim],
                        dim=-1,
                    )
                    swa_q = swa_q.contiguous()
                    swa_k = swa_k.contiguous()
                    swa_v = swa_v.contiguous()
                    hsa_q = hsa_q.contiguous()
                    hsa_k = hsa_k.contiguous()
                    hsa_v = hsa_v.contiguous()
                else:
                    hsa_q, hsa_k, hsa_v = q_full, k_full, v_full
        else:
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
        if unified_path:
            # Unified per-head norm on the entire q_full / k_full.  All heads
            # use the same head_dim-wide norm weight, so this is mathematically
            # identical to per-branch norm but with one launch instead of two.
            q_full = self.q_norm(q_full.view(-1, self.head_dim)).view(q_full.shape)
            k_full = self.k_norm(k_full.view(-1, self.head_dim)).view(k_full.shape)
        elif self._layerwise_qk:
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
        if unified_path:
            # Single RoPE call on q_full/k_full (instead of one per branch).
            q_full, k_full = self.rotary_emb(positions, q_full, k_full)
        else:
            if self.has_swa_branch:
                swa_q, swa_k = self.rotary_emb(positions, swa_q, swa_k)
            if self.apply_hsa_rope:
                hsa_q, hsa_k = self.rotary_emb(positions, hsa_q, hsa_k)

        # --- Selection query ---
        if self.enable_lmk_q_proj and self.lmk_q_proj is not None:
            # Separate lmk_q_proj for page selection (NO RoPE).
            if lmk_q_fused is not None:
                lmk_q = lmk_q_fused  # already projected as part of fused GEMM
            else:
                lmk_q, _ = self.lmk_q_proj(hidden_states)  # [T, out_dim // tp_size]
            attn_tp_size = get_attention_tp_size()
            if self.lmk_q_full_dim:
                # 2026-05-21 new-arch path: lmk_q_proj output = hq_hsa_total * head_dim,
                # i.e. one head_dim per attention head. Reshape per-head, RMSNorm over
                # head_dim, then use directly as selector (no kv→q broadcast).
                if attn_tp_size > 1:
                    lmk_q = tensor_model_parallel_all_gather(lmk_q.contiguous())
                # After all-gather lmk_q has hq_hsa_total heads' worth.
                lmk_q_h = lmk_q.view(-1, self.hq_hsa_total, self.head_dim)
                lmk_q_h_norm = self.lmk_q_norm(lmk_q_h.reshape(-1, self.head_dim)).reshape(
                    lmk_q_h.shape
                )
                if attn_tp_size > 1:
                    # Slice to local q heads.
                    tp_rank = get_attention_tp_rank()
                    heads_per_rank = self.hq_hsa_total // attn_tp_size
                    lmk_q_h_norm = lmk_q_h_norm[
                        :, tp_rank * heads_per_rank : (tp_rank + 1) * heads_per_rank, :
                    ].contiguous()
                sel_q = lmk_q_h_norm  # [T, hq_hsa, head_dim]
            else:
                rhn = self.retrieval_head_num  # 1 for unified, hk_hsa for standard
                # All-gather → norm → split to match training semantics.
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
            # In unified_path, hsa_q wasn't materialised — slice the SWA-suffix
            # of q_full (post-RoPE) to recover it cheaply.
            if unified_path and self.has_swa_branch:
                hsa_q = q_full[..., self._fused_q_swa_dim:].contiguous()
            elif unified_path:
                hsa_q = q_full
            sel_q = hsa_q.view(-1, self.hq_hsa, self.head_dim)

        # Concatenate heads into the single KV cache layout: [SWA | HSA].
        if unified_path:
            # q_full/k_full/v_full were produced directly by the fused GEMM in
            # the right [SWA | HSA] layout; no explicit cat needed.
            pass
        elif self.has_swa_branch:
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
        # Per-q-head lmk_k path: compute and stash chunk-aggregated lmk_k
        # whenever this prefill completes new chunks.  Pure side-effect; the
        # selector at decode reads from the LandmarkLmkKPool.
        if self._per_qhead_lmk_k_active and forward_batch.forward_mode.is_extend():
            # In unified_path we never materialised hsa_k; slice it on demand
            # (only when actually needed, i.e. during prefill with active LMK).
            if unified_path and hsa_k is None:
                if self.has_swa_branch:
                    hsa_k = k_full[..., self._fused_k_swa_dim:].contiguous()
                else:
                    hsa_k = k_full
            self._maybe_write_chunk_lmk_k(forward_batch, sel_q, hsa_k)
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

    # ------------------------------------------------------------------
    # Per-q-head lmk_k cache write (prefill helper)
    # ------------------------------------------------------------------
    def _maybe_write_chunk_lmk_k(self, forward_batch, sel_q, hsa_k) -> None:
        """Compute chunk-aggregated h_q-headed lmk_k for any chunks that this
        prefill *completes*, store them in the backend's LandmarkLmkKPool, and
        record the (req, chunk) -> slot mapping.

        Inputs (local to this forward):
          * ``sel_q``: ``[T, hq_hsa, head_dim]`` post-norm landmark query.
          * ``hsa_k``: ``[T, hk_hsa * head_dim]`` HSA-branch keys.

        The math mirrors the official ``chunk_attn_pool`` MHA-style mode:

            mu_q  = sel_q[chunk_end_token]                 # (h_q, D)
            K_chk = hsa_k[chunk_start : chunk_end+1]       # (S, h_kv, D)
            G     = h_q // h_kv
            K_qhd = K_chk.repeat_interleave(G, dim=1)      # (S, h_q, D)
            logits = einsum("hd, shd -> sh", mu_q, K_qhd) * sm_scale
            logits[last_token_of_chunk] = -inf             # landmark anchor
            p     = softmax(logits, dim=0)
            lmk_k = einsum("sh, shd -> hd", p, K_qhd)     # (h_q, D)

        Only single-request batches are supported right now (alignment harness
        + low-volume serving cases).  Multi-batch / chunked-prefill / TP > 1
        all fall back to no-op silently — the selector will still work, just
        with sglang's existing h_kv-shared-K path.
        """
        backend = getattr(forward_batch, "attn_backend", None)
        if backend is None:
            return
        lmk_k_pool = getattr(backend, "lmk_k_pool", None)
        req_to_chunk = getattr(backend, "req_to_chunk_pool", None)
        if lmk_k_pool is None or req_to_chunk is None:
            return
        if get_attention_tp_size() > 1:
            # Per-q-head lmk_k under TP would need an all-gather on hsa_k AND
            # storing the full h_q heads (vs sliced).  Not in this POC.
            return
        if int(forward_batch.batch_size) != 1:
            return

        import math as _math

        chunk_size = int(self.chunk_size)
        h_q = int(self.hq_hsa)
        h_kv = int(self.hk_hsa)
        head_dim = int(self.head_dim)
        if h_q % h_kv != 0:
            return
        G = h_q // h_kv

        req_idx = int(forward_batch.req_pool_indices[0].item())
        prefix_len = int(
            forward_batch.extend_prefix_lens[0].item()
            if forward_batch.extend_prefix_lens is not None
            else 0
        )
        extend_len = int(forward_batch.extend_seq_lens[0].item())

        already_done = prefix_len // chunk_size
        total_done_after = (prefix_len + extend_len) // chunk_size
        if total_done_after <= already_done:
            return  # no new chunks finish in this extend

        # Reshape K to per-head view.
        k_local = hsa_k.view(extend_len, h_kv, head_dim)

        # Vectorise across all newly-complete chunks.  Each chunk's start/end
        # is required to lie fully inside this extend window.
        new_chunk_ids = list(range(already_done, total_done_after))
        starts = [c * chunk_size - prefix_len for c in new_chunk_ids]
        ends = [(c + 1) * chunk_size - 1 - prefix_len for c in new_chunk_ids]
        kept = [
            (c, s, e)
            for (c, s, e) in zip(new_chunk_ids, starts, ends)
            if 0 <= s and e < extend_len
        ]
        if not kept:
            return
        N = len(kept)

        # Build (N, head_dim_axes) batches.
        mu_q_batch = torch.stack([sel_q[e] for (_, _, e) in kept], dim=0)  # (N, h_q, D)
        k_chunk_batch = torch.stack(
            [k_local[s : s + chunk_size] for (_, s, _) in kept], dim=0
        )  # (N, S, h_kv, D)

        # Pure-pytorch chunk_attn_pool — fp32 internal, cast back at the end.
        # Computes both ``lmk_k`` (softmax-weighted V-substitute K used as the
        # landmark KEY in the topk selector) AND ``prior_b`` (entropy of the
        # per-q-head attention distribution, added as a bias to selection
        # scores — mirrors lhsa_layer's ``lmk_b``).
        sm_scale = 1.0 / _math.sqrt(head_dim)
        mu_f32 = mu_q_batch.float()
        k_f32 = k_chunk_batch.float()
        k_q = k_f32.repeat_interleave(G, dim=2) if G != 1 else k_f32  # (N, S, h_q, D)
        logits = torch.einsum("nhd,nshd->nsh", mu_f32, k_q) * sm_scale  # (N, S, h_q)
        last_mask = torch.zeros(chunk_size, dtype=torch.bool, device=mu_f32.device)
        last_mask[-1] = True
        logits = logits.masked_fill(last_mask.view(1, chunk_size, 1), float("-inf"))
        p = torch.softmax(logits, dim=1)  # (N, S, h_q)
        lmk_k = torch.einsum("nsh,nshd->nhd", p, k_q).to(lmk_k_pool.dtype)  # (N, h_q, D)
        # Entropy bias.  log_softmax puts -inf where logits were -inf; the
        # corresponding ``p`` is exactly 0 there, but ``p * log_p`` would be
        # ``0 * (-inf) = NaN``.  Replace the -inf entries with 0 so the sum is
        # well-defined.  Matches lhsa_layer.py:_chunk_attn_pool_impl.
        log_p = torch.log_softmax(logits, dim=1)
        log_p_safe = torch.where(torch.isfinite(log_p), log_p, log_p.new_zeros(()))
        prior_b = -(p * log_p_safe).sum(dim=1)  # (N, h_q) fp32

        # Allocate + write + register.
        slots = lmk_k_pool.alloc(N)
        if slots is None:
            # Pool exhausted — skip silently.  Selector will fall back to the
            # h_kv path for these chunks (their req_to_chunk row stays 0).
            return
        lmk_k_pool.set(self.layer_id, slots, lmk_k, prior_b=prior_b)

        chunk_ids_t = torch.tensor([c for c, _, _ in kept], dtype=torch.int32, device=slots.device)
        req_idx_t = torch.full((N,), req_idx, dtype=torch.int32, device=slots.device)
        req_to_chunk.assign(req_idx_t, chunk_ids_t, slots)


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

        self.mlp = Olmo2MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        # Decoder norm topology — set by FlashHSAConfig from model_type. "olmo"
        # is OLMo3 post-norm (residual + post_norm(attn|mlp(x))), "qwen" is
        # Qwen3 pre-norm (residual + attn|mlp(pre_norm(x))). The two share the
        # name `post_attention_layernorm` for the param that lives in the same
        # *position* in the state_dict; in olmo it's the post-attn output norm,
        # in qwen it's the pre-MLP input norm.  The extra slot is named
        # `post_feedforward_layernorm` for olmo and `input_layernorm` for qwen.
        self.decoder_variant = str(getattr(config, "decoder_variant", "olmo"))
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        if self.decoder_variant == "olmo":
            self.post_feedforward_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        else:  # qwen
            self.input_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)

        # Enable the custom triton fused_post_norm_add kernel for the olmo
        # branch when it's safe (no RMSNorm special modes that would change
        # numerics).  Disable via HSA_DISABLE_POST_NORM_FUSION=1.
        _has_rl_target = get_global_server_args().rl_on_policy_target is not None
        self._use_fused_post_norm = (
            self.decoder_variant == "olmo"
            and os.environ.get("HSA_DISABLE_POST_NORM_FUSION", "0") != "1"
            and not _has_rl_target
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.decoder_variant == "olmo":
            # OLMo3 post-norm — pattern is `residual + norm(x)`.  flashinfer's
            # fused_add_rmsnorm does `norm(x+residual)` so it doesn't fit, but
            # our triton fused_post_norm_add does exactly this pattern in a
            # single kernel (saves the explicit add launch).
            residual = hidden_states
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            if self._use_fused_post_norm:
                hidden_states = fused_post_norm_add(
                    hidden_states,
                    residual,
                    self.post_attention_layernorm.weight,
                    self.post_attention_layernorm.variance_epsilon,
                )
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = hidden_states + residual

            residual = hidden_states
            hidden_states = self.mlp(hidden_states)
            if self._use_fused_post_norm:
                hidden_states = fused_post_norm_add(
                    hidden_states,
                    residual,
                    self.post_feedforward_layernorm.weight,
                    self.post_feedforward_layernorm.variance_epsilon,
                )
            else:
                hidden_states = self.post_feedforward_layernorm(hidden_states)
                hidden_states = residual + hidden_states
            return hidden_states, None

        # Qwen3 pre-norm — collapse `residual + x` and `norm(...)` into the
        # fused_add_rmsnorm kernel by threading (hidden_states, residual)
        # through the layer.  The final residual is folded into model.norm.
        if residual is None:
            # First layer: no residual to fuse yet → plain norm.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # input_layernorm(x, residual) returns (norm(x+residual), x+residual)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # post_attention_layernorm fuses the post-attn residual add into the
        # pre-MLP norm: input is attn_out + residual, output is norm of that.
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # Defer the post-MLP residual add to the *next* layer's input_layernorm
        # (which will fuse it).  Return (mlp_out, residual_so_far).
        return hidden_states, residual


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

        # Thread (hidden_states, residual) through layers so the qwen variant
        # can fold per-layer residual adds into fused_add_rmsnorm in the next
        # layer's input_layernorm.  Olmo layers ignore `residual` and always
        # return (out, None) — same end-to-end shape.
        residual: Optional[torch.Tensor] = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                positions, hidden_states, forward_batch, residual
            )

        if residual is not None:
            # Fuse the trailing residual add into the final norm (qwen path).
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
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

        # Build fused QKV weights for all attention layers (both dense and HSA
        # variants implement _fuse_projections).  Cheap one-shot concat; later
        # forward calls reuse the cached tensor.
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and callable(getattr(attn, "_fuse_projections", None)):
                attn._fuse_projections()

    # Weight loading is now explicitly implemented above.


# Model registry entrypoint
EntryClass = HSAForCausalLM
