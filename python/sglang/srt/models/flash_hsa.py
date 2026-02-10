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

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
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
from sglang.srt.models.qwen2 import Qwen2MLP, Qwen2Model
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3 import Qwen3ForCausalLM as _Qwen3ForCausalLM
from sglang.srt.server_args import get_global_server_args
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
    """Qwen3-style attention with explicit q/k/v projections (InnerX reference)."""

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

        # Match Qwen3Attention kv-head partition/replication semantics.
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

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

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
            reduce_results=False,
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

        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
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
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.lmk_q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        # Projections (explicit matrices, matching ultra reference naming).
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
        self.lmk_q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hk_hsa_total * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("lmk_q_proj", prefix),
        )

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
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        # RoPE is applied ONLY to the SWA branch in the ultra reference.
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
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # --- SWA branch ---
        swa_q, _ = self.q_proj(hidden_states)
        swa_k, _ = self.k_proj(hidden_states)
        swa_v, _ = self.v_proj(hidden_states)
        swa_q, swa_k = apply_qk_norm(
            q=swa_q,
            k=swa_k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        swa_q, swa_k = self.rotary_emb(positions, swa_q, swa_k)

        # --- HSA branch (NO RoPE, ultra reference) ---
        hsa_q, _ = self.hsa_q_proj(hidden_states)
        hsa_k, _ = self.hsa_k_proj(hidden_states)
        hsa_v, _ = self.hsa_v_proj(hidden_states)
        hsa_q, hsa_k = apply_qk_norm(
            q=hsa_q,
            k=hsa_k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )

        # --- Selection query (LMK-Q, NO RoPE) ---
        lmk_q, _ = self.lmk_q_proj(hidden_states)  # [T, hk_hsa*D]
        lmk_q_3 = lmk_q.view(-1, self.hk_hsa, self.head_dim)
        lmk_q_norm = self.lmk_q_norm(lmk_q_3)  # [T, hk_hsa, D]
        # Expand to q-head space for selector: [T, hq_hsa, D]
        assert self.hq_hsa % self.hk_hsa == 0
        G = self.hq_hsa // self.hk_hsa
        sel_q = (
            lmk_q_norm[:, :, None, :]
            .expand(lmk_q_norm.shape[0], self.hk_hsa, G, self.head_dim)
            .reshape(lmk_q_norm.shape[0], self.hq_hsa, self.head_dim)
            .contiguous()
        )

        # Concatenate heads into the single KV cache layout: [SWA | HSA].
        q_full = torch.cat([swa_q, hsa_q], dim=-1)
        k_full = torch.cat([swa_k, hsa_k], dim=-1)
        v_full = torch.cat([swa_v, hsa_v], dim=-1)

        window = None
        if bool(getattr(self.config, "use_sliding_window_merging", False)):
            window = _get_sliding_window_merging_size(self.config)

        hsa_split_head_info = dict(
            hq_swa=int(self.hq_swa),
            hq_hsa=int(self.hq_hsa),
            h_swa=int(self.hk_swa),
            h_hsa=int(self.hk_hsa),
            swa_window_size=window,
            # ultra reference: SWA sees LMK (no explicit exclusion)
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
        is_hsa_layer = bool(interleave > 0 and (self.layer_id % interleave) == (interleave - 1))

        if is_hsa_layer:
            self.self_attn = FlashHSAInnerXHierarchicalSparseAttention(
                config=config,
                layer_id=self.layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
                alt_stream=alt_stream,
            )
        else:
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
                prefix=add_prefix("self_attn", prefix),
                alt_stream=alt_stream,
            )

        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=int(getattr(config, "intermediate_size")),
            hidden_act=str(getattr(config, "hidden_act", "silu")),
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            self.hidden_size, eps=float(getattr(config, "rms_norm_eps", 1e-6)), **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, eps=float(getattr(config, "rms_norm_eps", 1e-6)), **norm_kwargs
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=self.layer_id,
            num_layers=int(getattr(config, "num_hidden_layers")),
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            post_residual_addition=post_residual_addition,
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


class FlashHSAInnerXModel(Qwen2Model):
    """Qwen2Model skeleton but with InnerX decoder layers."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=FlashHSAInnerXDecoderLayer,
            alt_stream=alt_stream,
        )

class HSAForCausalLM(_Qwen3ForCausalLM):
    """
    FlashHSA entry class for SGLang.

    We inherit Qwen3ForCausalLM to reuse:
      - logits processor / pooler plumbing
      - Qwen3 weight loading remaps
      - PP weight tying logic (we override sizes where necessary)
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        # We cannot call super().__init__ directly because it will build lm_head with base vocab size.
        nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        # We only support InnerX ultra (split-head) architecture.
        if getattr(config, "hsa_heads", None) is None or getattr(config, "hsa_qk_ratio", None) is None:
            raise ValueError(
                "InnerX ultra requires `hsa_heads` and `hsa_qk_ratio` in config. "
                "This repo no longer supports the non-InnerX FlashHSA variant."
            )
        self.model = FlashHSAInnerXModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        padded_vocab_size = _get_flashhsa_padded_vocab_size(config)

        # Replace embedding with padded vocab so LMK row exists.
        if self.pp_group.is_first_rank:
            self.model.embed_tokens = VocabParallelEmbedding(
                padded_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", add_prefix("model", prefix)),
                params_dtype=(
                    torch.float32
                    if get_global_server_args().rl_on_policy_target is not None
                    else None
                ),
            )

        # handle the lm head on different pp ranks (same as Qwen3ForCausalLM, but padded vocab size)
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    padded_vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP (same as Qwen3ForCausalLM, but padded vocab size)
        if self.pp_group.world_size > 1 and getattr(config, "tie_word_embeddings", False):
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.world_size - 1
                )
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=self.lm_head.weight.shape,
                    dtype=next(self.model.parameters()).dtype,
                    src=0,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        # Reuse Qwen3ForCausalLM components.
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.pooler import Pooler, PoolingType

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support (kept for parity)
        self.capture_aux_hidden_states = False

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
        if interleave <= 0 or num_layers <= 0:
            return []
        return [i for i in range(num_layers) if (i % interleave) == (interleave - 1)]

    def get_flashhsa_chunk_size(self) -> Optional[int]:
        if hasattr(self.config, "chunk_size") and getattr(self.config, "chunk_size") is not None:
            return int(getattr(self.config, "chunk_size"))
        return None

    # ---- Weight loading ----
    # We only support the InnerX ultra layout (explicit q_proj/k_proj/v_proj + hsa_* projections).
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
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


