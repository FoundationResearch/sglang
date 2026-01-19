import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict
from flash_attn.cute.interface import FlashAttnFunc
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers import LlamaConfig
from flash_attn_interface import flash_attn_func

from einops import rearrange
from veomni.ops import causallm_loss_function
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import slice_position_embedding
from liger_kernel.transformers.rms_norm import LigerRMSNorm as RMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP as SwiGLUMLP
import cutlass.cute as cute
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from ops.hsa_fwd_bwd_block_M_tilelang_head import HSA_block_M_head as HSA
ALL_LAYERNORM_LAYERS.append(RMSNorm)


logger = logging.get_logger(__name__)


class FlashHSA(nn.Module):
    def __init__(self, config, layer_idx):
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        self.head_dim = self.d_model // config.num_query_heads
        self.d_kv = self.head_dim * config.num_kv_heads
        self.h_kv = config.num_kv_heads
        self.h_q = config.num_query_heads
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.kv_proj = nn.Linear(self.d_model, self.d_kv * 2)
        self.topk = config.top_k
        self.window_size = config.sliding_window
        self.chunk_size = config.chunk_size
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.lmk_norm = RMSNorm(self.head_dim)
        self.enable_softmax1 = getattr(config, 'enable_softmax1', False)
        from ops.topk_head_softmax import online_softmax_topk_head
        self.topk_func = online_softmax_topk_head


    def forward(self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Dict] = None,
        shared_kv: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # x: (B, L, d)
        # lmks: (B, L // S, h_kv, d_kv)
        B, L, _ = hidden_states.shape
        swa_kv = self.kv_proj(hidden_states)  # (B, L, d_kv * 2)
        
        swa_k, swa_v = swa_kv.splits(self.d_kv, dim=2) # split swa_kv into 2 * (B, L, d_kv)
        hsa_k, hsa_v = swa_k, swa_v  # (B, L, d_kv)

        if shared_kv is not None:
            # 用shared kv作为kv cache，否则用x变换而来的key/values
            hsa_k, hsa_v = shared_kv.splti(self.d_kv, dim=2) # 

        chunk_lmks = hsa_k[:, ::self.chunk_size, :]
        chunk_lmks = rearrange(chunk_lmks, 'B L (h d)->B L h d', d=self.head_dim)
        chunk_lmks = self.lmk_norm(chunk_lmks)

        q = rearrange(q, 'B L (h d)->B L h d', d=self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states))  # q_norm
        hsa_k = rearrange(hsa_k, 'B L (h d)->B L h d', d=self.head_dim)
        hsa_k = self.k_norm(hsa_k)

        def create_causal_mask_with_window_size(window_size: int, chunk_size: int):
            @cute.jit
            def block_causal_mask(b, h, q_idx, kv_idx, aux_tensors):
                return q_idx >= (max(kv_idx, window_size) - window_size) // chunk_size * chunk_size
            return block_causal_mask

        # block_mask = create_block_mask(block_causal_mask, B=None, H=None, Q_LEN=L, KV_LEN=L)
        # swa_o, lse_sum = flex_attention(q, swa_k, swa_v, block_mask=block_mask, return_lse=True) # logits_sum: (B, L, h), swa_o (B, L, h, d)
        block_mask = create_causal_mask_with_window_size(self.window_size, self.chunk_size)
        swa_o, lse_sum = FlashAttnFunc(q, swa_k, swa_v, mask_mod=block_mask, pos_emb=position_embeddings)
        indices, scores = self.topk_func(q, chunk_lmks, lse_sum, self.topk, self.chunk_size, is_causal=True)  # (B, L, h_q, k), scores should be differential, q一组头和lmk一个头点积之和

        if not self.enable_softmax1:
            cat_scores = torch.cat(scores, lse_sum.unsqueeze(-1), dim=-1)  # (B, L, h_q, K + 1)
            swa_weight_idx = -1
        else:
            # softmax off by one
            cat_scores = torch.cat(scores, lse_sum.unsqueeze(-1), torch.ones(B, L, self.h_q, 1, device=x.device), dim=-1)
            swa_weight_idx = -2


        chunk_weights = F.softmax(cat_scores, dim=-1)
        hsa_o = HSA(q, hsa_k, hsa_v, chunk_weights, indices)  # 这里chunk weights最后一维是K+1, 需要只读到 K - 1
        swa_o_weight = chunk_weights[:, :, :, swa_weight_idx]
        o = hsa_o + swa_o * swa_o_weight.unsqueeze(-1)

        return self.o_proj(o)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SlidingWindowAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        assert self.sliding_window is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Dict] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # print(f'layer: {self.layer_idx} self.p_proj dtype: {self.q_proj.weight.dtype}, hidden states dtype: {hidden_states.dtype}')

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        assert cos.shape[1] == query_states.shape[-2]
        # print(f'before apply rope: {query_states.shape}, {cos.shape}, {sin.shape}')
        # if self.training:
        query_states, key_states = liger_rotary_pos_emb(query_states, key_states, cos, sin)
        # else:
        #     # issue: liger_rotary_pos_emb nan when eval

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            past_key_cache, past_value_cache = past_key_value.get(self.layer_idx, (None, None))
            if past_key_cache is None and past_value_cache is None:
                past_key_cache = key_states
                past_value_cache = value_states
            else:
                key_states = torch.cat([past_key_cache, key_states], dim=-2)
                value_states = torch.cat([past_value_cache, value_states], dim=-2)

            # TODO: implement with torch.roll
            past_key_value[self.layer_idx] = (key_states[:, :, -self.sliding_window:, :].contiguous(), value_states[:, :, -self.sliding_window:, :].contiguous())

        if query_states.dtype not in (torch.float16, torch.bfloat16):
            query_states = query_states.to(torch.bfloat16)
            key_states = key_states.to(torch.bfloat16)
            value_states = value_states.to(torch.bfloat16)

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            softmax_scale=None,
            causal=True,
            window_size=(self.sliding_window, 0),  # diff with Llama
        )
        # assert not torch.any(torch.isnan(attn_output))

        # print(f'attn output dtype: {attn_output.dtype} o_proj.dtype: {self.o_proj.weight.dtype}')
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

class DecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, attn_cls, layer_idx: int = None):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.self_attn = (
            attn_cls(config, layer_idx)
        )
        self.residual_in_fp32 = getattr(config, 'residual_in_fp32', True)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            position_embeddings=position_embeddings
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states.to(self.mlp.gate_proj.weight.dtype)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class DRTCache:
    def __init__(
        self, config: LlamaConfig, batch_size: int, dtype: torch.dtype = torch.bfloat16, device: Optional[str] = None
    ):
        self.past_key_values = {}
        self.gca_cache = HSACache(config, batch_size, dtype=dtype, device=device)

    @property
    def mem_mgr(self):
        return self.gca_cache.mem_mgr

    @property
    def key_value_memory_dict(self):
        return self.gca_cache.key_value_memory_dict

    @property
    def chunk_k(self):
        return self.gca_cache.chunk_k

    @property
    def chunk_v(self):
        return self.gca_cache.chunk_v

    @property
    def weights(self):
        return self.gca_cache.weights

    @property
    def landmark_positions(self):
        return self.gca_cache.landmark_positions

    @landmark_positions.setter
    def landmark_positions(self, val):
        self.gca_cache.landmark_positions = val


class DRTPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SWADecoderLayer", "ChunkingLayer", "RetrievalLayer", "HierarchicalSparseAttention"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)


@dataclass
class DRTOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[
        Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]], # Tuple of Llama KV caches
        Optional[HSACache]                                      # Single HSACache object
    ]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    chunk_weights: Optional[torch.FloatTensor] = None
    indices: Optional[torch.LongTensor] = None
    cache_params: Optional[DRTCache] = None


@dataclass
class DRTCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None
    past_key_values: Optional[DRTCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    chunk_weights: Optional[torch.FloatTensor] = None
    indices: Optional[torch.LongTensor] = None
    cache_params: Optional[DRTCache] = None


class FlashHSAModel(DRTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config=config)

        layers = []

        for layer_idx in range(config.total_layers):
            if (1 + layer_idx) % config.full_attn_stride:
                attn_cls = FlashHSA
            else:
                attn_cls = SlidingWindowAttention
            layers.append(DecoderLayer(config, attn_cls, layer_idx))
        
        print(f'total layers in ModuleList: {len(layers)}')
        self.config.num_hidden_layers = len(layers)
        # print(f'Number of Llama-like layers for KV cache: {self.num_llama_like_layers}')
        
        self.layers = nn.ModuleList(layers)
        self.lmk_id = config.lmk_token_id

        self.gradient_checkpointing = False
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[DRTCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, DRTOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (True if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        # print(f'input cache params is None: {cache_params is None}')
        past_key_values = None
        if use_cache:
            if cache_params is None:
                cache_params = DRTCache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device
                )
            past_key_values = cache_params.past_key_values
        else:
            cache_params = None

        # print(f'mode: {self.training}, cache_position is None: {cache_position is None}')
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        else:
            past_seen_tokens = cache_position[0]

        assert not (get_parallel_state().sp_enabled and position_ids is None), 'position ids should not be none for sp'
        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

            position_ids = position_ids.unsqueeze(0)
        
        if cache_position is None:
            cache_position = position_ids[:, 0]

        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # --- slice position embedding if using sp ---
        sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
        position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
        # --- slice position embedding if using sp ---

        weights = None
        mem_k, mem_v = None, None
        landmarks = None
        indices = None

        all_hidden_states = () if output_hidden_states else None

        for layer_idx, mixer_block in enumerate(self.layers):
            # print (f"layer:{layer_idx}'s cache_params is None: {cache_params is None}")
            if self.is_tmf_layers[layer_idx]:
                if self.gradient_checkpointing and self.training:
                    hidden_states = self._gradient_checkpointing_func(
                        mixer_block.__call__, 
                        hidden_states,
                        None,
                        position_embeddings,
                        use_cache
                    )
                else:
                    # layer_past_key_values = cache_params.past_key_values.get(layer_idx, None) if use_cache else None
                    # print(f"lower decoder layers cache: {layer_past_key_values[0].shape if layer_past_key_values is not None else None}")
                    # print (f"input of lower decoder@{layer_idx}:\nhidden_states: {hidden_states.shape}\nattention_mask:{attention_mask}\nposition_ids:{position_ids}")
                    # print(f"layer_past_key_values: {layer_past_key_values[0].shape if layer_past_key_values is not None else None}")
                    # print (f"input of lower decoder@{layer_idx}: position_ids:{position_ids}")                    
                    attn_output = mixer_block(
                        hidden_states,
                        cache_params=cache_params,
                        position_embeddings=position_embeddings,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        **kwargs
                    )
                    hidden_states = attn_output

                    # if use_cache:
                    #     present_key_value = attn_output[-1]
                    #     if self.slide_window > 0:
                    #         # print (f"present_key_value: {present_key_value[0].shape}")
                    #         present_key_value = (present_key_value[0][:,:,-self.slide_window: ,:], present_key_value[1][:,:,-self.slide_window: ,:])
                    #     cache_params.past_key_values[layer_idx] = present_key_value
            else:
                if self.gradient_checkpointing and self.training:
                    hidden_states, weights, mem_k, mem_v, landmarks, indices = \
                        self._gradient_checkpointing_func(
                            mixer_block.__call__,
                            hidden_states,
                            weights,
                            mem_k,
                            mem_v,
                            landmarks,
                            indices,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                else:
                    hidden_states, weights, mem_k, mem_v, landmarks, indices = \
                        mixer_block(
                            hidden_states,
                            weights=weights,
                            mem_k=mem_k,
                            mem_v=mem_v,
                            landmarks=landmarks,
                            indices=indices,
                            cache_params=cache_params,
                            cache_position=cache_position,
                            attention_mask=attention_mask,
                            position_embeddings=position_embeddings,
                            **kwargs,
                        )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        # print(f'out cache params is None: {cache_params is None}')
        return DRTOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            chunk_weights=weights,
            indices=indices
        )


class DRTForCausalLM(DRTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.backbone = DRTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.chunk_size = config.chunk_size
        self.pad_id = config.pad_token_id
        self.lmk_id = config.lmk_token_id
        self.inference_segment = getattr(config, 'inference_segment', 16384)
        self.enable_flash_inference = getattr(config, 'flash_inference', False)
        self.loss_fn = causallm_loss_function
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # total_seq_len: Optional[int] = None,
        **kwargs,
    ):
        # Overwitten -- uses `cache_params` as opposed to `past_key_values`
        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1].unsqueeze(-1)

                # if attention_mask is not None:
                #     attention_mask = None

            # else:
            #     # we initialize the `cache_position` to full size of `conv_states` at prefill stage
            #     # considering padding will be applied when input length is shorter, and truncation
            #     # will be applied when it is longer, so it will be equivalent to always have it match
            #     # the length of `cache_params.conv_states`, which is `config.conv_kernel`
            #     cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # if total_seq_len is None:
        #     total_seq_len = 0
        model_inputs.update(
            {
                "attention_mask": None,
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
                # "total_seq_len": total_seq_len
            }
        )
        return model_inputs

    def _insert_special_tokens(self, input_ids, fill_id):
        N = input_ids.shape[0]
        input_ids_ = input_ids.view(N, -1, self.chunk_size - 1)  # (N, L / cz, cz)
        chunk_num = input_ids_.shape[1]
        chunk_id_padding = torch.ones(N, chunk_num, 1, device=input_ids.device, dtype=torch.long).fill_(fill_id)
        # chunked_input_ids = torch.cat([input_ids_, chunk_id_padding], dim=2)  # (N, L / cz, cz+1)
        chunked_input_ids = torch.cat([chunk_id_padding, input_ids_], dim=2)  # (N, L / cz, cz+1)
        chunked_input_ids = chunked_input_ids.view(N, -1)  # (N, L // cz * (cz + 1))
        return chunked_input_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[DRTCache] = None,
        # total_seq_len: Optional[int] = None,
        output_whole_logits: Optional[bool] = False,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, DRTCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids = torch.where(input_ids < 0, self.pad_id, input_ids)

        backbone_outputs = None
        attention_mask = attention_mask.bool() if attention_mask is not None else None

        if self.training:
            backbone_outputs = self.backbone(
                input_ids,
                position_ids=position_ids,
                cache_params=cache_params,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=use_cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
                **kwargs
            )
            hidden_states = backbone_outputs[0]
            # print(f'batch hidden states: {hidden_states[0, :, :4]}')
            # logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        else:       # Inference with cache
            # assert cache_position is not None
            N, L = input_ids.shape[0], input_ids.shape[1]
            if cache_position is not None:
                cache_position = cache_position.clone()
            else:
                cache_position = torch.zeros(N, dtype=torch.long, device=input_ids.device)
            # if total_seq_len is None:
            #     total_seq_len = L
            hidden_states = None
            l = 1 if self.enable_flash_inference else 0
            # print(f'input_ids: {input_ids.shape}, {L - l}, {self.inference_segment}')
            inference_segment = self.inference_segment
            if self.inference_segment == -1:
                inference_segment = L - l
            
            use_cache = True
            for offset in range(0, L - l, inference_segment):
                # print('enter for loop')
                segment_ids = input_ids[:, offset: min(L - l, offset + inference_segment)]
                # print(f'segment_ids: {segment_ids.shape}, offset: {offset}')
                segment_mask = None
                if attention_mask is not None:
                    segment_mask=attention_mask[:, offset: min(L - l, offset + inference_segment)]
                backbone_outputs = self.backbone(
                    segment_ids,
                    cache_params=cache_params,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    attention_mask=segment_mask,
                    **kwargs
                )
                cache_params = backbone_outputs.cache_params
                cache_position += segment_ids.shape[1]
                # prev_hidden = hidden_states
                hidden_states = backbone_outputs[0]
                # print(hidden_states[0, :, :4])
                # print(f'hidden_states is None? {hidden_states is None}')
            

            if l > 0:
                segment_ids = input_ids[:, -l:]
                # print(f'segment_ids: {segment_ids.shape}, offset: {offset}')
                segment_mask = None
                if attention_mask is not None:
                    segment_mask=attention_mask[:, -l:]
                backbone_outputs = self.backbone(
                    segment_ids,
                    cache_params=cache_params,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    attention_mask=segment_mask,
                    **kwargs
                )
                cache_params = backbone_outputs.cache_params
                hidden_states = backbone_outputs[0]
                # print(f'hidden_states2 is None? {hidden_states is None}')
            
            # logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()


        loss = None
        logits = None
        if labels is not None:
            loss, logits = self.loss_fn(hidden_states.to(self.lm_head.weight.dtype), self.lm_head.weight, labels)
        else:
            logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()


        if not return_dict:
            output = (logits,) + backbone_outputs[1:]
            return ((loss,) + output) if loss is not None else output


        return DRTCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_outputs.hidden_states,
            cache_params=backbone_outputs.cache_params
        )


__all__ = ["DRTForCausalLM", "DRTModel", "DRTPreTrainedModel"]





    
