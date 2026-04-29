"""
Bootstrap module for HSA alignment training/comparison scripts.

Importing this module triggers:
  * veomni mocks (so InfiniteLongLM/models/DRT can be imported without veomni)
  * AutoConfig / configuration_hsa shim
  * patches to LandmarkHSA / Olmo3Attention / eager_attention_forward (sliding window)
  * tilelang kernel kwargs wrappers
  * sglang dist init for single-GPU TP=1

Exports:
  * OfficialModel  — DRT.modeling_olmo_lhsa.HSAForCausalLM
  * OfficialConfig — FlashHSA.configuration_hsa.HSAConfig
  * SGModel        — sglang.srt.models.flash_hsa.HSAForCausalLM
  * SGConfig       — sglang.srt.configs.flash_hsa.FlashHSAConfig
  * init_sglang_dist() — call once on entry

This file is a pure side-effect module; it must be imported BEFORE any DRT/sglang
import so that the mocks are in place.
"""
from __future__ import annotations

import os
import sys
import types
import logging
import tempfile

# ---------------------------------------------------------------------------
# 1. veomni mocks
# ---------------------------------------------------------------------------
for sub in [
    'veomni', 'veomni.distributed', 'veomni.distributed.parallel_state',
    'veomni.distributed.sequence_parallel', 'veomni.utils', 'veomni.utils.logging',
    'veomni.utils.import_utils', 'veomni.models', 'veomni.models.module_utils',
    'veomni.models.loader',
]:
    sys.modules[sub] = types.ModuleType(sub)


class _PS:
    sp_enabled = False
    sp_group = None


sys.modules['veomni.distributed.parallel_state'].get_parallel_state = lambda: _PS()
sys.modules['veomni.distributed.sequence_parallel'].slice_position_embedding = lambda x, **kw: x


def _gl(n=None):
    l = logging.getLogger(n)
    if not hasattr(l, 'info_rank0'):
        l.info_rank0 = l.info
    if not hasattr(l, 'warning_rank0'):
        l.warning_rank0 = l.warning
    return l


sys.modules['veomni.utils.logging'].get_logger = _gl
sys.modules['veomni.utils.import_utils'].is_liger_kernel_available = lambda: False
sys.modules['veomni.utils.import_utils'].is_torch_npu_available = lambda: False
sys.modules['veomni.utils.import_utils'].is_transformers_version_greater_or_equal_to = lambda v: True

import torch
import torch.nn as nn

sys.modules['veomni.models.module_utils'].GradientCheckpointingLayer = nn.Module

# liger_kernel is required only by InfiniteLongLM/ops/hsa_fwd_bwd_group.py at
# top-level import. The model code itself doesn't use it (we mock
# `is_liger_kernel_available()` to False above so the official model uses its
# native RMSNorm path). The real package is a hard dep.


class _FakeRegistry:
    def register(self, n):
        return lambda f: f


sys.modules['veomni.models.loader'].MODELING_REGISTRY = _FakeRegistry()
sys.modules['veomni.models.loader'].MODEL_CONFIG_REGISTRY = _FakeRegistry()

for _n in ['models.SWANGPT', 'models.SWANNSA']:
    sys.modules[_n] = types.ModuleType(_n)


# Mock utils.flex_attn with a PyTorch reference impl (chunk-aligned SWA + LSE).
def _flex_attn_ref(q, k, v, window_size, chunk_size, training=False, cu_seq_lens=None):
    B, H, L, D = q.shape
    scale = D ** -0.5
    qi = torch.arange(L, device=q.device)
    ki = torch.arange(L, device=q.device)
    causal = ki[None, :] <= qi[:, None]
    raw_start = qi - window_size + 1
    chunk_start = torch.clamp((raw_start // chunk_size) * chunk_size, min=0)
    swa = ki[None, :] >= chunk_start[:, None]
    not_lmk = ((ki + 1) % chunk_size) != 0
    mask = causal & swa & not_lmk[None, :]
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    output = torch.matmul(weights, v.float()).to(q.dtype)
    lse = torch.logsumexp(scores, dim=-1)
    return output, lse


# Mark fake `utils` as a namespace-package by setting __path__ to the real
# InfiniteLongLM/utils directory. This way `from utils.flex_attn import ...`
# uses the mock, but `from utils.landmark_utils import ...` finds the real file.
_HERE_BS = os.path.dirname(os.path.abspath(__file__))
_ILL_UTILS = os.path.join(os.path.dirname(_HERE_BS), 'InfiniteLongLM', 'utils')
_utils_pkg = types.ModuleType('utils')
_utils_pkg.__path__ = [_ILL_UTILS]
sys.modules['utils'] = _utils_pkg
_flex_mod = types.ModuleType('utils.flex_attn')
_flex_mod.flex_attn = _flex_attn_ref
sys.modules['utils.flex_attn'] = _flex_mod


# Mock veomni.utils.seqlen_pos_transform_utils.prepare_fa_kwargs_from_position_ids
def _prepare_fa_kwargs_from_position_ids(*a, **kw):
    return {}


_spt_mod = types.ModuleType('veomni.utils.seqlen_pos_transform_utils')
_spt_mod.prepare_fa_kwargs_from_position_ids = _prepare_fa_kwargs_from_position_ids
sys.modules['veomni.utils.seqlen_pos_transform_utils'] = _spt_mod


# ---------------------------------------------------------------------------
# 2. Path setup: InfiniteLongLM
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEV = os.path.dirname(_HERE)
_ILL = os.path.join(_DEV, 'InfiniteLongLM')
if _ILL not in sys.path:
    sys.path.insert(0, _ILL)

# DRT's modeling_olmo_lhsa imports `from .configuration_hsa import HSAConfig`,
# but the file lives in FlashHSA/. Pre-register it under the expected name.
from models.FlashHSA.configuration_hsa import HSAConfig as _HSAConfig

_drt_cfg = types.ModuleType('models.DRT.configuration_hsa')
_drt_cfg.HSAConfig = _HSAConfig
sys.modules['models.DRT.configuration_hsa'] = _drt_cfg

# ---------------------------------------------------------------------------
# 3. Import official model + apply patches
# ---------------------------------------------------------------------------
from models.DRT.modeling_olmo_lhsa import (
    HSAForCausalLM as OfficialModel,
    HSAModel,
    Olmo3Attention,
)
from models.DRT.lhsa_layer import LandmarkHSA
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks  # noqa: F401

OfficialConfig = _HSAConfig

# Eager attention path silently ignores `sliding_window` kwarg in HF.
# Patch it to apply a real SWA mask so the reference matches FA2.
import models.DRT.lhsa_layer as _lhsa_mod

_orig_eager_attn = _lhsa_mod.eager_attention_forward


def _eager_with_sliding_window(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    sw = kwargs.pop('sliding_window', None)
    if sw is not None and sw > 0 and attention_mask is None:
        L_q = query.shape[2]
        L_k = key.shape[2]
        qi = torch.arange(L_q, device=query.device)
        ki = torch.arange(L_k, device=query.device)
        causal = ki[None, :] <= qi[:, None]
        in_window = ki[None, :] >= (qi[:, None] - sw + 1)
        mask = causal & in_window
        attn_mask = torch.zeros(L_q, L_k, device=query.device, dtype=query.dtype)
        attn_mask.masked_fill_(~mask, float('-inf'))
        attention_mask = attn_mask[None, None, :, :]
    return _orig_eager_attn(module, query, key, value, attention_mask, scaling, dropout, **kwargs)


_lhsa_mod.eager_attention_forward = _eager_with_sliding_window

# LandmarkHSA / Olmo3Attention need num_key_value_groups for eager_attention_forward.
_orig_lhsa_init = LandmarkHSA.__init__


def _patched_lhsa_init(self, config, layer_idx, norm_cls=None, **kwargs):
    _orig_lhsa_init(self, config, layer_idx, norm_cls)
    if not hasattr(self, 'num_key_value_groups'):
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads


LandmarkHSA.__init__ = _patched_lhsa_init

_orig_olmo3_attn_init = Olmo3Attention.__init__


def _patched_olmo3_attn_init(self, config, layer_idx, **kwargs):
    _orig_olmo3_attn_init(self, config, layer_idx)
    if not hasattr(self, 'num_key_value_groups'):
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads


Olmo3Attention.__init__ = _patched_olmo3_attn_init

# Wrap tilelang kernels so extra kwargs from newer DRT are accepted, and inputs
# are made contiguous (otherwise tilelang trips on strides).
import ops.topk_group as _tm
import ops.hsa_fwd_bwd_group as _hm

_orig_topk = _tm.online_topk_group
_orig_hsa = _hm.HSA_block_M_group


def _patched_topk(q, k, topk, block_size, window_size, memory_window_size=-1, is_causal=True, **kw):
    return _orig_topk(
        q.contiguous(), k.contiguous(),
        topk,
        block_size=block_size,
        window_size=window_size,
        memory_window_size=memory_window_size,
        is_causal=is_causal,
    )


def _patched_hsa(q, k, v, weights, indices, block_size, mask_last_token=True, **kw):
    return _orig_hsa(
        q.contiguous(), k.contiguous(), v.contiguous(),
        weights=weights, indices=indices,
        block_size=block_size, mask_last_token=mask_last_token,
    )


_tm.online_topk_group = _patched_topk
_hm.HSA_block_M_group = _patched_hsa

# ---------------------------------------------------------------------------
# 4. sglang imports + dist init
# ---------------------------------------------------------------------------
from sglang.srt.layers import dp_attention as dpa
dpa.get_attention_tp_size = lambda: 1
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.configs.flash_hsa import FlashHSAConfig as SGConfig
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch, ForwardMode, compute_position, compute_decode_positions_landmark,
)
from sglang.srt.models.flash_hsa import HSAForCausalLM as SGModel
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

import torch.distributed as dist


def init_sglang_dist():
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix='hsa_align_', suffix='.t')
        dist.init_process_group(backend='gloo', init_method=f'file://{p}', rank=0, world_size=1)
    if not ps.model_parallel_is_initialized():
        ps._WORLD = ps.init_world_group(ranks=[0], local_rank=0, backend='gloo')
        ps._TP = ps.init_model_parallel_group(
            group_ranks=[[0]], local_rank=0, backend='gloo',
            use_custom_allreduce=False, use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False, group_name='tp',
        )
        ps._PP = ps.init_model_parallel_group(
            group_ranks=[[0]], local_rank=0, backend='gloo',
            use_custom_allreduce=False, use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False, group_name='pp',
        )
    if getattr(dpa, '_ATTN_TP_RANK', None) is None:
        dpa._ATTN_TP_RANK = 0
        dpa._ATTN_TP_SIZE = 1
        dpa._ATTN_DP_RANK = 0
        dpa._ATTN_DP_SIZE = 1
        dpa._LOCAL_ATTN_DP_RANK = 0
        dpa._LOCAL_ATTN_DP_SIZE = 1
        dpa._ENABLE_DP_ATTENTION_FLAG = False
        dpa._ATTN_TP_GROUP = ps.get_tp_group()
    try:
        sa = ServerArgs(model_path='dummy')
        sa.attention_backend = 'hsa'
        sa.enable_dp_lm_head = False
        set_global_server_args_for_scheduler(sa)
    except Exception:
        pass


def transfer_official_to_sglang(src, dst):
    """Copy weights from official OLMo-LHSA into sglang flash_hsa, handling the
    gate_proj+up_proj → gate_up_proj merge."""
    ssd = src.state_dict()
    dsd = dst.state_dict()
    loaded, total, skipped = 0, len(dsd), []
    for name, param in dsd.items():
        if name in ssd and ssd[name].shape == param.shape:
            param.data.copy_(ssd[name])
            loaded += 1
        elif ('embed_tokens' in name or 'lm_head' in name) and name in ssd:
            mr = min(ssd[name].shape[0], param.shape[0])
            param.data[:mr].copy_(ssd[name][:mr])
            loaded += 1
        elif 'gate_up_proj' in name:
            gn = name.replace('gate_up_proj', 'gate_proj')
            un = name.replace('gate_up_proj', 'up_proj')
            if gn in ssd and un in ssd:
                f = torch.cat([ssd[gn], ssd[un]], dim=0)
                if f.shape == param.shape:
                    param.data.copy_(f)
                    loaded += 1
                else:
                    skipped.append(name)
            else:
                skipped.append(name)
        else:
            skipped.append(name)
    return loaded, total, skipped


def force_native_rmsnorm(model):
    """sgl_kernel's bf16 RMSNorm CUDA kernel has a numerical bug; force the
    native (PyTorch) path on every RMSNorm."""
    from sglang.srt.layers.layernorm import RMSNorm as _RMSNorm
    for m in model.modules():
        if isinstance(m, _RMSNorm):
            m._forward_method = m.forward_native


__all__ = [
    'OfficialModel', 'OfficialConfig', 'HSAModel',
    'SGModel', 'SGConfig',
    'init_sglang_dist',
    'transfer_official_to_sglang',
    'force_native_rmsnorm',
    'insert_special_tokens', 'create_position_ids_with_landmarks',
    'HSAAttnBackend', 'MHATokenToKVPool',
    'ForwardBatch', 'ForwardMode', 'compute_position', 'compute_decode_positions_landmark',
    'Req',
]
