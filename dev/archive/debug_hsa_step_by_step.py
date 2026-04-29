"""
Step-by-step comparison of HSA attention pipeline.
Hooks into both official and sglang models to capture intermediates
at each HSA computation step for positions on page 1 (pos 64-65).

Official model runs full prefill of 65 real tokens (66 engine tokens).
sglang model also runs full prefill of 65 real tokens (66 engine tokens).
Both use identical weights. Compare at each step.
"""
import sys, os, types, logging

# veomni mock
for sub in ['veomni','veomni.distributed','veomni.distributed.parallel_state',
            'veomni.distributed.sequence_parallel','veomni.utils','veomni.utils.logging',
            'veomni.utils.import_utils','veomni.models','veomni.models.module_utils','veomni.models.loader']:
    sys.modules[sub] = types.ModuleType(sub)
class _PS: sp_enabled = False
sys.modules['veomni.distributed.parallel_state'].get_parallel_state = lambda: _PS()
sys.modules['veomni.distributed.sequence_parallel'].slice_position_embedding = lambda x, **kw: x
def _gl(n=None):
    l = logging.getLogger(n)
    if not hasattr(l, 'info_rank0'): l.info_rank0 = l.info
    if not hasattr(l, 'warning_rank0'): l.warning_rank0 = l.warning
    return l
sys.modules['veomni.utils.logging'].get_logger = _gl
sys.modules['veomni.utils.import_utils'].is_liger_kernel_available = lambda: True
sys.modules['veomni.utils.import_utils'].is_torch_npu_available = lambda: False
sys.modules['veomni.utils.import_utils'].is_transformers_version_greater_or_equal_to = lambda v: True
import torch.nn as nn
sys.modules['veomni.models.module_utils'].GradientCheckpointingLayer = nn.Module
class _FR:
    def register(self, n): return lambda f: f
sys.modules['veomni.models.loader'].MODELING_REGISTRY = _FR()
sys.modules['veomni.models.loader'].MODEL_CONFIG_REGISTRY = _FR()
for _n in ['models.SWANGPT', 'models.DRT', 'models.SWANNSA']:
    sys.modules[_n] = types.ModuleType(_n)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'InfiniteLongLM'))

import torch, tempfile, torch.distributed as dist

# Official model
from models.FlashHSA.modeling_hsa_lmk import HSAForCausalLM as OM, HierarchicalSparseAttention as HA
from models.FlashHSA.configuration_hsa import HSAConfig as OC
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks
_oi = HA.__init__
def _pi(s, c, l):
    _oi(s, c, l)
    if not hasattr(s, 'num_key_value_groups'):
        s.num_key_value_groups = c.num_attention_heads // c.num_key_value_heads
HA.__init__ = _pi
def _d(t): return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)
import ops.topk_group as _tm, ops.hsa_fwd_bwd_group as _hm
_ot = _tm.online_topk_group; _oh = _hm.HSA_block_M_group
_tm.online_topk_group = lambda q, k, *a, **kw: _ot(_d(q), _d(k), *a, **kw)
_hm.HSA_block_M_group = lambda q, k, v, *a, **kw: _oh(_d(q), _d(k), _d(v), *a, **kw)

# sglang
from sglang.srt.layers import dp_attention as dpa
dpa.get_attention_tp_size = lambda: 1
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.configs.flash_hsa import FlashHSAConfig
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch, ForwardMode, compute_position,
)
from sglang.srt.models.flash_hsa import (
    HSAForCausalLM as SM, FlashHSAInnerXHierarchicalSparseAttention as SHSA,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

if not dist.is_initialized():
    _, p = tempfile.mkstemp(prefix='dbg_', suffix='.t')
    dist.init_process_group(backend='gloo', init_method=f'file://{p}', rank=0, world_size=1)
if not ps.model_parallel_is_initialized():
    ps._WORLD = ps.init_world_group(ranks=[0], local_rank=0, backend='gloo')
    ps._TP = ps.init_model_parallel_group(group_ranks=[[0]], local_rank=0, backend='gloo',
        use_custom_allreduce=False, use_mscclpp_allreduce=False,
        use_torch_symm_mem_allreduce=False, group_name='tp')
    ps._PP = ps.init_model_parallel_group(group_ranks=[[0]], local_rank=0, backend='gloo',
        use_custom_allreduce=False, use_mscclpp_allreduce=False,
        use_torch_symm_mem_allreduce=False, group_name='pp')
if getattr(dpa, '_ATTN_TP_RANK', None) is None:
    dpa._ATTN_TP_RANK = 0; dpa._ATTN_TP_SIZE = 1
    dpa._ATTN_DP_RANK = 0; dpa._ATTN_DP_SIZE = 1
    dpa._LOCAL_ATTN_DP_RANK = 0; dpa._LOCAL_ATTN_DP_SIZE = 1
    dpa._ENABLE_DP_ATTENTION_FLAG = False; dpa._ATTN_TP_GROUP = ps.get_tp_group()
try:
    sa = ServerArgs(model_path='dummy'); sa.attention_backend = 'hsa'; sa.enable_dp_lm_head = False
    set_global_server_args_for_scheduler(sa)
except: pass

def _cmp(name, a, b, pos=None):
    """Compare tensors, optionally at specific positions."""
    if a is None or b is None:
        print(f'  {name}: SKIP (None)')
        return
    a = a.float(); b = b.float()
    if pos is not None and a.dim() >= 2 and b.dim() >= 2:
        # Extract specific positions
        if a.dim() == 4:  # [B, L, H, D]
            a = a[0, pos]; b = b[0, pos]
        elif a.dim() == 3:  # [B, L, X]
            a = a[0, pos]; b = b[0, pos]
        elif a.dim() == 2:  # [L, X]
            a = a[pos]; b = b[pos]
    d = (a - b).abs()
    mx = d.max().item(); mn = d.mean().item()
    s = 'EXACT' if mx == 0 else ('OK' if mx < 0.02 else ('CLOSE' if mx < 0.2 else 'DIFF'))
    print(f'  {name}: max={mx:.6f} mean={mn:.6f} [{s}]')

def main():
    device = torch.device('cuda'); dtype = torch.bfloat16
    torch.manual_seed(42)
    PS = 64; VS = 256

    cd = dict(vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
              num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
              chunk_size=64, hsa_topk=2, hsa_mode='sparse', full_attn_interleave=1,
              hsa_heads=8, hsa_qk_ratio=4, use_sliding_window=True, sliding_window=64,
              tie_word_embeddings=False, rope_theta=10000.0, hidden_act='silu')

    # ---- Official model with captured intermediates ----
    oc = OC(**cd); oc._attn_implementation = 'eager'
    om = OM(oc).to(device=device, dtype=dtype); om.eval()
    for mod in om.modules():
        if isinstance(mod, HA):
            _tf = mod.topk_func; _hf = mod.hsa_func
            mod.topk_func = lambda q, k, *a, _f=_tf, **kw: _f(_d(q), _d(k), *a, **kw)
            mod.hsa_func = lambda q, k, v, *a, _f=_hf, **kw: _f(_d(q), _d(k), _d(v), *a, **kw)

    # Monkey-patch official HSA forward to capture intermediates
    off_cap = {}
    off_attn = [m for m in om.modules() if isinstance(m, HA)][0]
    _orig_off_fwd = off_attn.forward
    def _cap_off_fwd(hidden_states, **kwargs):
        from einops import rearrange
        B, L, _ = hidden_states.shape
        # Capture Q/K/V projections
        swa_q = off_attn.q_proj(hidden_states)
        swa_q = rearrange(swa_q, 'B L (h d)->B L h d', d=off_attn.head_dim)
        swa_q_norm = off_attn.q_norm(swa_q)
        swa_k = off_attn.k_proj(hidden_states)
        swa_k = rearrange(swa_k, 'B L (h d)->B L h d', d=off_attn.head_dim)
        swa_k_norm = off_attn.k_norm(swa_k)

        hsa_q = off_attn.hsa_q_proj(hidden_states)
        hsa_q = rearrange(hsa_q, 'B L (h d)->B L h d', d=off_attn.head_dim)
        hsa_q_norm = off_attn.q_norm(hsa_q)
        hsa_k = off_attn.hsa_k_proj(hidden_states)
        hsa_k = rearrange(hsa_k, 'B L (h d)->B L h d', d=off_attn.head_dim)
        hsa_k_norm = off_attn.k_norm(hsa_k)

        off_cap['hidden_in'] = hidden_states.detach().clone()
        off_cap['swa_q_norm'] = swa_q_norm.detach().clone()
        off_cap['swa_k_norm'] = swa_k_norm.detach().clone()
        off_cap['hsa_q_norm'] = hsa_q_norm.detach().clone()
        off_cap['hsa_k_norm'] = hsa_k_norm.detach().clone()

        result = _orig_off_fwd(hidden_states, **kwargs)
        off_cap['attn_out'] = result[0].detach().clone() if isinstance(result, tuple) else result.detach().clone()
        return result
    off_attn.forward = _cap_off_fwd

    # ---- sglang model ----
    cfg = FlashHSAConfig(model_type='flash_hsa_innerx', architectures=['HSAForCausalLM'],
        vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
        chunk_size=64, hsa_topk=2, hsa_mode='sparse', full_attn_interleave=1,
        hsa_heads=8, hsa_qk_ratio=4, use_sliding_window_merging=True,
        sliding_window_merging_size=64, tie_word_embeddings=False, rope_theta=10000.0,
        hidden_act='silu')
    sm = SM(cfg).to(device=device, dtype=dtype); sm.eval()
    # Transfer weights
    osd = om.state_dict(); ssd = sm.state_dict()
    for n, p in ssd.items():
        if n in osd and osd[n].shape == p.shape: p.data.copy_(osd[n])
        elif ('embed_tokens' in n or 'lm_head' in n) and n in osd:
            mr = min(osd[n].shape[0], p.shape[0]); p.data[:mr].copy_(osd[n][:mr])
        elif 'gate_up_proj' in n:
            gn = n.replace('gate_up_proj','gate_proj'); un = n.replace('gate_up_proj','up_proj')
            if gn in osd and un in osd:
                f = torch.cat([osd[gn], osd[un]], dim=0)
                if f.shape == p.shape: p.data.copy_(f)
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    # Hook sglang attention
    sg_cap = {}
    sg_attn = [m for m in sm.modules() if isinstance(m, SHSA)][0]
    sg_radix = sg_attn.attn
    def _hook_sg_radix(mod, args, kwargs, out):
        sg_cap['q_full'] = args[0].detach().clone()
        sg_cap['k_full'] = args[1].detach().clone()
        sg_cap['v_full'] = args[2].detach().clone()
        sg_cap['radix_out'] = out.detach().clone()
    h_sg = sg_radix.register_forward_hook(_hook_sg_radix, with_kwargs=True)

    # ---- Run both on 65 real tokens ----
    real_tokens = list(range(5, 70))  # 65 real
    ids = insert_special_tokens(torch.tensor([real_tokens]), VS, PS)
    pos_ids = create_position_ids_with_landmarks(len(real_tokens), PS, device)
    mask = torch.ones((1, ids.shape[1]), dtype=torch.long, device=device)

    print(f'Input: {len(real_tokens)} real → {ids.shape[1]} engine tokens')

    # Official
    with torch.no_grad():
        off_out = om(input_ids=ids.to(device), position_ids=pos_ids.to(device),
                     attention_mask=mask, use_cache=False)

    # sglang
    mc = 256
    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(size=512, page_size=PS, dtype=dtype, head_num=4, head_dim=64,
                             layer_num=1, device=device, enable_memory_saver=False,
                             enable_alt_stream=False)
    mr = types.SimpleNamespace(device=device, page_size=PS, sliding_window_size=None, model=sm,
        model_config=types.SimpleNamespace(is_encoder_decoder=False, context_len=mc,
            num_attention_heads=16, get_num_kv_heads=lambda tp: 4 // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(attention_backend='hsa', speculative_num_draft_tokens=0,
            speculative_num_steps=0, triton_attention_num_kv_splits=8,
            triton_attention_split_tile_size=None, enable_deterministic_inference=False,
            hsa_topk=None, hsa_selection_strategy=None, hsa_layers=None,
            hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=VS))
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool; mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)

    fi = Req._hsa_insert_lmk_prompt(real_tokens, page_size=PS, lmk_id=VS)
    pl = len(fi)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device); r2t[0, :pl] = tl[:pl]
    ep = torch.tensor([0], device=device, dtype=torch.int32)
    es = torch.tensor([pl], device=device, dtype=torch.int32)
    pos_sg, esl = compute_position('hsa', ep, es, pl, page_size=PS, enable_landmark_positions=True)
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.tensor(fi, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([pl], device=device, dtype=torch.int32),
        out_cache_loc=tl[:pl].to(torch.int64), seq_lens_sum=pl,
        seq_lens_cpu=torch.tensor([pl], device='cpu', dtype=torch.int32),
        positions=pos_sg, extend_prefix_lens=ep, extend_seq_lens=es,
        extend_start_loc=esl, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[pl],
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool, attn_backend=be)
    be.init_forward_metadata(fb)
    with torch.no_grad():
        sg_out = sm.model(fb.input_ids, fb.positions, fb)
    h_sg.remove()
    if isinstance(sg_out, tuple): sg_out = sg_out[0]

    # ---- Compare step by step ----
    print(f'\n{"="*60}')
    print('STEP-BY-STEP HSA COMPARISON (positions 64-65 = page 1)')
    print(f'{"="*60}')

    # 1. Hidden states input to attention
    print('\n1. Hidden states input to attention:')
    sg_hidden_in = sg_cap.get('q_full')  # q_full is [T, HQ*D], but we need the pre-attn hidden
    # Actually we need to hook deeper. For now compare the Q/K/V.

    # 2. Q/K/V after projection + norm
    print('\n2. Q/K/V after projection + QK norm (before RoPE):')
    # Official: swa_q_norm [B, L, H_swa, D], hsa_q_norm [B, L, H_hsa, D]
    # sglang: q_full [T, HQ_total*D] = cat([swa_q_after_rope, hsa_q_normed])
    # We can't directly compare because sglang applies RoPE before cat.
    # But we captured the RadixAttention inputs: q_full, k_full, v_full.
    off_swa_qn = off_cap.get('swa_q_norm')  # [1, 66, 8, 64] before RoPE
    off_hsa_qn = off_cap.get('hsa_q_norm')  # [1, 66, 8, 64] no RoPE
    off_swa_kn = off_cap.get('swa_k_norm')  # [1, 66, 2, 64] before RoPE
    off_hsa_kn = off_cap.get('hsa_k_norm')  # [1, 66, 2, 64] no RoPE

    # sglang's q_full = [T, HQ_total*D] = cat([swa_q_after_rope, hsa_q])
    sg_q = sg_cap.get('q_full')  # [66, 1024] = [66, 16*64]
    sg_k = sg_cap.get('k_full')  # [66, 256] = [66, 4*64]
    if sg_q is not None:
        sg_q3 = sg_q.view(66, 16, 64)  # [T, HQ, D]
        sg_k3 = sg_k.view(66, 4, 64)   # [T, H, D]
        # HSA Q heads are at indices 8-15 (after SWA heads 0-7)
        # HSA K heads are at indices 2-3 (after SWA heads 0-1)
        sg_hsa_q = sg_q3[:, 8:, :].unsqueeze(0)  # [1, 66, 8, 64]
        sg_hsa_k = sg_k3[:, 2:, :].unsqueeze(0)  # [1, 66, 2, 64]
        sg_swa_q = sg_q3[:, :8, :].unsqueeze(0)  # [1, 66, 8, 64] (has RoPE)
        sg_swa_k = sg_k3[:, :2, :].unsqueeze(0)  # [1, 66, 2, 64] (has RoPE)

        # Compare HSA Q/K (no RoPE in either)
        _cmp('HSA Q (no RoPE) all positions', off_hsa_qn, sg_hsa_q)
        _cmp('HSA K (no RoPE) all positions', off_hsa_kn, sg_hsa_k)
        _cmp('HSA Q (no RoPE) pos 64', off_hsa_qn, sg_hsa_q, pos=64)
        _cmp('HSA K (no RoPE) pos 64', off_hsa_kn, sg_hsa_k, pos=64)

    # 3. Final attention output
    print('\n3. Final attention output (after o_proj):')
    off_attn_out = off_cap.get('attn_out')  # [1, 66, 1024]
    sg_attn_out = sg_cap.get('radix_out')   # [66, 1024]
    if off_attn_out is not None and sg_attn_out is not None:
        sg_ao = sg_attn_out.unsqueeze(0) if sg_attn_out.dim() == 2 else sg_attn_out
        _cmp('attn output all positions', off_attn_out, sg_ao)
        _cmp('attn output page 0 (pos 0-63)', off_attn_out[:, :64], sg_ao[:, :64])
        _cmp('attn output pos 64', off_attn_out, sg_ao, pos=64)
        _cmp('attn output pos 65', off_attn_out, sg_ao, pos=65)

        # Split into SWA and HSA head contributions (before o_proj)
        # We can't do this after o_proj. But we can check if the attn output
        # at page 0 matches (which we know from earlier should be exact).

    # 4. Final logits
    print('\n4. Final logits:')
    off_logits = off_out.logits[0, :, :VS].float()
    sg_h_normed = sm.model.norm(sg_out)
    sg_logits = (sg_h_normed @ sm.lm_head.weight.t())[:, :VS].float()
    _cmp('logits all positions', off_logits.unsqueeze(0), sg_logits.unsqueeze(0))
    _cmp('logits page 0', off_logits[:64].unsqueeze(0), sg_logits[:64].unsqueeze(0))
    _cmp('logits pos 64', off_logits[64:65].unsqueeze(0), sg_logits[64:65].unsqueeze(0))
    _cmp('logits pos 65', off_logits[65:66].unsqueeze(0), sg_logits[65:66].unsqueeze(0))


if __name__ == '__main__':
    main()
