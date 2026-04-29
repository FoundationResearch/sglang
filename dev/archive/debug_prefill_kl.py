"""
Debug: find exact source of prefill KL divergence between official and sglang.

Compares intermediate values (embedding, QK norm, projection, attention output)
step by step to isolate where the divergence first appears.

Usage:
  CUDA_VISIBLE_DEVICES=0 python dev/debug_prefill_kl.py
"""
import sys, os, types, logging, tempfile, time

# veomni mock
for sub in ['veomni','veomni.distributed','veomni.distributed.parallel_state',
            'veomni.distributed.sequence_parallel','veomni.utils','veomni.utils.logging',
            'veomni.utils.import_utils','veomni.models','veomni.models.module_utils','veomni.models.loader']:
    sys.modules[sub] = types.ModuleType(sub)
class _PS: sp_enabled = False; sp_group = None
sys.modules['veomni.distributed.parallel_state'].get_parallel_state = lambda: _PS()
sys.modules['veomni.distributed.sequence_parallel'].slice_position_embedding = lambda x, **kw: x
def _gl(n=None):
    l = logging.getLogger(n)
    if not hasattr(l, 'info_rank0'): l.info_rank0 = l.info
    if not hasattr(l, 'warning_rank0'): l.warning_rank0 = l.warning
    return l
sys.modules['veomni.utils.logging'].get_logger = _gl
sys.modules['veomni.utils.import_utils'].is_liger_kernel_available = lambda: False
sys.modules['veomni.utils.import_utils'].is_torch_npu_available = lambda: False
sys.modules['veomni.utils.import_utils'].is_transformers_version_greater_or_equal_to = lambda v: True
import torch, torch.nn as nn
sys.modules['veomni.models.module_utils'].GradientCheckpointingLayer = nn.Module
class _FR:
    def register(self, n): return lambda f: f
sys.modules['veomni.models.loader'].MODELING_REGISTRY = _FR()
sys.modules['veomni.models.loader'].MODEL_CONFIG_REGISTRY = _FR()
for _n in ['models.SWANGPT', 'models.SWANNSA']:
    sys.modules[_n] = types.ModuleType(_n)

# flex_attn mock
def _flex_attn_ref(q, k, v, window_size, chunk_size, training=False, cu_seq_lens=None):
    B, H, L, D = q.shape; scale = D ** -0.5
    qi = torch.arange(L, device=q.device); ki = torch.arange(L, device=q.device)
    causal = ki[None, :] <= qi[:, None]
    raw_start = qi - window_size + 1
    chunk_start = torch.clamp((raw_start // chunk_size) * chunk_size, min=0)
    swa = ki[None, :] >= chunk_start[:, None]
    not_lmk = ((ki + 1) % chunk_size) != 0
    mask = causal & swa & not_lmk[None, :]
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))
    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v.float())
    return output.to(q.dtype), lse
_fm = types.ModuleType('utils.flex_attn'); _fm.flex_attn = _flex_attn_ref; sys.modules['utils.flex_attn'] = _fm
_spt = types.ModuleType('veomni.utils.seqlen_pos_transform_utils')
_spt.prepare_fa_kwargs_from_position_ids = lambda p: ((torch.arange(0, (p.shape[0]+1)*p.shape[1], p.shape[1], device=p.device, dtype=torch.int32),)*2, (p.shape[1],)*2)
sys.modules['veomni.utils.seqlen_pos_transform_utils'] = _spt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'InfiniteLongLM'))
from models.FlashHSA.configuration_hsa import HSAConfig
_dcm = types.ModuleType('models.DRT.configuration_hsa'); _dcm.HSAConfig = HSAConfig; sys.modules['models.DRT.configuration_hsa'] = _dcm
from models.DRT.modeling_olmo_lhsa import HSAForCausalLM as OfficialModel, Olmo3Attention
from models.DRT.lhsa_layer import LandmarkHSA
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks
_oi = LandmarkHSA.__init__
def _pi(self, config=None, layer_idx=0, norm_cls=None, **kw):
    _oi(self, config, layer_idx, norm_cls)
    if not hasattr(self, 'num_key_value_groups'):
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
LandmarkHSA.__init__ = _pi
_oai = Olmo3Attention.__init__
def _pai(self, config, layer_idx, **kw):
    _oai(self, config, layer_idx)
    if not hasattr(self, 'num_key_value_groups'):
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
Olmo3Attention.__init__ = _pai
import ops.topk_group as _tm, ops.hsa_fwd_bwd_group as _hm
_ot = _tm.online_topk_group; _oh = _hm.HSA_block_M_group
_tm.online_topk_group = lambda q, k, topk, block_size, window_size, memory_window_size=-1, is_causal=True, **kw: _ot(q.contiguous(), k.contiguous(), topk, block_size=block_size, window_size=window_size, memory_window_size=memory_window_size, is_causal=is_causal)
_hm.HSA_block_M_group = lambda q, k, v, weights, indices, block_size, mask_last_token=True, **kw: _oh(q.contiguous(), k.contiguous(), v.contiguous(), weights=weights, indices=indices, block_size=block_size, mask_last_token=mask_last_token)

from einops import rearrange
import torch.distributed as dist

# Import sglang in correct order to avoid circular imports
from sglang.srt.layers import dp_attention as dpa
dpa.get_attention_tp_size = lambda: 1
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.configs.flash_hsa import FlashHSAConfig
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, compute_position
from sglang.srt.models.flash_hsa import HSAForCausalLM as SGModel
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

def init_sglang():
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix='dbg_', suffix='.t')
        dist.init_process_group(backend='gloo', init_method=f'file://{p}', rank=0, world_size=1)
    if not ps.model_parallel_is_initialized():
        ps._WORLD = ps.init_world_group(ranks=[0], local_rank=0, backend='gloo')
        ps._TP = ps.init_model_parallel_group(group_ranks=[[0]], local_rank=0, backend='gloo',
            use_custom_allreduce=False, use_mscclpp_allreduce=False, use_torch_symm_mem_allreduce=False, group_name='tp')
        ps._PP = ps.init_model_parallel_group(group_ranks=[[0]], local_rank=0, backend='gloo',
            use_custom_allreduce=False, use_mscclpp_allreduce=False, use_torch_symm_mem_allreduce=False, group_name='pp')
    if getattr(dpa, '_ATTN_TP_RANK', None) is None:
        dpa._ATTN_TP_RANK = 0; dpa._ATTN_TP_SIZE = 1
        dpa._ATTN_DP_RANK = 0; dpa._ATTN_DP_SIZE = 1
        dpa._LOCAL_ATTN_DP_RANK = 0; dpa._LOCAL_ATTN_DP_SIZE = 1
        dpa._ENABLE_DP_ATTENTION_FLAG = False; dpa._ATTN_TP_GROUP = ps.get_tp_group()
    try:
        sa = ServerArgs(model_path='dummy'); sa.attention_backend = 'hsa'; sa.enable_dp_lm_head = False
        set_global_server_args_for_scheduler(sa)
    except: pass


def main():
    device = torch.device('cuda'); dtype = torch.bfloat16
    init_sglang()
    torch.manual_seed(42)
    PS = 64; VS = 256

    # Build official model
    oc = HSAConfig(vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
        chunk_size=PS, hsa_topk=8, hsa_mode='sparse', full_attn_interleave=1, hsa_heads=4,
        hsa_qk_ratio=4, use_sliding_window=True, sliding_window=PS,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act='silu', enable_softmax1=True, enable_lmk_q_proj=False)
    oc._attn_implementation = 'sdpa'; oc.pad_token_id = None; oc.num_swa_layers = 0
    om = OfficialModel(oc).to(device=device, dtype=dtype); om.eval()

    # Build sglang model with SAME weights
    sc = FlashHSAConfig(model_type='flash_hsa_innerx', architectures=['HSAForCausalLM'],
        vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
        chunk_size=PS, hsa_topk=8, hsa_mode='sparse', full_attn_interleave=1, hsa_heads=4,
        hsa_qk_ratio=4, use_sliding_window_merging=True, sliding_window_merging_size=PS,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act='silu', hsa_sliding_window=PS)
    sm = SGModel(sc).to(device=device, dtype=dtype); sm.eval()

    # Transfer weights
    ssd = om.state_dict(); dsd = sm.state_dict()
    for name, param in dsd.items():
        if name in ssd and ssd[name].shape == param.shape:
            param.data.copy_(ssd[name])
        elif 'gate_up_proj' in name:
            gn = name.replace('gate_up_proj', 'gate_proj'); un = name.replace('gate_up_proj', 'up_proj')
            if gn in ssd and un in ssd:
                param.data.copy_(torch.cat([ssd[gn], ssd[un]], dim=0))
        elif ('embed_tokens' in name or 'lm_head' in name) and name in ssd:
            mr = min(ssd[name].shape[0], param.shape[0]); param.data[:mr].copy_(ssd[name][:mr])
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    # 10 tokens, SWA-only (no HSA)
    real = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # === Step 1: Compare embeddings ===
    ids_off = insert_special_tokens(torch.tensor([real]), VS, PS)
    pos_off = create_position_ids_with_landmarks(len(real), PS, device)
    fi = Req._hsa_insert_lmk_prompt(real, page_size=PS, lmk_id=VS)

    with torch.no_grad():
        emb_off = om.model.embed_tokens(ids_off.to(device))  # [1, 10, 1024]
        emb_sg = sm.model.embed_tokens(torch.tensor(fi, device=device, dtype=torch.int64))  # [10, 1024]

    diff_emb = (emb_off[0].float() - emb_sg.float()).abs()
    print(f"1. Embedding: max_diff={diff_emb.max():.8f}")

    # === Step 2: Compare positions ===
    pl = len(fi)
    ep = torch.tensor([0], device=device, dtype=torch.int32)
    es = torch.tensor([pl], device=device, dtype=torch.int32)
    pos_sg, _ = compute_position('hsa', ep, es, pl, page_size=PS, enable_landmark_positions=True)
    print(f"2. Positions: off={pos_off[0,:10].tolist()} sg={pos_sg[:10].tolist()}")
    pos_match = (pos_off[0, :pl].to(device) == pos_sg).all().item()
    print(f"   Match: {pos_match}")

    # === Step 3: Compare QK norm on SWA branch ===
    off_attn = list(om.model.layers)[0].self_attn  # LandmarkHSA
    sg_attn = list(sm.model.layers)[0].self_attn   # FlashHSAInnerXHierarchicalSparseAttention

    # Official: q_proj -> rearrange to [B,L,h,d] -> q_norm(per-head)
    off_swa_q_raw = off_attn.q_proj(emb_off)  # [1, 10, 768]
    off_swa_q_heads = rearrange(off_swa_q_raw, 'B L (h d)->B L h d', d=64)
    off_swa_q_normed = off_attn.q_norm(off_swa_q_heads)  # per-head norm

    # sglang: q_proj -> q_norm(reshape(-1,64)).view(shape)
    sg_swa_q_raw, _ = sg_attn.q_proj(emb_sg)  # [10, 768]
    sg_swa_q_normed = sg_attn.q_norm(sg_swa_q_raw.view(-1, 64)).view(sg_swa_q_raw.shape)

    # Compare projections first
    diff_proj = (off_swa_q_raw[0].float() - sg_swa_q_raw.float()).abs()
    print(f"\n3. SWA q_proj output: max_diff={diff_proj.max():.8f}")

    # Compare norms
    off_flat = off_swa_q_normed.reshape(-1, 64)
    sg_flat = sg_swa_q_normed.view(-1, 64)
    diff_norm = (off_flat.float() - sg_flat.float()).abs()
    print(f"4. SWA q_norm output: max_diff={diff_norm.max():.8f} mean={diff_norm.mean():.8f}")

    if diff_norm.max() > 0.01:
        print(f"   >>> SIGNIFICANT NORM DIFFERENCE FOUND <<<")
        # Check norm weight match
        print(f"   Official q_norm weight shape: {off_attn.q_norm.weight.shape}")
        print(f"   sglang q_norm weight shape: {sg_attn.q_norm.weight.shape}")
        w_diff = (off_attn.q_norm.weight.float() - sg_attn.q_norm.weight.float()).abs().max()
        print(f"   Weight diff: {w_diff:.8f}")

        # Check variance computation
        x_off = off_swa_q_heads[0, 0, 0, :].float()  # first token, first head
        x_sg = sg_swa_q_raw[0, :64].view(-1, 64)[0, :].float()  # same
        print(f"   Norm input first head: off={x_off[:4]} sg={x_sg[:4]}")
        var_off = x_off.pow(2).mean()
        var_sg = x_sg.pow(2).mean()
        print(f"   Variance: off={var_off:.8f} sg={var_sg:.8f}")
    else:
        print(f"   Norm outputs match well")

    # === Step 4b: Compare HSA branch QK norm ===
    off_hsa_q_raw = off_attn.hsa_q_proj(emb_off)
    off_hsa_q_heads = rearrange(off_hsa_q_raw, 'B L (h d)->B L h d', d=64)
    off_hsa_q_normed = off_attn.q_norm(off_hsa_q_heads)

    sg_hsa_q_raw, _ = sg_attn.hsa_q_proj(emb_sg)
    sg_hsa_q_normed = sg_attn.q_norm(sg_hsa_q_raw.view(-1, 64)).view(sg_hsa_q_raw.shape)

    diff_hsa_proj = (off_hsa_q_raw[0].float() - sg_hsa_q_raw.float()).abs()
    print(f"4b. HSA q_proj output: max_diff={diff_hsa_proj.max():.8f}")

    diff_hsa_norm = (off_hsa_q_normed.reshape(-1, 64).float() - sg_hsa_q_normed.view(-1, 64).float()).abs()
    print(f"4c. HSA q_norm output: max_diff={diff_hsa_norm.max():.8f}")

    # === Step 4d: Compare o_proj weight ===
    off_o = off_attn.o_proj.weight
    sg_o = sg_attn.o_proj.weight
    diff_o = (off_o.float() - sg_o.float()).abs().max()
    print(f"4d. o_proj weight diff: {diff_o:.8f}")

    # === Step 4e: Run official attention layer standalone ===
    rope_emb = om.model.rotary_emb(emb_off, pos_off.to(device))
    with torch.no_grad():
        off_attn_out, _ = off_attn(emb_off, position_embeddings=rope_emb, attention_mask=None)
    print(f"4e. Official attn output: shape={off_attn_out.shape} norm={off_attn_out.norm():.4f}")

    # === Step 4f: Compare post_attention_layernorm weights ===
    off_pan = list(om.model.layers)[0].post_attention_layernorm
    sg_pan = list(sm.model.layers)[0].post_attention_layernorm
    diff_pan = (off_pan.weight.float() - sg_pan.weight.float()).abs().max()
    print(f"4f. post_attention_layernorm weight diff: {diff_pan:.8f}")

    off_pfn = list(om.model.layers)[0].post_feedforward_layernorm
    sg_pfn = list(sm.model.layers)[0].post_feedforward_layernorm
    diff_pfn = (off_pfn.weight.float() - sg_pfn.weight.float()).abs().max()
    print(f"4g. post_feedforward_layernorm weight diff: {diff_pfn:.8f}")

    # === Step 4h: Compare MLP weights ===
    off_gate = list(om.model.layers)[0].mlp.gate_proj.weight
    off_up = list(om.model.layers)[0].mlp.up_proj.weight
    off_down = list(om.model.layers)[0].mlp.down_proj.weight
    sg_gate_up = list(sm.model.layers)[0].mlp.gate_up_proj.weight
    sg_down = list(sm.model.layers)[0].mlp.down_proj.weight

    expected_gate_up = torch.cat([off_gate, off_up], dim=0)
    diff_gu = (expected_gate_up.float() - sg_gate_up.float()).abs().max()
    diff_down = (off_down.float() - sg_down.float()).abs().max()
    print(f"4h. MLP gate_up_proj weight diff: {diff_gu:.8f}")
    print(f"4i. MLP down_proj weight diff: {diff_down:.8f}")

    # === Step 4j: Run official decoder layer manually, compare each step ===
    off_layer = list(om.model.layers)[0]
    with torch.no_grad():
        # Step A: attention
        off_attn_out2, _ = off_layer.self_attn(emb_off, position_embeddings=rope_emb, attention_mask=None)
        # Step B: post-attn norm
        off_normed = off_layer.post_attention_layernorm(off_attn_out2)
        # Step C: residual
        off_after_attn = off_normed + emb_off
        # Step D: MLP
        off_mlp_out = off_layer.mlp(off_after_attn)
        # Step E: post-ff norm
        off_ff_normed = off_layer.post_feedforward_layernorm(off_mlp_out)
        # Step F: residual
        off_final = off_after_attn + off_ff_normed

    print(f"\n4j. Manual layer decomposition:")
    print(f"  Attn output norm: {off_attn_out2.norm():.4f}")
    print(f"  Post-attn-norm output norm: {off_normed.norm():.4f}")
    print(f"  After attn residual norm: {off_after_attn.norm():.4f}")
    print(f"  MLP output norm: {off_mlp_out.norm():.4f}")
    print(f"  Final output norm: {off_final.norm():.4f}")

    # === Step 4k: Hook sglang attention layer output ===
    _sg_attn_out = {}
    def _hook(mod, args, output):
        _sg_attn_out['out'] = output.detach().clone() if isinstance(output, torch.Tensor) else output[0].detach().clone()
    _h = sg_attn.register_forward_hook(_hook)

    # === Step 4: Compare full layer output ===
    print(f"\n5. Full model prefill comparison:")
    with torch.no_grad():
        off_out = om(input_ids=ids_off.to(device), position_ids=pos_off.to(device),
                     attention_mask=None, use_cache=False)
    off_logits = off_out.logits[:, :, :VS].float()

    mc = 256; r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device); r2t[0, :pl] = tl[:pl]
    pool = MHATokenToKVPool(size=mc+64, page_size=PS, dtype=dtype, head_num=4, head_dim=64,
        layer_num=1, device=device, enable_memory_saver=False, enable_alt_stream=False)
    mr = types.SimpleNamespace(device=device, page_size=PS, sliding_window_size=None, model=sm,
        model_config=types.SimpleNamespace(is_encoder_decoder=False, context_len=mc,
            num_attention_heads=16, head_dim=64, get_num_kv_heads=lambda tp: 4//tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(attention_backend='hsa', speculative_num_draft_tokens=0,
            speculative_num_steps=0, triton_attention_num_kv_splits=8,
            triton_attention_split_tile_size=None, enable_deterministic_inference=False,
            hsa_topk=None, hsa_selection_strategy=None, hsa_layers=None,
            hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=VS))
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool; mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)
    esl = torch.tensor([0], device=device, dtype=torch.int32)
    fb = ForwardBatch(forward_mode=ForwardMode.EXTEND, batch_size=1,
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
        sg_h = sm.model(fb.input_ids, fb.positions, fb)  # hidden states
    sg_h_normed = sm.model.norm(sg_h)
    sg_logits = (sg_h_normed @ sm.lm_head.weight[:VS, :].t()).float()

    _h.remove()
    if 'out' in _sg_attn_out:
        sg_attn_output = _sg_attn_out['out']
        print(f"   sglang attn output: shape={sg_attn_output.shape} norm={sg_attn_output.norm():.4f}")
        diff_attn = (off_attn_out2[0].float() - sg_attn_output.float()).abs()
        print(f"   Attn output diff: max={diff_attn.max():.4f} mean={diff_attn.mean():.4f}")

        # Compare per-head
        off_heads = off_attn_out2[0].view(10, 16, 64)  # [L, H, D]
        sg_heads = sg_attn_output.view(10, 16, 64)
        for h in range(16):
            d = (off_heads[:, h, :].float() - sg_heads[:, h, :].float()).abs().max().item()
            if d > 0.01:
                print(f"   Head {h}: max_diff={d:.4f}")

    # Per-position KL
    p_off = torch.softmax(off_logits[0], dim=-1)
    p_sg = torch.softmax(sg_logits, dim=-1)
    kl = (p_off * (torch.log(p_off.clamp(min=1e-10)) - torch.log(p_sg.clamp(min=1e-10)))).sum(dim=-1)
    print(f"   Prefill KL: mean={kl.mean():.6f} max={kl.max():.6f}")

    # Per-position logit diff
    logit_diff = (off_logits[0].float() - sg_logits.float()).abs()
    print(f"   Prefill logit diff: max={logit_diff.max():.4f} mean={logit_diff.mean():.4f}")

    # Hidden state comparison before final norm
    off_h_prenorm = list(om.model.layers)[0](emb_off, position_embeddings=om.model.rotary_emb(emb_off, pos_off.to(device)), attention_mask=None)
    off_h_prenorm = off_h_prenorm[0] if isinstance(off_h_prenorm, tuple) else off_h_prenorm

    # Compare hidden states after decoder layer
    diff_h = (off_h_prenorm[0].float() - sg_h.float()).abs()
    print(f"\n6. Hidden state after decoder layer: max_diff={diff_h.max():.4f} mean={diff_h.mean():.4f}")


if __name__ == '__main__':
    main()
