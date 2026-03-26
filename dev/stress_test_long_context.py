"""
Stress test: long context prefill + multi-step decode.

Tests official prefill(N+K) vs sglang prefill(N) + K decode steps
at various context lengths and decode depths.

Usage:
  CUDA_VISIBLE_DEVICES=4 python dev/stress_test_long_context.py
"""
import sys, os, types, logging, tempfile, time

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

import torch, torch.distributed as dist

# Official model (import first to avoid utils shadowing)
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
    ForwardBatch, ForwardMode, compute_position, compute_decode_positions_landmark,
)
from sglang.srt.models.flash_hsa import HSAForCausalLM as SM
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

def init_sglang():
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix='st_', suffix='.t')
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


def transfer_weights(src, dst):
    ssd = src.state_dict(); dsd = dst.state_dict(); n = 0
    for name, param in dsd.items():
        if name in ssd and ssd[name].shape == param.shape:
            param.data.copy_(ssd[name]); n += 1
        elif ('embed_tokens' in name or 'lm_head' in name) and name in ssd:
            mr = min(ssd[name].shape[0], param.shape[0])
            param.data[:mr].copy_(ssd[name][:mr]); n += 1
        elif 'gate_up_proj' in name:
            gn = name.replace('gate_up_proj', 'gate_proj')
            un = name.replace('gate_up_proj', 'up_proj')
            if gn in ssd and un in ssd:
                f = torch.cat([ssd[gn], ssd[un]], dim=0)
                if f.shape == param.shape: param.data.copy_(f); n += 1
    return n, len(dsd)


def official_prefill(model, real_tokens, PS, VS, device):
    ids = insert_special_tokens(torch.tensor([real_tokens]), VS, PS)
    pos = create_position_ids_with_landmarks(len(real_tokens), PS, device)
    mask = torch.ones((1, ids.shape[1]), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids.to(device), position_ids=pos.to(device),
                     attention_mask=mask, use_cache=False)
    return out.logits[:, :, :VS].float()


def sglang_prefill_then_decode(sg_model, cfg, real_base, decode_tokens, device, dtype):
    """Prefill real_base, then decode each token in decode_tokens sequentially.
    Returns logits for each decode step."""
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); VS = int(cfg.vocab_size)
    mc = max(len(real_base) * 2, 512) + len(decode_tokens) * 2 + 128

    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(size=mc + 256, page_size=PS, dtype=dtype,
                             head_num=int(cfg.num_key_value_heads), head_dim=int(cfg.head_dim),
                             layer_num=int(cfg.num_hidden_layers), device=device,
                             enable_memory_saver=False, enable_alt_stream=False)
    mr = types.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=sg_model,
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, context_len=mc,
            num_attention_heads=int(cfg.num_attention_heads),
            get_num_kv_heads=lambda tp: int(cfg.num_key_value_heads) // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend='hsa', speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id))
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool; mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)

    # Prefill
    fi = Req._hsa_insert_lmk_prompt(real_base, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    r2t[0, :pl] = tl[:pl]
    ep = torch.tensor([0], device=device, dtype=torch.int32)
    es = torch.tensor([pl], device=device, dtype=torch.int32)
    pos, esl = compute_position('hsa', ep, es, pl, page_size=PS, enable_landmark_positions=True)
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.tensor(fi, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([pl], device=device, dtype=torch.int32),
        out_cache_loc=tl[:pl].to(torch.int64), seq_lens_sum=pl,
        seq_lens_cpu=torch.tensor([pl], device='cpu', dtype=torch.int32),
        positions=pos, extend_prefix_lens=ep, extend_seq_lens=es,
        extend_start_loc=esl, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[pl],
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool, attn_backend=be)
    be.init_forward_metadata(fb)
    with torch.no_grad():
        sg_model.model(fb.input_ids, fb.positions, fb)

    # Multi-step decode
    current_len = pl
    decode_logits = []
    for tok in decode_tokens:
        current_len += 1
        r2t[0, current_len - 1] = tl[current_len - 1]
        dp = compute_decode_positions_landmark(
            torch.tensor([current_len], device=device, dtype=torch.int32), page_size=PS)
        fbd = ForwardBatch(
            forward_mode=ForwardMode.DECODE, batch_size=1,
            input_ids=torch.tensor([tok], device=device, dtype=torch.int64),
            req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
            seq_lens=torch.tensor([current_len], device=device, dtype=torch.int32),
            out_cache_loc=tl[current_len-1:current_len].to(torch.int64),
            seq_lens_sum=current_len,
            seq_lens_cpu=torch.tensor([current_len], device='cpu', dtype=torch.int32),
            positions=dp, req_to_token_pool=mr.req_to_token_pool,
            token_to_kv_pool=pool, attn_backend=be)
        be.init_forward_metadata(fbd)
        with torch.no_grad():
            h = sg_model.model(fbd.input_ids, fbd.positions, fbd)
        if isinstance(h, tuple): h = h[0]
        h = sg_model.model.norm(h)
        logits = (h @ sg_model.lm_head.weight.t())[:, :VS].float()
        decode_logits.append(logits.view(-1))

    return decode_logits


def main():
    device = torch.device('cuda'); dtype = torch.bfloat16
    init_sglang()
    torch.manual_seed(42)

    PS = 64; VS = 256
    cd = dict(vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
              num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
              chunk_size=64, hsa_topk=2, hsa_mode='sparse', full_attn_interleave=1,
              hsa_heads=8, hsa_qk_ratio=4, use_sliding_window=True, sliding_window=64,
              tie_word_embeddings=False, rope_theta=10000.0, hidden_act='silu')

    oc = OC(**cd); oc._attn_implementation = 'eager'
    om = OM(oc).to(device=device, dtype=dtype); om.eval()
    for mod in om.modules():
        if isinstance(mod, HA):
            _tf = mod.topk_func; _hf = mod.hsa_func
            mod.topk_func = lambda q, k, *a, _f=_tf, **kw: _f(_d(q), _d(k), *a, **kw)
            mod.hsa_func = lambda q, k, v, *a, _f=_hf, **kw: _f(_d(q), _d(k), _d(v), *a, **kw)

    cfg = FlashHSAConfig(
        model_type='flash_hsa_innerx', architectures=['HSAForCausalLM'],
        vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
        chunk_size=64, hsa_topk=2, hsa_mode='sparse', full_attn_interleave=1,
        hsa_heads=8, hsa_qk_ratio=4, use_sliding_window_merging=True,
        sliding_window_merging_size=64, tie_word_embeddings=False, rope_theta=10000.0,
        hidden_act='silu')
    sm = SM(cfg).to(device=device, dtype=dtype); sm.eval()
    loaded, total = transfer_weights(om, sm)
    print(f'Weights: {loaded}/{total}')
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    print('=' * 70)
    print('STRESS TEST: Long Context Prefill + Multi-Step Decode')
    print('=' * 70)

    test_cases = [
        # (n_prefill_real, decode_tokens, description)
        (10, [42, 100, 200], '10 prefill, 3 decode, 0 pages'),
        (63, [42, 100, 200], '63 prefill, 3 decode, 1 page'),
        (65, [42, 100, 200, 50, 150], '65 prefill, 5 decode, 1 page'),
        (126, [42, 100], '126 prefill, 2 decode, 2 pages'),
        (130, [42, 100, 200, 50, 150, 75], '130 prefill, 6 decode, 2+ pages'),
        (200, [42, 100, 200], '200 prefill, 3 decode, 3+ pages'),
    ]

    for n_prefill, decode_toks, desc in test_cases:
        real_base = list(range(5, 5 + n_prefill))
        real_full = real_base + decode_toks

        print(f'\n--- {desc} ---')
        t0 = time.time()

        # Official: prefill the full sequence, extract logits at decode positions
        try:
            off_logits = official_prefill(om, real_full, PS, VS, device)
        except Exception as e:
            print(f'  Official prefill FAILED: {e}')
            continue

        # Find decode positions in the official output
        fi_base = Req._hsa_insert_lmk_prompt(real_base, page_size=PS, lmk_id=VS)
        base_eng_len = len(fi_base)

        # For each decode token, find its engine position
        off_decode_logits = []
        current_real = list(real_base)
        for tok in decode_toks:
            current_real.append(tok)
            fi_current = Req._hsa_insert_lmk_prompt(current_real, page_size=PS, lmk_id=VS)
            # The token 'tok' is at engine position = len(fi_previous)
            fi_prev = Req._hsa_insert_lmk_prompt(current_real[:-1], page_size=PS, lmk_id=VS)
            eng_pos = len(fi_prev)
            if eng_pos < off_logits.shape[1]:
                off_decode_logits.append(off_logits[0, eng_pos])
            else:
                off_decode_logits.append(None)

        # sglang: prefill + multi-step decode
        sg_decode_logits = sglang_prefill_then_decode(
            sm, cfg, real_base, decode_toks, device, dtype)

        t1 = time.time()

        # Compare each decode step
        all_pass = True
        for i, (off_l, sg_l, tok) in enumerate(zip(off_decode_logits, sg_decode_logits, decode_toks)):
            if off_l is None:
                print(f'  Step {i} (tok={tok}): SKIP (official out of range)')
                continue
            diff = (off_l - sg_l).abs()
            mx = diff.max().item()
            cos = torch.nn.functional.cosine_similarity(
                off_l.unsqueeze(0), sg_l.unsqueeze(0)).item()
            am_off = off_l.argmax().item()
            am_sg = sg_l.argmax().item()
            match = am_off == am_sg
            status = 'PASS' if mx < 0.05 else ('CLOSE' if mx < 0.5 else 'FAIL')
            if status == 'FAIL': all_pass = False
            print(f'  Step {i} (tok={tok}): max_err={mx:.4f} cos={cos:.6f} '
                  f'argmax off={am_off} sg={am_sg} {"OK" if match else "MISS"} [{status}]')

        print(f'  Time: {t1-t0:.1f}s  Overall: {"ALL PASS" if all_pass else "HAS FAILURES"}')

    print(f'\n{"="*70}')
    print('STRESS TEST COMPLETE')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
