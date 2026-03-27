"""
Comprehensive HSA test: correctness + efficiency at short, mid, long context.

Short:  64, 128, 256 real tokens   (0-4 pages)
Mid:    512, 1024, 2048             (8-32 pages)
Long:   4096, 8192, 16384           (64-256 pages)

Tests:
  1. Correctness: official prefill(N+K) vs sglang prefill(N) + K decode steps
  2. Efficiency: HSA vs Dense latency for prefill and decode

Usage:
  CUDA_VISIBLE_DEVICES=2 python dev/test_short_mid_long.py
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
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch, ForwardMode, compute_position, compute_decode_positions_landmark,
)
from sglang.srt.models.flash_hsa import HSAForCausalLM as SM
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def init_sglang():
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix='sml_', suffix='.t')
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


def make_env(cfg, device, dtype, attention_backend='hsa', max_ctx=None):
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size)
    mc = max_ctx or 32768

    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(size=mc + 256, page_size=PS, dtype=dtype,
                             head_num=int(cfg.num_key_value_heads), head_dim=int(cfg.head_dim),
                             layer_num=int(cfg.num_hidden_layers), device=device,
                             enable_memory_saver=False, enable_alt_stream=False)
    mr = types.SimpleNamespace(
        device=device, page_size=PS,
        sliding_window_size=PS if attention_backend != 'hsa' else None,
        model=None,
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, context_len=mc,
            num_attention_heads=int(cfg.num_attention_heads),
            head_dim=int(cfg.head_dim),
            get_num_kv_heads=lambda tp: int(cfg.num_key_value_heads) // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend=attention_backend, speculative_num_draft_tokens=0,
            speculative_num_steps=0, triton_attention_num_kv_splits=8,
            triton_attention_split_tile_size=None, enable_deterministic_inference=False,
            hsa_topk=None, hsa_selection_strategy=None, hsa_layers=None,
            hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id))
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool; mr.token_to_kv_pool_allocator = object()

    if attention_backend == 'hsa':
        be = HSAAttnBackend(mr)
    else:
        be = TritonAttnBackend(mr)
    return be, mr, pool, r2t


def official_prefill(model, real_tokens, PS, VS, device):
    ids = insert_special_tokens(torch.tensor([real_tokens]), VS, PS)
    pos = create_position_ids_with_landmarks(len(real_tokens), PS, device)
    mask = torch.ones((1, ids.shape[1]), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids.to(device), position_ids=pos.to(device),
                     attention_mask=mask, use_cache=False)
    return out.logits[:, :, :VS].float()


def sglang_prefill(sg_model, cfg, be, mr, pool, r2t, real_tokens, device, dtype):
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size)
    fi = Req._hsa_insert_lmk_prompt(real_tokens, page_size=PS, lmk_id=lmk_id)
    pl = len(fi); mc = r2t.shape[1]
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    r2t[0, :] = 0; r2t[0, :pl] = tl[:pl]
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
    return pl


def sglang_decode_step(sg_model, cfg, be, mr, pool, r2t, current_len, tok_id, device, dtype):
    PS = cfg.chunk_size; mc = r2t.shape[1]; VS = int(cfg.vocab_size)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    current_len += 1
    r2t[0, current_len - 1] = tl[current_len - 1]
    dp = compute_decode_positions_landmark(
        torch.tensor([current_len], device=device, dtype=torch.int32), page_size=PS)
    fbd = ForwardBatch(
        forward_mode=ForwardMode.DECODE, batch_size=1,
        input_ids=torch.tensor([tok_id], device=device, dtype=torch.int64),
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
    return logits.view(-1), current_len


def bench_prefill_latency(model, cfg, be, mr, pool, r2t, n_real, device, dtype, warmup=3, iters=5):
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); mc = r2t.shape[1]
    real_tokens = [5 + (i % 250) for i in range(n_real)]
    fi = Req._hsa_insert_lmk_prompt(real_tokens, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)

    for _ in range(warmup):
        r2t[0, :] = 0; r2t[0, :pl] = tl[:pl]
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
        with torch.no_grad(): model.model(fb.input_ids, fb.positions, fb)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        r2t[0, :] = 0; r2t[0, :pl] = tl[:pl]
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
        with torch.no_grad(): model.model(fb.input_ids, fb.positions, fb)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000, pl


def bench_decode_latency(model, cfg, be, mr, pool, r2t, n_real, device, dtype,
                          n_decode=10, warmup=3, iters=5):
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); mc = r2t.shape[1]
    real_tokens = [5 + (i % 250) for i in range(n_real)]
    fi = Req._hsa_insert_lmk_prompt(real_tokens, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)

    # Prefill once
    r2t[0, :] = 0; r2t[0, :pl] = tl[:pl]
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
    with torch.no_grad(): model.model(fb.input_ids, fb.positions, fb)

    def run_decode_steps(start_len, n_steps):
        cur = start_len
        for i in range(n_steps):
            cur += 1
            r2t[0, cur - 1] = tl[cur - 1]
            dp = compute_decode_positions_landmark(
                torch.tensor([cur], device=device, dtype=torch.int32), page_size=PS)
            fbd = ForwardBatch(
                forward_mode=ForwardMode.DECODE, batch_size=1,
                input_ids=torch.tensor([42 + i], device=device, dtype=torch.int64),
                req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
                seq_lens=torch.tensor([cur], device=device, dtype=torch.int32),
                out_cache_loc=tl[cur-1:cur].to(torch.int64), seq_lens_sum=cur,
                seq_lens_cpu=torch.tensor([cur], device='cpu', dtype=torch.int32),
                positions=dp, req_to_token_pool=mr.req_to_token_pool,
                token_to_kv_pool=pool, attn_backend=be)
            be.init_forward_metadata(fbd)
            with torch.no_grad(): model.model(fbd.input_ids, fbd.positions, fbd)

    for _ in range(warmup):
        run_decode_steps(pl, n_decode)
        r2t[0, pl:pl + n_decode] = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_decode_steps(pl, n_decode)
        r2t[0, pl:pl + n_decode] = 0
    torch.cuda.synchronize()
    ms_total = (time.perf_counter() - t0) / iters * 1000
    return ms_total / n_decode, pl


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

    # Official model
    oc = OC(**cd); oc._attn_implementation = 'eager'
    om = OM(oc).to(device=device, dtype=dtype); om.eval()
    for mod in om.modules():
        if isinstance(mod, HA):
            _tf = mod.topk_func; _hf = mod.hsa_func
            mod.topk_func = lambda q, k, *a, _f=_tf, **kw: _f(_d(q), _d(k), *a, **kw)
            mod.hsa_func = lambda q, k, v, *a, _f=_hf, **kw: _f(_d(q), _d(k), _d(v), *a, **kw)

    # sglang model
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
    print(f'Weights transferred: {loaded}/{total}')
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    # ================================================================
    # PART 1: CORRECTNESS (official vs sglang)
    # ================================================================
    print()
    print('=' * 80)
    print('PART 1: CORRECTNESS — Official vs sglang (prefill + decode)')
    print('=' * 80)

    correctness_cases = [
        # (n_prefill, decode_tokens, label)
        # SHORT
        (32,   [42, 100, 200], 'SHORT: 32 tokens, 0 pages'),
        (64,   [42, 100, 200], 'SHORT: 64 tokens, 1 page'),
        (128,  [42, 100, 200], 'SHORT: 128 tokens, 2 pages'),
        (256,  [42, 100, 200], 'SHORT: 256 tokens, 4 pages'),
        # MID
        (512,  [42, 100, 200], 'MID: 512 tokens, 8 pages'),
        (1024, [42, 100, 200], 'MID: 1024 tokens, 16 pages'),
        (2048, [42, 100, 200], 'MID: 2048 tokens, 32 pages'),
        # LONG
        (4096, [42, 100, 200], 'LONG: 4096 tokens, 64 pages'),
        (8192, [42, 100, 200], 'LONG: 8192 tokens, 128 pages'),
    ]

    correctness_results = []
    for n_prefill, decode_toks, desc in correctness_cases:
        print(f'\n--- {desc} ---')
        real_base = [5 + (i % 250) for i in range(n_prefill)]
        real_full = real_base + decode_toks

        # Official prefill full sequence
        t0 = time.time()
        try:
            off_logits = official_prefill(om, real_full, PS, VS, device)
        except Exception as e:
            print(f'  Official prefill FAILED: {e}')
            correctness_results.append((desc, None, None, None, 'OFFICIAL_FAIL'))
            continue

        # sglang prefill + decode
        max_ctx = n_prefill * 2 + 256
        be, mr, pool, r2t = make_env(cfg, device, dtype, 'hsa', max_ctx)
        mr.model = sm
        try:
            eng_len = sglang_prefill(sm, cfg, be, mr, pool, r2t, real_base, device, dtype)
        except Exception as e:
            print(f'  sglang prefill FAILED: {e}')
            correctness_results.append((desc, None, None, None, 'SG_PREFILL_FAIL'))
            continue

        # Decode steps
        current_len = eng_len
        all_pass = True
        max_err_overall = 0
        for i, tok in enumerate(decode_toks):
            try:
                sg_logits, current_len = sglang_decode_step(
                    sm, cfg, be, mr, pool, r2t, current_len, tok, device, dtype)

                # Find official position for this decode token
                current_real = real_base + decode_toks[:i+1]
                fi_prev = Req._hsa_insert_lmk_prompt(current_real[:-1], page_size=PS, lmk_id=VS)
                eng_pos = len(fi_prev)
                if eng_pos < off_logits.shape[1]:
                    off_l = off_logits[0, eng_pos]
                    diff = (off_l - sg_logits).abs()
                    mx = diff.max().item()
                    cos = torch.nn.functional.cosine_similarity(
                        off_l.unsqueeze(0), sg_logits.unsqueeze(0)).item()
                    am_off = off_l.argmax().item()
                    am_sg = sg_logits.argmax().item()
                    match = am_off == am_sg
                    status = 'PASS' if mx < 0.1 else ('CLOSE' if mx < 1.0 else 'FAIL')
                    if status == 'FAIL': all_pass = False
                    max_err_overall = max(max_err_overall, mx)
                    print(f'  Step {i} (tok={tok}): max_err={mx:.4f} cos={cos:.6f} '
                          f'argmax off={am_off} sg={am_sg} {"OK" if match else "MISS"} [{status}]')
                else:
                    print(f'  Step {i}: SKIP (official out of range)')
            except Exception as e:
                print(f'  Step {i}: DECODE FAILED: {e}')
                all_pass = False

        elapsed = time.time() - t0
        overall = 'ALL PASS' if all_pass else 'HAS FAILURES'
        print(f'  max_err={max_err_overall:.4f} time={elapsed:.1f}s [{overall}]')
        correctness_results.append((desc, max_err_overall, elapsed, all_pass, overall))

    # ================================================================
    # PART 2: EFFICIENCY (HSA vs Dense)
    # ================================================================
    print()
    print('=' * 80)
    print('PART 2: EFFICIENCY — HSA vs Dense latency')
    print('=' * 80)
    print(f'Config: hidden=1024, 1 layer, heads=16, kv=4, D=64, chunk=64, topk=2')

    efficiency_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # Prefill benchmark
    print()
    print('--- PREFILL LATENCY ---')
    header = f'{"N_real":>8} {"Category":>8} {"HSA(ms)":>10} {"Dense(ms)":>10} {"Ratio":>8} {"Pages":>6}'
    print(header)
    print('-' * len(header))

    prefill_data = []
    for n in efficiency_lengths:
        cat = 'SHORT' if n <= 256 else ('MID' if n <= 2048 else 'LONG')
        max_ctx = n * 2 + 1024

        # HSA
        be_h, mr_h, pool_h, r2t_h = make_env(cfg, device, dtype, 'hsa', max_ctx)
        mr_h.model = sm
        try:
            hsa_ms, eng_tok = bench_prefill_latency(sm, cfg, be_h, mr_h, pool_h, r2t_h,
                                                      n, device, dtype, warmup=3, iters=5)
            pages = eng_tok // PS
        except Exception as e:
            print(f'{n:>8} {cat:>8} HSA FAILED: {e}')
            continue

        # Dense
        be_d, mr_d, pool_d, r2t_d = make_env(cfg, device, dtype, 'triton', max_ctx)
        mr_d.model = sm
        try:
            dense_ms, _ = bench_prefill_latency(sm, cfg, be_d, mr_d, pool_d, r2t_d,
                                                  n, device, dtype, warmup=3, iters=5)
        except Exception as e:
            print(f'{n:>8} {cat:>8} Dense FAILED: {e}')
            continue

        ratio = hsa_ms / dense_ms
        print(f'{n:>8} {cat:>8} {hsa_ms:>10.2f} {dense_ms:>10.2f} {ratio:>7.2f}x {pages:>6}')
        prefill_data.append((n, cat, hsa_ms, dense_ms, ratio, pages))

    # Decode benchmark
    print()
    print('--- DECODE LATENCY (ms/token) ---')
    header = f'{"N_real":>8} {"Category":>8} {"HSA(ms)":>10} {"Dense(ms)":>10} {"Ratio":>8} {"Pages":>6}'
    print(header)
    print('-' * len(header))

    decode_data = []
    for n in efficiency_lengths:
        cat = 'SHORT' if n <= 256 else ('MID' if n <= 2048 else 'LONG')
        max_ctx = n * 2 + 1024

        # HSA
        be_h, mr_h, pool_h, r2t_h = make_env(cfg, device, dtype, 'hsa', max_ctx)
        mr_h.model = sm
        try:
            hsa_ms, eng_tok = bench_decode_latency(sm, cfg, be_h, mr_h, pool_h, r2t_h,
                                                     n, device, dtype, n_decode=10,
                                                     warmup=2, iters=5)
            pages = eng_tok // PS
        except Exception as e:
            print(f'{n:>8} {cat:>8} HSA FAILED: {e}')
            continue

        # Dense
        be_d, mr_d, pool_d, r2t_d = make_env(cfg, device, dtype, 'triton', max_ctx)
        mr_d.model = sm
        try:
            dense_ms, _ = bench_decode_latency(sm, cfg, be_d, mr_d, pool_d, r2t_d,
                                                 n, device, dtype, n_decode=10,
                                                 warmup=2, iters=5)
        except Exception as e:
            print(f'{n:>8} {cat:>8} Dense FAILED: {e}')
            continue

        ratio = hsa_ms / dense_ms
        print(f'{n:>8} {cat:>8} {hsa_ms:>10.2f} {dense_ms:>10.2f} {ratio:>7.2f}x {pages:>6}')
        decode_data.append((n, cat, hsa_ms, dense_ms, ratio, pages))

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)

    print()
    print('--- Correctness ---')
    print(f'{"Test Case":<45} {"Max Err":>8} {"Time(s)":>8} {"Status":>12}')
    print('-' * 75)
    for desc, mx, elapsed, passed, status in correctness_results:
        mx_s = f'{mx:.4f}' if mx is not None else 'N/A'
        t_s = f'{elapsed:.1f}' if elapsed is not None else 'N/A'
        print(f'{desc:<45} {mx_s:>8} {t_s:>8} {status:>12}')

    print()
    print('--- Efficiency (Prefill) ---')
    print(f'{"N_real":>8} {"Cat":>6} {"HSA(ms)":>10} {"Dense(ms)":>10} {"HSA/Dense":>10}')
    print('-' * 48)
    for n, cat, hsa_ms, dense_ms, ratio, pages in prefill_data:
        print(f'{n:>8} {cat:>6} {hsa_ms:>10.2f} {dense_ms:>10.2f} {ratio:>9.2f}x')

    print()
    print('--- Efficiency (Decode ms/token) ---')
    print(f'{"N_real":>8} {"Cat":>6} {"HSA(ms)":>10} {"Dense(ms)":>10} {"HSA/Dense":>10}')
    print('-' * 48)
    for n, cat, hsa_ms, dense_ms, ratio, pages in decode_data:
        print(f'{n:>8} {cat:>6} {hsa_ms:>10.2f} {dense_ms:>10.2f} {ratio:>9.2f}x')


if __name__ == '__main__':
    main()
