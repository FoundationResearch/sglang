"""
Benchmark HSA prefill and decode efficiency at various context lengths.

Measures:
  - Prefill latency (ms) and throughput (tokens/s)
  - Decode latency (ms/token)
  - Comparison with dense (vanilla SWA) attention

Usage:
  CUDA_VISIBLE_DEVICES=4 python dev/bench_efficiency.py
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

# Import official to avoid shadowing
from models.FlashHSA.modeling_hsa_lmk import HSAForCausalLM as OM, HierarchicalSparseAttention as HA
_oi = HA.__init__
def _pi(s, c, l):
    _oi(s, c, l)
    if not hasattr(s, 'num_key_value_groups'):
        s.num_key_value_groups = c.num_attention_heads // c.num_key_value_heads
HA.__init__ = _pi

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
        _, p = tempfile.mkstemp(prefix='b_', suffix='.t')
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


def make_model_and_env(cfg, device, dtype, attention_backend='hsa'):
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size)
    mc = 8192

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


def bench_prefill(model, cfg, be, mr, pool, r2t, n_real, device, dtype, warmup=2, iters=5):
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); mc = r2t.shape[1]
    real_tokens = [5 + (i % 250) for i in range(n_real)]  # keep within vocab_size=256
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
    ms = (time.perf_counter() - t0) / iters * 1000
    return ms, pl


def bench_decode(model, cfg, be, mr, pool, r2t, n_real, device, dtype, n_decode=10, warmup=2, iters=5):
    """Prefill n_real tokens, then benchmark n_decode decode steps."""
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); mc = r2t.shape[1]
    real_tokens = [5 + (i % 250) for i in range(n_real)]  # keep within vocab_size=256
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

    # Warmup
    for _ in range(warmup):
        run_decode_steps(pl, n_decode)
        # Reset r2t for repeated runs
        r2t[0, pl:pl + n_decode] = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_decode_steps(pl, n_decode)
        r2t[0, pl:pl + n_decode] = 0
    torch.cuda.synchronize()
    ms_total = (time.perf_counter() - t0) / iters * 1000
    ms_per_token = ms_total / n_decode
    return ms_per_token, pl


def main():
    device = torch.device('cuda'); dtype = torch.bfloat16
    init_sglang()
    torch.manual_seed(42)

    cfg = FlashHSAConfig(
        model_type='flash_hsa_innerx', architectures=['HSAForCausalLM'],
        vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
        chunk_size=64, hsa_topk=2, hsa_mode='sparse', full_attn_interleave=1,
        hsa_heads=8, hsa_qk_ratio=4, use_sliding_window_merging=True,
        sliding_window_merging_size=64, tie_word_embeddings=False, rope_theta=10000.0,
        hidden_act='silu')

    sm = SM(cfg).to(device=device, dtype=dtype); sm.eval()
    # Initialize weights with small std to avoid numerical overflow at long sequences
    for name, param in sm.named_parameters():
        if param.data.norm() == 0 and param.dim() >= 2:
            torch.nn.init.normal_(param.data, mean=0.0, std=0.005)
        elif param.data.norm() == 0 and param.dim() == 1:
            torch.nn.init.ones_(param.data)
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    test_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    n_decode_steps = 10
    warmup = 3
    iters = 5

    print('=' * 80)
    print('HSA EFFICIENCY BENCHMARK')
    print('=' * 80)
    print(f'Config: hidden=1024, 1 layer, heads=16, kv=4, D=64, chunk=64, topk=2')
    print(f'Decode: {n_decode_steps} steps per measurement, warmup={warmup}, iters={iters}')
    print()

    # HSA prefill
    print('--- PREFILL (HSA) ---')
    print(f'{"N_real":>8} {"Eng_tok":>8} {"Pages":>6} {"Latency(ms)":>12} {"Throughput(tok/s)":>18}')
    print('-' * 55)
    prefill_results = []
    for n in test_lengths:
        be, mr, pool, r2t = make_model_and_env(cfg, device, dtype, 'hsa')
        mr.model = sm
        try:
            ms, eng_tok = bench_prefill(sm, cfg, be, mr, pool, r2t, n, device, dtype, warmup, iters)
            pages = eng_tok // 64
            tps = eng_tok / (ms / 1000)
            print(f'{n:>8} {eng_tok:>8} {pages:>6} {ms:>12.2f} {tps:>18.0f}')
            prefill_results.append((n, eng_tok, pages, ms, tps))
        except Exception as e:
            print(f'{n:>8} ERROR: {e}')

    # HSA decode
    print()
    print('--- DECODE (HSA) ---')
    print(f'{"N_real":>8} {"Eng_tok":>8} {"Pages":>6} {"Latency(ms/tok)":>16}')
    print('-' * 42)
    decode_results = []
    for n in test_lengths:
        be, mr, pool, r2t = make_model_and_env(cfg, device, dtype, 'hsa')
        mr.model = sm
        try:
            ms_per_tok, eng_tok = bench_decode(sm, cfg, be, mr, pool, r2t, n, device, dtype,
                                                n_decode_steps, warmup, iters)
            pages = eng_tok // 64
            print(f'{n:>8} {eng_tok:>8} {pages:>6} {ms_per_tok:>16.2f}')
            decode_results.append((n, eng_tok, pages, ms_per_tok))
        except Exception as e:
            print(f'{n:>8} ERROR: {e}')

    # Dense (vanilla) prefill for comparison
    print()
    print('--- PREFILL (Dense/Triton, no HSA) ---')
    print(f'{"N_real":>8} {"Eng_tok":>8} {"Latency(ms)":>12} {"Throughput(tok/s)":>18}')
    print('-' * 48)
    dense_prefill = []
    for n in test_lengths:
        be, mr, pool, r2t = make_model_and_env(cfg, device, dtype, 'triton')
        mr.model = sm
        try:
            ms, eng_tok = bench_prefill(sm, cfg, be, mr, pool, r2t, n, device, dtype, warmup, iters)
            tps = eng_tok / (ms / 1000)
            print(f'{n:>8} {eng_tok:>8} {ms:>12.2f} {tps:>18.0f}')
            dense_prefill.append((n, eng_tok, ms, tps))
        except Exception as e:
            print(f'{n:>8} ERROR: {e}')

    # Dense decode for comparison
    print()
    print('--- DECODE (Dense/Triton, no HSA) ---')
    print(f'{"N_real":>8} {"Eng_tok":>8} {"Latency(ms/tok)":>16}')
    print('-' * 36)
    dense_decode = []
    for n in test_lengths:
        be, mr, pool, r2t = make_model_and_env(cfg, device, dtype, 'triton')
        mr.model = sm
        try:
            ms_per_tok, eng_tok = bench_decode(sm, cfg, be, mr, pool, r2t, n, device, dtype,
                                                n_decode_steps, warmup, iters)
            print(f'{n:>8} {eng_tok:>8} {ms_per_tok:>16.2f}')
            dense_decode.append((n, eng_tok, ms_per_tok))
        except Exception as e:
            print(f'{n:>8} ERROR: {e}')

    # Summary
    print()
    print('=' * 80)
    print('SUMMARY: HSA vs Dense')
    print('=' * 80)
    print(f'{"N_real":>8} {"HSA_pf(ms)":>11} {"Dense_pf(ms)":>13} {"PF_ratio":>9} '
          f'{"HSA_dc(ms)":>11} {"Dense_dc(ms)":>13} {"DC_ratio":>9}')
    print('-' * 77)
    for i, n in enumerate(test_lengths):
        hsa_pf = prefill_results[i][3] if i < len(prefill_results) else float('nan')
        den_pf = dense_prefill[i][2] if i < len(dense_prefill) else float('nan')
        hsa_dc = decode_results[i][3] if i < len(decode_results) else float('nan')
        den_dc = dense_decode[i][2] if i < len(dense_decode) else float('nan')
        pf_ratio = hsa_pf / den_pf if den_pf > 0 else float('nan')
        dc_ratio = hsa_dc / den_dc if den_dc > 0 else float('nan')
        print(f'{n:>8} {hsa_pf:>11.2f} {den_pf:>13.2f} {pf_ratio:>9.2f}x '
              f'{hsa_dc:>11.2f} {den_dc:>13.2f} {dc_ratio:>9.2f}x')


if __name__ == '__main__':
    main()
