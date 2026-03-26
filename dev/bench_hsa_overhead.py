"""
Benchmark HSA overhead: sglang HSA layer vs sglang standard attention layer.

Uses the same model architecture but toggles full_attn_interleave:
  - interleave=1: layer 0 is HSA (split-head SWA+HSA attention)
  - interleave=0: layer 0 is standard (full causal Qwen3-style attention)

Same Triton kernels, same infrastructure, same weights (where possible).
This isolates the pure HSA overhead.

Usage:
  CUDA_VISIBLE_DEVICES=4 python dev/bench_hsa_overhead.py
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
# Need official import first for utils
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
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch, ForwardMode, compute_position, compute_decode_positions_landmark,
)
from sglang.srt.models.flash_hsa import HSAForCausalLM as SM
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

if not dist.is_initialized():
    _, p = tempfile.mkstemp(prefix='bo_', suffix='.t')
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


def make_model(interleave, device, dtype):
    """Create model. interleave=1 → HSA layer, interleave=0 → standard attn layer."""
    cfg = FlashHSAConfig(
        model_type='flash_hsa_innerx', architectures=['HSAForCausalLM'],
        vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
        chunk_size=64, hsa_topk=2, hsa_mode='sparse',
        full_attn_interleave=interleave,  # 1=HSA, 0=standard
        hsa_heads=8, hsa_qk_ratio=4,
        use_sliding_window_merging=True, sliding_window_merging_size=64,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act='silu')
    model = SM(cfg).to(device=device, dtype=dtype); model.eval()
    for name, param in model.named_parameters():
        if param.data.norm() == 0 and param.dim() >= 2:
            torch.nn.init.normal_(param.data, mean=0.0, std=0.005)
        elif param.data.norm() == 0 and param.dim() == 1:
            torch.nn.init.ones_(param.data)
    for m in model.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)
    return model, cfg


def make_env(model, cfg, device, dtype, is_hsa=True):
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); mc = 8192
    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(size=mc + 256, page_size=PS, dtype=dtype,
                             head_num=int(cfg.num_key_value_heads), head_dim=int(cfg.head_dim),
                             layer_num=1, device=device,
                             enable_memory_saver=False, enable_alt_stream=False)
    mr = types.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=model,
        model_config=types.SimpleNamespace(is_encoder_decoder=False, context_len=mc,
            num_attention_heads=int(cfg.num_attention_heads),
            get_num_kv_heads=lambda tp: int(cfg.num_key_value_heads) // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend='hsa', speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers='0' if is_hsa else '',  # empty = no HSA layers
            hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id))
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool; mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)
    return be, mr, pool, r2t


def run_prefill(model, cfg, be, mr, pool, r2t, n_real, device, dtype):
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); mc = r2t.shape[1]
    real_tokens = [5 + (i % 250) for i in range(n_real)]
    fi = Req._hsa_insert_lmk_prompt(real_tokens, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
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
    with torch.no_grad(): model.model(fb.input_ids, fb.positions, fb)
    return pl


def run_decode_steps(model, cfg, be, mr, pool, r2t, prefill_len, n_steps, device, dtype):
    PS = cfg.chunk_size; mc = r2t.shape[1]
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    cur = prefill_len
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


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    device = torch.device('cuda'); dtype = torch.bfloat16
    torch.manual_seed(42)

    warmup = 3; iters = 10; n_decode = 10
    test_lengths = [64, 128, 256, 512, 1024, 2048, 4096]

    # Create both models
    print('Creating models...', flush=True)
    model_hsa, cfg_hsa = make_model(interleave=1, device=device, dtype=dtype)  # HSA layer
    model_swa, cfg_swa = make_model(interleave=0, device=device, dtype=dtype)  # Standard attn layer
    layer_type_hsa = type(model_hsa.model.layers[0].self_attn).__name__
    layer_type_swa = type(model_swa.model.layers[0].self_attn).__name__
    print(f'HSA model layer: {layer_type_hsa}', flush=True)
    print(f'SWA model layer: {layer_type_swa}', flush=True)

    print()
    print('=' * 90)
    print('HSA OVERHEAD: sglang HSA layer vs sglang Standard Attention layer')
    print('=' * 90)
    print(f'Config: hidden=1024, 1 layer, heads=16, kv=4, D=64, chunk=64, topk=2')
    print(f'warmup={warmup}, iters={iters}, decode_steps={n_decode}')
    print()

    # ---- Prefill ----
    print('PREFILL')
    print(f'{"N_real":>8} {"Eng":>6} {"SWA(ms)":>9} {"HSA(ms)":>9} {"Overhead":>9} {"SWA_tps":>10} {"HSA_tps":>10}')
    print('-' * 65)

    for n_real in test_lengths:
        # SWA prefill
        be_s, mr_s, pool_s, r2t_s = make_env(model_swa, cfg_swa, device, dtype, is_hsa=False)
        try:
            def fn_swa():
                run_prefill(model_swa, cfg_swa, be_s, mr_s, pool_s, r2t_s, n_real, device, dtype)
            ms_swa = timed(fn_swa, warmup, iters)
        except Exception as e:
            print(f'{n_real:>8} SWA ERROR: {e}', flush=True); continue

        # HSA prefill
        be_h, mr_h, pool_h, r2t_h = make_env(model_hsa, cfg_hsa, device, dtype)
        try:
            def fn_hsa():
                run_prefill(model_hsa, cfg_hsa, be_h, mr_h, pool_h, r2t_h, n_real, device, dtype)
            ms_hsa = timed(fn_hsa, warmup, iters)
        except Exception as e:
            print(f'{n_real:>8} HSA ERROR: {e}', flush=True); continue

        fi = Req._hsa_insert_lmk_prompt([5 + (i % 250) for i in range(n_real)],
                                         page_size=64, lmk_id=256)
        eng = len(fi)
        overhead = ms_hsa / ms_swa
        swa_tps = eng / (ms_swa / 1000)
        hsa_tps = eng / (ms_hsa / 1000)
        print(f'{n_real:>8} {eng:>6} {ms_swa:>9.2f} {ms_hsa:>9.2f} {overhead:>9.2f}x {swa_tps:>10.0f} {hsa_tps:>10.0f}',
              flush=True)

    # ---- Decode ----
    print()
    print('DECODE (ms/token)')
    print(f'{"N_real":>8} {"Eng":>6} {"SWA(ms)":>9} {"HSA(ms)":>9} {"Overhead":>9}')
    print('-' * 45)

    for n_real in test_lengths:
        # SWA decode
        be_s, mr_s, pool_s, r2t_s = make_env(model_swa, cfg_swa, device, dtype, is_hsa=False)
        try:
            pl_s = run_prefill(model_swa, cfg_swa, be_s, mr_s, pool_s, r2t_s, n_real, device, dtype)
            def fn_swa_dec():
                run_decode_steps(model_swa, cfg_swa, be_s, mr_s, pool_s, r2t_s, pl_s, n_decode, device, dtype)
                r2t_s[0, pl_s:pl_s + n_decode] = 0
            ms_swa_total = timed(fn_swa_dec, warmup, iters)
            ms_swa = ms_swa_total / n_decode
        except Exception as e:
            print(f'{n_real:>8} SWA ERROR: {e}', flush=True); continue

        # HSA decode
        be_h, mr_h, pool_h, r2t_h = make_env(model_hsa, cfg_hsa, device, dtype)
        try:
            pl_h = run_prefill(model_hsa, cfg_hsa, be_h, mr_h, pool_h, r2t_h, n_real, device, dtype)
            def fn_hsa_dec():
                run_decode_steps(model_hsa, cfg_hsa, be_h, mr_h, pool_h, r2t_h, pl_h, n_decode, device, dtype)
                r2t_h[0, pl_h:pl_h + n_decode] = 0
            ms_hsa_total = timed(fn_hsa_dec, warmup, iters)
            ms_hsa = ms_hsa_total / n_decode
        except Exception as e:
            print(f'{n_real:>8} HSA ERROR: {e}', flush=True); continue

        fi = Req._hsa_insert_lmk_prompt([5 + (i % 250) for i in range(n_real)],
                                         page_size=64, lmk_id=256)
        eng = len(fi)
        overhead = ms_hsa / ms_swa
        print(f'{n_real:>8} {eng:>6} {ms_swa:>9.2f} {ms_hsa:>9.2f} {overhead:>9.2f}x', flush=True)


if __name__ == '__main__':
    main()
