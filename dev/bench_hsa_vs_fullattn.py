"""
Benchmark 1 layer: HSA (sglang) vs Full Attention (official eager_attention).

Both use the same model weights. The official model runs full-sequence prefill
(no KV cache, recomputes everything). This gives us a clean comparison of
HSA attention vs standard full causal attention at various context lengths.

Usage:
  CUDA_VISIBLE_DEVICES=4 python dev/bench_hsa_vs_fullattn.py
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

import torch

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

    PS = 64; VS = 256
    cd = dict(vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
              num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
              chunk_size=64, hsa_topk=2, hsa_mode='sparse', full_attn_interleave=1,
              hsa_heads=8, hsa_qk_ratio=4, use_sliding_window=True, sliding_window=64,
              tie_word_embeddings=False, rope_theta=10000.0, hidden_act='silu')

    # Official model (full attention via eager_attention_forward + compiled_flex_attention)
    oc = OC(**cd); oc._attn_implementation = 'eager'
    om = OM(oc).to(device=device, dtype=dtype); om.eval()
    for mod in om.modules():
        if isinstance(mod, HA):
            _tf = mod.topk_func; _hf = mod.hsa_func
            mod.topk_func = lambda q, k, *a, _f=_tf, **kw: _f(_d(q), _d(k), *a, **kw)
            mod.hsa_func = lambda q, k, v, *a, _f=_hf, **kw: _f(_d(q), _d(k), _d(v), *a, **kw)

    # sglang HSA model
    import torch.distributed as dist
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
        _, p = tempfile.mkstemp(prefix='bh_', suffix='.t')
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

    cfg = FlashHSAConfig(
        model_type='flash_hsa_innerx', architectures=['HSAForCausalLM'],
        vocab_size=256, hidden_size=1024, intermediate_size=4096, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6,
        chunk_size=64, hsa_topk=2, hsa_mode='sparse', full_attn_interleave=1,
        hsa_heads=8, hsa_qk_ratio=4, use_sliding_window_merging=True,
        sliding_window_merging_size=64, tie_word_embeddings=False, rope_theta=10000.0,
        hidden_act='silu')
    sm = SM(cfg).to(device=device, dtype=dtype); sm.eval()
    # Transfer weights from official
    osd = om.state_dict(); ssd = sm.state_dict()
    for n2, p in ssd.items():
        if n2 in osd and osd[n2].shape == p.shape: p.data.copy_(osd[n2])
        elif ('embed_tokens' in n2 or 'lm_head' in n2) and n2 in osd:
            mr = min(osd[n2].shape[0], p.shape[0]); p.data[:mr].copy_(osd[n2][:mr])
        elif 'gate_up_proj' in n2:
            gn = n2.replace('gate_up_proj', 'gate_proj'); un = n2.replace('gate_up_proj', 'up_proj')
            if gn in osd and un in osd:
                f = torch.cat([osd[gn], osd[un]], dim=0)
                if f.shape == p.shape: p.data.copy_(f)
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    lmk_id = VS; mc = 8192
    warmup = 3; iters = 10

    test_lengths = [64, 128, 256, 512, 1024, 2048, 4096]

    print('=' * 80)
    print('1-LAYER BENCHMARK: HSA (sglang) vs Full Attention (official)')
    print('=' * 80)
    print(f'Config: hidden=1024, heads=16, kv=4, D=64, chunk=64, topk=2')
    print(f'warmup={warmup}, iters={iters}')
    print()

    # ---- Prefill ----
    print(f'{"N_real":>8} {"Eng":>6} {"Pages":>5} '
          f'{"Official(ms)":>12} {"sglang(ms)":>11} {"Ratio":>7} '
          f'{"Off_tps":>10} {"SG_tps":>10}')
    print('-' * 80)

    for n_real in test_lengths:
        real_tokens = [5 + (i % 250) for i in range(n_real)]

        # Official prefill
        ids = insert_special_tokens(torch.tensor([real_tokens]), VS, PS)
        pos_ids = create_position_ids_with_landmarks(n_real, PS, device)
        mask = torch.ones((1, ids.shape[1]), dtype=torch.long, device=device)
        ids_dev = ids.to(device)
        pos_dev = pos_ids.to(device)

        try:
            ms_off = timed(
                lambda: om(input_ids=ids_dev, position_ids=pos_dev, attention_mask=mask, use_cache=False),
                warmup=warmup, iters=iters
            )
        except Exception as e:
            print(f'{n_real:>8} Official ERROR: {e}')
            continue

        eng_off = ids.shape[1]

        # sglang prefill
        fi = Req._hsa_insert_lmk_prompt(real_tokens, page_size=PS, lmk_id=lmk_id)
        pl = len(fi)
        r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
        pool = MHATokenToKVPool(size=mc + 256, page_size=PS, dtype=dtype, head_num=4,
                                 head_dim=64, layer_num=1, device=device,
                                 enable_memory_saver=False, enable_alt_stream=False)
        mr_obj = types.SimpleNamespace(
            device=device, page_size=PS, sliding_window_size=None, model=sm,
            model_config=types.SimpleNamespace(is_encoder_decoder=False, context_len=mc,
                num_attention_heads=16, get_num_kv_heads=lambda tp: 4 // tp),
            hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
            server_args=types.SimpleNamespace(
                attention_backend='hsa', speculative_num_draft_tokens=0, speculative_num_steps=0,
                triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
                enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
                hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id))
        mr_obj.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
        mr_obj.token_to_kv_pool = pool; mr_obj.token_to_kv_pool_allocator = object()
        be = HSAAttnBackend(mr_obj)
        tl = torch.arange(0, mc, dtype=torch.int32, device=device)

        def run_sg_prefill():
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
                req_to_token_pool=mr_obj.req_to_token_pool, token_to_kv_pool=pool, attn_backend=be)
            be.init_forward_metadata(fb)
            with torch.no_grad(): sm.model(fb.input_ids, fb.positions, fb)

        try:
            ms_sg = timed(run_sg_prefill, warmup=warmup, iters=iters)
        except Exception as e:
            print(f'{n_real:>8} sglang ERROR: {e}')
            continue

        pages = pl // PS
        ratio = ms_sg / ms_off if ms_off > 0 else float('nan')
        off_tps = eng_off / (ms_off / 1000)
        sg_tps = pl / (ms_sg / 1000)
        print(f'{n_real:>8} {pl:>6} {pages:>5} {ms_off:>12.2f} {ms_sg:>11.2f} {ratio:>7.2f}x '
              f'{off_tps:>10.0f} {sg_tps:>10.0f}')

    print()
    print('Note: Official uses eager_attention (PyTorch matmul) + compiled_flex_attention +')
    print('      tilelang kernels. sglang uses Triton extend kernel + batched HSA selection.')
    print('      Both process the FULL sequence in one forward pass (no KV cache reuse).')
    print('      "Ratio" = sglang_time / official_time (lower is better for sglang).')


if __name__ == '__main__':
    main()
