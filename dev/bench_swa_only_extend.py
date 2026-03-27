"""
A/B benchmark: SWA-only extend (new) vs all-heads extend (old).

Measures prefill latency with the efficiency fix (dense extend on SWA heads only)
vs the old path (dense extend on all heads, then overwrite HSA heads).

Usage:
  CUDA_VISIBLE_DEVICES=0 python dev/bench_swa_only_extend.py
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
    l = logging.getLogger(n); l.info_rank0 = l.info; l.warning_rank0 = l.warning; return l
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
from sglang.srt.layers import dp_attention as dpa
dpa.get_attention_tp_size = lambda: 1
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.configs.flash_hsa import FlashHSAConfig
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch, ForwardMode, compute_position,
)
from sglang.srt.models.flash_hsa import HSAForCausalLM as SM
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def init_sglang():
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix='ab_', suffix='.t')
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


def bench_prefill(sm, cfg, be, mr, pool, r2t, n_real, device, dtype, warmup=3, iters=10):
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
        with torch.no_grad(): sm.model(fb.input_ids, fb.positions, fb)

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
        with torch.no_grad(): sm.model(fb.input_ids, fb.positions, fb)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000, pl


def main():
    device = torch.device('cuda'); dtype = torch.bfloat16
    init_sglang()
    torch.manual_seed(42)

    # Config: realistic 7B-like, 1 layer
    # hidden=4096, 32 q-heads, 8 kv-heads, D=128
    # hsa_heads=8 → denom=4, h_hsa=2, h_swa=6, HQ_hsa=8, HQ_swa=24 (75/25 split)
    PS = 64; VS = 256
    cfg = FlashHSAConfig(
        model_type='flash_hsa_innerx', architectures=['HSAForCausalLM'],
        vocab_size=256, hidden_size=4096, intermediate_size=16384,
        num_hidden_layers=1, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, rms_norm_eps=1e-6, chunk_size=PS, hsa_topk=2, hsa_mode='sparse',
        full_attn_interleave=1, hsa_heads=8, hsa_qk_ratio=4,
        use_sliding_window_merging=True, sliding_window_merging_size=PS,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act='silu')

    sm = SM(cfg).to(device=device, dtype=dtype); sm.eval()
    for name, param in sm.named_parameters():
        if param.data.norm() == 0 and param.dim() >= 2:
            torch.nn.init.normal_(param.data, mean=0.0, std=0.005)
        elif param.data.norm() == 0 and param.dim() == 1:
            torch.nn.init.ones_(param.data)
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    lmk_id = VS; mc = 8192
    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)

    # Get split info
    for layer in sm.model.layers:
        if hasattr(layer.self_attn, 'hq_swa'):
            a = layer.self_attn
            print(f'Head split: HQ_swa={a.hq_swa}, HQ_hsa={a.hq_hsa}, H_swa={a.hk_swa}, H_hsa={a.hk_hsa}')
            break

    print()
    print('=' * 70)
    print('A/B BENCHMARK: SWA-only extend vs all-heads extend')
    print('=' * 70)
    print(f'Config: hidden=4096, 1 layer, heads=32, kv=8, D=128, chunk=64, topk=2')
    print()

    test_lengths = [256, 512, 1024, 2048, 4096]

    # --- NEW PATH (SWA-only) ---
    print('--- NEW: Dense extend on SWA heads only ---')
    pool_new = MHATokenToKVPool(size=mc+256, page_size=PS, dtype=dtype,
        head_num=8, head_dim=128, layer_num=1, device=device,
        enable_memory_saver=False, enable_alt_stream=False)
    mr_new = types.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=sm,
        model_config=types.SimpleNamespace(is_encoder_decoder=False, context_len=mc,
            num_attention_heads=32, head_dim=128, get_num_kv_heads=lambda tp: 8 // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend='hsa', speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id))
    mr_new.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr_new.token_to_kv_pool = pool_new; mr_new.token_to_kv_pool_allocator = object()
    be_new = HSAAttnBackend(mr_new)

    new_results = {}
    for n in test_lengths:
        ms, eng = bench_prefill(sm, cfg, be_new, mr_new, pool_new, r2t, n, device, dtype)
        new_results[n] = ms
        print(f'  N={n:>5}  eng={eng:>5}  latency={ms:.2f} ms')

    # --- OLD PATH (all heads) via env var or monkey-patch ---
    print()
    print('--- OLD: Dense extend on ALL heads (monkey-patched) ---')
    pool_old = MHATokenToKVPool(size=mc+256, page_size=PS, dtype=dtype,
        head_num=8, head_dim=128, layer_num=1, device=device,
        enable_memory_saver=False, enable_alt_stream=False)
    mr_old = types.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=sm,
        model_config=types.SimpleNamespace(is_encoder_decoder=False, context_len=mc,
            num_attention_heads=32, head_dim=128, get_num_kv_heads=lambda tp: 8 // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend='hsa', speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id))
    mr_old.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr_old.token_to_kv_pool = pool_old; mr_old.token_to_kv_pool_allocator = object()
    be_old = HSAAttnBackend(mr_old)

    # Monkey-patch: force old path (run dense on ALL heads)
    _orig_forward_extend = be_old.forward_extend.__func__

    def _old_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        split_info = kwargs.get("hsa_split_head_info", None)
        if not self._is_hsa_layer(layer.layer_id) or split_info is None:
            kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
            return self._dense_backend.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs_clean)
        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None:
            kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
            return self._dense_backend.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs_clean)
        if save_kv_cache:
            pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        HQ_swa = int(split_info.get("hq_swa", 0))
        HQ_hsa = int(split_info.get("hq_hsa", 0))
        H_swa = int(split_info.get("h_swa", 0))
        H_hsa = int(split_info.get("h_hsa", 0))
        hsa_window = int(split_info.get("swa_window_size", 0) or 0)
        HQ_total = int(layer.tp_q_head_num)
        D = int(layer.qk_head_dim)
        T = q.shape[0]
        q3 = q.view(T, HQ_total, D)
        q_hsa = q3[:, HQ_swa:, :]
        # OLD: dense on ALL heads
        kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
        dense_out = self._dense_backend.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=False, **kwargs_clean)
        dense_out_3 = dense_out.view(T, HQ_total, D)
        page_table_1 = md.page_table_1
        selection_q = kwargs.get("hsa_selection_q", None)
        swa_o_inner, lse_kv = self._compute_internal_swa_extend(q_hsa=q_hsa, layer=layer, forward_batch=forward_batch,
            page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa, HQ_hsa=HQ_hsa, hsa_window=hsa_window)
        self._run_selection_extend(q=q_hsa, layer=layer, forward_batch=forward_batch, selection_q=selection_q,
            page_table_1=page_table_1, kv_head_offset=H_swa, kv_head_count=H_hsa, hsa_window=hsa_window)
        selected_page_ids = md.hsa_ext_selected_page_ids
        selected_scores = md.hsa_ext_selected_scores
        if selected_page_ids is None or selected_scores is None:
            return dense_out
        Gh = HQ_hsa // H_hsa; TOPK = int(selected_page_ids.shape[2])
        valid = selected_page_ids >= 0; scores = selected_scores.masked_fill(~valid, float("-inf"))
        if hsa_window > 0:
            if not self.enable_softmax1:
                cat_scores = torch.cat([scores, lse_kv.unsqueeze(-1)], dim=-1); swa_weight_idx = -1
            else:
                cat_scores = torch.cat([scores, lse_kv.unsqueeze(-1), torch.zeros(T, H_hsa, 1, device=scores.device, dtype=scores.dtype)], dim=-1); swa_weight_idx = -2
            merged_w = torch.softmax(cat_scores, dim=-1); merged_w = torch.nan_to_num(merged_w, nan=0.0)
            w_kv = merged_w[:, :, :TOPK].to(q_hsa.dtype); swa_w_kv = merged_w[:, :, swa_weight_idx]
        else:
            w_kv = torch.softmax(scores, dim=-1); w_kv = torch.nan_to_num(w_kv, nan=0.0).to(q_hsa.dtype); swa_w_kv = None
        w_q = w_kv[:, :, None, :].expand(T, H_hsa, Gh, TOPK).reshape(T, HQ_hsa, TOPK).contiguous()
        from sglang.srt.layers.attention.hsa.kernels import hsa_decode_paged_fwd
        k_cache_hsa = pool.get_key_buffer(layer.layer_id)[:, H_swa:H_swa+H_hsa, :]
        v_cache_hsa = pool.get_value_buffer(layer.layer_id)[:, H_swa:H_swa+H_hsa, :]
        out_hsa = self._hsa_sparse_attn_extend(q_hsa=q_hsa, k_cache=k_cache_hsa, v_cache=v_cache_hsa,
            page_table_1=page_table_1, selected_page_ids=selected_page_ids, hsa_weights=w_q,
            H_hsa=H_hsa, HQ_hsa=HQ_hsa, sm_scale=getattr(layer, "scaling", None))
        if swa_w_kv is not None:
            swa_w_q = swa_w_kv[:, :, None].expand(T, H_hsa, Gh).reshape(T, HQ_hsa)
            out_hsa = out_hsa.to(torch.float32) + swa_o_inner * swa_w_q[:, :, None]
        out_all = dense_out_3.clone()
        out_all[:, HQ_swa:, :] = out_hsa.to(torch.bfloat16)
        return out_all.reshape(T, HQ_total * D)

    import types as _types
    be_old.forward_extend = _types.MethodType(_old_forward_extend, be_old)

    old_results = {}
    for n in test_lengths:
        ms, eng = bench_prefill(sm, cfg, be_old, mr_old, pool_old, r2t, n, device, dtype)
        old_results[n] = ms
        print(f'  N={n:>5}  eng={eng:>5}  latency={ms:.2f} ms')

    # Summary
    print()
    print('=' * 70)
    print('COMPARISON')
    print('=' * 70)
    print(f'{"N_real":>8} {"Old(ms)":>10} {"New(ms)":>10} {"Speedup":>10}')
    print('-' * 40)
    for n in test_lengths:
        old = old_results[n]; new = new_results[n]
        speedup = old / new
        print(f'{n:>8} {old:>10.2f} {new:>10.2f} {speedup:>9.2f}x')


if __name__ == '__main__':
    main()
