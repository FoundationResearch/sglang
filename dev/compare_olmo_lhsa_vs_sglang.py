"""
Compare official OLMo3-LHSA (modeling_olmo_lhsa.py / lhsa_layer.py) vs sglang HSA
using identical random weights on the same input.

Tests both prefill and decode alignment.

Usage:
  CUDA_VISIBLE_DEVICES=0 python dev/compare_olmo_lhsa_vs_sglang.py
"""
import os, sys, types, logging, tempfile, time

# ============================================================
# 1. Veomni mock
# ============================================================
for sub in ['veomni','veomni.distributed','veomni.distributed.parallel_state',
            'veomni.distributed.sequence_parallel','veomni.utils','veomni.utils.logging',
            'veomni.utils.import_utils','veomni.models','veomni.models.module_utils',
            'veomni.models.loader']:
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
sys.modules['veomni.utils.import_utils'].is_liger_kernel_available = lambda: False  # disable liger to keep OLMo3 norms
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

# Mock utils.flex_attn with a PyTorch reference implementation
def _flex_attn_ref(q, k, v, window_size, chunk_size, training=False, cu_seq_lens=None):
    """PyTorch reference for flex_attn: chunk-aligned SWA with LSE return.

    Args:
        q, k, v: [B, H, L, D]
        window_size: sliding window size
        chunk_size: chunk alignment size
    Returns:
        output: [B, H, L, D]
        lse: [B, H, L] (logsumexp per position)
    """
    B, H, L, D = q.shape
    scale = D ** -0.5

    # Build causal + chunk-aligned SWA mask
    qi = torch.arange(L, device=q.device)
    ki = torch.arange(L, device=q.device)

    # Causal: ki <= qi
    causal = ki[None, :] <= qi[:, None]  # [L, L]

    # SWA window: ki >= qi - window_size + 1, aligned to chunk boundary
    # chunk_start = max(0, ((qi - window_size + 1) // chunk_size) * chunk_size)
    raw_start = qi - window_size + 1
    chunk_start = torch.clamp((raw_start // chunk_size) * chunk_size, min=0)
    swa = ki[None, :] >= chunk_start[:, None]  # [L, L]

    # Exclude LMK positions (chunk_size - 1, 2*chunk_size - 1, ...)
    not_lmk = ((ki + 1) % chunk_size) != 0  # [L]

    mask = causal & swa & not_lmk[None, :]  # [L, L]

    # Compute attention
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale  # [B, H, L, L]
    scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))

    lse = torch.logsumexp(scores, dim=-1)  # [B, H, L]
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v.float())

    return output.to(q.dtype), lse

# Create mock module for utils.flex_attn
_flex_mod = types.ModuleType('utils.flex_attn')
_flex_mod.flex_attn = _flex_attn_ref
sys.modules['utils.flex_attn'] = _flex_mod

# Also need to mock utils.seqlen_pos_transform_utils
_spt_mod = types.ModuleType('veomni.utils.seqlen_pos_transform_utils')
def _prepare_fa_kwargs_from_position_ids(position_ids):
    B, L = position_ids.shape
    cu = torch.arange(0, (B+1)*L, L, device=position_ids.device, dtype=torch.int32)
    return (cu, cu), (L, L)
_spt_mod.prepare_fa_kwargs_from_position_ids = _prepare_fa_kwargs_from_position_ids
sys.modules['veomni.utils.seqlen_pos_transform_utils'] = _spt_mod

# Add InfiniteLongLM to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ILL_DIR = os.path.join(_SCRIPT_DIR, 'InfiniteLongLM')
sys.path.insert(0, _ILL_DIR)

# Patch: DRT's modeling_olmo_lhsa imports `from .configuration_hsa import HSAConfig`
# but the actual file is in FlashHSA/. Pre-import it under the expected module name.
from models.FlashHSA.configuration_hsa import HSAConfig as _HSAConfig
_drt_config_mod = types.ModuleType('models.DRT.configuration_hsa')
_drt_config_mod.HSAConfig = _HSAConfig
sys.modules['models.DRT.configuration_hsa'] = _drt_config_mod

import torch.distributed as dist

# ============================================================
# 2. Import official model
# ============================================================
from models.DRT.modeling_olmo_lhsa import HSAForCausalLM as OfficialModel, HSAModel
from models.DRT.lhsa_layer import LandmarkHSA
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks
OfficialConfig = _HSAConfig

# ============================================================
# 3. Import sglang
# ============================================================
from sglang.srt.layers import dp_attention as dpa
dpa.get_attention_tp_size = lambda: 1
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.configs.flash_hsa import FlashHSAConfig
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch, ForwardMode, compute_position, compute_decode_positions_landmark,
)
from sglang.srt.models.flash_hsa import HSAForCausalLM as SGModel
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def init_sglang():
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix='olmo_cmp_', suffix='.t')
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
    """Transfer weights from official OLMo3-LHSA model to sglang model."""
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
            # OLMo3 has separate gate_proj + up_proj, sglang merges into gate_up_proj
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


def official_prefill(model, real_tokens, PS, VS, device):
    """Run official model prefill (full sequence, no KV cache)."""
    ids = insert_special_tokens(torch.tensor([real_tokens]), VS, PS)
    pos = create_position_ids_with_landmarks(len(real_tokens), PS, device)
    with torch.no_grad():
        out = model(input_ids=ids.to(device), position_ids=pos.to(device),
                    attention_mask=None, use_cache=False)
    return out.logits[:, :, :VS].float()


def sglang_prefill_and_decode(sg_model, cfg, real_base, decode_tokens, device, dtype):
    """sglang: prefill real_base, then decode each token. Returns logits per decode step."""
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

    # Config: 1 layer, OLMo3 base
    PS = 64; VS = 256
    # Official config
    oc = OfficialConfig(
        vocab_size=256, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4,
        head_dim=64, rms_norm_eps=1e-6, chunk_size=PS, hsa_topk=2,
        hsa_mode='sparse', full_attn_interleave=1, hsa_heads=4,
        hsa_qk_ratio=4, use_sliding_window=True, sliding_window=PS,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act='silu',
        enable_softmax1=True, enable_lmk_q_proj=False,
    )
    oc._attn_implementation = 'eager'
    oc.pad_token_id = None
    oc.num_swa_layers = 0

    om = OfficialModel(oc).to(device=device, dtype=dtype); om.eval()

    # sglang config
    sc = FlashHSAConfig(
        model_type='flash_hsa_innerx', architectures=['HSAForCausalLM'],
        vocab_size=256, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4,
        head_dim=64, rms_norm_eps=1e-6, chunk_size=PS, hsa_topk=2,
        hsa_mode='sparse', full_attn_interleave=1, hsa_heads=4,
        hsa_qk_ratio=4, use_sliding_window_merging=True,
        sliding_window_merging_size=PS, tie_word_embeddings=False,
        rope_theta=10000.0, hidden_act='silu', enable_lmk_q_proj=False,
        hsa_sliding_window=PS,
    )
    sm = SGModel(sc).to(device=device, dtype=dtype); sm.eval()

    loaded, total, skipped = transfer_weights(om, sm)
    print(f'Weights: {loaded}/{total} loaded, {len(skipped)} skipped')
    if skipped:
        for s in skipped[:10]:
            print(f'  SKIP: {s}')
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    print()
    print('=' * 70)
    print('OLMo3-LHSA: Official vs sglang Alignment')
    print('=' * 70)

    test_cases = [
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

        # Official: prefill full sequence
        try:
            off_logits = official_prefill(om, real_full, PS, VS, device)
        except Exception as e:
            print(f'  Official prefill FAILED: {e}')
            continue

        # sglang: prefill + decode
        sg_decode_logits = sglang_prefill_and_decode(
            sm, sc, real_base, decode_toks, device, dtype)

        # Compare each decode step
        all_pass = True
        for i, (tok, sg_l) in enumerate(zip(decode_toks, sg_decode_logits)):
            current_real = real_base + decode_toks[:i+1]
            fi_prev = Req._hsa_insert_lmk_prompt(current_real[:-1], page_size=PS, lmk_id=VS)
            eng_pos = len(fi_prev)
            if eng_pos < off_logits.shape[1]:
                off_l = off_logits[0, eng_pos]
                diff = (off_l - sg_l).abs()
                mx = diff.max().item()
                cos = torch.nn.functional.cosine_similarity(
                    off_l.unsqueeze(0), sg_l.unsqueeze(0)).item()
                am_off = off_l.argmax().item()
                am_sg = sg_l.argmax().item()
                match = am_off == am_sg
                status = 'PASS' if mx < 0.1 else ('CLOSE' if mx < 1.0 else 'FAIL')
                if status == 'FAIL': all_pass = False
                print(f'  Step {i} (tok={tok}): max_err={mx:.4f} cos={cos:.6f} '
                      f'argmax off={am_off} sg={am_sg} {"OK" if match else "MISS"} [{status}]')
            else:
                print(f'  Step {i}: SKIP (official out of range)')

        print(f'  Overall: {"ALL PASS" if all_pass else "HAS FAILURES"}')

    print(f'\n{"="*70}')
    print('ALIGNMENT TEST COMPLETE')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
