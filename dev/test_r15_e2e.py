"""End-to-end correctness test for R15 buffer path using the aligned 345M
weights (known to produce finite output).

Approach:
  1. Load HSA model via dev/align/compare.py infrastructure
  2. Run a normal decode step — save HSA per-layer attention output + logits
  3. Trigger R15 buffer allocation via init_cuda_graph_state(...)
  4. Run the SAME decode step again — buffer path is now active
  5. Compare both outputs

If the buffer path produces matching output, R15 is numerically correct
(within bf16 reduction-order noise).  This isolates the buffer plumbing
from cuda graph capture/replay (which is a separate concern — if the
buffer path's outputs match, the captured kernels' outputs do too,
because the kernels are identical).
"""
import sys, os, json, types, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "align"))

import bootstrap  # noqa
import torch
from safetensors.torch import load_file

from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

OfficialConfig = bootstrap.OfficialConfig
SGConfig = bootstrap.SGConfig
SGModel = bootstrap.SGModel
compute_position = bootstrap.compute_position
Req = bootstrap.Req


def build_sglang(cfg_dict, device, dtype):
    kw = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
    sw = kw.pop("sliding_window", 64)
    use_sw = kw.pop("use_sliding_window", True)
    kw["hsa_sliding_window"] = sw
    kw["use_sliding_window_merging"] = use_sw
    kw["sliding_window_merging_size"] = sw
    if "decoder_variant" not in kw:
        kw["decoder_variant"] = "qwen" if cfg_dict.get("model_type") == "qwen_lhsa" else "olmo"
    cfg = SGConfig(**kw)
    return cfg, SGModel(cfg).to(device=device, dtype=dtype).eval()


def run_decode_step(sg_model, sg_cfg, real_prompt_tokens, decode_token_id, device, dtype,
                    allocate_cg_buffers: bool):
    """Run one prefill + one decode step.  If `allocate_cg_buffers`, also call
    init_cuda_graph_state so init_forward_metadata takes the R15 buffer path."""
    PS = sg_cfg.chunk_size
    VS = int(sg_cfg.vocab_size)
    lmk_id = int(sg_cfg.vocab_size)

    real = list(real_prompt_tokens)
    decode_tokens = [decode_token_id]
    fi_full_len_est = (len(real) + len(decode_tokens)) + (
        (len(real) + len(decode_tokens)) // (PS - 1) + 2
    )
    mc = max(fi_full_len_est * 2, 512) + 1024

    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(
        size=mc + 256, page_size=PS, dtype=dtype,
        head_num=int(sg_cfg.num_key_value_heads), head_dim=int(sg_cfg.head_dim),
        layer_num=int(sg_cfg.num_hidden_layers), device=device,
        enable_memory_saver=False, enable_alt_stream=False,
    )
    mr = types.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=sg_model,
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, context_len=mc,
            num_attention_heads=int(sg_cfg.num_attention_heads),
            get_num_kv_heads=lambda tp: int(sg_cfg.num_key_value_heads) // tp,
        ),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend="hsa", speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id,
        ),
    )
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool
    mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)

    # Hookup per-q-head pools when needed (matches compare.py).
    _hq = int(getattr(sg_cfg, "num_attention_heads"))
    _hk = int(getattr(sg_cfg, "num_key_value_heads"))
    if _hq > _hk:
        from sglang.srt.mem_cache.landmark_pool import LandmarkLmkKPool, ReqToChunkPool
        _max_chunks = (mc + PS - 1) // PS
        be.lmk_k_pool = LandmarkLmkKPool(
            num_chunk_slots=_max_chunks * 2,
            num_layers=int(sg_cfg.num_hidden_layers), h_q=_hq, head_dim=int(sg_cfg.head_dim),
            dtype=dtype, device=device,
        )
        be.req_to_chunk_pool = ReqToChunkPool(num_reqs=1, max_chunks_per_req=_max_chunks, device=device)

    # CRITICAL: maybe trigger R15 buffer path
    if allocate_cg_buffers:
        be.init_cuda_graph_state(max_bs=4, max_num_tokens=mc)

    # ---- Prefill ----
    fi = Req._hsa_insert_lmk_prompt(real, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    r2t[0, :pl] = tl[:pl]
    ep = torch.tensor([0], device=device, dtype=torch.int32)
    es = torch.tensor([pl], device=device, dtype=torch.int32)
    pos, esl = compute_position("hsa", ep, es, pl, page_size=PS, enable_landmark_positions=True)
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.tensor(fi, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([pl], device=device, dtype=torch.int32),
        out_cache_loc=tl[:pl].to(torch.int64), seq_lens_sum=pl,
        seq_lens_cpu=torch.tensor([pl], device="cpu", dtype=torch.int32),
        positions=pos, extend_prefix_lens=ep, extend_seq_lens=es,
        extend_start_loc=esl, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[pl],
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool, attn_backend=be,
    )
    be.init_forward_metadata(fb)
    with torch.no_grad():
        _ = sg_model.model(fb.input_ids, fb.positions, fb)

    # ---- One decode step ----
    dpos = bootstrap.compute_decode_positions_landmark(
        torch.tensor([pl], device=device, dtype=torch.int32),
        page_size=PS,
    )
    r2t[0, pl] = pl  # next token slot
    fb_d = ForwardBatch(
        forward_mode=ForwardMode.DECODE, batch_size=1,
        input_ids=torch.tensor([decode_token_id], device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([pl + 1], device=device, dtype=torch.int32),
        out_cache_loc=torch.tensor([pl], device=device, dtype=torch.int64),
        seq_lens_sum=pl + 1,
        seq_lens_cpu=torch.tensor([pl + 1], device="cpu", dtype=torch.int32),
        positions=dpos,
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool, attn_backend=be,
    )
    be.init_forward_metadata(fb_d)
    with torch.no_grad():
        h = sg_model.model(fb_d.input_ids, fb_d.positions, fb_d)
    if isinstance(h, tuple):
        h = h[0]
    h_normed = sg_model.model.norm(h)
    logits = (h_normed @ sg_model.lm_head.weight[:VS, :].t()).float()
    return logits.detach().cpu(), be


def main():
    bootstrap.init_sglang_dist()

    device = "cuda"
    dtype = torch.bfloat16

    cfg_path = HERE / "align" / "config_345m.json"
    wts_path = HERE / "align" / "weights_345m" / "model.safetensors"
    cfg_dict = json.loads(cfg_path.read_text())
    sg_cfg, sg_model = build_sglang(cfg_dict, device, dtype)
    state = load_file(str(wts_path))
    sg_model.load_state_dict(state, strict=False)
    sg_model.eval()

    real_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    decode_token = 100

    print("=== Run 1: regular path (no R15 buffers) ===")
    logits_off, _ = run_decode_step(
        sg_model, sg_cfg, real_prompt, decode_token, device, dtype,
        allocate_cg_buffers=False,
    )
    print(f"  logits.shape={tuple(logits_off.shape)}  finite={torch.isfinite(logits_off).all().item()}")
    print(f"  argmax={logits_off.argmax(-1).tolist()}  top5={logits_off.topk(5).indices.tolist()}")
    print(f"  max={logits_off.max().item():.4f}  min={logits_off.min().item():.4f}")

    print("\n=== Run 2: R15 buffer path (cuda graph buffers allocated) ===")
    logits_on, _ = run_decode_step(
        sg_model, sg_cfg, real_prompt, decode_token, device, dtype,
        allocate_cg_buffers=True,
    )
    print(f"  logits.shape={tuple(logits_on.shape)}  finite={torch.isfinite(logits_on).all().item()}")
    print(f"  argmax={logits_on.argmax(-1).tolist()}  top5={logits_on.topk(5).indices.tolist()}")
    print(f"  max={logits_on.max().item():.4f}  min={logits_on.min().item():.4f}")

    print("\n=== Comparison ===")
    if logits_off.shape != logits_on.shape:
        print(f"SHAPE MISMATCH: off={logits_off.shape} on={logits_on.shape}")
        return
    max_abs = (logits_off - logits_on).abs().max().item()
    max_rel = max_abs / (logits_off.abs().max().item() + 1e-9)
    print(f"  max_abs_diff = {max_abs:.6e}")
    print(f"  max_rel_diff = {max_rel:.6e}")
    print(f"  argmax_match = {logits_off.argmax(-1).equal(logits_on.argmax(-1))}")
    print(f"  top5_match = {torch.equal(logits_off.topk(5).indices, logits_on.topk(5).indices)}")

    # bf16 tolerance: 2^-7 ≈ 7.8e-3 per ULP at scale 1.
    if max_abs < 1e-2:
        print("\n*** R15 BUFFER PATH NUMERICALLY MATCHES REFERENCE ***")
    else:
        print(f"\nWARNING: max_abs_diff={max_abs:.4e} > 1e-2 threshold")


if __name__ == "__main__":
    main()
