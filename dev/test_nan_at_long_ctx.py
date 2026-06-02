"""Check whether the long-context NaN is sglang-specific or a model property.

For each L in [8K, 16K, 32K, 64K, 128K]:
  1. Generate random tokens (same seed as test_long_ctx_cg_correctness.py).
  2. Run OFFICIAL HSAForCausalLM (eager attn, transformers-style) → check finiteness.
  3. Run SGLANG HSA (eager, --disable-cuda-graph) → check finiteness + KL vs official.

============================================================================
IMPORTANT — RUN EACH LENGTH IN A FRESH PROCESS.
============================================================================
Running multiple lengths in a single python process produces FALSE-POSITIVE
NaN reports for later lengths (likely GPU memory fragmentation interacting
with the triton/tilelang JIT cache after the first length's tensors are
freed).  Always invoke as:

    for L in 8192 16384 32768 65536 131072; do
        LENGTHS=$L python dev/test_nan_at_long_ctx.py
    done

Per-process verified results (HSA-345M, R35):
  L=8K   KL=4.5e-5  (sglang 100% finite)
  L=16K  KL=3.2e-5  (sglang 100% finite)
  L=32K  KL=2.4e-5  (sglang 100% finite)
  L=64K  KL=3.6e-5  (sglang 100% finite)
  L=128K KL=3.6e-5  (sglang 100% finite)
  L=256K official OOMs (eager attn materialises N² scores) — cannot compare
  L=512K official OOMs

bench (sglang.bench_one_batch) runs each length in its own python process,
so all the perf numbers in dev/hsa_optimization_log.md are valid.
============================================================================
"""
import sys, os, json, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "align"))

import bootstrap  # noqa
bootstrap.init_sglang_dist()

import torch
from safetensors.torch import load_file

# Import sglang-side
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

# Import official-side via bootstrap mocks (kicks in transformers-side HSAForCausalLM)
from compare import _build_official, official_prefill_logits, insert_special_tokens

SGConfig = bootstrap.SGConfig
SGModel = bootstrap.SGModel
compute_position = bootstrap.compute_position
Req = bootstrap.Req


def build_sglang_model(cfg_dict, device, dtype):
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


def setup_sglang_prefill(sg_model, sg_cfg, real_prompt, device, dtype):
    PS = sg_cfg.chunk_size
    lmk_id = int(sg_cfg.vocab_size)
    mc = max(2 * (len(real_prompt) * 2 + 16), 512) + 1024
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

    _hq, _hk = int(sg_cfg.num_attention_heads), int(sg_cfg.num_key_value_heads)
    if _hq > _hk:
        from sglang.srt.mem_cache.landmark_pool import LandmarkLmkKPool, ReqToChunkPool
        _max_chunks = (mc + PS - 1) // PS
        be.lmk_k_pool = LandmarkLmkKPool(
            num_chunk_slots=_max_chunks * 2,
            num_layers=int(sg_cfg.num_hidden_layers), h_q=_hq,
            head_dim=int(sg_cfg.head_dim), dtype=dtype, device=device,
        )
        be.req_to_chunk_pool = ReqToChunkPool(num_reqs=1, max_chunks_per_req=_max_chunks, device=device)
    be.init_cuda_graph_state(max_bs=1, max_num_tokens=1)

    fi = Req._hsa_insert_lmk_prompt(real_prompt, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    r2t[0, :pl] = tl[:pl]
    ep = torch.tensor([0], device=device, dtype=torch.int32)
    es = torch.tensor([pl], device=device, dtype=torch.int32)
    pos, esl = compute_position("hsa", ep, es, pl, page_size=PS, enable_landmark_positions=True)
    fb_p = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.tensor(fi, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([pl], dtype=torch.int32, device=device),
        out_cache_loc=tl[:pl],
        positions=pos, extend_prefix_lens=ep, extend_seq_lens=es,
        extend_start_loc=esl, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[pl],
        seq_lens_sum=int(pl), seq_lens_cpu=torch.tensor([pl], dtype=torch.int32),
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool,
    )
    fb_p.attn_backend = be
    be.init_forward_metadata(fb_p)
    with torch.no_grad():
        h = sg_model.model(fb_p.input_ids, fb_p.positions, fb_p)
    if isinstance(h, tuple):
        h = h[0]
    return be, mr, pool, r2t, pl, h


def test_one(L: int, cfg_dict, device, dtype, results):
    print(f"\n{'=' * 60}\n  L = {L}\n{'=' * 60}")

    # Same input as test_long_ctx_cg_correctness.py
    g = torch.Generator().manual_seed(42 + L)
    VS = int(cfg_dict["vocab_size"])
    real_prompt = torch.randint(5, VS - 5, (L,), generator=g).tolist()
    PS = int(cfg_dict.get("chunk_size", 64))

    # ---- 1. OFFICIAL ----
    print("  [1/2] Building official model...")
    try:
        oc, om = _build_official(cfg_dict, device, dtype)
        state = load_file(str(HERE / "align" / "weights_345m" / "model.safetensors"))
        om.load_state_dict(state, strict=False)
        om.eval()
    except Exception as e:
        print(f"    [SKIP] official build/load failed: {e}")
        results.append({"L": L, "official": "BUILD_FAIL", "sglang": None})
        return

    print(f"  [1/2] Running official prefill at L={L}...")
    try:
        with torch.no_grad():
            off_logits = official_prefill_logits(om, real_prompt, PS, VS, device)
    except torch.OutOfMemoryError:
        print(f"    [OOM] official OOM at L={L}")
        results.append({"L": L, "official": "OOM", "sglang": None})
        del om
        torch.cuda.empty_cache()
        return
    except Exception as e:
        print(f"    [ERROR] official forward failed: {e}")
        results.append({"L": L, "official": f"ERR: {e}", "sglang": None})
        del om
        torch.cuda.empty_cache()
        return

    off_finite_per_pos = torch.isfinite(off_logits).all(dim=-1)[0]  # [eng_len]
    off_finite_count = int(off_finite_per_pos.sum())
    off_total = int(off_finite_per_pos.numel())
    off_last_pos = off_logits[0, -1, :]
    off_last_finite = bool(torch.isfinite(off_last_pos).all())
    print(f"    official finite positions: {off_finite_count}/{off_total}  "
          f"last_pos_finite: {off_last_finite}")
    if off_last_finite:
        off_argmax = int(off_last_pos.argmax())
        off_max = float(off_last_pos.abs().max())
        print(f"    official last argmax={off_argmax}  max|logit|={off_max:.3f}")
    else:
        off_argmax = None
        off_max = float("nan")

    del om
    torch.cuda.empty_cache()

    # ---- 2. SGLANG ----
    print("  [2/2] Building sglang model + prefill...")
    sg_cfg, sg_model = build_sglang_model(cfg_dict, device, dtype)
    sg_model.load_state_dict(state, strict=False)
    sg_model.eval()
    for m in sg_model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    try:
        be, mr, pool, r2t, pl, h_prefill = setup_sglang_prefill(
            sg_model, sg_cfg, real_prompt, device, dtype
        )
    except torch.OutOfMemoryError:
        print(f"    [OOM] sglang prefill OOM at L={L}")
        results.append({
            "L": L, "official_finite_count": off_finite_count, "official_total": off_total,
            "official_last_finite": off_last_finite, "sglang": "OOM",
        })
        del sg_model
        torch.cuda.empty_cache()
        return

    # Check finiteness on HIDDEN states (memory-cheap), but compute logits
    # only for the last position (full materialization is 13GB+ at 128K).
    h_finite_per_pos = torch.isfinite(h_prefill).all(dim=-1)
    sg_pf_finite_count = int(h_finite_per_pos.sum())
    sg_pf_total = int(h_finite_per_pos.numel())
    sg_pf_last = (sg_model.model.norm(h_prefill[-1:]) @ sg_model.lm_head.weight[:VS].t()).float().squeeze(0)
    sg_pf_last_finite = bool(torch.isfinite(sg_pf_last).all())
    print(f"    sglang prefill finite positions: {sg_pf_finite_count}/{sg_pf_total}  "
          f"last_pos_finite: {sg_pf_last_finite}")
    if sg_pf_last_finite:
        sg_argmax = int(sg_pf_last.argmax())
        sg_max = float(sg_pf_last.abs().max())
        print(f"    sglang last argmax={sg_argmax}  max|logit|={sg_max:.3f}")

    # KL between off_last_pos and sg_pf_last (if both finite)
    if off_last_finite and sg_pf_last_finite:
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(sg_pf_last.cpu().float(), dim=-1),
            torch.log_softmax(off_last_pos[:VS].cpu().float(), dim=-1),
            reduction="batchmean", log_target=True,
        ).item()
        print(f"    KL(sglang || official) on last logit: {kl:.4e}")
    else:
        kl = None

    results.append({
        "L": L,
        "official_finite_count": off_finite_count, "official_total": off_total,
        "official_last_finite": off_last_finite, "off_argmax": off_argmax,
        "sglang_pf_finite_count": sg_pf_finite_count, "sglang_pf_total": sg_pf_total,
        "sglang_pf_last_finite": sg_pf_last_finite,
        "kl_pf_last": kl,
    })

    del sg_model
    torch.cuda.empty_cache()


def main():
    device = "cuda"
    dtype = torch.bfloat16

    cfg_path = HERE / "align" / "config_345m.json"
    cfg_dict = json.loads(cfg_path.read_text())

    lengths_env = os.environ.get("LENGTHS", "8192,16384,32768,65536")
    lengths = [int(x) for x in lengths_env.split(",") if x.strip()]

    results = []
    for L in lengths:
        test_one(L, cfg_dict, device, dtype, results)

    print(f"\n{'=' * 80}\n  SUMMARY\n{'=' * 80}")
    print(f"{'L':>8}  {'official last':<14}  {'sglang pf last':<15}  "
          f"{'official %finite':<16}  {'sglang %finite':<14}  KL(pf last)")
    for r in results:
        ot = r.get("official_total", 0) or 1
        st = r.get("sglang_pf_total", 0) or 1
        of_pct = 100 * (r.get("official_finite_count", 0) / ot)
        sf_pct = 100 * (r.get("sglang_pf_finite_count", 0) / st)
        print(f"{r['L']:>8}  "
              f"{str(r.get('official_last_finite', '?')):<14}  "
              f"{str(r.get('sglang_pf_last_finite', '?')):<15}  "
              f"{of_pct:>6.1f}%          "
              f"{sf_pct:>6.1f}%        "
              f"{r.get('kl_pf_last', '-')}")


if __name__ == "__main__":
    main()
