# Agent runbook: HSA logits-alignment workflow

This file is for **future Claude / agent sessions**. It is the canonical
playbook for keeping the SGLang HSA inference path numerically aligned with
the official OLMo-LHSA reference (`InfiniteLongLM/models/DRT/`). Read it
top-to-bottom before touching `dev/align/`. The user-facing quickstart lives in
`dev/align/README.md` — that one is shorter and assumes a human reader.

---

## 1. What this workflow exists to catch

We are testing a sparse-attention engine. SGLang's HSA backend has two
sources of divergence from the official reference:

1. **Top-k selection noise.** With random init, scores are nearly uniform. A
   sub-LSB difference in the score reduction order between the two
   implementations causes them to pick disjoint top-k pages, and the entire
   downstream attention computation differs. This is a **comparison artifact**,
   not a real bug, and it dominates the alignment signal at random init.
2. **Real backend numerics.** Different kernels (PyTorch eager reference vs.
   SGLang's Triton/tilelang HSA path), different reduction orders in fp16/bf16,
   different RMSNorm paths, etc. This is what we actually want to measure.

**Training removes (1).** A few hundred steps of LM training on synthetic data
gives the selector heads enough signal that top-k becomes robust to
sub-LSB perturbations. After that, residual KL reflects (2) only.

The empirical signature: after training, KL is **constant across context
length** (10 tokens vs 2048 tokens vs 2048+1024 decode all show KL ≈ 0.0023).
If KL grew with context, top-k noise would still be dominant. It doesn't, so
backend numerics are the floor — and that floor is well below any FAIL band.

## 2. Entry points

| File | Role |
|------|------|
| `dev/align/bootstrap.py`  | Side-effect import: veomni mocks, AutoConfig shim, eager-attn SWA patch, tilelang kwarg wrappers, sglang dist init. **Must be imported before any DRT/sglang import.** |
| `dev/align/config.json`   | 1-layer OLMo-LHSA arch. SHA-256 tracked in manifest. Edit deliberately. |
| `dev/align/train.py`      | Builds official model from `config.json`, trains on synthetic LM data, writes `weights/model.safetensors` + `manifest.json`. |
| `dev/align/compare.py`    | Loads weights into both impls, refuses to run on SHA mismatch, runs prefill (10..2048) + decode (5..1024) test cases, reports KL / argmax / max-err. |
| `dev/align/manifest.json` | Provenance: git commit, SHAs, training dynamics, env versions, train-at UTC. |
| `dev/align/weights/model.safetensors` | fp16 trained state_dict (~8 MB). Committed. |

`compare.py` returns non-zero on SHA mismatch — that's intentional. Don't
silence it without understanding why.

## 3. When to retrain

Retrain whenever any of these change:

- HSA selection / kernel logic — `python/sglang/srt/layers/attention/hsa/`,
  `python/sglang/srt/layers/attention/hsa_backend.py`,
  `python/sglang/srt/models/flash_hsa.py`,
  `dev/hsa-kernel-main/ops/`
- The official reference — `dev/InfiniteLongLM/models/DRT/`
- Positional encoding (RoPE → ALiBi → sinusoidal etc.)
- `dev/align/config.json` itself
- Training procedure (loss, optimizer, seed)
- Major dependency upgrade (torch, tilelang, transformers)

The comparison script's SHA check exists *specifically* to prevent silently
running stale weights against new code. If you see "config SHA mismatch" or
"weights SHA mismatch", **retrain**. Don't pass `--no-verify-sha` unless you
are intentionally validating a transient debug change you won't commit.

## 4. Standard retrain protocol

```bash
conda activate alexsg

# 1. Make your change(s) — config, kernel, model, etc.
$EDITOR ...

# 2. Tree must be clean before retraining; the manifest's git_commit field is
#    only meaningful from a clean tree. (`--allow-dirty` exists as an escape
#    hatch but flag dirty=true in the manifest, which compare.py displays.)
git status
git add ... && git commit -m "..."

# 3. Retrain. ~1-2 min on H100 with defaults (200 steps, batch 2, seq 384).
python dev/align/train.py

# 4. Compare. Should print [PASS] (KL < 0.01) on every line.
python dev/align/compare.py 2>&1 | tee /tmp/align.log

# 5. Commit weights + manifest as a SEPARATE commit from the code change. The
#    diff stays minimal and bisectable.
git add dev/align/manifest.json dev/align/weights/model.safetensors
git commit -m "align: retrain after <change description>

<one-paragraph summary of the comparison results: KL bands across cases>"
git push
```

## 5a. Sequential-churn stress (the benchmark proxy)

`compare.py` runs a sequential-churn block by default (disable with
`--no-stress`). It builds ONE shared `MHATokenToKVPool` /
`TokenToKVPoolAllocator` / `ReqToTokenPool`, then drives N=64 requests of
varied (prefill, decode) sizes through it back-to-back. After each
request: alloc → prefill → decode → free → next. The block reports four
invariants that must hold:

  * `pool_avail_delta == 0` per request: every alloc was matched by a free.
  * `req_avail_delta == 0` per request: same for the request-pool.
  * Cumulative CUDA memory growth bounded (workspace variation is fine;
    monotonic growth is a leak).
  * Per-request KL stays in PASS (with occasional CLOSE on noisy decodes
    is acceptable; persistent CLOSE/WARN means cross-request state is
    bleeding through).

The shared pool is sized at 8192 slots; peak observed in the default
workload is ~1600 slots, so the test exercises recycling, not capacity.
If you change the workload to push peak slots > 4096, increase
`max_ctx` in the call from `main()`.

Why this is the right benchmark proxy: the engine-level benchmark in
`bench_hsa_overhead.py` / `bench_hsa_vs_fullattn.py` does many requests
through a long-lived scheduler; if the HSA backend caches per-request
metadata across `init_forward_metadata` calls without resetting it, the
benchmark will silently produce wrong logits on later requests. The
churn stress catches that without spinning up the full Scheduler.

## 5b. The XLONG case (don't skip it)

`compare.py` includes a **XLONG 2048 + 1024 decode** test case. This is the
only case that exercises:

- Decode-time KV writes (16 new pages written *during* decode)
- LMK auto-insertion at page boundaries during decode
  (mirrors `Req.hsa_decode_postprocess_sampled_token`)
- Top-k selection over a KV cache that grew *during* the run
- `select_topk_pages_decode` over 48 candidate pages, of which 16 are
  freshly-written

If subsequent decode steps fail to attend to newly-decoded tokens, KL will
drift up over the 1024 decode steps. We instead see **flat KL ≈ 0.0023 with
max=0.0036** across all 1023 real-decode steps, which is the same band as
the short tests. That's the signature of a correct decode-write path.

Per-page-boundary LMK steps are reported separately as `decode_lmk(n=16)` —
divergence on those would point to LMK-handling bugs in either implementation.

If you are about to claim the HSA backend is correct, **make sure the XLONG
case passed in your latest run**. It's the only one that actually stresses
cross-decode KV.

## 6. Pitfalls discovered during construction (don't re-debug these)

**`utils` package shim.** `dev/InfiniteLongLM/utils/` has no `__init__.py`,
so it's a namespace package. We mock `utils.flex_attn` for the reference
attention, but the shim must set `sys.modules['utils'].__path__ =
[<real-utils-dir>]` so that `from utils.landmark_utils import …` still finds
the real file. Without `__path__`, you get `'utils' is not a package`.

**`liger_kernel` is required, not optional.** Despite the official model
guarding it via `is_liger_kernel_available()`, `dev/InfiniteLongLM/ops/
hsa_fwd_bwd_group.py` has a top-level `from liger_kernel.transformers.rope
import liger_rotary_pos_emb`. Pip-install the real package
(`liger_kernel==0.7.0`); mocks break transformers' `find_spec("liger_kernel")`
because the fake module has no `__spec__`.

**Train mode is broken.** `OfficialModel.forward()` in train mode calls
`create_position_ids_with_landmarks(position_ids, seq_len, chunk_size,
device)` — a 4-arg signature that doesn't match the 3-arg version in
`utils/landmark_utils.py` of this branch. We work around this in `train.py`
by calling `model.eval()` then `p.requires_grad_(True)` so gradients flow
without triggering the broken train-mode branch. We pre-insert LMK in
`synthetic_batch` to compensate. Dropout is off, which is fine.

**HSA head divisibility.** `hsa_heads % hsa_qk_ratio == 0` and
`num_attention_heads % hsa_heads == 0` and `num_key_value_heads % (h_q /
hsa_heads) == 0`. Working values for `config.json`: `(num_attention_heads=8,
num_key_value_heads=2, hsa_heads=4, hsa_qk_ratio=4)` → `hsa_denom=2`,
`h_hsa_kv=1`. Don't blindly shrink — check the asserts in
`dev/InfiniteLongLM/models/DRT/lhsa_layer.py:155-167`.

**`sgl_kernel` bf16 RMSNorm CUDA bug.** Zeros every other bf16 element. We
override every `RMSNorm._forward_method = forward_native` after model
construction. See `bootstrap.force_native_rmsnorm`.

**Cos/sin RoPE cache.** `cos_sin_cache` must be cast to fp32 before sglang
forward, otherwise bf16 rounding offsets RoPE positions slightly.
`compare.py` does this; if you build a new entry point, replicate it.

**Dropout / training flag.** Every comparison run uses `model.eval()` on
both impls. The training-mode forward path inserts LMK and shifts labels
internally — we already pre-insert, so train-mode would double-insert.

## 7. Rough KL bands and what they mean

| KL          | Band      | Action |
|-------------|-----------|--------|
| < 0.001     | IDENTICAL | Probably suspicious — verify weights actually loaded into both models. |
| < 0.01      | PASS      | Expected for a correct HSA backend on bf16 forward. |
| 0.01 - 0.1  | CLOSE     | Investigate. Likely a kernel reduction-order regression. |
| 0.1 - 1.0   | WARN      | Real divergence. Check recent diffs to `hsa_backend.py` / `selector.py`. |
| ≥ 1.0       | FAIL      | Serious. Bisect against last green manifest. |

## 8. When *adding* test cases

If the user asks you to verify behaviour at a new context length / decode
length / config, add a tuple to the `test_cases` list in `compare.py`:

```python
(prefill_len, decode_tokens_list, 'description string')
```

Use `decode_toks_huge` (1024 random tokens) when you specifically want to
stress decode-write KV. Use shorter lists otherwise — long decode is ~30s
per case on H100.

## 8b. Known gaps

**Batch_size > 1 in a single `ForwardBatch` is not covered here.** The
sequential-churn block uses batch_size=1 per request. The existing
pytest e2e tests
`test_innerx_radix_kv_reuse_end_to_end_gpu.py`,
`test_innerx_radix_branching_prefix_reuse_end_to_end_gpu.py`,
`test_hsa_swa_dual_pool_eviction_gpu.py`,
and `test_innerx_model_runner_like_end_to_end_decode_gpu.py`
exercise B>1 and currently fail with NaN (43/47 pytest pass). Don't
add a B>1 stress to `compare.py` until those tests are green —
otherwise this harness will start failing for the same reason and
muddy the alignment signal. When those tests are fixed, add a
`stress_batched_decode` block that prefills 2-4 requests separately
then runs a single batched decode step with `batch_size=B`.

## 9. Architectural changes that need wider thought (don't auto-resolve)

If the user wants to:

- **Swap positional encoding** (RoPE → ALiBi / sinusoidal): retrain is
  straightforward, but `bootstrap._eager_with_sliding_window` and
  `flash_hsa.py`'s rotary embedding path both need to be consistent with the
  new encoding. Confirm the user has updated both reference and SGLang side.
- **Multi-layer model**: doable; bump `num_hidden_layers` in `config.json`,
  retrain. Per-layer alignment is automatic since the loop iterates layers.
  But weights size grows linearly — if the safetensors gets large (>30 MB),
  switch to a manifest-only commit and `.gitignore` the weights file.
- **Different HSA variant** (innerx vs flash_hsa): adjust `model_type` in
  `_build_sglang` and confirm the official reference matches.

In all of these, **ask the user before making the architectural change** —
don't assume their intent.

## 10. Files this workflow does NOT touch

- `python/sglang/test/attention/hsa/` — pytest unit tests. They have their
  own pass/fail; don't conflate them with align KL numbers.
- `dev/compare_olmo_lhsa_vs_sglang.py` — predecessor. Kept temporarily.
  When `dev/align/compare.py` has been stable through a config change cycle,
  archive `compare_olmo_lhsa_vs_sglang.py` to `dev/archive/`.
- `dev/InfiniteLongLM/train.sh` — official training entry point with full
  veomni setup. Don't go there for alignment work; the synthetic in-script
  training is intentionally tiny.

## 11. Last verified

- 2026-04-28, on H100 PCIe 80GB, alexsg env (torch 2.9.1+cu128,
  tilelang 0.1.9, transformers 4.57.1, triton 3.5.1, liger_kernel 0.7.0).
- Baseline manifest: `dev/align/manifest.json`.
- Single-request: all 9 cases PASS, including XLONG 2048+1024 decode
  (1023 real-decode steps, max KL = 0.003634, mean KL = 0.002322).
- Sequential churn (64 requests, varied sizes 64..1536 prefill + 16..200
  decode, programmatic workload): pool_leak=0, req_leak=0,
  peak_used=1690/8192, peak_cuda_mem=253.9MB, cum_mem_growth=+114MB
  (mostly tilelang JIT cache for the wider shape coverage — bounded
  by the number of unique shapes, not monotonic per-request). KL prefill
  max=0.002765 [PASS] (n=64, mean=0.002316, p95=0.002444). KL last-decode
  max=0.011519 [CLOSE on 1 req of 64, rest PASS] (mean=0.003294,
  p95=0.005313).
