from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Optional, Set

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hsa.kernels import (
    hsa_decode_paged_fwd,
    hsa_extend_paged_fwd,
)
from sglang.srt.layers.attention.hsa.kernels.chunk_weight import (
    fused_chunk_weight_per_qhead_decode,
    fused_chunk_weight_h_kv_decode,
)
from sglang.srt.layers.attention.hsa.kernels.cuda_graph_buffers import (
    update_hsa_cg_buffers,
)
from sglang.srt.layers.attention.hsa.metadata import HSAMetadata
from sglang.srt.mem_cache.landmark_pool import LandmarkLmkKPool, ReqToChunkPool
from sglang.srt.layers.attention.hsa.selector import (
    _online_topk_group,
    _online_topk_head_maxpool,
    build_active_page_candidates,
    select_topk_pages_decode,
    select_topk_pages_decode_fused,
)
from sglang.srt.layers.attention.hsa.utils import transform_page_table_1_to_real
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.distributed import tensor_model_parallel_all_gather
from sglang.srt.layers.dp_attention import get_attention_tp_size

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


logger = logging.getLogger(__name__)

_COMPILED_FLEX_ATTN = None
_FLEX_FALLBACK_WARNED = False


def _get_compiled_flex_attention():
    """Return a cached torch.compile'd flex_attention callable (best-effort)."""
    global _COMPILED_FLEX_ATTN
    if _COMPILED_FLEX_ATTN is not None:
        return _COMPILED_FLEX_ATTN
    from torch.nn.attention.flex_attention import flex_attention

    # Compiling the base entrypoint (as recommended by PyTorch) avoids the unfused slow path.
    _COMPILED_FLEX_ATTN = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTN


class HSAAttnBackend(AttentionBackend):
    """
    HSA (Hierarchical Sparse Attention) backend entrypoint.

    Phase 1 (this file): implement the "core scheduling point" so we can select the backend
    via CLI and have it run end-to-end. We build HSA-style metadata (page_table_1/real_page_table)
    and delegate dense attention compute + CUDA-graph plumbing to `TritonAttnBackend`.

    Phase 2+: replace delegation with real HSA selection + paged HSA kernels.

    Docs:
    - dev/hsa_dev_roadmap.md
    - dev/hsa_sglang_impl_todo.md
    """

    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__()
        self.model_runner = model_runner
        self.device = model_runner.device
        self.page_size = model_runner.page_size

        # HSA config resolution (override-only server args; model config is source-of-truth).
        server_args = getattr(model_runner, "server_args", None)
        cfg = getattr(getattr(model_runner, "model", None), "config", None)

        # R75: align fallback with FlashHSAConfig default (16, see
        # configs/flash_hsa.py:66) so that if a future config class change
        # introduces a different default, the backend won't silently use a
        # stale value. R72 was caused by exactly this kind of drift —
        # backend defaulted to False while config class defaulted to True,
        # masking the maxpool kernel for months.
        default_topk = getattr(cfg, "hsa_topk", 16)
        override_topk = getattr(server_args, "hsa_topk", None)
        self.hsa_topk = int(override_topk) if override_topk is not None else int(default_topk)

        default_sel = getattr(cfg, "hsa_selection_strategy", "head")
        override_sel = getattr(server_args, "hsa_selection_strategy", None)
        self.hsa_selection_strategy = (
            str(override_sel) if override_sel is not None else str(default_sel)
        )

        # Prefill topk strategy:
        #   True  -> softmax-then-max-pool (training kernel via online_softmax_topk_head;
        #            computes hsa_lse + logaddexp with swa_lse + softmax-normalized topk).
        #   False -> max-pooling-only (online_topk_head): skip hsa_lse, pass swa_lse
        #            directly to the selector as a per-query offset.  ~2x faster prefill
        #            topk on upstream measurements with negligible quality impact.
        # CLI override (--hsa-headwise-topk-softmax / --no-hsa-headwise-topk-softmax) wins;
        # otherwise read from model config (default True for back-compat).
        # R62: default to False (maxpool fast path) when cfg doesn't specify.
        # Alignment is bit-equivalent for trained-with-softmax models per the
        # InfiniteLongLM upstream notes, and the kernel is 12-20% faster at
        # long-context prefill (dev/bench: 32K -12%, 64K -20% on hsa345m_real).
        # Checkpoints that need the exact softmax path can set
        # `headwise_topk_softmax=True` in config.json.
        default_headwise = bool(getattr(cfg, "headwise_topk_softmax", False))
        override_headwise = getattr(server_args, "hsa_headwise_topk_softmax", None)
        self.headwise_topk_softmax = (
            bool(override_headwise) if override_headwise is not None else default_headwise
        )

        self.hsa_layers = getattr(server_args, "hsa_layers", None)
        self._hsa_layer_ids: Optional[Set[int]] = self._resolve_hsa_layer_ids()

        # LHSA merged softmax: enable_softmax1 adds a zero logit to the
        # denominator. R75: align fallback with FlashHSAConfig default
        # (True, see configs/flash_hsa.py:64) so the training-time choice
        # is preserved if a checkpoint omits the key. Backend used to
        # default to False, which differs from training and would silently
        # produce slightly different attention weights for any config that
        # didn't load through FlashHSAConfig.
        self.enable_softmax1 = bool(getattr(cfg, "enable_softmax1", True))

        # InnerX ultra only: we always run split-head stitch for HSA layers.
        # SWA window and LMK visibility are controlled by per-layer kwargs.
        self.hsa_window_size = None
        self.hsa_enable_swa_merging = False

        # R15: cuda-graph persistent buffers (allocated in init_cuda_graph_state).
        self._cg_page_table_1 = None
        self._cg_cache_seqlens_i32 = None
        self._cg_max_seqlen_k = 0

        # Delegate dense attention implementation for now.
        # NOTE: For encoder-decoder models, triton backend is not supported.
        if model_runner.model_config.is_encoder_decoder:
            raise ValueError(
                "HSAAttnBackend currently delegates to TritonAttnBackend, which does not support "
                "encoder-decoder (cross attention). Please use --attention-backend flashinfer for "
                "encoder-decoder models, or wait for HSA cross-attention support."
            )

        self._dense_backend = TritonAttnBackend(model_runner, **kwargs)
        self.forward_metadata: Optional[HSAMetadata] = None

        # R52: auto-init per-q-head lmk_k pools when the config asks for the
        # chunk_attn_pool path. Previously these stayed None in production
        # sglang and `_maybe_write_chunk_lmk_k` returned early — silently
        # disabling all of R47 (slot reuse), R48 (prior_b), and R51 (decode
        # chunk write). The pools are only set externally by dev/align tests,
        # which is why alignment tests PASSed while production diverged.
        self.lmk_k_pool = None
        self.req_to_chunk_pool = None
        # Resolve config: prefer model.config (works for both production
        # model_runner and the SimpleNamespace mock used by dev/align/compare.py).
        _hsa_cfg = getattr(getattr(model_runner, "model", None), "config", None)
        if _hsa_cfg is None:
            _mc = getattr(model_runner, "model_config", None)
            _hsa_cfg = getattr(_mc, "hf_text_config", _mc)
        # Env knob — used by dev/test_engine_alignment.py to reproduce the
        # pre-R52 silent disable state on demand. Set to "1" to skip auto-init
        # and force pools to stay None (matches pre-R52 production behaviour).
        _disable_auto_init = os.environ.get(
            "SGLANG_HSA_DISABLE_AUTO_POOL_INIT", "0"
        ) == "1"
        if (
            not _disable_auto_init
            and _hsa_cfg is not None
            and bool(getattr(_hsa_cfg, "enable_prior_query", False))
            and bool(getattr(_hsa_cfg, "enable_lmk_q_proj", False))
        ):
            cfg = _hsa_cfg
            try:
                num_layers = int(getattr(cfg, "num_hidden_layers", 1))
                h_q = int(getattr(cfg, "num_attention_heads"))
                head_dim = int(
                    getattr(cfg, "head_dim", getattr(cfg, "hidden_size") // h_q)
                )
                # Pool sizing — base on PER-REQUEST context (model_config.context_len
                # or max_req_input_len), NOT on the global KV-pool size. Using
                # max_total_num_tokens here is wrong: it's the *aggregate* token
                # capacity across all concurrent reqs AND all layers, so it
                # over-counts by req_pool_size × num_layers and easily yields
                # 10^10+ slots → multi-TB allocation. We just need per-req chunks.
                _mc = getattr(model_runner, "model_config", None)
                _ctx_len = int(getattr(_mc, "context_len", 0) or 0)
                if _ctx_len <= 0:
                    _ctx_len = int(getattr(_mc, "max_req_input_len", 0) or 0)
                if _ctx_len <= 0:
                    _server_args = getattr(model_runner, "server_args", None)
                    _ctx_len = int(
                        getattr(_server_args, "context_length", None) or 0
                    )
                if _ctx_len <= 0:
                    _ctx_len = 131072  # last-ditch fallback
                # Allow LMK insertion (~+3%) + small slack.
                max_chunks_per_req = max(
                    int(_ctx_len * 33 // 32 // int(self.page_size)) + 16, 16
                )
                req_pool = getattr(model_runner, "req_to_token_pool", None)
                req_pool_size = int(getattr(req_pool, "size", 1) or 1)
                # Each concurrent req can occupy up to max_chunks_per_req slots.
                # Cap total slots so the lmk_k_pool itself stays bounded: at
                # 16 heads × 64 D × 2B/elem × 16 layers ≈ 32 KiB per slot;
                # we want ≤ ~4 GiB → ~128K slots ceiling. Production rarely
                # has all reqs holding the full context simultaneously, so cap
                # at min(req_pool_size, 128) × max_chunks_per_req.
                effective_reqs = min(max(req_pool_size, 1), 128)
                num_chunk_slots = max(
                    max_chunks_per_req * effective_reqs + 16, 16
                )
                self.lmk_k_pool = LandmarkLmkKPool(
                    num_chunk_slots=num_chunk_slots,
                    num_layers=num_layers,
                    h_q=h_q,
                    head_dim=head_dim,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                self.req_to_chunk_pool = ReqToChunkPool(
                    num_reqs=req_pool_size,
                    max_chunks_per_req=max_chunks_per_req,
                    device=self.device,
                )
                if os.environ.get("SGLANG_HSA_ALIGN_DEBUG", "0") == "1":
                    print(
                        f"[HSA-align] init lmk_k_pool slots={num_chunk_slots}, "
                        f"layers={num_layers}, h_q={h_q}, head_dim={head_dim}, "
                        f"req_pool={req_pool_size}, "
                        f"max_chunks_per_req={max_chunks_per_req}",
                        flush=True,
                    )
            except Exception as e:
                logger.warning("Failed to initialize HSA lmk_k_pool: %s", e)
                self.lmk_k_pool = None
                self.req_to_chunk_pool = None
        # When per_qhead is active, the selector also computes per-q-head
        # scores (gathered at the chunks selected by max-over-G) and the SWA
        # branch exposes per-q-head LSE.  These are consumed by the
        # forward_decode / forward_extend fusion blocks to do the chunk_weight
        # softmax at h_q granularity — matching the official lhsa_layer.py
        # line 782 ``cat([scores, lse_sum.unsqueeze(-1)], dim=-1)``.  Empty
        # default means "fall back to h_kv fusion" (current behaviour).
        self._last_per_qhead_scores_decode = None  # [B, h_q, K]
        self._last_per_qhead_scores_extend = None  # [T, h_q, K]
        self._last_swa_lse_hq_decode = None        # [B, h_q]
        self._last_swa_lse_hq_extend = None        # [T, h_q]

    def _resolve_hsa_layer_ids(self) -> Optional[Set[int]]:
        """Resolve per-layer HSA enable set.

        Priority:
        1) CLI: --hsa-layers (string/list)
        2) Model-provided default (FlashHSA): model.get_flashhsa_hsa_layer_ids()
        3) None => apply HSA selection to all layers (current default behavior)
        """

        layers = self.hsa_layers
        if layers is not None:
            try:
                if isinstance(layers, (list, tuple, set)):
                    return set(int(x) for x in layers)
                if isinstance(layers, str):
                    parts = [p for p in re.split(r"[,\s]+", layers.strip()) if p]
                    return set(int(p) for p in parts)
            except Exception:
                return None

        get_default = getattr(getattr(self.model_runner, "model", None), "get_flashhsa_hsa_layer_ids", None)
        if callable(get_default):
            try:
                default_layers = get_default()
                if default_layers is not None:
                    return set(int(x) for x in default_layers)
            except Exception:
                return None

        return None

    def _is_hsa_layer(self, layer_id: int) -> bool:
        if self._hsa_layer_ids is None:
            return True
        return int(layer_id) in self._hsa_layer_ids

    # R64: shared scalar cache for prefill hot paths (selection + internal SWA +
    # chunk_attn_pool write all want the same (req_idx, prefix_len, extend_len)
    # tuple). Compute once at layer 0, reuse across all 16 layers — drops ~80
    # .item() CUDA syncs per prefill.
    def _get_prefill_scalars(self, forward_batch):
        scalars = getattr(forward_batch, "_hsa_prefill_scalars", None)
        if scalars is not None:
            return scalars
        req_idx = int(forward_batch.req_pool_indices[0].item())
        ep_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        es_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if ep_cpu is not None and len(ep_cpu) > 0:
            prefix_len = int(ep_cpu[0])
        elif forward_batch.extend_prefix_lens is not None:
            prefix_len = int(forward_batch.extend_prefix_lens[0].item())
        else:
            prefix_len = 0
        if es_cpu is not None and len(es_cpu) > 0:
            extend_len = int(es_cpu[0])
        elif forward_batch.extend_seq_lens is not None:
            extend_len = int(forward_batch.extend_seq_lens[0].item())
        else:
            extend_len = 0
        scalars = (req_idx, prefix_len, extend_len)
        forward_batch._hsa_prefill_scalars = scalars
        return scalars

    # (removed) _get_effective_window_size: InnerX ultra passes window via kwargs

    def _run_selection_decode(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        selection_q: Optional[torch.Tensor] = None,
        window_size_override: Optional[int] = None,
        kv_head_offset: int = 0,
        kv_head_count: Optional[int] = None,
        split_info: Optional[dict] = None,
    ) -> None:
        """Top-k page selection for decode.

        Uses logical token positions (not KV slot indices) for candidate
        building and window exclusion, matching the official FlashHSA semantics:
          limit_chunk = (query_pos - window_size + 1) // page_size
          valid candidates: completed pages with page_id < limit_chunk
        """
        md = self.forward_metadata
        if md is None:
            return
        if not self._is_hsa_layer(layer.layer_id):
            return

        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if pool is None or not hasattr(pool, "get_key_buffer"):
            return

        hsa_window = 0
        if window_size_override is not None:
            hsa_window = int(window_size_override)

        page_size = int(self.page_size)
        B = int(forward_batch.batch_size)
        H_sel = int(kv_head_count) if kv_head_count is not None else int(layer.tp_k_head_num)
        device = q.device

        # page_table_1 already has the out_cache_loc overlay applied once per
        # step in init_forward_metadata (R12).  No per-layer clone needed.
        page_table_1 = md.page_table_1

        # R29: effective_cands (= clamp(min(seq_lens // ps, (seq_lens - hsa_window) // ps)))
        # is now computed inline inside the selector kernel from cache_seqlens.
        # Per-layer host-side compute (4 small ops × 16 layers ≈ 200µs) is gone.
        # Slow paths below still need it materialised (rare; fp32 fallback OK).

        # R10: avoid `effective_cands.max().item()` sync.  Use the metadata-known
        # upper bound (max_seqlen_k // page_size); cand_mask handles unused entries
        # below the upper bound so the selection kernel still ignores padding.
        # The early-exit for "all batches have 0 candidates" (context < page_size)
        # is folded into the regular kernel path — at that point context is so
        # short that the extra work is negligible compared to a saved sync.
        C_max = max(int(md.max_seqlen_k) // page_size, 0)

        if C_max == 0:
            md.hsa_cand_page_ids = page_table_1.new_full((B, 0), -1, dtype=torch.int32)
            md.hsa_cand_mask = page_table_1.new_zeros((B, 0), dtype=torch.bool)
            md.hsa_selected_page_ids = page_table_1.new_full(
                (B, H_sel, int(self.hsa_topk)), -1, dtype=torch.int32
            )
            md.hsa_selected_scores = q.new_full(
                (B, H_sel, int(self.hsa_topk)), float("-inf"), dtype=torch.float32
            )
            return

        # Per-q-head lmk_k path: when the HSA layer pre-computed chunk-aggregated
        # h_q-headed lmk_k into the LandmarkLmkKPool (mirrors the official
        # chunk_attn_pool default MHA mode), gather from there instead of
        # synthesising lmk_k from the last-token K in the paged KV cache.
        # The split_info carries hq_hsa / h_hsa so we know G = hq_hsa / h_hsa.
        per_qhead_active = (
            getattr(self, "lmk_k_pool", None) is not None
            and getattr(self, "req_to_chunk_pool", None) is not None
            and split_info is not None
            and int(split_info.get("hq_hsa", 0)) > int(split_info.get("h_hsa", 0))
        )

        # R24+R26: legacy h_kv fast path — fused_selector_score_kernel internalises
        # the whole arange/expand/lt/masked_fill/sum chain plus gather+Q·K+mask
        # in one launch.  Followed by torch.topk (R22, sorted=False).
        unified = split_info is not None and split_info.get("unified_retrieval", False)
        if not per_qhead_active and not unified:
            from sglang.srt.layers.attention.hsa.kernels.fused_selector_score import (
                fused_selector_score_decode,
            )
            import math as _m
            k_cache_full = pool.get_key_buffer(layer.layer_id)
            D_d = int(k_cache_full.shape[2])
            HQ = int(q.shape[-1]) // D_d if q.dim() == 2 else int(q.shape[1])
            H_sel_d = int(H_sel)
            assert HQ % H_sel_d == 0, f"HQ={HQ} not divisible by H_sel={H_sel_d}"
            q3 = q.view(B, HQ, D_d) if q.dim() == 2 else q

            sm_scale_d = (
                float(sm_scale_val) if (sm_scale_val := getattr(layer, "scaling", None)) is not None
                else (1.0 / _m.sqrt(D_d))
            )
            H_offset = int(kv_head_offset) if kv_head_offset is not None else 0

            # R30: pass raw q (not GQA-summed) — kernel does the sum over G inline.
            scores_d = fused_selector_score_decode(
                q=q3,
                H_sel=H_sel_d,
                cache_seqlens=md.cache_seqlens_int32,
                C_max=int(C_max),
                page_table_1=page_table_1,
                k_cache_full=k_cache_full,
                H_offset=H_offset,
                sm_scale=sm_scale_d,
                page_size=int(self.page_size),
                hsa_window=int(hsa_window),
            )

            # Topk (R22 sorted=False).
            eff_topk = min(int(self.hsa_topk), int(C_max))
            top_scores_d, top_idx_d = scores_d.topk(eff_topk, dim=-1, sorted=False)

            if eff_topk < int(self.hsa_topk):
                pad_n = int(self.hsa_topk) - eff_topk
                top_idx_d = torch.cat(
                    [top_idx_d, top_idx_d.new_full((B, H_sel_d, pad_n), -1)], dim=-1,
                )
                top_scores_d = torch.cat(
                    [top_scores_d, top_scores_d.new_full((B, H_sel_d, pad_n), float("-inf"))], dim=-1,
                )

            # In the fast path cand_page_ids[b, c] == c (logical chunk index),
            # so selected_page_ids equals top_idx_d directly.  R31: drop the
            # `.masked_fill(top_scores == -inf, -1)` post-process — downstream
            # `fused_chunk_weight_h_kv_kernel` masks via the score's -inf (which
            # is preserved by sbtopk), not via the page-id's -1 sentinel.
            # R33: keep page_ids as int64 — both chunk_weight and hsa_decode
            # kernels now accept either dtype (load is in-register cast).
            md.hsa_selected_page_ids = top_idx_d
            # R28: keep selected_scores in bf16 — chunk_weight kernel casts inline.
            md.hsa_selected_scores = top_scores_d
            self._per_qhead_G = None
            self._per_qhead_prior_b = None
            self._last_per_qhead_scores_decode = None
            return

        # Slow path still needs the materialised cand_page_ids/cand_mask + the
        # effective_cands [B] tensor.  R54a: cache layer-invariant scratch in
        # forward_metadata so we only build it once per step (was 16x per step
        # for 16-layer 345M HSA — ~96 launches saved).
        cached_pi = getattr(md, "hsa_per_step_cand_page_ids", None)
        cached_cm = getattr(md, "hsa_per_step_cand_mask", None)
        cached_window = getattr(md, "hsa_per_step_hsa_window", None)
        if (
            cached_pi is not None
            and cached_cm is not None
            and cached_window == hsa_window
            and cached_pi.shape == (B, C_max)
        ):
            cand_page_ids = cached_pi
            cand_mask = cached_cm
        else:
            seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
            completed_pages = seq_lens_i64 // page_size
            if hsa_window > 0:
                limit_chunk = (seq_lens_i64 - hsa_window).clamp(min=0) // page_size
                effective_cands = torch.min(completed_pages, limit_chunk).clamp(min=0)
            else:
                effective_cands = completed_pages.clamp(min=0)
            cand_range = torch.arange(C_max, device=device, dtype=torch.int32)
            cand_page_ids = cand_range.unsqueeze(0).expand(B, C_max).contiguous()
            cand_mask = cand_page_ids < effective_cands.unsqueeze(1).to(torch.int32)
            cand_page_ids = cand_page_ids.masked_fill(~cand_mask, -1)
            md.hsa_per_step_cand_page_ids = cand_page_ids
            md.hsa_per_step_cand_mask = cand_mask
            md.hsa_per_step_hsa_window = hsa_window

        if per_qhead_active:
            # R54a: slots = req_to_chunk_pool.gather_slots(req_pool_indices,
            # cand_page_ids) is layer-invariant (same req_pool_indices +
            # same cand_page_ids); cache it too.
            slots = getattr(md, "hsa_per_step_slots", None)
            if slots is None or slots.shape != cand_page_ids.shape:
                slots = self.req_to_chunk_pool.gather_slots(
                    forward_batch.req_pool_indices, cand_page_ids
                )
                md.hsa_per_step_slots = slots
            # R54c: batch the lmk_k + prior_b gathers across ALL layers in one
            # multi-layer index_select instead of per-layer (saves ~94 launches/
            # step for 16-layer 345M HSA). Lazily on the first per_qhead layer.
            all_lmk_k = getattr(md, "hsa_per_step_all_lmk_k", None)
            all_prior_b = getattr(md, "hsa_per_step_all_prior_b", None)
            if all_lmk_k is None or all_prior_b is None:
                # pool.pool: [L, num_chunk_slots+1, h_q, head_dim]
                # prior_b_pool: [L, num_chunk_slots+1, h_q]
                pool_pool = self.lmk_k_pool.pool
                pb_pool = self.lmk_k_pool.prior_b_pool
                flat_idx = (
                    slots.clamp(min=0).to(pool_pool.device, torch.int64).reshape(-1)
                )
                # [L, B*C, h_q, D] / [L, B*C, h_q]
                gathered_k = pool_pool.index_select(1, flat_idx)
                gathered_pb = pb_pool.index_select(1, flat_idx)
                all_lmk_k = gathered_k.view(
                    pool_pool.shape[0], *slots.shape,
                    pool_pool.shape[2], pool_pool.shape[3],
                )
                all_prior_b = gathered_pb.view(
                    pb_pool.shape[0], *slots.shape, pb_pool.shape[2]
                )
                md.hsa_per_step_all_lmk_k = all_lmk_k
                md.hsa_per_step_all_prior_b = all_prior_b
            cand_repr = all_lmk_k[int(layer.layer_id)]
            # prior_b (entropy bias) per chunk per q-head, [B, C_max, h_q]
            self._per_qhead_prior_b = all_prior_b[int(layer.layer_id)]
            H_sel = cand_repr.shape[2]
            D = cand_repr.shape[3]
            # G for the topk kernel — switches it to per-q-head mode.
            self._per_qhead_G = int(split_info["hq_hsa"]) // int(split_info["h_hsa"])
        else:
            # Existing path: gather last-token K from KV cache (h_kv shape).
            self._per_qhead_G = None
            self._per_qhead_prior_b = None
            safe_page_ids = cand_page_ids.clamp(min=0).to(torch.int64)
            lmk_token_pos = safe_page_ids * page_size + (page_size - 1)  # [B, C_max]
            lmk_token_pos_safe = lmk_token_pos.clamp(max=page_table_1.shape[1] - 1)
            lmk_locs = torch.gather(
                page_table_1.to(torch.int64), 1, lmk_token_pos_safe
            )  # [B, C_max]

            k_cache = pool.get_key_buffer(layer.layer_id)  # [num_locs, H_total, D]
            D = int(k_cache.shape[2])
            flat_lmk_locs = lmk_locs.reshape(-1)
            flat_repr = k_cache[flat_lmk_locs]  # [B*C_max, H_total, D]
            if kv_head_count is not None:
                flat_repr = flat_repr[:, int(kv_head_offset):int(kv_head_offset) + int(kv_head_count), :]
            cand_repr = flat_repr.view(B, C_max, H_sel, D)

        # unified_retrieval: group+sum KV heads to match retrieval_dim.
        unified = split_info is not None and split_info.get("unified_retrieval", False)
        retrieval_dim = split_info.get("retrieval_dim", None) if split_info is not None else None
        if unified and retrieval_dim is not None:
            tp_size = get_attention_tp_size()
            if tp_size > 1:
                # All-gather KV head landmark keys across TP ranks so every rank
                # sees the full head dimension, matching training semantics where
                # a single retrieval head operates on all KV heads.
                cand_repr = tensor_model_parallel_all_gather(cand_repr.contiguous(), dim=2)
                H_sel = cand_repr.shape[2]  # updated to full H_hsa
            flat_dim = H_sel * D
            if flat_dim != retrieval_dim:
                G_groups = flat_dim // retrieval_dim
                cand_repr = cand_repr.reshape(B, C_max, 1, G_groups, retrieval_dim)
                cand_repr = cand_repr.sum(dim=3)
            else:
                cand_repr = cand_repr.reshape(B, C_max, 1, flat_dim)

        q_sel = selection_q if selection_q is not None else q
        if unified and retrieval_dim is not None:
            tp_size = get_attention_tp_size()
            if tp_size > 1:
                # All-gather sel_q across TP ranks to reconstruct the full
                # retrieval_dim vector (ColumnParallelLinear splits along last dim).
                q_sel = tensor_model_parallel_all_gather(q_sel.contiguous(), dim=-1)
        sm_scale_val = getattr(layer, "scaling", None)

        # R54b: for the per-qhead path, hand the kernel q_offset+hsa_window so
        # it can do causal+window masking internally (no separate masked_fill).
        # max_seqlen_k is already an int on CPU (set by init_forward_metadata).
        sel = select_topk_pages_decode_fused(
            q=q_sel,
            cand_page_ids=cand_page_ids,
            cand_mask=cand_mask,
            cand_repr=cand_repr,
            topk=int(self.hsa_topk),
            page_size=int(self.page_size),
            sm_scale=sm_scale_val,
            selection_strategy=str(self.hsa_selection_strategy),
            G=getattr(self, "_per_qhead_G", None),
            per_qhead_prior_b=getattr(self, "_per_qhead_prior_b", None),
            _decode_q_offset=int(md.max_seqlen_k) - 1,
            _decode_hsa_window=int(hsa_window),
        )
        if sel is None:
            sel = select_topk_pages_decode(
                q=q_sel,
                cand_page_ids=cand_page_ids,
                cand_mask=cand_mask,
                cand_chunk_repr=cand_repr,
                cand_chunk_repr_valid=cand_mask,
                topk=int(self.hsa_topk),
                selection_strategy=str(self.hsa_selection_strategy),
                sm_scale=sm_scale_val,
            )

        md.hsa_cand_page_ids = sel.cand_page_ids
        md.hsa_cand_mask = sel.cand_mask
        md.hsa_selected_page_ids = sel.selected_page_ids
        md.hsa_selected_scores = sel.selected_scores
        # Per-q-head scores side-channel — used by the per-q-head fusion in
        # forward_decode when active.  Stash on self (single-request scope).
        self._last_per_qhead_scores_decode = getattr(sel, "_per_qhead_scores", None)

    def _compute_internal_swa_decode(
        self,
        *,
        q_hsa: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        page_table_1: torch.Tensor,
        H_swa: int,
        H_hsa: int,
        HQ_hsa: int,
        hsa_window: int,
    ) -> tuple:
        """Internal SWA on HSA heads (LHSA semantics).

        Computes chunk-aligned sliding window attention on the HSA heads,
        excluding LMK positions.  Returns output and per-kv-head logsumexp
        for merged softmax.

        Returns
        -------
        swa_o : [B, HQ_hsa, D] float32
        lse_hq : [B, HQ_hsa] bf16  (per-q-head logsumexp; reduce to lse_kv
                                    in the consuming chunk_weight kernel via R27)
        """
        B, _, D = q_hsa.shape
        device = q_hsa.device

        if hsa_window <= 0:
            swa_o = torch.zeros((B, HQ_hsa, D), device=device, dtype=torch.float32)
            lse_hq_neg = torch.full(
                (B, HQ_hsa), float("-inf"), device=device, dtype=q_hsa.dtype
            )
            self._last_swa_lse_hq_decode = lse_hq_neg
            return swa_o, lse_hq_neg

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None:
            swa_o = torch.zeros((B, HQ_hsa, D), device=device, dtype=torch.float32)
            lse_hq_neg = torch.full(
                (B, HQ_hsa), float("-inf"), device=device, dtype=q_hsa.dtype
            )
            self._last_swa_lse_hq_decode = lse_hq_neg
            return swa_o, lse_hq_neg

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        page_size = int(self.page_size)
        sm_scale = float(getattr(layer, "scaling", 1.0))

        # R21: fused streaming-flash-attention triton kernel.  Collapses ~15
        # ops (R11+R16 chain: window construction, K/V gather, two einsums,
        # softmax, nan_to_num, casts) into a single launch.  Each program
        # handles one (b, hq) and walks the chunk-aligned window in BLOCK_W
        # chunks with online-softmax running max/sum updates.  LMK exclusion
        # happens inside the kernel via `(pos+1) % page_size != 0`.
        from sglang.srt.layers.attention.hsa.kernels.internal_swa_decode import (
            fused_internal_swa_decode,
        )
        k_cache_all = pool.get_key_buffer(layer.layer_id)
        v_cache_all = pool.get_value_buffer(layer.layer_id)
        swa_o, lse_hq_bf16 = fused_internal_swa_decode(
            q_hsa=q_hsa,
            k_cache_full=k_cache_all,
            v_cache_full=v_cache_all,
            page_table_1=page_table_1,
            cache_seqlens=md.cache_seqlens_int32,
            H_swa=int(H_swa),
            H_hsa=int(H_hsa),
            HQ_hsa=int(HQ_hsa),
            page_size=page_size,
            hsa_window=int(hsa_window),
            sm_scale=sm_scale,
        )
        # R27: drop the torch.logsumexp(lse_hq → lse_kv) — the legacy h_kv
        # chunk_weight kernel now reduces over Gh inline.  Just return
        # lse_hq directly; the per-q-head path already used lse_hq.
        self._last_swa_lse_hq_decode = lse_hq_bf16
        return swa_o, lse_hq_bf16

    def _build_hsa_lmk_excluded_indices(self, forward_batch: ForwardBatch):
        """Build KV indices with LMK positions excluded (for HSA layers only).

        FlashHSA mask semantics: LMK tokens are written to KV cache but never
        attended as KV.  We build a separate kv_indptr/kv_indices pair that
        skips LMK slots so HSA layers see only real tokens.
        """
        bs = forward_batch.batch_size
        device = self.device
        page_size = int(self.page_size)

        max_seqlen = int(forward_batch.seq_lens.max().item())
        pos = torch.arange(max_seqlen, device=device, dtype=torch.int64)
        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen
        ]
        valid = pos[None, :] < forward_batch.seq_lens.to(torch.int64)[:, None]
        keep = valid & ((pos[None, :] + 1) % page_size != 0)
        kv_lens = keep.sum(dim=1, dtype=torch.int32)

        kv_indptr = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)
        kv_indices = page_table[keep].to(torch.int64)

        dense_backend = self._dense_backend
        num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=device)
        dense_backend.get_num_kv_splits(num_kv_splits, kv_lens)

        return kv_indptr, kv_indices, kv_lens, num_kv_splits

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # First, initialize dense backend metadata (standard indices, no LMK exclusion).
        # SWA layers will use these indices directly.
        self._dense_backend.init_forward_metadata(forward_batch)

        # R15: cuda-graph buffer path.  When persistent buffers are allocated
        # AND we're in decode-or-idle (the only mode cuda graph captures),
        # write into the buffers via triton kernels so the captured kernel
        # pointers stay valid across replays.  R10 already makes max_seqlen_k
        # constant (=graph max) so all derived shapes are stable.
        if (
            getattr(self, "_cg_page_table_1", None) is not None
            and forward_batch.forward_mode.is_decode_or_idle()
        ):
            B = int(forward_batch.batch_size)
            max_seqlen_k = self._cg_max_seqlen_k
            update_hsa_cg_buffers(
                bs=B,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                req_to_token=forward_batch.req_to_token_pool.req_to_token,
                page_table_1_buf=self._cg_page_table_1,
                cache_seqlens_i32_buf=self._cg_cache_seqlens_i32,
            )
            page_table_1 = self._cg_page_table_1[:B]
            cache_seqlens_int32 = self._cg_cache_seqlens_i32[:B]
            real_page_table = None  # only NSA uses this field
            dense_md = getattr(self._dense_backend, "forward_metadata", None)
            self.forward_metadata = HSAMetadata(
                page_size=self.page_size,
                cache_seqlens_int32=cache_seqlens_int32,
                max_seqlen_k=max_seqlen_k,
                page_table_1=page_table_1,
                real_page_table=real_page_table,
                kv_indptr=getattr(dense_md, "kv_indptr", None),
                kv_indices=getattr(dense_md, "kv_indices", None),
                window_kv_indptr=getattr(dense_md, "window_kv_indptr", None),
                window_kv_indices=getattr(dense_md, "window_kv_indices", None),
                hsa_kv_indptr=None,
                hsa_kv_indices=None,
                hsa_kv_lens=None,
                hsa_num_kv_splits=None,
                token_positions=None,
                token_to_seq_id=None,
                extend_seq_lens=None,
                extend_prefix_lens=None,
                engine_indices=None,
            )
            return

        # ---- Non-cuda-graph path (extend, or HSA without --cuda-graph-max-bs) ----
        # Build HSA metadata scaffold (paged-KV-first).
        if forward_batch.seq_lens_cpu is not None:
            max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
        else:
            max_seqlen_k = int(forward_batch.seq_lens.max().item())

        page_table_1 = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        # R12: overlay out_cache_loc into page_table_1 at position (seq_len - 1)
        # once per step instead of once per layer.
        if (
            forward_batch.forward_mode.is_decode_or_idle()
            and getattr(forward_batch, "out_cache_loc", None) is not None
        ):
            B = int(forward_batch.batch_size)
            if B > 0:
                out_locs = forward_batch.out_cache_loc[:B].to(torch.int32)
                seq_lens_i64 = forward_batch.seq_lens[:B].to(torch.int64)
                cap = page_table_1.shape[1] - 1
                positions = (seq_lens_i64 - 1).clamp_(min=0, max=cap)
                batch_idx = torch.arange(
                    B, device=page_table_1.device, dtype=torch.int64
                )
                page_table_1[batch_idx, positions] = out_locs

        real_page_table = transform_page_table_1_to_real(page_table_1, self.page_size)

        dense_md = getattr(self._dense_backend, "forward_metadata", None)
        kv_indptr = getattr(dense_md, "kv_indptr", None)
        kv_indices = getattr(dense_md, "kv_indices", None)
        window_kv_indptr = getattr(dense_md, "window_kv_indptr", None)
        window_kv_indices = getattr(dense_md, "window_kv_indices", None)

        # Build HSA-specific LMK-excluded indices for decode.
        hsa_kv_indptr = None
        hsa_kv_indices = None
        hsa_kv_lens = None
        hsa_num_kv_splits = None
        if forward_batch.forward_mode.is_decode_or_idle():
            hsa_kv_indptr, hsa_kv_indices, hsa_kv_lens, hsa_num_kv_splits = (
                self._build_hsa_lmk_excluded_indices(forward_batch)
            )

        # Build extend-specific fields when in extend mode.
        token_positions = None
        token_to_seq_id = None
        extend_seq_lens = None
        extend_prefix_lens = None
        engine_indices = None
        if forward_batch.forward_mode.is_extend():
            token_positions = forward_batch.positions
            extend_seq_lens = forward_batch.extend_seq_lens
            extend_prefix_lens = forward_batch.extend_prefix_lens
            if extend_seq_lens is not None:
                token_to_seq_id = torch.repeat_interleave(
                    torch.arange(len(extend_seq_lens), device=self.device, dtype=torch.int32),
                    extend_seq_lens.to(torch.int64),
                )
                engine_indices = torch.cat([
                    torch.arange(int(plen), int(plen) + int(elen),
                                 device=self.device, dtype=torch.int64)
                    for plen, elen in zip(extend_prefix_lens, extend_seq_lens)
                ])

        self.forward_metadata = HSAMetadata(
            page_size=self.page_size,
            cache_seqlens_int32=forward_batch.seq_lens.to(torch.int32),
            max_seqlen_k=max_seqlen_k,
            page_table_1=page_table_1,
            real_page_table=real_page_table,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            window_kv_indptr=window_kv_indptr,
            window_kv_indices=window_kv_indices,
            hsa_kv_indptr=hsa_kv_indptr,
            hsa_kv_indices=hsa_kv_indices,
            hsa_kv_lens=hsa_kv_lens,
            hsa_num_kv_splits=hsa_num_kv_splits,
            token_positions=token_positions,
            token_to_seq_id=token_to_seq_id,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            engine_indices=engine_indices,
        )

    # ---- CUDA graph plumbing (R15) ----
    # Dense state is still set up via delegation, so kv_indptr / kv_indices
    # / window_kv_indices are populated by the dense backend.  HSA owns two
    # extra persistent buffers (page_table_1 + cache_seqlens) so the captured
    # graph's HSA kernels read from stable addresses; replay-time updates go
    # in through `update_hsa_cg_buffers` (graph-safe triton kernels — no
    # intermediate Python allocations).

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self._dense_backend.init_cuda_graph_state(max_bs, max_num_tokens)

        max_context_len = int(self.model_runner.model_config.context_len)
        device = self.device

        # Force max_seqlen_k to graph-max so C_max / cand_* / selected_* all
        # have stable shapes across captures and replays.
        self._cg_max_seqlen_k = max_context_len
        self._cg_page_table_1 = torch.zeros(
            (max_bs, max_context_len), dtype=torch.int32, device=device
        )
        self._cg_cache_seqlens_i32 = torch.zeros(
            (max_bs,), dtype=torch.int32, device=device
        )

    def _build_hsa_cg_metadata(self, bs: int):
        """Populate self.forward_metadata from the pre-allocated cuda-graph
        buffers.  Called by both capture and replay hooks so the captured
        graph reads from the buffer pointers and replay just refreshes the
        buffer contents.
        """
        max_seqlen_k = self._cg_max_seqlen_k
        dense_md = getattr(self._dense_backend, "forward_metadata", None)
        self.forward_metadata = HSAMetadata(
            page_size=self.page_size,
            cache_seqlens_int32=self._cg_cache_seqlens_i32[:bs],
            max_seqlen_k=max_seqlen_k,
            page_table_1=self._cg_page_table_1[:bs],
            real_page_table=None,  # unused by HSA path
            kv_indptr=getattr(dense_md, "kv_indptr", None),
            kv_indices=getattr(dense_md, "kv_indices", None),
            window_kv_indptr=getattr(dense_md, "window_kv_indptr", None),
            window_kv_indices=getattr(dense_md, "window_kv_indices", None),
            hsa_kv_indptr=None,
            hsa_kv_indices=None,
            hsa_kv_lens=None,
            hsa_num_kv_splits=None,
            token_positions=None,
            token_to_seq_id=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            engine_indices=None,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        self._dense_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        # R15.2: capture's run_once calls model.forward directly (NOT
        # model_runner.forward_decode), so init_forward_metadata is never
        # invoked during capture.  We must populate HSA's buffers + metadata
        # here, otherwise self.forward_metadata stays stale from the previous
        # EXTEND init and forward_decode silently falls back to dense.
        if forward_mode.is_decode_or_idle() and self._cg_page_table_1 is not None:
            update_hsa_cg_buffers(
                bs=int(bs),
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                page_table_1_buf=self._cg_page_table_1,
                cache_seqlens_i32_buf=self._cg_cache_seqlens_i32,
            )
            self._build_hsa_cg_metadata(int(bs))

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        self._dense_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )
        # Update HSA buffers in place (so captured kernels see fresh data).
        # Note: must use the SAME buffer pointers as capture, otherwise the
        # captured graph reads from stale addresses.  Also re-publish
        # self.forward_metadata in case anything (e.g. forward_decode fast-path
        # before graph replay) reads it.
        if forward_mode.is_decode_or_idle() and self._cg_page_table_1 is not None:
            update_hsa_cg_buffers(
                bs=int(bs),
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                page_table_1_buf=self._cg_page_table_1,
                cache_seqlens_i32_buf=self._cg_cache_seqlens_i32,
            )
            self._build_hsa_cg_metadata(int(bs))

    def get_cuda_graph_seq_len_fill_value(self):
        return self._dense_backend.get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        return self._dense_backend.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        return self._dense_backend.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Optional: InnerX-style split-head HSA passes extra tensors via kwargs.
        split_info = kwargs.get("hsa_split_head_info", None)
        selection_q = kwargs.get("hsa_selection_q", None)

        # Step 4: run selection to populate metadata.
        if split_info is not None:
            h_swa = int(split_info.get("h_swa", 0))
            h_hsa = int(split_info.get("h_hsa", 0))
            hsa_window = int(split_info.get("swa_window_size", 0) or 0)
            self._run_selection_decode(
                q=q,
                layer=layer,
                forward_batch=forward_batch,
                selection_q=selection_q,
                window_size_override=hsa_window,
                kv_head_offset=h_swa,
                kv_head_count=h_hsa,
                split_info=split_info,
            )
        else:
            self._run_selection_decode(q=q, layer=layer, forward_batch=forward_batch)

        # Non-HSA layers: delegate to dense backend.
        if not self._is_hsa_layer(layer.layer_id):
            kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
            return self._dense_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs_clean
            )

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None:
            kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
            return self._dense_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs_clean
            )

        # Save current KV into the paged KV cache (same as TritonAttnBackend).
        if save_kv_cache:
            pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        # Prepare q: [B, HQ, D]
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError(
                "HSAAttnBackend.hsa_decode_paged_fwd currently assumes qk_head_dim == v_head_dim."
            )

        # page_table_1 already has the out_cache_loc overlay applied once per
        # step in init_forward_metadata (R12).
        page_table_1 = md.page_table_1

        # ---- InnerX split-head HSA (ultra reference) ----
        # In this mode, the layer output is a head-wise concatenation:
        #   out = cat([out_swa_heads, out_hsa_heads], dim=head)
        # where:
        #   - SWA heads run local sliding-window attention
        #   - HSA heads run sparse paged attention with per-page weights
        if split_info is not None:
            selected_page_ids = md.hsa_selected_page_ids
            selected_scores = md.hsa_selected_scores
            if selected_page_ids is None or selected_scores is None:
                raise RuntimeError("HSA selection did not populate metadata")

            HQ_total = int(layer.tp_q_head_num)
            H_total = int(layer.tp_k_head_num)
            HQ_swa = int(split_info.get("hq_swa", 0))
            HQ_hsa = int(split_info.get("hq_hsa", 0))
            H_swa = int(split_info.get("h_swa", 0))
            H_hsa = int(split_info.get("h_hsa", 0))
            window_size = split_info.get("swa_window_size", None)
            swa_exclude_lmk = bool(split_info.get("swa_exclude_lmk", False))

            if HQ_swa + HQ_hsa != HQ_total:
                raise ValueError(
                    f"hsa_split_head_info mismatch: HQ_swa+HQ_hsa={HQ_swa+HQ_hsa} != HQ_total={HQ_total}"
                )
            if H_swa + H_hsa != H_total:
                raise ValueError(
                    f"hsa_split_head_info mismatch: H_swa+H_hsa={H_swa+H_hsa} != H_total={H_total}"
                )
            if HQ_hsa <= 0 or H_hsa <= 0:
                raise ValueError("InnerX split-head requires non-empty HSA head partitions")
            # unified_retrieval: selection returns [B, 1, K]; expand to [B, H_hsa, K]
            if int(selected_page_ids.shape[1]) == 1 and H_hsa > 1:
                selected_page_ids = selected_page_ids.expand(B, H_hsa, -1).contiguous()
                selected_scores = selected_scores.expand(B, H_hsa, -1).contiguous()
                md.hsa_selected_page_ids = selected_page_ids
                md.hsa_selected_scores = selected_scores
            elif int(selected_page_ids.shape[1]) != H_hsa:
                raise ValueError(
                    f"Selection H mismatch: got {int(selected_page_ids.shape[1])}, expected H_hsa={H_hsa}"
                )

            B = int(q3.shape[0])
            D = int(q3.shape[2])
            q_swa = q3[:, :HQ_swa, :]
            q_hsa = q3[:, HQ_swa:, :]

            # SWA branch: sliding window attention on upper (SWA) heads.
            upper_swa_window = int(split_info.get("upper_swa_window_size", 0) or -1)
            out_swa = torch.zeros((B, HQ_swa, D), device=q3.device, dtype=torch.float32)
            if HQ_swa > 0 and H_swa > 0:
                import torch.nn.functional as F

                seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
                k_cache_all = pool.get_key_buffer(layer.layer_id)
                v_cache_all = pool.get_value_buffer(layer.layer_id)
                swa_sliding_window = upper_swa_window if upper_swa_window > 0 else None
                sm_scale = float(getattr(layer, "scaling", 1.0))
                assert HQ_swa % H_swa == 0
                Gs = HQ_swa // H_swa

                for b in range(B):
                    seqlen = int(seq_lens_i64[b].item())
                    if seqlen <= 0:
                        continue
                    if swa_sliding_window is not None and int(swa_sliding_window) > 0:
                        w = min(seqlen, int(swa_sliding_window))
                    else:
                        w = seqlen
                    start = seqlen - w
                    tok_pos = torch.arange(start, seqlen, device=q3.device, dtype=torch.int64)
                    if swa_exclude_lmk:
                        keep = (tok_pos % int(self.page_size)) != (int(self.page_size) - 1)
                        tok_pos = tok_pos[keep]
                    if tok_pos.numel() == 0:
                        continue
                    token_locs = page_table_1[b, tok_pos].to(torch.int64)

                    # Gather K/V: [H_swa, S, D]
                    k_win = k_cache_all[token_locs, :H_swa, :].transpose(0, 1)
                    v_win = v_cache_all[token_locs, :H_swa, :].transpose(0, 1)
                    # Expand for GQA: [HQ_swa, S, D]
                    k_win = k_win.repeat_interleave(Gs, dim=0)
                    v_win = v_win.repeat_interleave(Gs, dim=0)
                    # Q: [HQ_swa, 1, D]
                    q_b = q_swa[b].unsqueeze(1)

                    # SDPA: no mask needed for decode (Q_LEN=1, attend to all)
                    o = F.scaled_dot_product_attention(
                        q_b.to(torch.float32), k_win.to(torch.float32),
                        v_win.to(torch.float32), scale=sm_scale,
                    )  # [HQ_swa, 1, D]
                    out_swa[b] = o.squeeze(1)

            # Internal SWA on HSA heads (LHSA semantics).
            # R27: returns per-q-head lse_hq directly (kernel reduces over Gh inline).
            swa_o_inner, lse_hq_bf16 = self._compute_internal_swa_decode(
                q_hsa=q_hsa,
                layer=layer,
                forward_batch=forward_batch,
                page_table_1=page_table_1,
                H_swa=H_swa,
                H_hsa=H_hsa,
                HQ_hsa=HQ_hsa,
                hsa_window=hsa_window,
            )

            # HSA branch weights via merged softmax (LHSA reference lines 371-384).
            assert HQ_hsa % H_hsa == 0
            Gh = HQ_hsa // H_hsa
            TOPK = int(selected_page_ids.shape[2])
            # R31: `valid = selected_page_ids >= 0` was used only in the
            # hsa_window <= 0 fallback branch — defer its compute to there
            # (saves ~24µs/step at 16K, ~1.5µs × 16 layers).

            # Per-q-head fusion path: when the selector exposed per-q-head
            # scores AND the SWA branch exposed per-q-head LSE, do the entire
            # cat+softmax at h_q granularity (matching official Qwen3-LHSA
            # line 782).  Otherwise fall back to the h_kv path then broadcast.
            per_qhead_scores = self._last_per_qhead_scores_decode  # [B, h_q, K] or None
            per_qhead_lse = self._last_swa_lse_hq_decode           # [B, h_q] or None
            per_qhead_fusion = (
                per_qhead_scores is not None
                and per_qhead_lse is not None
                and per_qhead_scores.shape == (B, HQ_hsa, TOPK)
                and per_qhead_lse.shape == (B, HQ_hsa)
            )

            if per_qhead_fusion and hsa_window > 0:
                # R13: fused chunk-weight softmax (8 ops -> 1 triton kernel).
                # Replaces masked_fill + expand/reshape + cat + softmax +
                # nan_to_num + slice + contiguous + cast.
                w_q, swa_w_q = fused_chunk_weight_per_qhead_decode(
                    per_qhead_scores=per_qhead_scores,
                    per_qhead_lse=per_qhead_lse,
                    selected_page_ids=selected_page_ids.to(torch.int32).contiguous()
                    if selected_page_ids.dtype != torch.int32 or not selected_page_ids.is_contiguous()
                    else selected_page_ids,
                    Gh=Gh,
                    enable_softmax1=bool(self.enable_softmax1),
                    out_dtype=q_hsa.dtype,
                )
                swa_w_kv = None  # signal: use swa_w_q downstream
            elif hsa_window > 0:
                # R18+R20+R27: fused legacy h_kv chunk_weight + GQA broadcast +
                # HQ-granular swa_w output + inline logsumexp(lse_hq → lse_kv).
                # R33: kernel accepts int32 OR int64 page_ids; no host-side cast.
                w_q, swa_w_q = fused_chunk_weight_h_kv_decode(
                    selected_scores=selected_scores,
                    lse_hq=lse_hq_bf16,
                    selected_page_ids=selected_page_ids if selected_page_ids.is_contiguous()
                    else selected_page_ids.contiguous(),
                    Gh=Gh,
                    enable_softmax1=bool(self.enable_softmax1),
                    out_dtype=q_hsa.dtype,
                )
                swa_w_kv = None  # use swa_w_q directly downstream
            else:
                # No internal SWA: independent softmax over scores.
                valid = selected_page_ids >= 0
                scores = selected_scores.masked_fill(~valid, float("-inf"))
                w_kv = torch.softmax(scores, dim=-1)
                w_kv = torch.nan_to_num(w_kv, nan=0.0).to(q_hsa.dtype)
                swa_w_kv = None
                w_q = (
                    w_kv[:, :, None, :]
                    .expand(B, H_hsa, Gh, TOPK)
                    .reshape(B, HQ_hsa, TOPK)
                    .contiguous()
                )
                swa_w_q = None

            # R25: fuse the SWA blend into hsa_decode_paged_fwd's epilogue.
            # Need swa_w_q at [B, HQ_hsa] granularity.  Both per_qhead and
            # R20-modified legacy h_kv paths now ALWAYS output swa_w_q already
            # at HQ granularity (swa_w_kv is None on both branches above), so
            # the legacy expansion path is dead.  The fallback (`hsa_window
            # <= 0` else branch) sets swa_w_q = None — handled in
            # hsa_decode_paged_fwd by BLEND_SWA=False, no expansion needed.

            k_cache_hsa = pool.get_key_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :]
            v_cache_hsa = pool.get_value_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :]
            out_hsa = hsa_decode_paged_fwd(
                q=q_hsa,
                k_cache=k_cache_hsa,
                v_cache=v_cache_hsa,
                page_table_1=page_table_1,
                selected_page_ids=selected_page_ids,
                hsa_weights=w_q,
                page_size=int(self.page_size),
                sm_scale=getattr(layer, "scaling", None),
                mask_last_token=True,
                swa_o_inner=swa_o_inner if swa_w_q is not None else None,
                swa_w_q=swa_w_q,
            )  # [B, HQ_hsa, D] bf16 (blend already applied inside kernel)

            # R34: HSA-345M has HQ_swa == 0 — skip the empty+slice-write that
            # otherwise copies out_hsa into a fresh out_all (≈2KB/layer × 16
            # layers under CG = ~30µs/step that's pure data motion for nothing).
            if HQ_swa == 0:
                return out_hsa.reshape(q.shape[0], HQ_total * layer.v_head_dim)
            out_all = torch.empty((B, HQ_total, D), device=q3.device, dtype=torch.bfloat16)
            out_all[:, :HQ_swa, :] = out_swa.to(torch.bfloat16)
            out_all[:, HQ_swa:, :] = out_hsa
            return out_all.reshape(q.shape[0], HQ_total * layer.v_head_dim)
        raise RuntimeError(
            "InnerX ultra only: HSA layers must pass `hsa_split_head_info` kwargs."
        )

    # ---- Extend helpers (production + reference) ----

    _USE_EXTEND_REFERENCE = os.getenv("SGLANG_HSA_EXTEND_REFERENCE", "0") == "1"
    # Independent SWA-only reference toggle (used to bisect whether the
    # remaining alignment gap is in the triton extend SWA kernel or in the
    # HSA aggregation kernel).  When True it overrides the SWA path
    # specifically while leaving selection + HSA aggregation on the
    # production path (so per-q-head fixes still apply).
    _USE_SWA_EXTEND_REFERENCE = os.getenv("SGLANG_HSA_SWA_REFERENCE", "0") == "1"

    def _compute_internal_swa_extend(
        self,
        *,
        q_hsa: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        page_table_1: torch.Tensor,
        H_swa: int,
        H_hsa: int,
        HQ_hsa: int,
        hsa_window: int,
    ) -> tuple:
        """Dispatch to batched (production) or reference implementation."""
        if self._USE_EXTEND_REFERENCE or self._USE_SWA_EXTEND_REFERENCE:
            return self._compute_internal_swa_extend_reference(
                q_hsa=q_hsa, layer=layer, forward_batch=forward_batch,
                page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
                HQ_hsa=HQ_hsa, hsa_window=hsa_window,
            )
        return self._compute_internal_swa_extend_batched(
            q_hsa=q_hsa, layer=layer, forward_batch=forward_batch,
            page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
            HQ_hsa=HQ_hsa, hsa_window=hsa_window,
        )

    def _compute_internal_swa_extend_batched(
        self,
        *,
        q_hsa: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        page_table_1: torch.Tensor,
        H_swa: int,
        H_hsa: int,
        HQ_hsa: int,
        hsa_window: int,
    ) -> tuple:
        """Internal SWA on HSA heads for extend via triton extend kernel.

        R36: single-batch fused mask path. The triton extend kernel now natively
        supports BOTH chunk-aligned sliding window AND LMK exclusion via two new
        constexpr params (CHUNK_ALIGNED_SW_PAGE_SIZE, LMK_PERIOD).  We collapse the
        previous "T sequences of 1 query each" structure into "1 sequence of T
        queries", letting the kernel use its full BLOCK_M Q-tile and amortize KV
        loads across consecutive Q tokens — eliminates the per-token launch-and-Q-
        tile-waste that made prefill ~8× slower than dense at L=16K.

        Mask equivalence to old per-token kv_indices construction:
          * Chunk-aligned SW lower bound:
                raw_start  = q_abs - hsa_window + 1
                chunk_start = max(0, floor(raw_start / page_size) * page_size)
                K visible iff k_abs >= chunk_start
            (kernel: window_mask when CHUNK_ALIGNED_SW_PAGE_SIZE > 0)
          * LMK exclusion:
                K visible iff (k_abs + 1) % page_size != 0
            (kernel: lmk_mask when LMK_PERIOD > 0)
          * Causal between extend Q and extend K (q_abs >= k_abs):
            (kernel: IS_CAUSAL with prefix_lens passed through)

        Only supports batch=1 (SGLang continuous batching guarantee).

        Returns
        -------
        swa_o : [T, HQ_hsa, D] float32
        lse_hq : [T, HQ_hsa] float32  (per-q-head logsumexp; caller does
                 GQA aggregation only if it needs lse_kv [T, H_hsa]).
        """
        T, _, D = q_hsa.shape
        device = q_hsa.device

        swa_o = torch.zeros((T, HQ_hsa, D), device=device, dtype=torch.float32)
        # R77: placeholder per-q-head LSE for empty/no-window early returns.
        lse_hq_empty = torch.full((T, HQ_hsa), float("-inf"), device=device, dtype=torch.float32)

        if hsa_window <= 0:
            return swa_o, lse_hq_empty

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None or md.token_positions is None or md.token_to_seq_id is None:
            return swa_o, lse_hq_empty

        # --- batch=1 only ---
        extend_seq_lens = md.extend_seq_lens
        extend_prefix_lens = md.extend_prefix_lens
        assert extend_seq_lens is not None and extend_prefix_lens is not None
        B = int(extend_seq_lens.shape[0])
        assert B == 1, (
            f"_compute_internal_swa_extend_batched 当前只支持 batch=1，"
            f"但收到 B={B}。"
        )

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        page_size = int(self.page_size)
        sm_scale = float(getattr(layer, "scaling", 1.0))

        # R64: shared per-prefill scalar cache (was 2 .item() per layer here).
        _req_idx_cached, prefix_len, extend_len = self._get_prefill_scalars(forward_batch)
        total_kv = prefix_len + extend_len  # full sequence length in cache

        if total_kv == 0 or extend_len == 0:
            return swa_o, lse_hq_empty

        # --- Build single-sequence kv_indices over [0, total_kv) ---
        # Each linear position i in kv_indices maps to cache slot for engine pos i.
        # R78: keep page_table_1's int32 — the kernel internally casts each
        # loaded slot to int64 (saves a per-layer dtype copy + halves bw).
        kv_indices = page_table_1[0, :total_kv].contiguous()

        # HSA heads slice of KV cache.
        pool_k = pool.get_key_buffer(layer.layer_id)[:, H_swa:H_swa + H_hsa, :]
        pool_v = pool.get_value_buffer(layer.layer_id)[:, H_swa:H_swa + H_hsa, :]

        # R39: dedicated G-fused HSA-SWA kernel — packs all GQA q-heads per
        # program (M=BLOCK_M*G=128 at G=4, hitting TC sweet spot) instead of
        # the generic _fwd_kernel_unified which reloaded K once per q-head.
        from sglang.srt.layers.attention.hsa.kernels import hsa_swa_extend_fwd
        swa_o, lse_raw = hsa_swa_extend_fwd(
            q_hsa=q_hsa,
            k_cache_hsa=pool_k,
            v_cache_hsa=pool_v,
            kv_indices=kv_indices,
            prefix_len=prefix_len,
            extend_len=extend_len,
            hsa_window=hsa_window,
            page_size=page_size,
            sm_scale=sm_scale,
        )
        # R66: keep swa_o in bf16 — the fusion at forward_extend step 7 mixes
        # it with bf16 out_hsa anyway, so the .to(fp32) here just burned
        # 64MB of memory bandwidth per layer for nothing. Skipping saves
        # ~1GB of memory traffic per 16K prefill (16 layers).

        # R71: stash per-q-head LSE for the per-q-head fusion path. Keep in
        # fp32 — both downstream consumers (the maxpool topk kernel and the
        # cat-with-scores at forward_extend step 5) want fp32, so casting to
        # bf16 here was a per-layer kernel launch that the kernel then undid
        # with .to(fp32) internally. Skipping saves 16 launches per prefill.
        self._last_swa_lse_hq_extend = lse_raw

        # R77: skip the per-layer torch.logsumexp(lse_raw, dim=Gh) — only the
        # legacy h_kv chunk_weight branch in forward_extend uses lse_kv, and
        # the per_qhead branch (active whenever the selector is per-q-head)
        # uses lse_raw directly. Returning lse_raw lets the caller compute
        # lse_kv on demand. Removed call had ~480us CPU dispatch overhead
        # per layer (max + sub + exp + sum + log decomposition).
        return swa_o, lse_raw

    def _compute_internal_swa_extend_reference(
        self,
        *,
        q_hsa: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        page_table_1: torch.Tensor,
        H_swa: int,
        H_hsa: int,
        HQ_hsa: int,
        hsa_window: int,
    ) -> tuple:
        """Internal SWA on HSA heads for extend (per-token).

        For each query token t at position pos_t:
          1. Chunk-aligned window: start = max(0, ((pos_t - window + 1) // PS) * PS)
          2. Range [chunk_start, pos_t], excluding LMK positions
          3. Compute softmax(q @ K^T * sm_scale) @ V → output + LSE
          4. Aggregate LSE across GQA groups → per-kv-head LSE

        Returns
        -------
        swa_o : [T, HQ_hsa, D] float32
        lse_hq : [T, HQ_hsa] float32  (R77: per-q-head; caller aggregates if needed)
        """
        T, _, D = q_hsa.shape
        device = q_hsa.device

        swa_o = torch.zeros((T, HQ_hsa, D), device=device, dtype=torch.float32)
        lse_hq_empty = torch.full((T, HQ_hsa), float("-inf"), device=device, dtype=torch.float32)

        if hsa_window <= 0:
            return swa_o, lse_hq_empty

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None or md.token_positions is None or md.token_to_seq_id is None:
            return swa_o, lse_hq_empty

        k_cache_all = pool.get_key_buffer(layer.layer_id)
        v_cache_all = pool.get_value_buffer(layer.layer_id)
        token_to_seq_id = md.token_to_seq_id

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        page_size = int(self.page_size)
        sm_scale = float(getattr(layer, "scaling", 1.0))

        engine_indices = md.engine_indices
        if engine_indices is None:
            return swa_o, lse_hq_empty

        # Per-q-head LSE (h_q-shape) for the per-q-head fusion path.
        lse_hq = torch.full((T, HQ_hsa), float("-inf"), device=device, dtype=torch.float32)

        for t in range(T):
            eng_idx = int(engine_indices[t].item())
            b = int(token_to_seq_id[t].item())

            # Chunk-aligned window start using ENGINE index.
            raw_start = eng_idx - hsa_window + 1
            chunk_start = max(0, (raw_start // page_size) * page_size) if raw_start >= 0 else 0

            # Engine indices [chunk_start, eng_idx], excluding LMK slots.
            # LMK exclusion: (engine_idx + 1) % page_size == 0
            tok_pos = torch.arange(chunk_start, eng_idx + 1, device=device, dtype=torch.int64)
            keep = ((tok_pos + 1) % page_size) != 0
            tok_pos = tok_pos[keep]
            if tok_pos.numel() == 0:
                continue

            token_locs = page_table_1[b, tok_pos].to(torch.int64)
            q_hgd = q_hsa[t].view(H_hsa, Gh, D).to(torch.float32)

            lse_per_q = torch.full((H_hsa, Gh), float("-inf"), device=device, dtype=torch.float32)
            for kv_h in range(H_hsa):
                kv_h_global = H_swa + kv_h
                k_win = k_cache_all[token_locs, kv_h_global, :].to(torch.float32)
                v_win = v_cache_all[token_locs, kv_h_global, :].to(torch.float32)
                logits = (q_hgd[kv_h] @ k_win.transpose(0, 1)) * sm_scale
                lse_per_q[kv_h] = torch.logsumexp(logits, dim=-1)
                p = torch.softmax(logits, dim=-1)
                o = p @ v_win
                hq_start = kv_h * Gh
                swa_o[t, hq_start : hq_start + Gh, :] = o

            # Per-q-head LSE for the per-q-head fusion path (matches the
            # batched variant's stash so the fusion code finds it).
            lse_hq[t] = lse_per_q.reshape(HQ_hsa)

        # R77: return per-q-head LSE (matches _batched API); caller lazily
        # aggregates to per-kv-head if it needs lse_kv.
        lse_hq_f32 = lse_hq.contiguous()
        self._last_swa_lse_hq_extend = lse_hq_f32
        return swa_o, lse_hq_f32

    def _run_selection_extend(
        self,
        *,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        selection_q: Optional[torch.Tensor] = None,
        page_table_1: torch.Tensor,
        kv_head_offset: int = 0,
        kv_head_count: Optional[int] = None,
        hsa_window: int = 0,
        split_info: Optional[dict] = None,
    ) -> None:
        """Dispatch to batched (production) or reference implementation."""
        if self._USE_EXTEND_REFERENCE:
            return self._run_selection_extend_reference(
                q=q, layer=layer, forward_batch=forward_batch,
                selection_q=selection_q, page_table_1=page_table_1,
                kv_head_offset=kv_head_offset, kv_head_count=kv_head_count,
                hsa_window=hsa_window,
            )
        return self._run_selection_extend_batched(
            q=q, layer=layer, forward_batch=forward_batch,
            selection_q=selection_q, page_table_1=page_table_1,
            kv_head_offset=kv_head_offset, kv_head_count=kv_head_count,
            hsa_window=hsa_window,
            split_info=split_info,
        )

    def _run_selection_extend_batched(
        self,
        *,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        selection_q: Optional[torch.Tensor] = None,
        split_info: Optional[dict] = None,
        page_table_1: torch.Tensor,
        kv_head_offset: int = 0,
        kv_head_count: Optional[int] = None,
        hsa_window: int = 0,
    ) -> None:
        """Memory-efficient extend selection via shared LMK + online_topk_group.

        复用 training 的 online_topk_group kernel 调用方式：
        只 gather 一次共享的 LMK [1, S, H, D]，然后调用
        online_topk_group(B=1, L=T, is_causal=True, q_offset=prefix_len)，
        kernel 内部自动处理 causal mask + head group sum。
        内存从 O(T × C_max × H × D) 降到 O(S × H × D)。
        当前只支持 batch=1（SGLang continuous batching 保证）。
        """
        md = self.forward_metadata
        if md is None:
            return
        if not self._is_hsa_layer(layer.layer_id):
            return

        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if pool is None or not hasattr(pool, "get_key_buffer"):
            return
        if md.token_positions is None or md.token_to_seq_id is None:
            return

        # --- 只支持 batch=1，多序列直接报错暴露问题 ---
        extend_seq_lens = md.extend_seq_lens
        extend_prefix_lens = md.extend_prefix_lens
        assert extend_seq_lens is not None and extend_prefix_lens is not None, \
            "extend_seq_lens and extend_prefix_lens must be set for extend mode"
        B = int(extend_seq_lens.shape[0])
        assert B == 1, (
            f"_run_selection_extend_batched 当前只支持 batch=1，"
            f"但收到 B={B}。SGLang continuous batching 应保证 extend 阶段 B=1。"
        )

        T = int(md.token_positions.shape[0])
        H_sel = int(kv_head_count) if kv_head_count is not None else int(layer.tp_k_head_num)
        page_size = int(self.page_size)
        device = q.device

        k_cache = pool.get_key_buffer(layer.layer_id)  # [num_locs, H_total, D]
        D = int(k_cache.shape[2])

        q_sel = selection_q if selection_q is not None else q
        if q_sel.dim() == 2:
            HQ_sel = q_sel.shape[1] // D
            q_sel_3 = q_sel.view(T, HQ_sel, D)
        else:
            q_sel_3 = q_sel
            HQ_sel = int(q_sel_3.shape[1])

        # --- 计算 prefix_len 和总序列长度 ---
        # R64: shared per-prefill scalar cache (was 2 .item() + 1 req_idx .item() per layer).
        _req_idx_cached, prefix_len, _extend_len_cached = self._get_prefill_scalars(forward_batch)
        total_seq_len = prefix_len + _extend_len_cached  # prefix + extend
        S = total_seq_len // page_size  # 总 LMK chunk 数

        if S == 0:
            md.hsa_ext_selected_page_ids = torch.full(
                (T, H_sel, int(self.hsa_topk)), -1, device=device, dtype=torch.int32
            )
            md.hsa_ext_selected_scores = torch.full(
                (T, H_sel, int(self.hsa_topk)), float("-inf"), device=device, dtype=torch.float32
            )
            return

        # Per-q-head lmk_k path mirrors the decode-side fast-path: when the
        # HSA layer has stashed chunk-aggregated h_q-headed lmk_k into the
        # LandmarkLmkKPool (mode triggered by hq_hsa > h_hsa, see flash_hsa.py
        # `_maybe_write_chunk_lmk_k`), gather those instead of synthesising
        # an h_kv-shape surrogate from the chunk's last-token K. The layer's
        # writer runs BEFORE this call (attn() comes after the write hook in
        # the layer's forward), so all S chunks of the current prefill are
        # already in the pool.
        per_qhead_ext_active = (
            getattr(self, "lmk_k_pool", None) is not None
            and getattr(self, "req_to_chunk_pool", None) is not None
            and split_info is not None
            and int(split_info.get("hq_hsa", 0)) > int(split_info.get("h_hsa", 0))
        )
        self._per_qhead_G_ext = None
        self._per_qhead_lmk_keys_full = None  # cache for the per-token-aware pytorch path
        if per_qhead_ext_active:
            # R79: cache (S, slots_i64) once per prefill — slots only depends
            # on (req_idx, S), both invariant within a prefill. The int64
            # cast lets get/get_prior_b take their no-clamp fast path. Saves
            # per-layer arange + tensor(req_idx) + gather_slots dispatches.
            slots_cache = getattr(forward_batch, "_hsa_prefill_chunk_slots_S", None)
            if slots_cache is None or slots_cache[0] != S:
                req_idx = _req_idx_cached  # R64 cached
                chunk_ids = torch.arange(S, device=device, dtype=torch.int32).unsqueeze(0)
                req_idx_t = torch.tensor([req_idx], dtype=torch.int32, device=device)
                slots = (
                    self.req_to_chunk_pool.gather_slots(req_idx_t, chunk_ids)[0]
                    .to(torch.int64)
                )
                forward_batch._hsa_prefill_chunk_slots_S = (S, slots)
            else:
                slots = slots_cache[1]
            lmk_keys_full = self.lmk_k_pool.get(int(layer.layer_id), slots)  # [S, h_q, D]
            self._per_qhead_lmk_keys_full = lmk_keys_full
            self._per_qhead_prior_b_ext = self.lmk_k_pool.get_prior_b(int(layer.layer_id), slots)  # [S, h_q]
            self._per_qhead_G_ext = int(split_info["hq_hsa"]) // int(split_info["h_hsa"])
            # Surface to the kernel as h_kv-shaped (mean over G) so the topk
            # kernel runs in shared-K mode (no shape mismatch).  The pytorch
            # post-pass below replaces the kernel's output with the
            # algorithmically correct max-over-G per-q-head topk.
            h_q_e = lmk_keys_full.shape[1]
            G_e = self._per_qhead_G_ext
            h_kv_e = h_q_e // G_e
            lmk_keys = lmk_keys_full.view(S, h_kv_e, G_e, lmk_keys_full.shape[-1]).mean(dim=2)
            H_sel = h_kv_e
        else:
            # --- 1. 只 gather 一次共享的 LMK [S, H, D] ---
            lmk_positions = torch.arange(S, device=device, dtype=torch.int64) * page_size + (page_size - 1)  # [S]
            lmk_positions = lmk_positions.clamp(max=page_table_1.shape[1] - 1)
            lmk_locs = page_table_1[0, lmk_positions].to(torch.int64)  # [S]
            lmk_keys = k_cache[lmk_locs]  # [S, H_total, D]
            if kv_head_count is not None:
                lmk_keys = lmk_keys[:, int(kv_head_offset):int(kv_head_offset) + int(kv_head_count), :]
            # lmk_keys: [S, H_sel, D]

        # --- 2. unified_retrieval 降维（跟 training 一致）---
        unified = split_info is not None and split_info.get("unified_retrieval", False)
        retrieval_dim = split_info.get("retrieval_dim", None) if split_info is not None else None
        if unified and retrieval_dim is not None:
            tp_size = get_attention_tp_size()
            if tp_size > 1:
                # All-gather KV head landmark keys across TP ranks
                lmk_keys = tensor_model_parallel_all_gather(lmk_keys.contiguous(), dim=1)
                H_sel = lmk_keys.shape[1]  # updated to full H_hsa
                # All-gather sel_q across TP ranks
                q_sel_3 = tensor_model_parallel_all_gather(q_sel_3.contiguous(), dim=-1)
            # 跟 training 一致: rearrange('S H D -> S 1 (H D)')
            lmk_keys = lmk_keys.reshape(S, 1, H_sel * D)  # [S, 1, H_sel*D]
            H_sel_kernel = 1
            D_kernel = H_sel * D  # = retrieval_dim (e.g. 1024)
        else:
            H_sel_kernel = H_sel
            D_kernel = D

        # --- 3. 调用 online_topk_group（跟 training 一致）---
        assert _online_topk_group is not None, (
            "Fused online_topk_group kernel 未加载，无法执行 extend selection。"
            "请确保 hsa-kernel-main/ops/topk_group.py 可正常导入。"
        )

        # Q: [T, HQ_sel, D] → [1, T, HQ_sel, D]
        # 对于 unified_retrieval，HQ_sel=1, D=retrieval_dim
        q_4d = q_sel_3.unsqueeze(0)  # [1, T, HQ_sel, D_sel]
        lmk_4d = lmk_keys.unsqueeze(0)  # [1, S, H_sel_kernel, D_kernel]

        # q_offset = prefix_len，表示 extend 的第一个 token 在全局序列中的位置
        if self._per_qhead_G_ext is not None and self._per_qhead_lmk_keys_full is not None:
            # Per-q-head path. Two implementations:
            #   (a) headwise_topk_softmax=True (default, back-compat) — pure-PyTorch
            #       max-over-G + topk. Slow (fp32 SIMT einsum) but kept as the safe
            #       reference. Trained models that already use softmax-topk get
            #       bit-equivalent results; H100 prefill is dominated by this path.
            #   (b) headwise_topk_softmax=False — call the fused max-pooling kernel
            #       (`online_topk_head` from topk_head_maxpool) which skips hsa_lse
            #       and takes swa_lse directly. ~2x faster prefill topk per upstream.
            #       Uses _last_swa_lse_hq_extend that the SWA branch (step 3 in
            #       forward_extend) just produced.
            use_maxpool = (
                not self.headwise_topk_softmax
                and _online_topk_head_maxpool is not None
                and self._last_swa_lse_hq_extend is not None
            )

            G_e = int(self._per_qhead_G_ext)
            lmk_full = self._per_qhead_lmk_keys_full  # [S, h_q, D]
            h_q_e = lmk_full.shape[1]
            h_kv_e = h_q_e // G_e
            D_e = lmk_full.shape[2]

            if use_maxpool:
                # ---- Fast path: fused max-pooling kernel (skips hsa_lse) ----
                # Diagnostic counter (read by tests to confirm wire-up).
                self._maxpool_call_count = getattr(self, "_maxpool_call_count", 0) + 1
                # q_sel_3: [T, h_q, D] -> [1, T, h_q, D]
                # lmk_full: [S, h_q, D] -> [1, S, h_q, D]
                # lse_swa:  [T, h_q]   -> [1, T, h_q]
                q_4d_mp = q_sel_3.unsqueeze(0).contiguous()
                lmk_4d_mp = lmk_full.unsqueeze(0).contiguous()
                lse_swa_arg = self._last_swa_lse_hq_extend.unsqueeze(0).contiguous()
                # Returns indices [1, T, h_kv, K] int32, scores [1, T, h_q, K] raw scaled qk.
                fused_indices_mp, scores_hq_mp = _online_topk_head_maxpool(
                    q_4d_mp,
                    lmk_4d_mp,
                    topk=int(self.hsa_topk),
                    block_size=page_size,
                    window_size=hsa_window if hsa_window > 0 else 0,
                    is_causal=True,
                    G=G_e,
                    lse_swa=lse_swa_arg,
                    q_offset=prefix_len,
                    is_training=False,
                )
                fused_indices = fused_indices_mp.to(torch.int32)         # [1, T, h_kv, K]
                # R81: keep scores in their native dtype (bf16 from maxpool).
                # fused_chunk_weight_per_qhead_decode now accepts bf16 scores
                # via SCORES_IS_BF16 — saves the per-layer fp32 cast.
                scores_hq_sel_t = scores_hq_mp.squeeze(0)  # [T, h_q, K] bf16
                T_q_mp = q_sel_3.shape[0]
                K_eff = int(fused_indices_mp.shape[-1])
                # h_kv-level fused_scores: max over G of per-q-head scores at selected K
                # (matching the slow path's topk_scores_kv semantics).
                scores_kv_sel_t = (
                    scores_hq_sel_t.view(T_q_mp, h_kv_e, G_e, K_eff).max(dim=2).values
                )
                fused_scores = scores_kv_sel_t.unsqueeze(0).to(torch.float32)
                self._last_per_qhead_scores_extend = scores_hq_sel_t
            else:
                # ---- Pure-PyTorch fallback (default, headwise_topk_softmax=True) ----
                import math as _m
                sm_scale_ref = 1.0 / _m.sqrt(D_e)
                # q_sel_3 here is the full h_q query [T, h_q, D] (per_qhead path
                # is non-unified so q_sel_3 stays at the original h_q layout).
                q_full = q_sel_3.float()                                   # [T, h_q, D]
                lmk_full_f = lmk_full.float()                              # [S, h_q, D]
                # Per-q-head scores: einsum(t,h,d ; s,h,d -> t,h,s)
                scores_pqh = torch.einsum("thd,shd->ths", q_full, lmk_full_f) * sm_scale_ref
                # NOTE: prior_b deliberately not added (see selector.py for the
                # rationale — adding entropy bias hurts alignment until the
                # chunk_attn_pool intermediate matches the official exactly).
                # Causal: query at global pos p sees chunks with end_pos < p.
                # chunk c's end_pos = (c+1)*page_size - 1.  Visible iff end_pos < p
                # i.e. c < p // page_size.  When sliding window, also exclude chunks
                # whose start_pos >= p - window_size + 1 -> c >= (p - window + 1) / page_size.
                T_q = q_full.shape[0]
                S_k = lmk_full_f.shape[0]
                q_pos = torch.arange(prefix_len, prefix_len + T_q, device=device)
                chunk_end_pos = torch.arange(1, S_k + 1, device=device) * page_size - 1
                chunk_start_pos = torch.arange(0, S_k, device=device) * page_size
                visible_causal = chunk_end_pos.unsqueeze(0) < q_pos.unsqueeze(1)         # [T, S]
                if hsa_window > 0:
                    outside_window = chunk_start_pos.unsqueeze(0) < (q_pos.unsqueeze(1) - hsa_window + 1)
                    visible = visible_causal & outside_window
                else:
                    visible = visible_causal
                # Max over G: [T, h_q, S] -> [T, h_kv, S] for selection.
                scores_kv_sel = scores_pqh.view(T_q, h_kv_e, G_e, S_k).max(dim=2).values
                scores_kv_sel = scores_kv_sel.masked_fill(~visible.unsqueeze(1), float("-inf"))
                eff_topk = min(int(self.hsa_topk), S_k)
                _, topk_idx_kv = scores_kv_sel.topk(eff_topk, dim=-1)                # [T, h_kv, K]
                topk_idx_kv, _ = torch.sort(topk_idx_kv, dim=-1)
                # Per-q-head scores at selected chunks (broadcast h_kv idx to h_q).
                topk_idx_hq = topk_idx_kv.unsqueeze(2).expand(T_q, h_kv_e, G_e, eff_topk).reshape(
                    T_q, h_q_e, eff_topk
                )
                scores_hq_sel = torch.gather(scores_pqh, dim=-1, index=topk_idx_hq.clamp_min(0))
                invalid = topk_idx_kv < 0
                topk_scores_kv = torch.gather(scores_kv_sel, dim=-1, index=topk_idx_kv.clamp_min(0))
                topk_scores_kv = topk_scores_kv.masked_fill(invalid, float("-inf"))
                scores_hq_sel = scores_hq_sel.masked_fill(
                    invalid.unsqueeze(2).expand(T_q, h_kv_e, G_e, eff_topk).reshape(T_q, h_q_e, eff_topk),
                    float("-inf"),
                )
                if eff_topk < int(self.hsa_topk):
                    pad = int(self.hsa_topk) - eff_topk
                    topk_idx_kv = torch.cat(
                        [topk_idx_kv,
                         topk_idx_kv.new_full((T_q, h_kv_e, pad), -1, dtype=topk_idx_kv.dtype)],
                        dim=-1,
                    )
                    topk_scores_kv = torch.cat(
                        [topk_scores_kv,
                         topk_scores_kv.new_full((T_q, h_kv_e, pad), float("-inf"))],
                        dim=-1,
                    )
                    scores_hq_sel = torch.cat(
                        [scores_hq_sel,
                         scores_hq_sel.new_full((T_q, h_q_e, pad), float("-inf"))],
                        dim=-1,
                    )
                # Match kernel output layout: fused_indices [1, T, h_kv, K], fused_scores same.
                fused_indices = topk_idx_kv.unsqueeze(0).to(torch.int32)
                fused_scores = topk_scores_kv.unsqueeze(0).to(torch.float32)
                # Stash per-q-head scores for the fusion path: [T, h_q, K]
                self._last_per_qhead_scores_extend = scores_hq_sel.to(torch.float32)
        else:
            fused_indices, fused_scores = _online_topk_group(
                q=q_4d,
                lmks=lmk_4d,
                topk=int(self.hsa_topk),
                block_size=page_size,
                window_size=hsa_window if hsa_window > 0 else 0,
                is_causal=True,
                q_offset=prefix_len,
                is_training=False,
            )
        # fused_indices: [1, T, h_shared, topk], fused_scores: [1, T, h_shared, topk]

        # --- 4. 转换输出格式 ---
        # squeeze batch dim: [T, h_shared, topk]
        selected_page_ids = fused_indices.squeeze(0).to(torch.int32)
        selected_scores = fused_scores.squeeze(0).to(torch.float32)

        # kernel 返回的 index 是 chunk index（0-based），即 page_id
        # 无效位（-1）保持不变
        selected_page_ids = selected_page_ids.masked_fill(selected_page_ids < 0, -1)

        # 如果 unified_retrieval，h_shared=1，需要 repeat 到 H_sel（原始 kv head 数）
        if unified and retrieval_dim is not None:
            orig_H_sel = int(kv_head_count) if kv_head_count is not None else int(layer.tp_k_head_num)
            if selected_page_ids.shape[1] != orig_H_sel:
                selected_page_ids = selected_page_ids.repeat_interleave(
                    orig_H_sel // selected_page_ids.shape[1], dim=1
                )
                selected_scores = selected_scores.repeat_interleave(
                    orig_H_sel // selected_scores.shape[1], dim=1
                )

        md.hsa_ext_selected_page_ids = selected_page_ids  # [T, H_sel, K]
        md.hsa_ext_selected_scores = selected_scores  # [T, H_sel, K]

    def _run_selection_extend_reference(
        self,
        *,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        selection_q: Optional[torch.Tensor] = None,
        page_table_1: torch.Tensor,
        kv_head_offset: int = 0,
        kv_head_count: Optional[int] = None,
        hsa_window: int = 0,
    ) -> None:
        """Per-token top-k page selection for extend (reference implementation).

        For each query token t at position pos_t in sequence b:
          1. Completed pages before pos_t: page_ids 0..(pos_t // page_size - 1)
          2. Exclude pages within the SWA window (chunk-aligned)
          3. Load LMK keys and score
          4. TopK → selected_page_ids[t, H, K], scores[t, H, K]
        """
        md = self.forward_metadata
        if md is None:
            return
        if not self._is_hsa_layer(layer.layer_id):
            return

        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if pool is None or not hasattr(pool, "get_key_buffer"):
            return
        if md.token_positions is None or md.token_to_seq_id is None:
            return

        T = int(md.token_positions.shape[0])
        H_sel = int(kv_head_count) if kv_head_count is not None else int(layer.tp_k_head_num)
        page_size = int(self.page_size)
        token_positions = md.token_positions
        token_to_seq_id = md.token_to_seq_id
        device = q.device

        k_cache = pool.get_key_buffer(layer.layer_id)  # [num_locs, H_total, D]
        D = int(k_cache.shape[2])
        sm_scale = getattr(layer, "scaling", None)

        q_sel = selection_q if selection_q is not None else q
        if q_sel.dim() == 2:
            HQ_sel = q_sel.shape[1] // D
            q_sel_3 = q_sel.view(T, HQ_sel, D)
        else:
            q_sel_3 = q_sel
            HQ_sel = int(q_sel_3.shape[1])

        all_page_ids = []
        all_scores = []

        _engine_indices = md.engine_indices
        if _engine_indices is None:
            return

        for t in range(T):
            eng_idx = int(_engine_indices[t].item())
            b = int(token_to_seq_id[t].item())

            # Completed pages using engine index.
            completed = eng_idx // page_size
            if completed <= 0:
                all_page_ids.append(
                    torch.full((1, H_sel, self.hsa_topk), -1, device=device, dtype=torch.int32)
                )
                all_scores.append(
                    torch.full((1, H_sel, self.hsa_topk), float("-inf"), device=device, dtype=torch.float32)
                )
                continue

            cand_pages = torch.arange(completed, device=device, dtype=torch.int32)

            # Exclude pages within the SWA window using engine index.
            # Official: limit_chunk = (engine_idx - window_size + 1) // block_size
            if hsa_window > 0:
                raw_start = eng_idx - hsa_window + 1
                limit_chunk = max(0, raw_start // page_size) if raw_start >= 0 else 0
                cand_pages = cand_pages[cand_pages < limit_chunk]

            C = int(cand_pages.numel())
            if C == 0:
                all_page_ids.append(
                    torch.full((1, H_sel, self.hsa_topk), -1, device=device, dtype=torch.int32)
                )
                all_scores.append(
                    torch.full((1, H_sel, self.hsa_topk), float("-inf"), device=device, dtype=torch.float32)
                )
                continue

            # Load LMK keys: K[page_id * page_size + (page_size - 1)].
            lmk_token_pos = cand_pages.to(torch.int64) * page_size + (page_size - 1)
            lmk_locs = page_table_1[b, lmk_token_pos].to(torch.int64)
            lmk_repr = k_cache[lmk_locs]  # [C, H_total, D]
            if kv_head_count is not None:
                lmk_repr = lmk_repr[:, int(kv_head_offset):int(kv_head_offset) + int(kv_head_count), :]
            # lmk_repr: [C, H_sel, D]

            # Use select_topk_pages_decode with batch_size=1.
            cand_page_ids_1 = cand_pages.unsqueeze(0)  # [1, C]
            cand_mask_1 = torch.ones((1, C), device=device, dtype=torch.bool)
            cand_repr_1 = lmk_repr.unsqueeze(0)  # [1, C, H_sel, D]

            sel = select_topk_pages_decode(
                q=q_sel_3[t:t+1],  # [1, HQ_sel, D]
                cand_page_ids=cand_page_ids_1,
                cand_mask=cand_mask_1,
                cand_chunk_repr=cand_repr_1,
                cand_chunk_repr_valid=cand_mask_1,
                topk=int(self.hsa_topk),
                selection_strategy=str(self.hsa_selection_strategy),
                sm_scale=sm_scale,
            )
            all_page_ids.append(sel.selected_page_ids)  # [1, H_sel, K]
            all_scores.append(sel.selected_scores)  # [1, H_sel, K]

        md.hsa_ext_selected_page_ids = torch.cat(all_page_ids, dim=0)  # [T, H_sel, K]
        md.hsa_ext_selected_scores = torch.cat(all_scores, dim=0)  # [T, H_sel, K]

    def _hsa_sparse_attn_extend(
        self,
        *,
        q_hsa: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        selected_page_ids: torch.Tensor,
        hsa_weights: torch.Tensor,
        H_hsa: int,
        HQ_hsa: int,
        sm_scale: Optional[float] = None,
        swa_o_inner: Optional[torch.Tensor] = None,
        swa_w_q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dispatch to Triton kernel (production) or reference implementation."""
        if self._USE_EXTEND_REFERENCE:
            out = self._hsa_sparse_attn_extend_reference(
                q_hsa=q_hsa, k_cache=k_cache, v_cache=v_cache,
                page_table_1=page_table_1, selected_page_ids=selected_page_ids,
                hsa_weights=hsa_weights, H_hsa=H_hsa, HQ_hsa=HQ_hsa,
                sm_scale=sm_scale,
            )
            if swa_o_inner is not None and swa_w_q is not None:
                out = torch.addcmul(out, swa_o_inner, swa_w_q[:, :, None])
            return out
        # R37: Q-batched extend kernel (collapses (T, HQ) grid to (T/BLOCK_M, HQ)).
        md = self.forward_metadata
        import os as _os
        return hsa_extend_paged_fwd(
            q=q_hsa,                              # [T, HQ_hsa, D]
            k_cache=k_cache,                       # [Nloc, H_hsa, D]
            v_cache=v_cache,                       # [Nloc, H_hsa, D]
            page_table_1=page_table_1,             # [B, MAX_T]
            selected_page_ids=selected_page_ids,   # [T, H_hsa, K]
            hsa_weights=hsa_weights,               # [T, HQ_hsa, K]
            page_size=int(self.page_size),
            sm_scale=sm_scale,
            mask_last_token=True,
            token_to_seq_id=md.token_to_seq_id,    # [T] int32
            block_m=int(_os.environ.get("HSA_SPARSE_BM", 1)),
            num_warps=int(_os.environ.get("HSA_SPARSE_NW", 2)),
            num_stages=int(_os.environ.get("HSA_SPARSE_NS", 2)),
            swa_o_inner=swa_o_inner,
            swa_w_q=swa_w_q,
        )

    def _hsa_sparse_attn_extend_reference(
        self,
        *,
        q_hsa: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        selected_page_ids: torch.Tensor,
        hsa_weights: torch.Tensor,
        H_hsa: int,
        HQ_hsa: int,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Paged HSA sparse attention for extend (per-token, PyTorch reference).

        For each query token t, for each selected page k:
          1. Load PAGE_SIZE tokens from paged cache via page_table_1
          2. softmax(q @ K^T * sm_scale) @ V, with mask_last_token
          3. Multiply by hsa_weights[t, hq, k] and accumulate

        Returns: out [T, HQ_hsa, D] bf16
        """
        T, _, D = q_hsa.shape
        device = q_hsa.device
        TOPK = int(selected_page_ids.shape[2])
        page_size = int(self.page_size)

        md = self.forward_metadata
        token_to_seq_id = md.token_to_seq_id

        if sm_scale is None:
            sm_scale = float(D) ** -0.5

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa

        out = torch.zeros((T, HQ_hsa, D), device=device, dtype=torch.float32)

        for t in range(T):
            b = int(token_to_seq_id[t].item())
            q_t = q_hsa[t].view(H_hsa, Gh, D).to(torch.float32)  # [H_hsa, Gh, D]

            for k_i in range(TOPK):
                page_id = int(selected_page_ids[t, :, k_i].max().item())  # same across H
                if page_id < 0:
                    continue

                # Token indices for this page.
                tok_pos = torch.arange(page_size, device=device, dtype=torch.int64) + page_id * page_size
                token_locs = page_table_1[b, tok_pos].to(torch.int64)

                for kv_h in range(H_hsa):
                    pid = int(selected_page_ids[t, kv_h, k_i].item())
                    if pid < 0:
                        continue
                    w = hsa_weights[t, kv_h * Gh : (kv_h + 1) * Gh, k_i].to(torch.float32)  # [Gh]

                    k_page = k_cache[token_locs, kv_h, :].to(torch.float32)  # [PS, D]
                    v_page = v_cache[token_locs, kv_h, :].to(torch.float32)  # [PS, D]

                    logits = (q_t[kv_h] @ k_page.transpose(0, 1)) * sm_scale  # [Gh, PS]
                    # Mask last token (LMK).
                    logits[:, page_size - 1] = float("-inf")
                    p = torch.softmax(logits, dim=-1)  # [Gh, PS]
                    o_page = p @ v_page  # [Gh, D]

                    hq_start = kv_h * Gh
                    out[t, hq_start : hq_start + Gh, :] += w[:, None] * o_page

        return out.to(torch.bfloat16)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        split_info = kwargs.get("hsa_split_head_info", None)
        selection_q = kwargs.get("hsa_selection_q", None)

        # Non-HSA layers or no split_info: delegate to dense.
        if not self._is_hsa_layer(layer.layer_id) or split_info is None:
            kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
            return self._dense_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs_clean
            )

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None:
            kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
            return self._dense_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs_clean
            )

        # Step 1: Save full KV to paged cache (all heads).
        if save_kv_cache:
            pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        # Extract head split info.
        HQ_swa = int(split_info.get("hq_swa", 0))
        HQ_hsa = int(split_info.get("hq_hsa", 0))
        H_swa = int(split_info.get("h_swa", 0))
        H_hsa = int(split_info.get("h_hsa", 0))
        hsa_window = int(split_info.get("swa_window_size", 0) or 0)
        upper_swa_window = int(split_info.get("upper_swa_window_size", 0) or -1)

        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError(
                "HSA extend currently assumes qk_head_dim == v_head_dim."
            )

        # Reshape q to [T, HQ, D].
        HQ_total = int(layer.tp_q_head_num)
        D = int(layer.qk_head_dim)
        T = q.shape[0]
        q3 = q.view(T, HQ_total, D)
        q_hsa = q3[:, HQ_swa:, :]   # [T, HQ_hsa, D]

        # Step 2: Run dense extend on SWA heads ONLY.
        # When HQ_swa == 0 (all heads are HSA, e.g. hsa_denom=1), skip this step.
        H_total = int(layer.tp_k_head_num)

        # R83: HSA-345M has HQ_swa==0 — skip the dense_out_3 alloc + slice-write
        # of out_hsa into it, mirroring R34's decode fast path. Per layer saves
        # 1 empty + 1 slice copy (~7us CPU + 50us CUDA each).
        dense_out_3 = (
            torch.empty((T, HQ_total, D), device=q.device, dtype=q.dtype)
            if HQ_swa > 0
            else None
        )

        if HQ_swa > 0 and H_swa > 0:
            # We call the triton kernel directly with head-sliced tensor views,
            # avoiding the `forward_extend` Python-level overhead.
            # The kernel computes `kv_group_num = q.shape[1] // k_buffer.shape[1]`, so
            # passing sliced Q (HQ_swa) and sliced KV buffer (H_swa) gives the correct GQA ratio.
            pool_k = pool.get_key_buffer(layer.layer_id)[:, :H_swa, :]
            pool_v = pool.get_value_buffer(layer.layer_id)[:, :H_swa, :]

            db = self._dense_backend
            dense_md = db.forward_metadata

            k3 = k.view(-1, H_total, D)
            v3 = v.view(-1, H_total, D)

            if db.enable_deterministic:
                # 1-stage unified kernel: uses kv_indices for both prefix + extend
                db.extend_attention_fwd_unified(
                    q3[:, :HQ_swa, :],
                    dense_out_3[:, :HQ_swa, :],
                    pool_k, pool_v,
                    dense_md.qo_indptr,
                    dense_md.kv_indptr,
                    dense_md.kv_indices,
                    forward_batch.extend_prefix_lens.to(torch.int32),
                    dense_md.max_extend_len,
                    sm_scale=layer.scaling,
                    sliding_window_size=upper_swa_window,
                )
            else:
                # 2-stage kernel: uses k_extend/v_extend + k_buffer/v_buffer separately
                kv_indptr = dense_md.kv_indptr
                kv_indices = dense_md.kv_indices
                db.extend_attention_fwd(
                    q3[:, :HQ_swa, :],
                    k3[:, :H_swa, :].contiguous(),
                    v3[:, :H_swa, :].contiguous(),
                    dense_out_3[:, :HQ_swa, :],
                    pool_k, pool_v,
                    dense_md.qo_indptr,
                    kv_indptr, kv_indices,
                    dense_md.custom_mask,
                    True,  # is_causal
                    dense_md.mask_indptr,
                    dense_md.max_extend_len,
                    sm_scale=layer.scaling,
                    sliding_window_size=upper_swa_window,
                )

        page_table_1 = md.page_table_1

        # Step 3: Internal SWA on HSA heads.
        # R77: second return is now `lse_raw` [T, HQ_hsa] (per-q-head) — the
        # legacy h_kv branch computes lse_kv on demand below.
        swa_o_inner, lse_raw = self._compute_internal_swa_extend(
            q_hsa=q_hsa,
            layer=layer,
            forward_batch=forward_batch,
            page_table_1=page_table_1,
            H_swa=H_swa,
            H_hsa=H_hsa,
            HQ_hsa=HQ_hsa,
            hsa_window=hsa_window,
        )

        # Step 4: TopK selection per token.
        self._run_selection_extend(
            q=q_hsa,
            layer=layer,
            forward_batch=forward_batch,
            selection_q=selection_q,
            page_table_1=page_table_1,
            kv_head_offset=H_swa,
            kv_head_count=H_hsa,
            hsa_window=hsa_window,
            split_info=split_info,
        )

        selected_page_ids = md.hsa_ext_selected_page_ids  # [T, H_hsa, K]
        selected_scores = md.hsa_ext_selected_scores  # [T, H_hsa, K]

        if selected_page_ids is None or selected_scores is None:
            # No selection results (e.g. all tokens at early positions).
            # HSA heads uninitialized; zero them.
            if dense_out_3 is None:
                out_hsa_zero = torch.zeros((T, HQ_hsa, D), device=q.device, dtype=q.dtype)
                return out_hsa_zero.reshape(T, HQ_total * D)
            dense_out_3[:, HQ_swa:, :] = 0
            return dense_out_3.reshape(T, HQ_total * D)

        # Step 5: Merged softmax.
        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        TOPK = int(selected_page_ids.shape[2])

        # unified_retrieval: selection returns [T, 1, K]; expand to [T, H_hsa, K]
        if selected_page_ids.shape[1] == 1 and H_hsa > 1:
            selected_page_ids = selected_page_ids.expand(T, H_hsa, TOPK).contiguous()
            selected_scores = selected_scores.expand(T, H_hsa, TOPK).contiguous()

        valid = selected_page_ids >= 0
        # Per-q-head fusion path: when the selector exposed per-q-head scores
        # AND the SWA branch exposed per-q-head LSE, do the entire
        # cat+softmax at h_q granularity (matching official Qwen3-LHSA
        # line 782).  Otherwise fall back to the h_kv path then broadcast.
        per_qhead_scores = self._last_per_qhead_scores_extend  # [T, h_q, K] or None
        per_qhead_lse = self._last_swa_lse_hq_extend           # [T, h_q] or None
        per_qhead_fusion = (
            per_qhead_scores is not None
            and per_qhead_lse is not None
            and per_qhead_scores.shape == (T, HQ_hsa, TOPK)
            and per_qhead_lse.shape == (T, HQ_hsa)
        )

        if per_qhead_fusion and hsa_window > 0:
            # R76: reuse the existing fused_chunk_weight_per_qhead_decode kernel
            # for extend too. Kernel handles valid-mask + cat(lse, [0]) +
            # softmax + nan_to_num + bf16 cast + swa_w_q split in one launch.
            # Replaces the per-layer chain:
            #   scores_4d.view -> valid_4d.expand -> masked_fill -> view -> cat(2/3)
            #   -> softmax -> nan_to_num -> slice+to(bf16)+contiguous -> slice
            # (~10 ops -> 1 kernel per layer; saves ~150 launches per 16K prefill).
            sel_pages_i32 = (
                selected_page_ids
                if selected_page_ids.dtype == torch.int32 and selected_page_ids.is_contiguous()
                else selected_page_ids.to(torch.int32).contiguous()
            )
            scores_contig = (
                per_qhead_scores
                if per_qhead_scores.is_contiguous()
                else per_qhead_scores.contiguous()
            )
            w_q, swa_w_q = fused_chunk_weight_per_qhead_decode(
                per_qhead_scores=scores_contig,
                per_qhead_lse=per_qhead_lse,
                selected_page_ids=sel_pages_i32,
                Gh=Gh,
                enable_softmax1=bool(self.enable_softmax1),
                out_dtype=q_hsa.dtype,
            )
            swa_w_kv = None
        elif hsa_window > 0:
            # R77: compute lse_kv on demand here (per_qhead branch above
            # doesn't need it). lse_raw is [T, HQ_hsa] per-q-head fp32.
            lse_kv = torch.logsumexp(lse_raw.view(T, H_hsa, Gh), dim=-1)
            scores = selected_scores.masked_fill(~valid, float("-inf"))
            if not self.enable_softmax1:
                cat_scores = torch.cat(
                    [scores, lse_kv.unsqueeze(-1)], dim=-1
                )  # [T, H_hsa, K+1]
                swa_weight_idx = -1
            else:
                cat_scores = torch.cat(
                    [
                        scores,
                        lse_kv.unsqueeze(-1),
                        torch.zeros(T, H_hsa, 1, device=scores.device, dtype=scores.dtype),
                    ],
                    dim=-1,
                )  # [T, H_hsa, K+2]
                swa_weight_idx = -2
            merged_w = torch.softmax(cat_scores, dim=-1)
            merged_w = torch.nan_to_num(merged_w, nan=0.0)
            w_kv = merged_w[:, :, :TOPK].to(q_hsa.dtype)
            swa_w_kv = merged_w[:, :, swa_weight_idx]  # [T, H_hsa]
            swa_w_q = None
            w_q = (
                w_kv[:, :, None, :]
                .expand(T, H_hsa, Gh, TOPK)
                .reshape(T, HQ_hsa, TOPK)
                .contiguous()
            )
        else:
            scores = selected_scores.masked_fill(~valid, float("-inf"))
            w_kv = torch.softmax(scores, dim=-1)
            w_kv = torch.nan_to_num(w_kv, nan=0.0).to(q_hsa.dtype)
            swa_w_kv = None
            swa_w_q = None
            w_q = (
                w_kv[:, :, None, :]
                .expand(T, H_hsa, Gh, TOPK)
                .reshape(T, HQ_hsa, TOPK)
                .contiguous()
            )

        # Step 6+7 R80: fuse the SWA blend epilogue into the HSA sparse kernel.
        # When swa_w_q is per-HQ, hsa_extend_paged_fwd does:
        #   out = bf16(acc + swa_o_inner * swa_w_q)
        # in its store epilogue — saves the standalone addcmul launch.
        # Fallback path (swa_w_kv only, hsa_window <= 0 branch) still expands
        # swa_w_kv to swa_w_q first, then addcmul outside.
        if swa_w_q is None and swa_w_kv is not None:
            swa_w_q = (
                swa_w_kv[:, :, None]
                .expand(T, H_hsa, Gh)
                .reshape(T, HQ_hsa)
                .contiguous()
            )
        # R81: kernel internally casts loaded swa_w_q to fp32 (see hsa_extend.py
        # BLEND_SWA epilogue). Pass it in its native fp32 dtype — saves a
        # per-layer .to(bf16) cast.
        blend_swa_in_kernel = swa_w_q is not None
        swa_w_q_for_kernel = swa_w_q
        if blend_swa_in_kernel and not swa_w_q_for_kernel.is_contiguous():
            swa_w_q_for_kernel = swa_w_q_for_kernel.contiguous()
        k_cache_hsa = pool.get_key_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :]
        v_cache_hsa = pool.get_value_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :]
        out_hsa = self._hsa_sparse_attn_extend(
            q_hsa=q_hsa,
            k_cache=k_cache_hsa,
            v_cache=v_cache_hsa,
            page_table_1=page_table_1,
            selected_page_ids=selected_page_ids,
            hsa_weights=w_q,
            H_hsa=H_hsa,
            HQ_hsa=HQ_hsa,
            sm_scale=getattr(layer, "scaling", None),
            swa_o_inner=swa_o_inner if blend_swa_in_kernel else None,
            swa_w_q=swa_w_q_for_kernel if blend_swa_in_kernel else None,
        )  # [T, HQ_hsa, D] bf16, SWA blend already applied if requested

        # Step 8: Write HSA heads into the pre-allocated output tensor.
        # R83: when HQ_swa==0, out_hsa already covers all heads — return
        # directly without the empty+slice-write.
        if dense_out_3 is None:
            return out_hsa.reshape(T, HQ_total * D)
        dense_out_3[:, HQ_swa:, :] = out_hsa.to(dense_out_3.dtype)
        return dense_out_3.reshape(T, HQ_total * D)
