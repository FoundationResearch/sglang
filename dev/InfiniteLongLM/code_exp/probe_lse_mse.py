"""Probe MSE between estimated chunk LSE Z'_c and the true chunk LSE Z_c
on a real HSA OLMo checkpoint, bucketed by per-query topk selection rank.

This script does NOT modify any modeling code.  It monkey-patches each
``LandmarkHSA`` layer's ``topk_func`` and ``hsa_func`` instance attributes
so we can:

  1. capture ``scores`` (raw scaled qk, shape [B, L, h_q, K]) and
     ``indices`` (shape [B, L, h_kv, K]) returned by the topk kernel,
     plus the per-q-head entropy bias ``prior_b`` passed in;
  2. recompute the *real* per-chunk LSE on the q/k that actually enter
     the HSA kernel (via the wrapped ``hsa_func``), mirroring
     ``hsa_torch_ref`` style: gather the chunk's S keys, compute scaled
     qk, mask the last token, then logsumexp;
  3. accumulate per-(layer, rank) statistics, where ``rank`` is the
     intra-query order obtained by sorting Z'_c = scores + gathered
     prior_b in descending order (the score actually used at
     selection / final HSA softmax).

Group-wise topk path is intentionally NOT supported.

Loading mirrors ``code_exp/compare_olmo_hsa_with_trm.py``; data loading
mirrors ``eval/eval_ppl.py`` (numpy olmo3 dataset).

Example:
python code_exp/probe_lse_mse.py --max_seq_len 8192 --max_samples 4

"""

import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange
from torch.utils import data
from torch.utils.data import SequentialSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Trigger model registration side effects (mirrors eval_ppl.py).
import models  # noqa: F401  pylint: disable=unused-import

from data import build_numpy_dataset
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks


# ---------------------------------------------------------------------------
# Model registration / loading (mirrors compare_olmo_hsa_with_trm.py).
# ---------------------------------------------------------------------------
def register_hsa_model(model_type="olmo_lhsa"):
    from models.FlashHSA.configuration_hsa import HSAConfig

    if model_type == "qwen_lhsa":
        from models.FlashHSA.modeling_qwen_lhsa import HSAForCausalLM
    else:
        from models.FlashHSA.modeling_olmo_lhsa import HSAForCausalLM

    HSAConfig.model_type = "olmo_lhsa"
    AutoConfig.register("olmo_lhsa", HSAConfig, exist_ok=True)
    HSAForCausalLM.config_class = HSAConfig
    AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM, exist_ok=True)
    return HSAConfig, HSAForCausalLM


def parse_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_hsa_model(args, device, dtype):
    register_hsa_model(model_type=args.model_type)
    # The HF checkpoint ships its own ``config.json`` (with ``model_type ==
    # 'olmo_lhsa'`` and all HSA-specific fields like ``layerwise_qk_norm``),
    # so we just let ``from_pretrained`` pick it up directly.  An external
    # ``--hsa_config`` is optional and only used to override the on-disk
    # config when explicitly provided.
    if args.hsa_config:
        config = AutoConfig.from_pretrained(args.hsa_config)
    else:
        config = AutoConfig.from_pretrained(args.ckpt_path)
    if getattr(config, "insert_landmarks", True):
        config.auto_insert_lmk = True

    kwargs = {"torch_dtype": dtype}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if device.type == "cuda":
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt_path, config=config, **kwargs,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()
    if hasattr(model, "_gen_state"):
        model._gen_state.reset()
    return model


# ---------------------------------------------------------------------------
# LSE probe: monkey-patches every LandmarkHSA layer in the model.
# ---------------------------------------------------------------------------
class LSEProbe:
    """Accumulates per-(layer, rank) statistics over many forward passes.

    Stats per bucket:
        n          : number of (b, l, h_q) samples
        sse        : sum of (Z'-Z)^2
        sae        : sum of |Z'-Z|
        sum_pred   : sum of Z'
        sum_true   : sum of Z
        sse_pred   : sum of Z'^2
        sse_true   : sum of Z^2
        cross      : sum of Z' * Z
        sum_smape  : sum of 2*|tanh((Z'-Z)/2)|  (== sMAPE_exp per sample)

    Additional KL stats (only for queries with K_eff == K, i.e. full topk view):
        kl_stats[layer_idx] = {"n_query": int, "sum_kl": float}
        where each query contributes one scalar KL(P_true || P_pred) over the
        K-chunk softmax distribution.
    """

    def __init__(self, model, case_limit=5, case_layers=None, case_layer_window=2):
        self.stats = defaultdict(lambda: {
            "n": 0, "sse": 0.0, "sae": 0.0,
            "sum_pred": 0.0, "sum_true": 0.0,
            "sse_pred": 0.0, "sse_true": 0.0,
            "cross": 0.0,
            "sum_smape": 0.0,
        })
        # Global KL stats (only filled when K_eff == K for the query).
        # Aggregated across all layers and queries.
        self.kl_stats = {"n_query": 0, "sum_kl": 0.0}
        # Top8 query-level filter stats, aggregated across all layers.
        self.top8_filter_stats = defaultdict(lambda: {
            "n_query": 0,
            "n_chunk": 0,
            "sum_rel_log_pct": 0.0,
            "sum_smape": 0.0,
        })
        self.top8_all_active_rank_stats = defaultdict(lambda: {
            "n": 0,
            "sum_z_rel_err_pct": 0.0,
        })
        # Per-(layer, head) stats for heatmap visualization.
        self.per_layer_head_stats = defaultdict(lambda: {
            "n": 0,
            "sum_z_rel_err_pct": 0.0,
            "sum_smape": 0.0,
        })
        # Per-(layer, head) sMAPE stats WITHOUT Z_pred - Z_swa > 0 filter.
        self.per_layer_head_smape_stats = defaultdict(lambda: {
            "n": 0,
            "sum_smape": 0.0,
        })
        # ---- Per-(layer, head) distribution probe (no filter) -----------
        # All buckets accumulate over top8 chunks of queries with K_eff == K
        # and full top8 valid. NO Z_pred-Z_swa>0 filter, NO Z_true>0 filter.
        # Histogram bins for (Z_true - Z_swa):
        #   (-inf, 0], (0, 1], (1, 2], (2, 5], (5, +inf)
        # Histogram bins for P_true:
        #   [0, 1e-3], (1e-3, 1e-2], (1e-2, 1e-1], (1e-1, 1]
        self._dz_edges = [0.0, 1.0, 2.0, 5.0]   # 5 buckets
        self._pt_edges = [1e-3, 1e-2, 1e-1]     # 4 buckets
        self.per_layer_head_dist_stats = defaultdict(lambda: {
            "n": 0,
            # marginal sums for (Z_true - Z_swa) and P_true.
            "sum_dz": 0.0, "sum_dz_sq": 0.0,
            "sum_pt": 0.0, "sum_pt_sq": 0.0,
            # tail-fraction counts (chunks with dz > thr).
            "n_dz_gt_0": 0, "n_dz_gt_1": 0, "n_dz_gt_2": 0, "n_dz_gt_5": 0,
            # tail-fraction counts (chunks with P_true > thr).
            "n_pt_gt_1em3": 0, "n_pt_gt_1em2": 0, "n_pt_gt_1em1": 0,
            # Joint bucket: dz-bucket -> (count, sum_smape, sum_abs_diff, sum_pt).
            "dz_bucket_n":         [0] * 5,
            "dz_bucket_sum_smape": [0.0] * 5,
            "dz_bucket_sum_adiff": [0.0] * 5,
            "dz_bucket_sum_pt":    [0.0] * 5,
            # Joint bucket: pt-bucket -> (count, sum_smape, sum_abs_diff).
            "pt_bucket_n":         [0] * 4,
            "pt_bucket_sum_smape": [0.0] * 4,
            "pt_bucket_sum_adiff": [0.0] * 4,
            # Per-query P_swa_total stats (one value per kept query).
            # P_swa_total = exp(Z_swa) / (sum_c exp(Z_c) + exp(Z_swa))
            #            = total final-attention weight that goes to SWA.
            # Small P_swa_total => retrieval channel dominates the output.
            "n_query": 0,
            "sum_pswa": 0.0, "sum_pswa_sq": 0.0,
            # tail-fraction counts (queries with P_swa_total > thr).
            "n_pswa_gt_0p9": 0, "n_pswa_gt_0p7": 0, "n_pswa_gt_0p5": 0,
            # tail-fraction counts (queries with P_swa_total < thr).
            "n_pswa_lt_0p5": 0, "n_pswa_lt_0p3": 0, "n_pswa_lt_0p1": 0,
            # sMAPE on top8 chunks restricted to retrieval-dominant queries
            # (those with P_swa_total < 0.5). This is the most physically
            # meaningful prediction-error indicator: prior errors on these
            # queries directly perturb the final attention output.
            "n_chunk_retrdom": 0,
            "sum_smape_retrdom": 0.0,
            "sum_adiff_retrdom": 0.0,
        })
        # A few concrete examples for sanity-checking actual predicted/true values.
        self.case_limit = int(case_limit)
        self.cases = []
        self._case_counts = defaultdict(int)
        self._requested_case_layers = case_layers
        self.case_layer_window = int(case_layer_window)
        self.case_layers = set()
        # Per-layer scratch: topk wrapper writes, hsa wrapper consumes.
        self._scratch = {}
        self._patched_layers = []
        self._install(model)
        self._select_case_layers()

    def _select_case_layers(self):
        layer_indices = [int(getattr(layer, "layer_idx", id(layer)))
                         for layer in self._patched_layers]
        layer_indices = sorted(set(layer_indices))
        if not layer_indices:
            self.case_layers = set()
            return
        if self._requested_case_layers:
            self.case_layers = {int(x) for x in self._requested_case_layers}
            return
        w = max(0, self.case_layer_window)
        if w == 0:
            self.case_layers = set()
            return
        n = len(layer_indices)
        mid = n // 2
        mid_start = max(0, mid - w // 2)
        mid_layers = layer_indices[mid_start:mid_start + w]
        last_layers = layer_indices[-w:]
        self.case_layers = set(mid_layers + last_layers)

    # ---- installation ------------------------------------------------------
    def _install(self, model):
        for _, mod in model.named_modules():
            if mod.__class__.__name__ != "LandmarkHSA":
                continue
            if getattr(mod, "groupwise_topk", False):
                # Group-wise path is not supported by this probe.
                raise RuntimeError(
                    "LSEProbe does not support groupwise_topk=True layers."
                )
            self._wrap_one(mod)
            self._patched_layers.append(mod)

    def _wrap_one(self, layer):
        layer_idx = getattr(layer, "layer_idx", id(layer))
        orig_topk = layer.topk_func
        orig_hsa = layer.hsa_func
        probe = self

        # ---- topk wrapper: capture indices, scores, prior_b ---------------
        def topk_wrap(lmk_q_norm, lmk_k, lse_sum, topk, **kwargs):
            indices, scores = orig_topk(
                lmk_q_norm, lmk_k, lse_sum, topk, **kwargs,
            )
            probe._scratch[layer_idx] = {
                "indices": indices.detach(),
                "scores": scores.detach(),
                "lse_sum": lse_sum.detach(),
                "bias": (None if kwargs.get("bias", None) is None
                         else kwargs["bias"].detach()),
            }
            return indices, scores

        # ---- hsa wrapper: real q/k available; compute true LSE ------------
        def hsa_wrap(q, k, v, **kwargs):
            sc = probe._scratch.pop(layer_idx, None)
            if sc is not None:
                indices = kwargs.get("indices", None)
                block_size = kwargs.get("block_size", None)
                mask_last = kwargs.get("mask_last_token", True)
                if indices is not None and block_size is not None:
                    with torch.no_grad():
                        probe._accumulate(
                            layer_idx=layer_idx,
                            q=q, k=k,
                            indices=indices,
                            scratch=sc,
                            sm_scale=float(layer.scaling),
                            chunk_size=int(block_size),
                            mask_last_token=bool(mask_last),
                        )
            return orig_hsa(q, k, v, **kwargs)

        layer.topk_func = topk_wrap
        layer.hsa_func = hsa_wrap

    # ---- accumulation ------------------------------------------------------
    def _accumulate(self, layer_idx, q, k, indices, scratch,
                    sm_scale, chunk_size, mask_last_token):
        """Compute Z'_c and Z_c on the captured forward pass and update stats.

        Args:
            q          : [B, L, h_q, D]    -- final HSA q (rope or nope)
            k          : [B, kv_len, h_kv, D]
            indices    : [B, L, h_kv, K]   -- KV-head topk selection
            scratch    : dict from topk_wrap
            sm_scale   : float, == layer.scaling == 1/sqrt(D)
            chunk_size : int
            mask_last_token : whether to mask chunk's last position (matches
                              the HSA kernel call in lhsa_layer.forward).
        """
        scores = scratch["scores"]   # [B, L, h_q, K]  raw scaled qk
        lse_sum = scratch["lse_sum"]  # [B, L, h_q] or [B, L, h_kv]
        bias = scratch["bias"]       # [B, N, h_q] or None
        S = chunk_size

        B, Lq, h_q, D = q.shape
        _, kv_len, h_kv, _ = k.shape
        if h_q % h_kv != 0:
            raise RuntimeError(f"h_q ({h_q}) not divisible by h_kv ({h_kv})")
        G = h_q // h_kv
        K = indices.shape[-1]

        Z_swa = lse_sum.float()
        if Z_swa.dim() != 3:
            raise RuntimeError(f"Expected lse_sum to be 3-D, got shape {tuple(Z_swa.shape)}")
        if Z_swa.shape[2] == h_kv and G > 1:
            Z_swa = Z_swa.repeat_interleave(G, dim=2)
        if Z_swa.shape != (B, Lq, h_q):
            raise RuntimeError(
                f"Unexpected lse_sum shape after h_q alignment: {tuple(Z_swa.shape)}, "
                f"expected {(B, Lq, h_q)}"
            )

        N = kv_len // S
        if N == 0:
            return
        valid_kv = N * S
        # k_chunks: [B, N, S, h_kv, D]
        k_chunks = rearrange(k[:, :valid_kv], "b (n s) h d -> b n s h d", s=S)

        # valid mask in KV-head layout, then broadcast to h_q.
        valid_kv_layout = (indices >= 0)                                # [B, L, h_kv, K]
        if G > 1:
            valid_hq = valid_kv_layout.repeat_interleave(G, dim=2)      # [B, L, h_q, K]
            indices_hq = indices.repeat_interleave(G, dim=2)            # [B, L, h_q, K]
        else:
            valid_hq = valid_kv_layout
            indices_hq = indices

        if not valid_hq.any():
            return

        # ---- Build Z_pred = scores + gathered prior_b -------------------
        Z_pred = scores.float()                                         # [B, L, h_q, K]
        if bias is not None:
            prior_b = bias.float()                                      # [B, N, h_q]
            idx_for_bias = indices_hq.clamp_min(0).long()               # [B, L, h_q, K]
            src = prior_b.unsqueeze(-1).expand(-1, -1, -1, K)           # [B, N, h_q, K]
            gathered = torch.gather(src, dim=1, index=idx_for_bias)     # [B, L, h_q, K]
            Z_pred = Z_pred + gathered

        # ---- Build Z_true: real per-chunk LSE on (q, k) ------------------
        # Layout: convert k_chunks from per-h_kv to per-h_q gather buffer
        # by indexing along chunk axis with KV-head indices, then later
        # repeat to G in the GQA inner-product.
        # Strategy: for memory friendliness, gather chunk-by-chunk via
        # advanced indexing in the (N) axis on k_chunks per (b, h_kv).
        #
        # Reorder k_chunks to [B, h_kv, N, S, D] for easier gathering.
        k_perm = k_chunks.permute(0, 3, 1, 2, 4).contiguous()           # [B, h_kv, N, S, D]
        # Build gather index of shape [B, h_kv, L, K, S, D] from the
        # KV-layout indices.
        idx_kv = indices.clamp_min(0).long()                            # [B, L, h_kv, K]
        idx_kv = idx_kv.permute(0, 2, 1, 3).contiguous()                # [B, h_kv, L, K]
        idx_g = idx_kv[:, :, :, :, None, None].expand(-1, -1, -1, -1, S, D)
        # Expand k_perm with an L axis (broadcast), then gather along N.
        k_src = k_perm.unsqueeze(2).expand(-1, -1, Lq, -1, -1, -1)      # [B, h_kv, L, N, S, D]
        k_gather = torch.gather(k_src, dim=3, index=idx_g)              # [B, h_kv, L, K, S, D]
        # qk per q-head: q reshape [B, h_kv, G, L, D] (and broadcast K).
        q5 = q.view(B, Lq, h_kv, G, D).permute(0, 2, 3, 1, 4)           # [B, h_kv, G, L, D]
        # einsum: 'b H G L D, b H L K S D -> b H G L K S'
        qk = torch.einsum("bHGLD,bHLKSD->bHGLKS", q5.float(), k_gather.float())
        qk = qk * float(sm_scale)
        if mask_last_token:
            qk[..., -1] = float("-inf")
        Z_true = torch.logsumexp(qk, dim=-1)                            # [B, h_kv, G, L, K]
        # -> [B, L, h_q, K]
        Z_true = Z_true.permute(0, 3, 1, 2, 4).reshape(B, Lq, h_q, K)

        # ---- Re-rank by Z_pred descending; bucket = last-axis index ------
        Z_pred_for_rank = Z_pred.masked_fill(~valid_hq, float("-inf"))
        _, perm = torch.sort(Z_pred_for_rank, dim=-1, descending=True)  # [B, L, h_q, K]
        Zp = torch.gather(Z_pred, -1, perm)
        Zt = torch.gather(Z_true, -1, perm)
        vd = torch.gather(valid_hq.to(torch.int8), -1, perm).bool()

        # ---- Sample all ranks for the last valid query before reducing ----
        if self.case_limit > 0 and int(layer_idx) in self.case_layers \
                and self._case_counts[int(layer_idx)] < self.case_limit:
            logp_true_sorted = torch.log_softmax(Zt.masked_fill(~vd, float("-inf")), dim=-1)
            logp_pred_sorted = torch.log_softmax(Zp.masked_fill(~vd, float("-inf")), dim=-1)
            chunk_idx_sorted = torch.gather(indices_hq, -1, perm)
            query_has_valid = vd.any(dim=-1)                            # [B, L, h_q]
            coords = query_has_valid.nonzero(as_tuple=False)
            if coords.numel() > 0:
                max_query = int(coords[:, 1].max().item())
                last_query_coords = coords[coords[:, 1] == max_query]
                b, lq, h = last_query_coords[0].tolist()
                valid_ranks = vd[b, lq, h].nonzero(as_tuple=False).flatten()
                for r_tensor in valid_ranks.tolist():
                    r = int(r_tensor)
                    diff_val = float((Zp[b, lq, h, r] - Zt[b, lq, h, r]).item())
                    self.cases.append({
                        "layer": int(layer_idx),
                        "batch": int(b),
                        "query": int(lq),
                        "head": int(h),
                        "rank": int(r + 1),
                        "Z_pred": float(Zp[b, lq, h, r].item()),
                        "Z_true": float(Zt[b, lq, h, r].item()),
                        "diff": diff_val,
                        "smape_exp": float((2.0 * math.tanh(abs(diff_val) * 0.5))),
                        "P_pred": float(logp_pred_sorted[b, lq, h, r].exp().item()),
                        "P_true": float(logp_true_sorted[b, lq, h, r].exp().item()),
                        "chunk_idx": int(chunk_idx_sorted[b, lq, h, r].item()),
                        "K_eff": int(vd[b, lq, h].sum().item()),
                    })
                self._case_counts[int(layer_idx)] += 1

        # ---- Per-rank reductions over (B, L, h_q) ------------------------
        err = (Zp - Zt)
        for r in range(K):
            m = vd[..., r]
            n = int(m.sum().item())
            if n == 0:
                continue
            e = err[..., r][m]
            p = Zp[..., r][m]
            t = Zt[..., r][m]
            # sMAPE_exp per sample = 2 * |e^d - 1| / (e^d + 1) = 2 * |tanh(d/2)|.
            # The tanh form is bounded in [0, 2] and numerically stable for any d.
            smape = 2.0 * (e * 0.5).tanh().abs()
            s = self.stats[(int(layer_idx), r)]
            s["n"] += n
            s["sse"] += float((e * e).sum().item())
            s["sae"] += float(e.abs().sum().item())
            s["sum_pred"] += float(p.sum().item())
            s["sum_true"] += float(t.sum().item())
            s["sse_pred"] += float((p * p).sum().item())
            s["sse_true"] += float((t * t).sum().item())
            s["cross"] += float((p * t).sum().item())
            s["sum_smape"] += float(smape.sum().item())

        # ---- Top8 query-level filters based on Z_pred - Z_swa ------------
        # Mode 1: keep only queries whose whole top8 is active, then count top8.
        # Mode 2: keep queries with at least one active chunk in top8, then count top8.
        k_eff = valid_hq.sum(dim=-1)                                   # [B, L, h_q]
        full_mask = (k_eff == K)
        topn = min(8, K)
        top_valid = vd[..., :topn].all(dim=-1)                          # [B, L, h_q]
        top_active = (Zp[..., :topn] - Z_swa.unsqueeze(-1)) > 0
        all_active_query = full_mask & top_valid & top_active.all(dim=-1)
        any_active_query = full_mask & top_valid & top_active.any(dim=-1)
        top_e = err[..., :topn]
        top_t = Zt[..., :topn]
        top_smape = 2.0 * (top_e * 0.5).tanh().abs()
        # Only positive Z_true gives an interpretable signed-denominator ratio.
        top_pos_true = top_t > 0
        top_rel_log_pct = top_e.abs() / top_t * 100.0
        for mode, q_mask in (
            ("top8_all_active", all_active_query),
            ("top8_any_active", any_active_query),
        ):
            m = q_mask.unsqueeze(-1).expand(-1, -1, -1, topn) & top_pos_true
            kept_query = m.any(dim=-1)
            n_query = int(kept_query.sum().item())
            n_chunk = int(m.sum().item())
            if n_query == 0 or n_chunk == 0:
                continue
            s = self.top8_filter_stats[mode]
            s["n_query"] += n_query
            s["n_chunk"] += n_chunk
            s["sum_rel_log_pct"] += float(top_rel_log_pct[m].sum().item())
            s["sum_smape"] += float(top_smape[m].sum().item())

        if all_active_query.any():
            for r in range(topn):
                rank_mask = all_active_query & top_pos_true[..., r]
                vals = top_rel_log_pct[..., r][rank_mask]
                s = self.top8_all_active_rank_stats[r]
                s["n"] += int(vals.numel())
                s["sum_z_rel_err_pct"] += float(vals.sum().item())

            # Per-(layer, head) accumulation for heatmap.
            # For each head, gather all top8 chunks from all-active queries
            # that also have positive Z_true.
            for h in range(h_q):
                h_q_mask = all_active_query[:, :, h]                     # [B, L]
                if not h_q_mask.any():
                    continue
                h_pos = top_pos_true[:, :, h, :]                        # [B, L, topn]
                h_vals = top_rel_log_pct[:, :, h, :]                    # [B, L, topn]
                h_smape = top_smape[:, :, h, :]                         # [B, L, topn]
                chunk_mask = h_q_mask.unsqueeze(-1).expand_as(h_vals) & h_pos
                n_h = int(chunk_mask.sum().item())
                if n_h == 0:
                    continue
                s = self.per_layer_head_stats[(int(layer_idx), h)]
                s["n"] += n_h
                s["sum_z_rel_err_pct"] += float(h_vals[chunk_mask].sum().item())
                s["sum_smape"] += float(h_smape[chunk_mask].sum().item())

        # ---- Per-(layer, head) sMAPE without all-active filter -----------
        # Only requires K_eff == K and top8 valid (no Z_pred - Z_swa > 0 gate).
        base_query_mask = full_mask & top_valid                         # [B, L, h_q]
        if base_query_mask.any():
            for h in range(h_q):
                h_mask = base_query_mask[:, :, h]                       # [B, L]
                if not h_mask.any():
                    continue
                h_smape_vals = top_smape[:, :, h, :]                    # [B, L, topn]
                chunk_mask = h_mask.unsqueeze(-1).expand_as(h_smape_vals)
                n_h = int(chunk_mask.sum().item())
                if n_h == 0:
                    continue
                s = self.per_layer_head_smape_stats[(int(layer_idx), h)]
                s["n"] += n_h
                s["sum_smape"] += float(h_smape_vals[chunk_mask].sum().item())

        # ---- Per-(layer, head) distribution probe ------------------------
        # Same gating as the smape block above (full topk + top8 valid),
        # NO Z_pred-Z_swa>0 filter. For every retained top8 chunk, compute:
        #   * dz   = Z_true - Z_swa            (chunk influence in log-space)
        #   * P_true = softmax([Z_true_1..K, Z_swa])[chunk]   (final attn weight)
        # Then accumulate marginal moments / tail counts and joint
        # (sMAPE, |diff|) split by dz-bucket and pt-bucket.
        if base_query_mask.any():
            # Build sorted Z_true for the K+1 softmax (chunk + SWA) the same
            # way the KL block does, but using the rank-sorted layout so we
            # can index top8 directly.
            Zt_full = torch.cat(                                        # [B, L, h_q, K+1]
                [Zt, Z_swa.unsqueeze(-1)], dim=-1,
            )
            # Mask invalid chunk slots so they don't poison softmax.
            Zt_full_masked = Zt_full.clone()
            full_valid_mask = torch.cat(
                [vd, torch.ones_like(vd[..., :1])], dim=-1,
            )
            Zt_full_masked = Zt_full_masked.masked_fill(
                ~full_valid_mask, float("-inf"),
            )
            P_true_full = torch.softmax(Zt_full_masked, dim=-1)         # [B, L, h_q, K+1]
            P_true_top = P_true_full[..., :topn]                        # [B, L, h_q, topn]
            # Per-query total SWA weight in the K+1 softmax. This is the
            # physical answer to "how much does the retrieval channel
            # matter for this query": small P_swa_total => retrieval
            # dominates and prior errors hit the output hard.
            P_swa_total = P_true_full[..., -1]                          # [B, L, h_q]
            dz_top = (Zt[..., :topn] - Z_swa.unsqueeze(-1)).float()     # [B, L, h_q, topn]
            adiff_top = top_e.abs().float()                             # [B, L, h_q, topn]
            smape_top = top_smape.float()                               # [B, L, h_q, topn]

            dz_edges = self._dz_edges
            pt_edges = self._pt_edges

            for h in range(h_q):
                h_mask = base_query_mask[:, :, h]                       # [B, L]
                if not h_mask.any():
                    continue
                qm = h_mask.unsqueeze(-1).expand(-1, -1, topn)          # [B, L, topn]
                dz_h    = dz_top[:, :, h, :][qm]                        # [N_h]
                pt_h    = P_true_top[:, :, h, :][qm]                    # [N_h]
                smape_h = smape_top[:, :, h, :][qm]                     # [N_h]
                adiff_h = adiff_top[:, :, h, :][qm]                     # [N_h]
                n_h = int(dz_h.numel())
                if n_h == 0:
                    continue
                s = self.per_layer_head_dist_stats[(int(layer_idx), h)]
                s["n"] += n_h
                s["sum_dz"]    += float(dz_h.sum().item())
                s["sum_dz_sq"] += float((dz_h * dz_h).sum().item())
                s["sum_pt"]    += float(pt_h.sum().item())
                s["sum_pt_sq"] += float((pt_h * pt_h).sum().item())
                s["n_dz_gt_0"] += int((dz_h > 0.0).sum().item())
                s["n_dz_gt_1"] += int((dz_h > 1.0).sum().item())
                s["n_dz_gt_2"] += int((dz_h > 2.0).sum().item())
                s["n_dz_gt_5"] += int((dz_h > 5.0).sum().item())
                s["n_pt_gt_1em3"] += int((pt_h > 1e-3).sum().item())
                s["n_pt_gt_1em2"] += int((pt_h > 1e-2).sum().item())
                s["n_pt_gt_1em1"] += int((pt_h > 1e-1).sum().item())
                # ---- Per-query P_swa_total stats for this head -------
                pswa_h = P_swa_total[:, :, h][h_mask].float()           # [Q_h]
                n_q_h = int(pswa_h.numel())
                if n_q_h > 0:
                    s["n_query"] += n_q_h
                    s["sum_pswa"]    += float(pswa_h.sum().item())
                    s["sum_pswa_sq"] += float((pswa_h * pswa_h).sum().item())
                    s["n_pswa_gt_0p9"] += int((pswa_h > 0.9).sum().item())
                    s["n_pswa_gt_0p7"] += int((pswa_h > 0.7).sum().item())
                    s["n_pswa_gt_0p5"] += int((pswa_h > 0.5).sum().item())
                    s["n_pswa_lt_0p5"] += int((pswa_h < 0.5).sum().item())
                    s["n_pswa_lt_0p3"] += int((pswa_h < 0.3).sum().item())
                    s["n_pswa_lt_0p1"] += int((pswa_h < 0.1).sum().item())
                    # Retrieval-dominant subset: P_swa_total < 0.5.
                    retrdom_query = (P_swa_total[:, :, h] < 0.5) & h_mask  # [B, L]
                    if retrdom_query.any():
                        rqm = retrdom_query.unsqueeze(-1).expand(-1, -1, topn)
                        smape_rd = smape_top[:, :, h, :][rqm]
                        adiff_rd = adiff_top[:, :, h, :][rqm]
                        n_rd = int(smape_rd.numel())
                        if n_rd > 0:
                            s["n_chunk_retrdom"] += n_rd
                            s["sum_smape_retrdom"] += float(smape_rd.sum().item())
                            s["sum_adiff_retrdom"] += float(adiff_rd.sum().item())
                # dz-bucket index in [0, 4]: edges = [0, 1, 2, 5].
                dz_bidx = torch.bucketize(
                    dz_h, torch.tensor(dz_edges, device=dz_h.device, dtype=dz_h.dtype),
                    right=True,
                )
                # pt-bucket index in [0, 3]: edges = [1e-3, 1e-2, 1e-1].
                pt_bidx = torch.bucketize(
                    pt_h, torch.tensor(pt_edges, device=pt_h.device, dtype=pt_h.dtype),
                    right=True,
                )
                for b_i in range(5):
                    sel = (dz_bidx == b_i)
                    if not sel.any():
                        continue
                    s["dz_bucket_n"][b_i]         += int(sel.sum().item())
                    s["dz_bucket_sum_smape"][b_i] += float(smape_h[sel].sum().item())
                    s["dz_bucket_sum_adiff"][b_i] += float(adiff_h[sel].sum().item())
                    s["dz_bucket_sum_pt"][b_i]    += float(pt_h[sel].sum().item())
                for b_i in range(4):
                    sel = (pt_bidx == b_i)
                    if not sel.any():
                        continue
                    s["pt_bucket_n"][b_i]         += int(sel.sum().item())
                    s["pt_bucket_sum_smape"][b_i] += float(smape_h[sel].sum().item())
                    s["pt_bucket_sum_adiff"][b_i] += float(adiff_h[sel].sum().item())

        # ---- Per-query KL(P_true || P_pred), restricted to K_eff == K ----
        # The downstream HSA softmax is over K retrieval chunks plus one SWA
        # score, so KL is computed on [Z_chunk_1..K, Z_swa].
        if full_mask.any():
            Z_swa_full = Z_swa.unsqueeze(-1)
            Z_true_full = torch.cat([Z_true, Z_swa_full], dim=-1)
            Z_pred_full = torch.cat([Z_pred, Z_swa_full], dim=-1)
            log_p = torch.log_softmax(Z_true_full, dim=-1)             # true log-probs
            log_q = torch.log_softmax(Z_pred_full, dim=-1)             # pred log-probs
            p_prob = log_p.exp()
            # KL(P||Q) = sum_k P_k * (log P_k - log Q_k); 0*log0 := 0.
            kl_terms = torch.where(
                p_prob > 0,
                p_prob * (log_p - log_q),
                torch.zeros_like(p_prob),
            )
            kl_per_query = kl_terms.sum(dim=-1)                        # [B, L, h_q]
            kl_sel = kl_per_query[full_mask]
            self.kl_stats["n_query"] += int(kl_sel.numel())
            self.kl_stats["sum_kl"] += float(kl_sel.sum().item())

    # ---- reporting ---------------------------------------------------------
    def report(self):
        """Aggregate raw sums into per-rank metrics (over all layers)."""
        # Collect all (layer, rank) buckets observed.
        ranks_seen = set()
        layers_seen = set()
        for (lyr, r), s in self.stats.items():
            if s["n"] == 0:
                continue
            ranks_seen.add(r)
            layers_seen.add(lyr)

        # Aggregated-over-layers per-rank.
        per_rank = []
        for r in sorted(ranks_seen):
            n = sse = sae = 0
            sp = st = sse_p = sse_t = cr = 0.0
            sum_smape = 0.0
            for lyr in layers_seen:
                s = self.stats.get((lyr, r))
                if s is None or s["n"] == 0:
                    continue
                n += s["n"]
                sse += s["sse"]; sae += s["sae"]
                sp += s["sum_pred"]; st += s["sum_true"]
                sse_p += s["sse_pred"]; sse_t += s["sse_true"]
                cr += s["cross"]
                sum_smape += s.get("sum_smape", 0.0)
            if n == 0:
                continue
            mp = sp / n; mt = st / n
            var_p = sse_p / n - mp * mp
            var_t = sse_t / n - mt * mt
            cov = cr / n - mp * mt
            denom = math.sqrt(max(var_p, 0.0) * max(var_t, 0.0))
            pearson = (cov / denom) if denom > 0 else float("nan")
            per_rank.append({
                "rank": r + 1, "n": n,
                "mse": sse / n, "rmse": math.sqrt(sse / n),
                "mae": sae / n,
                "bias": mp - mt,
                "mean_pred": mp, "mean_true": mt,
                "pearson": pearson,
                "smape_exp": sum_smape / n,
            })

        # Global KL (only for K_eff == K queries), aggregated over all layers.
        n_q = self.kl_stats["n_query"]
        global_kl = {
            "n_query": n_q,
            "mean_kl": (self.kl_stats["sum_kl"] / n_q) if n_q > 0 else float("nan"),
        }

        top8_filters = {}
        for mode, s in self.top8_filter_stats.items():
            n_chunk = s["n_chunk"]
            top8_filters[mode] = {
                "n_query": s["n_query"],
                "n_chunk": n_chunk,
                "mean_rel_log_pct": (
                    s["sum_rel_log_pct"] / n_chunk if n_chunk > 0 else float("nan")
                ),
                "mean_smape_exp": (
                    s["sum_smape"] / n_chunk if n_chunk > 0 else float("nan")
                ),
            }

        top8_all_active_per_rank = []
        for r in sorted(self.top8_all_active_rank_stats):
            s = self.top8_all_active_rank_stats[r]
            n = s["n"]
            top8_all_active_per_rank.append({
                "rank": r + 1,
                "n": n,
                "z_rel_err_pct": (
                    s["sum_z_rel_err_pct"] / n if n > 0 else float("nan")
                ),
            })

        # Per-(layer, head) aggregation for heatmap.
        per_layer_head = []
        for (lyr, h), s in self.per_layer_head_stats.items():
            n = s["n"]
            if n == 0:
                continue
            per_layer_head.append({
                "layer": lyr, "head": h, "n": n,
                "z_rel_err_pct": s["sum_z_rel_err_pct"] / n,
                "smape_exp": s["sum_smape"] / n,
            })
        per_layer_head.sort(key=lambda x: x["z_rel_err_pct"])

        # Per-(layer, head) sMAPE (no all-active filter).
        per_layer_head_smape = []
        for (lyr, h), s in self.per_layer_head_smape_stats.items():
            n = s["n"]
            if n == 0:
                continue
            per_layer_head_smape.append({
                "layer": lyr, "head": h, "n": n,
                "smape_exp": s["sum_smape"] / n,
            })
        per_layer_head_smape.sort(key=lambda x: x["smape_exp"])

        # Per-(layer, head) distribution probe (no Z_pred-Z_swa>0 filter).
        per_layer_head_dist = []
        dz_edges = self._dz_edges
        pt_edges = self._pt_edges
        dz_labels = (
            ["(-inf, 0]"]
            + [f"({dz_edges[i]:g}, {dz_edges[i + 1]:g}]" for i in range(len(dz_edges) - 1)]
            + [f"({dz_edges[-1]:g}, +inf)"]
        )
        pt_labels = (
            [f"[0, {pt_edges[0]:g}]"]
            + [f"({pt_edges[i]:g}, {pt_edges[i + 1]:g}]" for i in range(len(pt_edges) - 1)]
            + [f"({pt_edges[-1]:g}, 1]"]
        )
        for (lyr, h), s in self.per_layer_head_dist_stats.items():
            n = s["n"]
            if n == 0:
                continue
            mean_dz = s["sum_dz"] / n
            var_dz = max(s["sum_dz_sq"] / n - mean_dz * mean_dz, 0.0)
            mean_pt = s["sum_pt"] / n
            var_pt = max(s["sum_pt_sq"] / n - mean_pt * mean_pt, 0.0)
            n_q = s.get("n_query", 0)
            if n_q > 0:
                mean_pswa = s["sum_pswa"] / n_q
                var_pswa = max(
                    s["sum_pswa_sq"] / n_q - mean_pswa * mean_pswa, 0.0
                )
                std_pswa = math.sqrt(var_pswa)
                frac_pswa_gt_0p9 = s["n_pswa_gt_0p9"] / n_q
                frac_pswa_gt_0p7 = s["n_pswa_gt_0p7"] / n_q
                frac_pswa_gt_0p5 = s["n_pswa_gt_0p5"] / n_q
                frac_pswa_lt_0p5 = s["n_pswa_lt_0p5"] / n_q
                frac_pswa_lt_0p3 = s["n_pswa_lt_0p3"] / n_q
                frac_pswa_lt_0p1 = s["n_pswa_lt_0p1"] / n_q
            else:
                mean_pswa = float("nan"); std_pswa = float("nan")
                frac_pswa_gt_0p9 = float("nan")
                frac_pswa_gt_0p7 = float("nan")
                frac_pswa_gt_0p5 = float("nan")
                frac_pswa_lt_0p5 = float("nan")
                frac_pswa_lt_0p3 = float("nan")
                frac_pswa_lt_0p1 = float("nan")
            n_rd = s.get("n_chunk_retrdom", 0)
            mean_smape_retrdom = (
                s["sum_smape_retrdom"] / n_rd if n_rd > 0 else float("nan")
            )
            mean_adiff_retrdom = (
                s["sum_adiff_retrdom"] / n_rd if n_rd > 0 else float("nan")
            )
            entry = {
                "layer": lyr, "head": h, "n": n,
                "mean_dz": mean_dz, "std_dz": math.sqrt(var_dz),
                "mean_pt": mean_pt, "std_pt": math.sqrt(var_pt),
                "frac_dz_gt_0": s["n_dz_gt_0"] / n,
                "frac_dz_gt_1": s["n_dz_gt_1"] / n,
                "frac_dz_gt_2": s["n_dz_gt_2"] / n,
                "frac_dz_gt_5": s["n_dz_gt_5"] / n,
                "frac_pt_gt_1em3": s["n_pt_gt_1em3"] / n,
                "frac_pt_gt_1em2": s["n_pt_gt_1em2"] / n,
                "frac_pt_gt_1em1": s["n_pt_gt_1em1"] / n,
                # Per-query P_swa_total marginal + tail counts.
                "n_query": n_q,
                "mean_pswa": mean_pswa, "std_pswa": std_pswa,
                "frac_pswa_gt_0p9": frac_pswa_gt_0p9,
                "frac_pswa_gt_0p7": frac_pswa_gt_0p7,
                "frac_pswa_gt_0p5": frac_pswa_gt_0p5,
                "frac_pswa_lt_0p5": frac_pswa_lt_0p5,
                "frac_pswa_lt_0p3": frac_pswa_lt_0p3,
                "frac_pswa_lt_0p1": frac_pswa_lt_0p1,
                # Retrieval-dominant subset (P_swa_total < 0.5) sMAPE / |diff|.
                "n_chunk_retrdom": n_rd,
                "mean_smape_retrdom": mean_smape_retrdom,
                "mean_abs_diff_retrdom": mean_adiff_retrdom,
                "dz_buckets": [],
                "pt_buckets": [],
            }
            for b_i in range(5):
                bn = s["dz_bucket_n"][b_i]
                entry["dz_buckets"].append({
                    "label": dz_labels[b_i],
                    "n": bn,
                    "frac": (bn / n) if n > 0 else 0.0,
                    "mean_smape": (s["dz_bucket_sum_smape"][b_i] / bn) if bn > 0 else float("nan"),
                    "mean_abs_diff": (s["dz_bucket_sum_adiff"][b_i] / bn) if bn > 0 else float("nan"),
                    "mean_pt": (s["dz_bucket_sum_pt"][b_i] / bn) if bn > 0 else float("nan"),
                })
            for b_i in range(4):
                bn = s["pt_bucket_n"][b_i]
                entry["pt_buckets"].append({
                    "label": pt_labels[b_i],
                    "n": bn,
                    "frac": (bn / n) if n > 0 else 0.0,
                    "mean_smape": (s["pt_bucket_sum_smape"][b_i] / bn) if bn > 0 else float("nan"),
                    "mean_abs_diff": (s["pt_bucket_sum_adiff"][b_i] / bn) if bn > 0 else float("nan"),
                })
            per_layer_head_dist.append(entry)
        per_layer_head_dist.sort(key=lambda x: (x["layer"], x["head"]))

        return {
            "per_rank": per_rank,
            "global_kl": global_kl,
            "top8_filters": top8_filters,
            "top8_all_active_per_rank": top8_all_active_per_rank,
            "per_layer_head": per_layer_head,
            "per_layer_head_smape": per_layer_head_smape,
            "per_layer_head_dist": per_layer_head_dist,
            "cases": list(self.cases),
        }


# ---------------------------------------------------------------------------
# Pretty printers.
# ---------------------------------------------------------------------------
def _fmt_row(cells, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def print_cases(cases):
    if not cases:
        return
    print("\n[Cases] concrete predicted vs. true chunk-LSE examples")
    headers = [
        "layer", "query", "head", "rank", "chunk", "K_eff",
        "Z_pred", "Z_true", "diff", "sMAPE_exp", "P_pred", "P_true",
    ]
    widths = [7, 7, 6, 6, 7, 7, 12, 12, 12, 12, 12, 12]
    print(_fmt_row(headers, widths))
    for c in cases:
        print(_fmt_row([
            c["layer"], c["query"], c["head"], c["rank"], c["chunk_idx"],
            c["K_eff"], f"{c['Z_pred']:.6f}", f"{c['Z_true']:.6f}",
            f"{c['diff']:.6f}", f"{c['smape_exp']:.6f}",
            f"{c['P_pred']:.6f}", f"{c['P_true']:.6f}",
        ], widths))


def print_per_rank(per_rank):
    print("\n[Aggregated over layers] per-rank metrics")
    headers = ["rank", "count", "MSE", "RMSE", "MAE", "sMAPE_exp"]
    widths = [6, 12, 12, 12, 12, 12]
    print(_fmt_row(headers, widths))
    for r in per_rank:
        print(_fmt_row([
            r["rank"], r["n"],
            f"{r['mse']:.6f}", f"{r['rmse']:.6f}", f"{r['mae']:.6f}",
            f"{r.get('smape_exp', float('nan')):.6f}",
        ], widths))


def print_kl(global_kl):
    print("\n[KL(P_true || P_pred)] global per-query KL over K+1 softmax "
          "(restricted to K_eff == K queries, aggregated across all layers)")
    headers = ["n_query", "mean_KL"]
    widths = [12, 14]
    print(_fmt_row(headers, widths))
    print(_fmt_row([
        global_kl["n_query"],
        f"{global_kl['mean_kl']:.6f}"
        if global_kl["mean_kl"] == global_kl["mean_kl"] else "nan",
    ], widths))


def print_top8_filters(top8_filters):
    if not top8_filters:
        return
    print("\n[Top8 query-filtered metrics] full topk queries only; each kept query counts all top8 chunks")
    headers = ["mode", "n_query", "n_chunk", "Z-RelErr (%)", "sMAPE_exp"]
    widths = [18, 12, 12, 20, 12]
    print(_fmt_row(headers, widths))
    for mode in ("top8_all_active", "top8_any_active"):
        s = top8_filters.get(mode)
        if s is None:
            continue
        rel = s["mean_rel_log_pct"]
        smape = s["mean_smape_exp"]
        print(_fmt_row([
            mode,
            s["n_query"],
            s["n_chunk"],
            f"{rel:.6f}" if rel == rel else "nan",
            f"{smape:.6f}" if smape == smape else "nan",
        ], widths))


def print_top8_all_active_per_rank(rows):
    if not rows:
        return
    print("\n[Top8 all-active] per-rank Z-RelErr (%)")
    headers = ["rank", "count", "Z-RelErr (%)"]
    widths = [6, 12, 16]
    print(_fmt_row(headers, widths))
    for row in rows:
        v = row["z_rel_err_pct"]
        print(_fmt_row([
            row["rank"],
            row["n"],
            f"{v:.6f}" if v == v else "nan",
        ], widths))


def print_per_layer_head(per_layer_head, top_n=20):
    """Print sorted per-(layer, head) Z-RelErr stats."""
    if not per_layer_head:
        return
    print(f"\n[Per-layer-head] top8 all-active Z-RelErr (%), sorted ascending (showing top/bottom {top_n})")
    headers = ["rank", "layer", "head", "n", "Z-RelErr (%)", "sMAPE_exp"]
    widths = [6, 7, 6, 10, 16, 12]
    print(_fmt_row(headers, widths))
    show = per_layer_head[:top_n]
    for i, row in enumerate(show):
        print(_fmt_row([
            i + 1, row["layer"], row["head"], row["n"],
            f"{row['z_rel_err_pct']:.4f}",
            f"{row['smape_exp']:.6f}",
        ], widths))
    if len(per_layer_head) > 2 * top_n:
        print("  ...")
        show_bottom = per_layer_head[-top_n:]
        for i, row in enumerate(show_bottom):
            idx = len(per_layer_head) - top_n + i + 1
            print(_fmt_row([
                idx, row["layer"], row["head"], row["n"],
                f"{row['z_rel_err_pct']:.4f}",
                f"{row['smape_exp']:.6f}",
            ], widths))


def print_per_layer_head_dist(per_layer_head_dist, max_rows=None):
    """Print per-(layer, head) distribution probe.

    For each head we print three blocks:
        1. marginal stats:  mean/std of dz=Z_true-Z_swa and P_true plus
           tail-fractions (fraction of top8 chunks with dz>0/1/2/5 and
           P_true>1e-3/1e-2/1e-1).
        2. joint stats split by dz-bucket: per-bucket count, fraction,
           mean(sMAPE), mean(|diff|), mean(P_true).  Tells us whether
           prediction error correlates with chunk influence.
        3. joint stats split by P_true-bucket: per-bucket count, fraction,
           mean(sMAPE), mean(|diff|).  This is the most physical view -
           large-weight chunks dominate the final attention output.
    """
    if not per_layer_head_dist:
        return
    rows = per_layer_head_dist
    if max_rows is not None:
        rows = rows[:max_rows]

    print("\n[Per-layer-head][Distribution] dz=Z_true-Z_swa and P_true over top8 chunks")
    print("  (gating: K_eff==K and top8 valid, NO Z_pred-Z_swa>0 filter)")

    # ---- 1) Marginal table ----------------------------------------------
    print("\n  -- 1. Marginal stats over top8 chunks --")
    headers = [
        "layer", "head", "n",
        "mean_dz", "std_dz",
        "%dz>0", "%dz>1", "%dz>2", "%dz>5",
        "mean_pt",
        "%pt>1e-3", "%pt>1e-2", "%pt>1e-1",
    ]
    widths = [6, 5, 10, 9, 9, 8, 8, 8, 8, 9, 10, 10, 10]
    print(_fmt_row(headers, widths))
    for row in rows:
        print(_fmt_row([
            row["layer"], row["head"], row["n"],
            f"{row['mean_dz']:.4f}", f"{row['std_dz']:.4f}",
            f"{row['frac_dz_gt_0'] * 100:.2f}",
            f"{row['frac_dz_gt_1'] * 100:.2f}",
            f"{row['frac_dz_gt_2'] * 100:.2f}",
            f"{row['frac_dz_gt_5'] * 100:.2f}",
            f"{row['mean_pt']:.4e}",
            f"{row['frac_pt_gt_1em3'] * 100:.2f}",
            f"{row['frac_pt_gt_1em2'] * 100:.2f}",
            f"{row['frac_pt_gt_1em1'] * 100:.2f}",
        ], widths))

    # ---- 2) Joint sMAPE split by dz-bucket ------------------------------
    print("\n  -- 2. sMAPE / |diff| / P_true split by dz=Z_true-Z_swa bucket --")
    if rows and rows[0]["dz_buckets"]:
        bucket_labels = [b["label"] for b in rows[0]["dz_buckets"]]
        headers = ["layer", "head"]
        widths = [6, 5]
        for lbl in bucket_labels:
            headers.append(f"n@{lbl}")
            widths.append(max(10, len(f"n@{lbl}") + 1))
            headers.append(f"sMAPE@{lbl}")
            widths.append(max(11, len(f"sMAPE@{lbl}") + 1))
        print(_fmt_row(headers, widths))
        for row in rows:
            cells = [row["layer"], row["head"]]
            for b in row["dz_buckets"]:
                cells.append(b["n"])
                v = b["mean_smape"]
                cells.append(f"{v:.4f}" if v == v else "nan")
            print(_fmt_row(cells, widths))

    # ---- 3) Joint sMAPE split by P_true-bucket --------------------------
    print("\n  -- 3. sMAPE / |diff| split by P_true bucket "
          "(physical: weight-of-final-attention) --")
    if rows and rows[0]["pt_buckets"]:
        bucket_labels = [b["label"] for b in rows[0]["pt_buckets"]]
        headers = ["layer", "head"]
        widths = [6, 5]
        for lbl in bucket_labels:
            headers.append(f"n@{lbl}")
            widths.append(max(10, len(f"n@{lbl}") + 1))
            headers.append(f"sMAPE@{lbl}")
            widths.append(max(11, len(f"sMAPE@{lbl}") + 1))
            headers.append(f"|d|@{lbl}")
            widths.append(max(10, len(f"|d|@{lbl}") + 1))
        print(_fmt_row(headers, widths))
        for row in rows:
            cells = [row["layer"], row["head"]]
            for b in row["pt_buckets"]:
                cells.append(b["n"])
                vs = b["mean_smape"]
                cells.append(f"{vs:.4f}" if vs == vs else "nan")
                vd = b["mean_abs_diff"]
                cells.append(f"{vd:.4f}" if vd == vd else "nan")
            print(_fmt_row(cells, widths))

    # ---- 4) Per-query P_swa_total stats and retrieval-dominant sMAPE ----
    # P_swa_total = exp(Z_swa) / (sum_c exp(Z_c) + exp(Z_swa)) is the total
    # final-attention weight assigned to SWA on this query. Small
    # P_swa_total => retrieval channel dominates the output, so prior
    # errors there directly hurt downstream attention.
    print("\n  -- 4. Per-query P_swa_total (total SWA weight in K+1 softmax) "
          "and retrieval-dominant sMAPE --")
    print("     retrdom subset := queries with P_swa_total < 0.5")
    headers = [
        "layer", "head", "n_query",
        "mean_pswa", "std_pswa",
        "%pswa>0.9", "%pswa>0.7", "%pswa>0.5",
        "%pswa<0.5", "%pswa<0.3", "%pswa<0.1",
        "n_rd", "sMAPE@rd", "|d|@rd",
    ]
    widths = [6, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    print(_fmt_row(headers, widths))
    for row in rows:
        n_q = row.get("n_query", 0)
        if n_q == 0:
            continue
        def _pct(v):
            return f"{v * 100:.2f}" if v == v else "nan"
        def _f4(v):
            return f"{v:.4f}" if v == v else "nan"
        print(_fmt_row([
            row["layer"], row["head"], n_q,
            _f4(row["mean_pswa"]), _f4(row["std_pswa"]),
            _pct(row["frac_pswa_gt_0p9"]),
            _pct(row["frac_pswa_gt_0p7"]),
            _pct(row["frac_pswa_gt_0p5"]),
            _pct(row["frac_pswa_lt_0p5"]),
            _pct(row["frac_pswa_lt_0p3"]),
            _pct(row["frac_pswa_lt_0p1"]),
            row.get("n_chunk_retrdom", 0),
            _f4(row.get("mean_smape_retrdom", float("nan"))),
            _f4(row.get("mean_abs_diff_retrdom", float("nan"))),
        ], widths))


def save_heatmap(per_layer_head, save_dir, ckpt_name=""):
    """Generate and save a heatmap of per-(layer, head) Z-RelErr.

    Uses matplotlib with Agg backend so it works headlessly.
    """
    if not per_layer_head:
        print("[Heatmap] no per_layer_head data, skipping.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Build the matrix.
    layers = sorted(set(r["layer"] for r in per_layer_head))
    heads = sorted(set(r["head"] for r in per_layer_head))
    layer_to_idx = {l: i for i, l in enumerate(layers)}
    head_to_idx = {h: i for i, h in enumerate(heads)}
    mat = np.full((len(layers), len(heads)), np.nan)
    for r in per_layer_head:
        li = layer_to_idx[r["layer"]]
        hi = head_to_idx[r["head"]]
        mat[li, hi] = r["z_rel_err_pct"]

    fig, ax = plt.subplots(figsize=(max(8, len(heads) * 0.5), max(6, len(layers) * 0.3)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels(heads, fontsize=6)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=6)
    ax.set_title(f"Top8 All-Active Z-RelErr (%) per (Layer, Head)\n{ckpt_name}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Z-RelErr (%)")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    safe_name = ckpt_name.replace("/", "_").replace(" ", "_") or "heatmap"
    path = os.path.join(save_dir, f"heatmap_zrelerr_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Heatmap] saved to {path}")


def save_heatmap_smape(per_layer_head_smape, save_dir, ckpt_name=""):
    """Generate and save a heatmap of per-(layer, head) sMAPE_exp (no all-active filter).

    Uses matplotlib with Agg backend so it works headlessly.
    """
    if not per_layer_head_smape:
        print("[Heatmap-sMAPE] no per_layer_head_smape data, skipping.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Build the matrix.
    layers = sorted(set(r["layer"] for r in per_layer_head_smape))
    heads = sorted(set(r["head"] for r in per_layer_head_smape))
    layer_to_idx = {l: i for i, l in enumerate(layers)}
    head_to_idx = {h: i for i, h in enumerate(heads)}
    mat = np.full((len(layers), len(heads)), np.nan)
    for r in per_layer_head_smape:
        li = layer_to_idx[r["layer"]]
        hi = head_to_idx[r["head"]]
        mat[li, hi] = r["smape_exp"]

    fig, ax = plt.subplots(figsize=(max(8, len(heads) * 0.5), max(6, len(layers) * 0.3)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels(heads, fontsize=6)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=6)
    ax.set_title(f"Top8 sMAPE_exp per (Layer, Head) [no all-active filter]\n{ckpt_name}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("sMAPE_exp")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    safe_name = ckpt_name.replace("/", "_").replace(" ", "_") or "heatmap"
    path = os.path.join(save_dir, f"heatmap_smape_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Heatmap-sMAPE] saved to {path}")


# ---------------------------------------------------------------------------
# Forward pass over a numpy olmo3 dataset (mirrors eval_ppl.py).
# ---------------------------------------------------------------------------
def run_probe(args, device, dtype, probe, model, tokenizer):
    dataset = build_numpy_dataset(args.data_path, args.max_seq_len, namespace="test")

    def collate(examples):
        return {"input_ids": torch.tensor(examples)}

    dataloader = data.DataLoader(
        dataset, batch_size=1, collate_fn=collate,
        sampler=SequentialSampler(dataset), num_workers=1,
    )

    chunk_size = getattr(model, "chunk_size", args.chunk_size)
    lmk_id = getattr(model, "lmk_id", tokenizer.vocab_size)
    insert_lmk = bool(args.insert_lmk)
    print(f"insert_lmk={insert_lmk}, chunk_size={chunk_size}, lmk_id={lmk_id}")

    steps = 0
    for inputs in dataloader:
        steps += 1
        input_ids = inputs["input_ids"].to(device)
        pos_ids = None
        if insert_lmk:
            orig_seq_len = input_ids.shape[1]
            input_ids = insert_special_tokens(
                input_ids, fill_id=lmk_id, chunk_size=chunk_size,
            )
            if args.adjust_lmk_pos:
                pos_ids = create_position_ids_with_landmarks(
                    None, orig_seq_len, chunk_size=chunk_size, device=device,
                )

        with torch.amp.autocast("cuda", dtype=dtype), torch.no_grad():
            _ = model(input_ids, position_ids=pos_ids, use_cache=False)

        print(f"step {steps}: input_len={input_ids.shape[1]}")
        if args.max_samples > 0 and steps >= args.max_samples:
            break

    return steps


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Comparison printer: side-by-side per-rank metrics across multiple ckpts.
# ---------------------------------------------------------------------------
def print_compare_per_rank(reports_by_name):
    """Side-by-side per-rank table across multiple runs.

    reports_by_name: list of (name, report_dict) preserving CLI order.
    """
    if not reports_by_name:
        return

    # Index per-rank rows by rank for each run.
    rank_set = set()
    indexed = []
    for name, rep in reports_by_name:
        d = {row["rank"]: row for row in rep["per_rank"]}
        indexed.append((name, d))
        rank_set.update(d.keys())
    ranks = sorted(rank_set)

    # Header: rank, then per-run blocks of {RMSE, sMAPE_exp}.
    metric_keys = [("rmse", "RMSE"),
                   ("smape_exp", "sMAPE_exp")]
    print("\n[Compare] per-rank metrics across ckpts (aggregated over layers)")
    headers = ["rank"]
    widths = [6]
    for name, _ in indexed:
        for _, lbl in metric_keys:
            headers.append(f"{lbl}@{name}")
            widths.append(max(12, len(lbl) + len(name) + 2))
    print(_fmt_row(headers, widths))
    for r in ranks:
        cells = [r]
        for _, d in indexed:
            row = d.get(r)
            if row is None:
                cells += ["-"] * len(metric_keys)
                continue
            for k, _ in metric_keys:
                v = row.get(k)
                if v is None or (isinstance(v, float) and v != v):
                    cells.append("nan")
                else:
                    cells.append(f"{v:.6f}")
        print(_fmt_row(cells, widths))


def print_compare_kl(reports_by_name):
    """Single-row global KL comparison across runs."""
    if not reports_by_name:
        return
    print("\n[Compare] global KL(P_true || P_pred), restricted to K_eff == K queries")
    headers = []
    widths = []
    cells = []
    for name, rep in reports_by_name:
        g = rep["global_kl"]
        headers.append(f"mean_KL@{name}")
        widths.append(max(14, len(name) + 10))
        v = g["mean_kl"]
        cells.append(f"{v:.6f}" if v == v else "nan")
    print(_fmt_row(headers, widths))
    print(_fmt_row(cells, widths))


def print_compare_top8_filters(reports_by_name):
    """Side-by-side top8 query-filtered metrics across runs."""
    if not reports_by_name:
        return
    print("\n[Compare] top8 query-filtered metrics across ckpts")
    headers = ["mode", "metric"]
    widths = [18, 20]
    for name, _ in reports_by_name:
        headers.append(name)
        widths.append(max(12, len(name) + 2))
    print(_fmt_row(headers, widths))
    for mode in ("top8_all_active", "top8_any_active"):
        for key, label in (
            ("mean_rel_log_pct", "Z-RelErr (%)"),
            ("mean_smape_exp", "sMAPE_exp"),
            ("n_query", "n_query"),
        ):
            cells = [mode, label]
            for _, rep in reports_by_name:
                stats = rep.get("top8_filters", {}).get(mode, {})
                v = stats.get(key, float("nan"))
                if isinstance(v, float):
                    cells.append(f"{v:.6f}" if v == v else "nan")
                else:
                    cells.append(v)
            print(_fmt_row(cells, widths))


def print_compare_top8_all_active_per_rank(reports_by_name):
    """Side-by-side top8_all_active per-rank Z-RelErr metrics across runs."""
    if not reports_by_name:
        return
    rank_set = set()
    indexed = []
    for name, rep in reports_by_name:
        d = {row["rank"]: row for row in rep.get("top8_all_active_per_rank", [])}
        indexed.append((name, d))
        rank_set.update(d.keys())
    ranks = sorted(rank_set)
    if not ranks:
        return
    print("\n[Compare] top8_all_active per-rank Z-RelErr (%) across ckpts")
    headers = ["rank"]
    widths = [6]
    for name, _ in indexed:
        headers.append(f"Z-RelErr@{name}")
        widths.append(max(16, len(name) + 10))
    print(_fmt_row(headers, widths))
    for r in ranks:
        cells = [r]
        for _, d in indexed:
            row = d.get(r)
            if row is None:
                cells.append("-")
                continue
            v = row["z_rel_err_pct"]
            cells.append(f"{v:.6f}" if v == v else "nan")
        print(_fmt_row(cells, widths))


def _latex_escape(s: str):
    """Escape underscores so column headers like 'no_priorq' compile in LaTeX."""
    return str(s).replace("_", r"\_")


def print_latex_compare(reports_by_name, fmt="{:.4f}",
                        col_a_label=r"w/ prop.~\ref{prop:logsumexp}",
                        col_b_label=r"w/o prop.~\ref{prop:logsumexp}"):
    r"""Emit a LaTeX table for top8_all_active per-rank Z-RelErr (%).

    Requires ``\usepackage{booktabs, siunitx}`` in the preamble.
    """
    if not reports_by_name:
        return

    # --- gather rows -------------------------------------------------------
    rank_set = set()
    z_rel_by_run = []
    for name, rep in reports_by_name:
        d = {row["rank"]: row.get("z_rel_err_pct", float("nan"))
             for row in rep.get("top8_all_active_per_rank", [])}
        z_rel_by_run.append((name, d))
        rank_set.update(d.keys())
    ranks = sorted(rank_set)

    # --- helpers -----------------------------------------------------------
    def _row_min_idx(values):
        """Return the index of the smallest finite value (None if all NaN)."""
        best_i, best_v = None, math.inf
        for i, v in enumerate(values):
            if v is None or (isinstance(v, float) and v != v):
                continue
            if v < best_v:
                best_v, best_i = v, i
        return best_i

    def _fmt_cell(v, is_best):
        if v is None or (isinstance(v, float) and v != v):
            return "{--}"
        s = fmt.format(v)
        # Wrap in {} so siunitx 'S' column accepts \bfseries content.
        return r"{\bfseries " + s + "}" if is_best else s

    n_runs = len(reports_by_name)

    # ===== Fallback: vertical layout for n_runs != 2 =======================
    if n_runs != 2:
        col_spec = "l " + " ".join(["S[table-format=2.4]"] * n_runs)
        name_cells = " & ".join(
            "{" + _latex_escape(name) + "}" for name, _ in reports_by_name
        )
        out = []
        out.append("")
        out.append("% ---- LaTeX comparison table (top8_all_active Z-RelErr) ----")
        out.append(r"% Requires: \usepackage{booktabs, siunitx}")
        out.append(r"\begin{table}[t]")
        out.append(r"\centering")
        out.append(
            r"\caption{Per-rank Z-RelErr on top-8 all-active queries. "
            r"Lower is better ($\downarrow$); the best entry per row is "
            r"in \textbf{bold}.}"
        )
        out.append(r"\label{tab:lse_probe_top8_all_active_zrelerr}")
        out.append(r"\begin{tabular}{" + col_spec + "}")
        out.append(r"\toprule")
        out.append(" & " + name_cells + r" \\")
        out.append(r"\midrule")
        for r in ranks:
            vals = [d.get(r, float("nan")) for _, d in z_rel_by_run]
            bi = _row_min_idx(vals)
            cells = [_fmt_cell(v, i == bi) for i, v in enumerate(vals)]
            out.append(f"\\quad top {r} & " + " & ".join(cells) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{table}")
        out.append("")
        print("\n[Compare][LaTeX] copy-paste-ready table")
        print("\n".join(out))
        return

    # ===== Main path: vertical layout for exactly 2 runs ==================
    d_a = z_rel_by_run[0][1]
    d_b = z_rel_by_run[1][1]

    def _rank_row_cells(rk):
        """(label, cell_a, cell_b) for one rank entry."""
        va = d_a.get(rk, float("nan"))
        vb = d_b.get(rk, float("nan"))
        bi = _row_min_idx([va, vb])
        ca = _fmt_cell(va, bi == 0)
        cb = _fmt_cell(vb, bi == 1)
        return (f"top {rk}", ca, cb)

    # --- emit --------------------------------------------------------------
    col_spec = "l l S[table-format=2.4] S[table-format=2.4]"

    z_rel_label = r"Z-RelErr (\%) ($\downarrow$)"

    out = []
    out.append("")
    out.append("% ---- LaTeX comparison table (top8_all_active Z-RelErr) ----")
    out.append(r"% Requires: \usepackage{booktabs, siunitx}")
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\begin{tabular}{" + col_spec + "}")
    out.append(r"\toprule")
    out.append(
        r"Metrics & Rank & {" + col_a_label + r"} & {" + col_b_label + r"} \\"
    )
    out.append(r"\midrule")
    for idx, r in enumerate(ranks):
        rank_label, cell_a, cell_b = _rank_row_cells(r)
        metric_cell = z_rel_label if idx == 0 else ""
        out.append(
            f"{metric_cell} & {rank_label} & {cell_a} & {cell_b} \\\\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(
        r"\caption{Per-rank Z-RelErr on top-8 all-active queries. "
        r"Lower is better ($\downarrow$); the best entry per row is "
        r"in \textbf{bold}.}"
    )
    out.append(r"\label{tab:lse_probe_top8_all_active_zrelerr}")
    out.append(r"\end{table}")
    out.append("")

    print("\n[Compare][LaTeX] copy-paste-ready table")
    print("\n".join(out))


def _ckpt_short_name(path: str) -> str:
    """Heuristic short name from a ckpt path for table headers / json filenames."""
    parts = [p for p in path.strip("/").split("/") if p]
    # Try the run-name segment (the one that typically encodes hyperparams).
    for p in parts:
        if "hsa_" in p or "345M" in p or "7B" in p:
            tag = "priorq" if "priorq" in p else "no_priorq"
            return tag
    return parts[-1] if parts else "ckpt"


def _run_one_ckpt(args, ckpt_path, device, dtype, tokenizer):
    """Run probe on a single ckpt and return (name, report, probe)."""
    sub_args = argparse.Namespace(**vars(args))
    sub_args.ckpt_path = ckpt_path
    name = _ckpt_short_name(ckpt_path)

    print("\n" + "=" * 80)
    print(f"[Run] ckpt={ckpt_path}")
    print(f"[Run] short_name={name}")
    print("=" * 80)

    model = load_hsa_model(sub_args, device, dtype)
    case_layers = None
    if args.case_layers:
        case_layers = [int(x) for x in args.case_layers.split(",") if x.strip()]
    probe = LSEProbe(
        model,
        case_limit=args.print_cases,
        case_layers=case_layers,
        case_layer_window=args.case_layer_window,
    )
    print(f"[Probe] patched {len(probe._patched_layers)} LandmarkHSA layers.")
    if args.print_cases > 0:
        print(f"[Probe] case layers={sorted(probe.case_layers)}")
    n_steps = run_probe(sub_args, device, dtype, probe, model, tokenizer)
    print(f"[Probe] processed {n_steps} sequences.")
    report = probe.report()
    print_per_rank(report["per_rank"])
    print_kl(report["global_kl"])
    print_top8_filters(report.get("top8_filters", {}))
    print_top8_all_active_per_rank(report.get("top8_all_active_per_rank", []))
    print_per_layer_head(report.get("per_layer_head", []))
    print_per_layer_head_dist(report.get("per_layer_head_dist", []))
    save_heatmap(
        report.get("per_layer_head", []),
        save_dir="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/hot_map/",
        ckpt_name=name,
    )
    save_heatmap_smape(
        report.get("per_layer_head_smape", []),
        save_dir="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/hot_map/",
        ckpt_name=name,
    )
    print_cases(report.get("cases", []))

    # Optional per-ckpt json dump: split args.output by short name.
    if args.output is not None:
        base, ext = os.path.splitext(args.output)
        out_path = f"{base}.{name}{ext or '.json'}"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "args": vars(sub_args),
                "report": report,
                "raw_stats": {
                    f"L{lyr}_R{r}": v for (lyr, r), v in probe.stats.items()
                },
                "raw_kl_stats": probe.kl_stats,
                "raw_top8_filter_stats": dict(probe.top8_filter_stats),
                "raw_top8_all_active_rank_stats": dict(probe.top8_all_active_rank_stats),
                "per_layer_head": report.get("per_layer_head", []),
                "per_layer_head_smape": report.get("per_layer_head_smape", []),
                "per_layer_head_dist": report.get("per_layer_head_dist", []),
                "cases": probe.cases,
            }, f, indent=2)
        print(f"[Probe] wrote stats to {out_path}")

    # Free GPU memory before the next ckpt.
    del probe
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return name, report


DEFAULT_CKPT_PATHS = [
    "/apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/"
    "hsa_8KA2K_HoPE_full_345M_dist-priorq-wloralmkq-loradim64/"
    "checkpoints/global_step_30000/hf_ckpt",
    "/apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/"
    "hsa_8KA2K_HoPE_full_345M_dist-wloralmkq-loradim64/"
    "checkpoints/global_step_30000/hf_ckpt",
]


def main():
    parser = argparse.ArgumentParser()
    # Model. Multiple ckpts are run sequentially and compared side by side.
    parser.add_argument("--ckpt_path", type=str, nargs="+",
                        default=DEFAULT_CKPT_PATHS,
                        help="One or more HF-format HSA checkpoint dirs. "
                             "When multiple are given, results are compared.")
    parser.add_argument("--model_type", type=str, default="qwen_lhsa",
                        choices=["olmo_lhsa", "qwen_lhsa"],
                        help="Which modeling file to use for loading.")
    parser.add_argument("--hsa_config",  type=str, default=None,
                        help="Optional override of the HSA config json. "
                             "By default the ckpt's own config.json is used "
                             "(recommended, since it always matches the weights).")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_3")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda:0")
    # Data.
    parser.add_argument("--vocab_dir", type=str,
                        default="./configs/olmo3_vocab/")
    parser.add_argument("--data_path", type=str,
                        default="/apdcephfs_tj5/share_300719894/shared/data/"
                                "dolma3_mix-6T-1025-partial-tokenized/",
                        help="Numpy olmo3 dataset path (eval_ppl.py style).")
    parser.add_argument("--max_seq_len", default=16384, type=int)
    parser.add_argument("--chunk_size", default=64, type=int,
                        help="Fallback chunk_size if model doesn't expose one.")
    parser.add_argument("--insert_lmk", action="store_true",
                        help="Externally insert landmark tokens (mirrors eval_ppl.py "
                             "with --insert_lmk). Leave off if the model does it.")
    parser.add_argument("--adjust_lmk_pos", action="store_true")
    parser.add_argument("--max_samples", default=4, type=int,
                        help="Number of sequences to feed.")
    parser.add_argument("--print_cases", default=1, type=int,
                        help="If > 0, print all ranks for the last valid query/head. "
                             "Set to 0 to disable.")
    parser.add_argument("--case_layer_window", default=2, type=int,
                        help="Number of middle layers and final layers to print cases for. "
                             "Default prints 2 middle layers and 2 final layers.")
    parser.add_argument("--case_layers", type=str, default=None,
                        help="Optional comma-separated layer indices for case printing, "
                             "overriding --case_layer_window.")
    # Output.
    parser.add_argument("--output", type=str, default=None,
                        help="If set, dump full stats and report to this json file.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype)

    ckpt_paths = args.ckpt_path if isinstance(args.ckpt_path, list) else [args.ckpt_path]
    print(f"ckpt_paths={ckpt_paths}")
    print(f"hsa_config={args.hsa_config}")
    print(f"device={device}, dtype={dtype}, attn_impl={args.attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    # Disambiguate short names if any collision occurs.
    raw_names = [_ckpt_short_name(p) for p in ckpt_paths]
    seen = {}
    final_names = []
    for n in raw_names:
        cnt = seen.get(n, 0)
        final_names.append(n if cnt == 0 else f"{n}_{cnt}")
        seen[n] = cnt + 1

    reports_by_name = []
    for ckpt_path, name_hint in zip(ckpt_paths, final_names):
        # Patch _ckpt_short_name's result by overriding via wrapper namespace.
        name, report = _run_one_ckpt(args, ckpt_path, device, dtype, tokenizer)
        # Use disambiguated name for the comparison table.
        reports_by_name.append((name_hint, report))

    if len(reports_by_name) >= 2:
        print_compare_per_rank(reports_by_name)
        print_compare_kl(reports_by_name)
        print_compare_top8_filters(reports_by_name)
        print_compare_top8_all_active_per_rank(reports_by_name)
        print_latex_compare(reports_by_name)


if __name__ == "__main__":
    main()
