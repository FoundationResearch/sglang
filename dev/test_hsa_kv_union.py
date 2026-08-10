#!/usr/bin/env python3
"""Unit test for the KV-union extend kernel.

Compares hsa_extend_paged_fwd with kv_union=False (base) vs kv_union=True
(union) on identical random inputs.  Both kernels compute the same weighted
per-page attention in fp32; only the accumulation order differs, so results
should match within bf16 resolution.

Usage:  python dev/test_hsa_kv_union.py [--block-m 4] [--g 8] [--topk 32]
"""
import argparse
import sys

import torch

sys.path.insert(0, "python")
from sglang.srt.layers.attention.hsa.kernels.hsa_extend import (
    _build_kv_union,
    hsa_extend_paged_fwd,
)


def run(
    T, hq, h, d, topk, page_size, max_pages, block_m, kv_union,
    seed=0, blend_swa=False,
):
    torch.manual_seed(seed)
    dev = "cuda"
    nloc = max_pages * page_size
    bs = max(T // 256, 1)  # pretend bs sequences

    q = torch.randn(T, hq, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(nloc, h, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(nloc, h, d, device=dev, dtype=torch.bfloat16)

    # page_table: [bs, max_pages * page_size]
    pt = torch.stack(
        [torch.randperm(nloc, device=dev)[:max_pages * page_size] for _ in range(bs)]
    ).int()

    # selected_page_ids: [T, H, topk]
    sel = torch.stack(
        [
            torch.stack(
                [torch.randperm(max_pages, device=dev)[:topk] for _ in range(h)]
            )
            for _ in range(T)
        ]
    ).int()

    # Make adjacent tokens share most pages (simulate real overlap)
    for t in range(1, T):
        for hh in range(h):
            n_shared = topk - topk // 4  # 75% overlap
            sel[t, hh, :n_shared] = sel[t - 1, hh, :n_shared]

    # hsa_weights: [T, HQ, topk] — softmax-like (positive, sums to ~1)
    raw = torch.randn(T, hq, topk, device=dev)
    weights = torch.softmax(raw, dim=-1).to(torch.bfloat16)

    # token_to_seq_id: [T] — assign consecutive tokens to sequences
    tokens_per_seq = T // bs
    seq_ids = torch.arange(T, device=dev) // tokens_per_seq
    seq_ids = seq_ids.clamp(max=bs - 1).int()

    swa_o = swa_w = None
    if blend_swa:
        swa_o = torch.randn(T, hq, d, device=dev, dtype=torch.bfloat16)
        swa_w = torch.rand(T, hq, device=dev, dtype=torch.bfloat16)

    out = hsa_extend_paged_fwd(
        q=q, k_cache=k, v_cache=v, page_table_1=pt,
        selected_page_ids=sel, hsa_weights=weights,
        page_size=page_size, token_to_seq_id=seq_ids,
        mask_last_token=True, block_m=block_m,
        swa_o_inner=swa_o, swa_w_q=swa_w,
        kv_union=kv_union,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--max-pages", type=int, default=256)
    ap.add_argument("--block-m", type=int, default=4)
    ap.add_argument("--blend-swa", action="store_true")
    a = ap.parse_args()

    h = a.hq // a.g
    print(
        f"T={a.T} HQ={a.hq} H={h} G={a.g} D={a.d} topk={a.topk} "
        f"page={a.page_size} pages={a.max_pages} block_m={a.block_m} "
        f"blend_swa={a.blend_swa}"
    )

    ref = run(
        a.T, a.hq, h, a.d, a.topk, a.page_size, a.max_pages,
        a.block_m, kv_union=False, blend_swa=a.blend_swa,
    )
    got = run(
        a.T, a.hq, h, a.d, a.topk, a.page_size, a.max_pages,
        a.block_m, kv_union=True, blend_swa=a.blend_swa,
    )

    rf, gf = ref.float(), got.float()
    denom = rf.abs().clamp(min=1e-3)
    rel = (gf - rf).abs() / denom
    exact = int((ref == got).sum()), ref.numel()

    print(f"\nbit-exact elements : {exact[0]}/{exact[1]} ({100 * exact[0] / exact[1]:.2f}%)")
    print(f"max |abs| diff     : {(gf - rf).abs().max().item():.3e}")
    print(f"max relative diff  : {rel.max().item():.3e}   (bf16 ulp ~ 7.8e-3)")
    print(f"mean relative diff : {rel.mean().item():.3e}")

    ok = rel.max().item() < 7.8e-3
    print("\nRESULT:", "PASS" if ok else "FAIL")

    # Also test the union-build stats
    from sglang.srt.layers.attention.hsa.kernels.hsa_extend import _build_kv_union
    torch.manual_seed(0)
    T_test = a.T
    sel_test = torch.stack(
        [torch.stack([torch.randperm(a.max_pages, device="cuda")[:a.topk] for _ in range(h)]) for _ in range(T_test)]
    ).int()
    for t in range(1, T_test):
        for hh in range(h):
            n_shared = a.topk - a.topk // 4
            sel_test[t, hh, :n_shared] = sel_test[t - 1, hh, :n_shared]
    w_test = torch.softmax(torch.randn(T_test, a.hq, a.topk, device="cuda"), dim=-1).bfloat16()
    seq_test = (torch.arange(T_test, device="cuda") // max(T_test // max(T_test // 256, 1), 1)).clamp(max=max(T_test // 256, 1) - 1).int()

    upids, useq, uw, max_u = _build_kv_union(sel_test, w_test, seq_test, a.block_m, a.hq, h)
    num_groups = (T_test + a.block_m - 1) // a.block_m
    valid_pages = (upids >= 0).sum(dim=-1).float()
    no_union = a.topk * a.block_m
    actual_mean = valid_pages.mean().item()
    print(f"\nUnion stats: MAX_U={max_u}  mean_union_size={actual_mean:.1f}  "
          f"no-dedup={no_union}  dedup_ratio={no_union / actual_mean:.2f}x")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
