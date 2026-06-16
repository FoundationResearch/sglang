"""Breakdown of the 3 internal kernels in topk_head_maxpool selector.

Times select / sort / recompute separately via torch.cuda.Event.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsa-kernel-main"))
from ops.topk_head_maxpool import TopKMaxPooling_Fused, TopKMaxPoolingFusedFn  # noqa: E402

LEN = int(os.environ.get("LEN", "32768"))
ITERS = int(os.environ.get("ITERS", "100"))
WARMUP = int(os.environ.get("WARMUP", "10"))
H_KV = 2
G = 8
D = 64
TOPK = 32
BLOCK = 64
WINDOW = 512


def main():
    print(f"==== selector breakdown  L={LEN} ====")
    L = LEN
    S = L // BLOCK
    q = (torch.randn(1, L, H_KV, G, D, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
    lmks = (torch.randn(1, S, H_KV * G, D, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
    lse_swa = torch.randn(1, L, H_KV, G, dtype=torch.float32, device="cuda").contiguous()
    bias_arg = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")
    head_mask_arg = torch.ones(1, 1, dtype=torch.int32, device="cuda")
    q_off = torch.zeros(1, dtype=torch.int32, device="cuda")

    mod = TopKMaxPooling_Fused(
        topk=TOPK, block_size=BLOCK, window_size=WINDOW, is_causal=True,
        use_bias=False, per_qhead_lmks=True, is_training=False,
    )
    # Trigger kernel cache build by calling once
    out = mod.forward(q, lmks, lse_swa=lse_swa, bias=None, q_offset=0)
    torch.cuda.synchronize()
    sel_k = mod._cached_select_kernel
    sort_k = mod._cached_sort_kernel
    rec_k = mod._cached_recompute_kernel

    # Warmup
    for _ in range(WARMUP):
        idx_raw = sel_k(q, lmks, lse_swa, bias_arg, head_mask_arg, q_off)
        idx_sorted = sort_k(idx_raw)
        rec_k(q, lmks, idx_sorted, head_mask_arg)
    torch.cuda.synchronize()

    def time_one(fn, iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        # warm
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ev0.record()
        for _ in range(iters):
            fn()
        ev1.record()
        torch.cuda.synchronize()
        return ev0.elapsed_time(ev1) / iters

    # cached intermediate to time sort/recompute separately
    idx_raw_cached = sel_k(q, lmks, lse_swa, bias_arg, head_mask_arg, q_off)
    idx_sorted_cached = sort_k(idx_raw_cached)

    t_sel = time_one(lambda: sel_k(q, lmks, lse_swa, bias_arg, head_mask_arg, q_off), ITERS)
    t_sort = time_one(lambda: sort_k(idx_raw_cached), ITERS)
    t_rec = time_one(lambda: rec_k(q, lmks, idx_sorted_cached, head_mask_arg), ITERS)

    t_total = t_sel + t_sort + t_rec
    print(f"  select   : {t_sel:7.3f} ms   ({100*t_sel/t_total:5.1f}%)")
    print(f"  sort     : {t_sort:7.3f} ms   ({100*t_sort/t_total:5.1f}%)")
    print(f"  recompute: {t_rec:7.3f} ms   ({100*t_rec/t_total:5.1f}%)")
    print(f"  -------- ---------")
    print(f"  sum      : {t_total:7.3f} ms")


if __name__ == "__main__":
    main()
