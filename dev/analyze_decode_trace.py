"""Parse a torch.profiler chrome trace for HSA decode and aggregate by:
  - kernel name (GPU side)
  - Python op stack (CPU side) - identify which torch.nn / functional ops dominate

Usage:
  python dev/analyze_decode_trace.py <trace.json.gz>
"""
import gzip, json, sys, collections


def main(path):
    with gzip.open(path, 'rb') as f:
        data = json.load(f)
    events = data['traceEvents']

    # --- GPU kernel aggregation ---
    gpu_total = 0
    gpu_by_name = collections.Counter()
    gpu_count = collections.Counter()
    for e in events:
        cat = e.get('cat', '')
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            dur = e.get('dur', 0)
            gpu_total += dur
            name = e.get('name', '<unknown>')[:90]
            gpu_by_name[name] += dur
            gpu_count[name] += 1

    print(f"\n=== GPU kernels (total {gpu_total/1000:.2f}ms) ===")
    print(f"{'kernel':<92} {'us':>10} {'cnt':>6} {'us/call':>8} {'pct':>6}")
    for name, dur in gpu_by_name.most_common(30):
        cnt = gpu_count[name]
        print(f"{name:<92} {dur:>10} {cnt:>6} {dur/cnt:>8.1f} {100*dur/gpu_total:>5.1f}%")

    # --- Aggregate by "category" of kernel ---
    def cat_kernel(n):
        n_low = n.lower()
        if 'nvjet' in n_low or 'cutlass' in n_low or 'cublas' in n_low or 'gemm' in n_low or 'jet_' in n_low:
            return 'GEMM/cuBLAS'
        if 'rmsnorm' in n_low or 'rms_norm' in n_low:
            return 'RMSNorm'
        if 'rotary' in n_low or 'rope' in n_low:
            return 'RoPE'
        if 'fmha' in n_low or 'flashattn' in n_low or 'flash_attn' in n_low or 'fwd_block_M' in n_low:
            return 'Attention(FA/FMHA)'
        if 'topk' in n_low or 'select' in n_low:
            return 'TopK/Select'
        if 'softmax' in n_low:
            return 'Softmax'
        if 'cat' in n_low and 'array' in n_low:
            return 'Concat'
        if 'memcpy' in n_low or 'memset' in n_low:
            return 'Memcpy/Memset'
        if 'elementwise' in n_low:
            return 'Elementwise'
        if 'reduce' in n_low:
            return 'Reduce'
        if 'store_kvcache' in n_low:
            return 'KV cache store'
        if 'act_and_mul' in n_low or 'silu' in n_low or 'gelu' in n_low:
            return 'MLP activation'
        return 'Other'
    by_cat = collections.Counter()
    cnt_cat = collections.Counter()
    for name, dur in gpu_by_name.items():
        c = cat_kernel(name)
        by_cat[c] += dur
        cnt_cat[c] += gpu_count[name]
    print(f"\n=== GPU time by category ===")
    print(f"{'category':<25} {'us':>10} {'cnt':>6} {'pct':>6}")
    for c, dur in by_cat.most_common():
        print(f"{c:<25} {dur:>10} {cnt_cat[c]:>6} {100*dur/gpu_total:>5.1f}%")

    # --- CPU op aggregation (filter to user-meaningful ops) ---
    # Look at cat='cpu_op' / cat='user_annotation'
    cpu_by_name = collections.Counter()
    for e in events:
        cat = e.get('cat', '')
        if cat in ('cpu_op', 'user_annotation'):
            name = e.get('name', '<unknown>')
            # Skip framework noise
            if name.startswith('aten::') and not name.startswith(('aten::linear', 'aten::matmul', 'aten::mm', 'aten::bmm', 'aten::sdp')):
                continue
            cpu_by_name[name] += e.get('dur', 0)
    print(f"\n=== Top CPU/user ops (informational) ===")
    for name, dur in cpu_by_name.most_common(25):
        print(f"  {name:<80} {dur/1000:>8.2f}ms")


if __name__ == '__main__':
    main(sys.argv[1])
