# G=8 baseline snapshot — BEFORE v3 md (G=1 + layerwise_lmkq_norm) changes

Commit: `2a2884271` (hsa_h200). Date: 2026-06-29. GPU 0 (H200, shared; MEM_FRAC=0.35 for sweep).
Purpose: prove the v3 md G=1/norm changes do NOT regress the existing G=8 path.

## Alignment — friend's verify workers (verify_sglang_worker.py + verify_sglang_vs_hf.py), CG ON
ckpt: `dev/align/ckpt_345m_bench_hf` (16 heads / 2 kv / hsa_heads 16 -> per-qhead G=8, prior_query=True)

| prompt | token match | top2 match | KL mean | KL max | verdict |
|---|---|---|---|---|---|
| SHORT PT=512  | 12/12 (100%) | 12/12 (100%) | 4.2e-4 | 7.4e-4 | 完全一致 |
| LONG  PT=2048 | 12/12 (100%) | 12/12 (100%) | 1.2e-4 | 2.2e-4 | 完全一致 |

## Speed — dev/sweep_8k_512k.sh (hsa345m_real G=8 vs dense345m_fair), prefill eager / decode CG ON
No real OOM at any length (err_lines=2 was a benign mem-fraction warning substring; latencies all valid).

| L | PREFILL hsa | PREFILL triton | DECODE hsa | DECODE triton |
|---|---|---|---|---|
| 8192   | 0.05050 | 0.03167 | 0.00371 | 0.00276 |
| 16384  | 0.08728 | 0.08503 | 0.00377 | 0.00412 |
| 32768  | 0.16640 | 0.29397 | 0.00384 | 0.00662 |   # 32K prefill re-measured clean on idle GPU0 (orig 0.24948 was shared-GPU noise); matches paper 154ms
| 65536  | 0.34040 | 1.05406 | 0.00395 | 0.01170 |
| 131072 | 0.76559 | 4.12570 | 0.00430 | 0.02201 |
| 262144 | 1.90008 | 16.62521 | 0.00472 | 0.04254 |
| 524288 | 5.21814 | 66.61138 | 0.00551 | 0.08355 |

(seconds; prefill output-len 4, decode median over 32, batch 1, page_size 64.)

## Re-run after v3 md changes to confirm no regression:
  bash dev/run_cg_vs_hf_verify.sh dev/align/ckpt_345m_bench_hf on 0 512 12 0
  bash dev/run_cg_vs_hf_verify.sh dev/align/ckpt_345m_bench_hf on 0 2048 12 0
  GPU=0 MEM_FRAC=0.35 bash dev/sweep_8k_512k.sh
