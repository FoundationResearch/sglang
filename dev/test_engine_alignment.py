"""Production-path sglang alignment regression test (R53).

Why this test exists
====================
Pre-R52, ``HSAAttnBackend.lmk_k_pool`` and ``req_to_chunk_pool`` defaulted to
``None``. ``_maybe_write_chunk_lmk_k`` returned at its very first check, so
R47 (slot reuse), R48 (prior_b in selector) and R51 (decode chunk write) were
all silently no-ops in production sglang.Engine.

Our existing alignment harness (``dev/align/compare.py``) initialised the
pools externally as part of its test scaffold, so KL stayed low and we
never noticed. The friend's ``code_exp/verify_sglang_worker.py`` (which
uses ``sgl.Engine`` directly, no scaffold) saw the real divergence.

R52 fixes this by auto-initialising the pools inside
``HSAAttnBackend.__init__`` when the config has ``enable_prior_query=True``
and ``enable_lmk_q_proj=True``.

What this test guards
=====================
The test asserts the *wiring*, not the *output KL*. Reason: at our 345M
config + greedy decode, the per-q-head chunk_attn_pool path happens to pick
the same top-K pages as the fast last-token-K fallback, so the final logit
is numerically the same. The friend's real production config + long prompt
*does* diverge, but reproducing that here would require pulling in HF
infra. Wiring assertions catch the regression class regardless of whether
this specific checkpoint exhibits the divergence.

Probes:

1. **Auto-init fires** when config has both flags and the env knob is unset.
2. **Disable env knob actually disables** auto-init (so R52 itself can be
   gated off for forensics without ripping it out).
3. **Existing greedy alignment harness still PASSes** in both states (with
   and without external init), proving R52 doesn't regress what compare.py
   already exercised.

Run:

    python dev/test_engine_alignment.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALIGN = HERE / "align"
ART_DIR = HERE / "_decode_greedy_artifacts"

L = int(os.environ.get("LEN", "16384"))
N = int(os.environ.get("N_DECODE", "4"))
KL_THRESH_PASS = 5e-4


# ---------------------------------------------------------------------------
# Probe 1+2: backend wiring assertions (in-process, no subprocess).
# ---------------------------------------------------------------------------
def _build_mock_runner(cfg_dict):
    """Minimal model_runner shim sufficient for HSAAttnBackend.__init__."""
    sys.path.insert(0, str(ALIGN))
    import bootstrap  # noqa: F401
    bootstrap.init_sglang_dist()

    import torch
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    PS = int(cfg_dict.get("chunk_size", 64))
    VS = int(cfg_dict["vocab_size"])
    L_ctx = max(L * 33 // 32 + 1024, 32_768)
    pool = MHATokenToKVPool(
        size=L_ctx + 256, page_size=PS, dtype=torch.bfloat16,
        head_num=int(cfg_dict["num_key_value_heads"]),
        head_dim=int(cfg_dict["head_dim"]),
        layer_num=int(cfg_dict["num_hidden_layers"]),
        device="cuda", enable_memory_saver=False, enable_alt_stream=False,
    )

    class _Cfg:
        pass

    cfg = _Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    mr = types.SimpleNamespace(
        device="cuda", page_size=PS, sliding_window_size=None,
        # The auto-init reads model.config first (matches production path);
        # supply the hsa-config dict here so the gate fires.
        model=types.SimpleNamespace(config=cfg),
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, context_len=L_ctx,
            num_attention_heads=int(cfg_dict["num_attention_heads"]),
            get_num_kv_heads=lambda tp: int(cfg_dict["num_key_value_heads"]) // tp,
            hf_text_config=cfg,
        ),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend="hsa", speculative_num_draft_tokens=0,
            speculative_num_steps=0, triton_attention_num_kv_splits=8,
            triton_attention_split_tile_size=None,
            enable_deterministic_inference=False,
            hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None,
            hsa_enable_swa_merging=None, hsa_lmk_id=VS,
            hsa_headwise_topk_softmax=None,
        ),
        max_total_num_tokens=L_ctx,
    )
    mr.req_to_token_pool = types.SimpleNamespace(
        size=1, req_to_token=torch.zeros((1, L_ctx), dtype=torch.int32, device="cuda")
    )
    mr.token_to_kv_pool = pool
    mr.token_to_kv_pool_allocator = object()
    return mr


def probe_wiring(cfg_dict):
    """Build HSAAttnBackend with and without the disable knob; assert the
    pool state matches expectation."""
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend

    print("[r53] --- Probe 1: auto-init fires when env unset ---", flush=True)
    os.environ.pop("SGLANG_HSA_DISABLE_AUTO_POOL_INIT", None)
    mr = _build_mock_runner(cfg_dict)
    be = HSAAttnBackend(mr)
    auto_pool = be.lmk_k_pool
    auto_r2c = be.req_to_chunk_pool
    print(f"  lmk_k_pool       = {type(auto_pool).__name__ if auto_pool else None}")
    print(f"  req_to_chunk_pool= {type(auto_r2c).__name__ if auto_r2c else None}")

    print("[r53] --- Probe 2: disable knob suppresses auto-init ---", flush=True)
    os.environ["SGLANG_HSA_DISABLE_AUTO_POOL_INIT"] = "1"
    mr2 = _build_mock_runner(cfg_dict)
    be2 = HSAAttnBackend(mr2)
    off_pool = be2.lmk_k_pool
    off_r2c = be2.req_to_chunk_pool
    print(f"  lmk_k_pool       = {type(off_pool).__name__ if off_pool else None}")
    print(f"  req_to_chunk_pool= {type(off_r2c).__name__ if off_r2c else None}")
    os.environ.pop("SGLANG_HSA_DISABLE_AUTO_POOL_INIT", None)

    failures: list[str] = []
    if auto_pool is None or auto_r2c is None:
        failures.append(
            f"R52 auto-init regressed: pools should be non-None for "
            f"enable_prior_query=True + enable_lmk_q_proj=True, got "
            f"lmk_k_pool={auto_pool}, req_to_chunk_pool={auto_r2c}"
        )
    if off_pool is not None or off_r2c is not None:
        failures.append(
            f"SGLANG_HSA_DISABLE_AUTO_POOL_INIT gate broken: pools should "
            f"stay None when set, got lmk_k_pool={off_pool}, "
            f"req_to_chunk_pool={off_r2c}"
        )
    return failures


# ---------------------------------------------------------------------------
# Probe 3: existing greedy alignment harness still PASSes in both states.
# ---------------------------------------------------------------------------
def ensure_official_artifact() -> Path:
    art = ART_DIR / f"official_L{L}_N{N}.pt"
    if art.exists():
        return art
    print(f"[r53] generating official artifact for L={L} N={N}...", flush=True)
    env = dict(os.environ)
    env["LEN"] = str(L)
    env["N_DECODE"] = str(N)
    subprocess.run(
        [sys.executable, str(HERE / "test_decode_greedy_vs_official.py"),
         "--stage", "official"],
        cwd=str(HERE.parent),
        env=env, check=True,
    )
    return art


def run_sglang_subprocess(*, label: str, disable_auto: bool, disable_external: bool):
    env = dict(os.environ)
    env["LEN"] = str(L)
    env["N_DECODE"] = str(N)
    env["SGLANG_HSA_DISABLE_AUTO_POOL_INIT"] = "1" if disable_auto else "0"
    env["SGLANG_HSA_DISABLE_EXTERNAL_POOL_INIT"] = "1" if disable_external else "0"
    print(f"[r53] === {label}: auto_init={'OFF' if disable_auto else 'ON'} "
          f"external_init={'OFF' if disable_external else 'ON'} ===", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(HERE / "test_decode_greedy_vs_official.py"),
         "--stage", "sglang"],
        cwd=str(HERE.parent),
        env=env, check=False,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"{label} stage_sglang failed (rc={result.returncode})")
    for line in result.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            payload = json.loads(line.split("@@RESULT@@", 1)[1].strip())
            payload["_label"] = label
            payload["_wall_s"] = time.time() - t0
            return payload
    raise RuntimeError(f"{label}: no @@RESULT@@ marker found")


def summarise(r: dict) -> str:
    rows = r["rows"]
    kls = [row["kl"] for row in rows]
    am = [row["argmax_match"] for row in rows]
    overlap = [row.get("top5_overlap", 0) for row in rows]
    return (
        f"L={r['L']} N={r['N']} bad_steps={r['bad_steps']}"
        f" | KL mean={sum(kls) / len(kls):.3e} max={max(kls):.3e}"
        f" | argmax_match={sum(am)}/{len(am)}"
        f" | top5_overlap_mean={sum(overlap) / len(overlap):.2f}"
        f" | wall={r['_wall_s']:.1f}s"
    )


def main():
    cfg_dict = json.loads((ALIGN / "config_345m.json").read_text())

    failures = probe_wiring(cfg_dict)

    print("[r53] --- Probe 3: greedy alignment runs ---", flush=True)
    ensure_official_artifact()

    # (b) R52 production — auto-init kicks in.
    b = run_sglang_subprocess(
        label="(b) R52 production (auto-init)",
        disable_auto=False, disable_external=True,
    )
    # (c) Test scaffold — external init runs.
    c = run_sglang_subprocess(
        label="(c) test scaffold (external init)",
        disable_auto=False, disable_external=False,
    )

    print("\n[r53] --- Summary ---")
    print(f"  (b) {summarise(b)}")
    print(f"  (c) {summarise(c)}")

    kls_b = [row["kl"] for row in b["rows"]]
    kls_c = [row["kl"] for row in c["rows"]]
    am_b = [row["argmax_match"] for row in b["rows"]]
    am_c = [row["argmax_match"] for row in c["rows"]]
    if max(kls_b) >= KL_THRESH_PASS or not all(am_b):
        failures.append(
            f"(b) R52 production greedy alignment failed: "
            f"max_KL={max(kls_b):.3e} argmax_match={sum(am_b)}/{len(am_b)}"
        )
    if max(kls_c) >= KL_THRESH_PASS or not all(am_c):
        failures.append(
            f"(c) test scaffold greedy alignment failed: "
            f"max_KL={max(kls_c):.3e} argmax_match={sum(am_c)}/{len(am_c)}"
        )

    if failures:
        print("\n[r53] FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(
        "\n[r53] PASS — R52 auto-init wires the chunk_attn_pool path, "
        "env-gate works, greedy alignment holds with and without the test "
        "scaffold's external pool init."
    )


if __name__ == "__main__":
    main()
