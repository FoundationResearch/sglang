"""Run dev/align/compare.py with R15 buffer path force-enabled, and
verify the alignment KL is unchanged.

Mechanism: monkey-patch HSAAttnBackend.__init__ to automatically call
init_cuda_graph_state right after construction.  This triggers the
R15 buffer path inside init_forward_metadata, exercising:
  * hsa_build_page_table_1_kernel (writing page_table_1 buffer)
  * hsa_copy_seq_lens_kernel (writing cache_seqlens_i32 buffer)
  * the buffer-aware metadata assembly branch

Expected: KL prefill mean ≈ 0.003649, KL last-dec mean ≈ 0.004414 — same
as the non-buffer path.  Any meaningful divergence means R15 is incorrect.
"""
import sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "align"))

# Force the R15 buffer path for ALL HSAAttnBackend instances.
import bootstrap  # noqa  (sets up sglang+veomni shims)

from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend

_orig_init = HSAAttnBackend.__init__


def _patched_init(self, model_runner, **kw):
    _orig_init(self, model_runner, **kw)
    # Pick max_bs and max_num_tokens that are >= the values compare.py uses
    # (bs=1, num_tokens up to context_len).
    max_bs = 4
    max_num_tokens = max_bs * int(
        getattr(model_runner.model_config, "context_len", 16384)
    )
    self.init_cuda_graph_state(max_bs, max_num_tokens)
    print(
        f"[R15 patch] init_cuda_graph_state called: "
        f"page_table_1 buf shape={tuple(self._cg_page_table_1.shape)}",
        flush=True,
    )


HSAAttnBackend.__init__ = _patched_init

sys.argv = ["compare.py"]
import importlib.util as _u
spec = _u.spec_from_file_location("compare", HERE / "align" / "compare.py")
mod = _u.module_from_spec(spec)
sys.modules["compare"] = mod
print("Loading compare.py inline...", flush=True)
spec.loader.exec_module(mod)
print("Calling compare.main()...", flush=True)
if hasattr(mod, "main"):
    mod.main()
else:
    print("compare.py has no main() — looking at __main__ flow", flush=True)
