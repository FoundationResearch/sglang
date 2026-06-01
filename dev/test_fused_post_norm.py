"""Quick correctness check for fused_post_norm_add vs reference (norm + add)."""
import torch
import torch.nn.functional as F

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.elementwise import fused_post_norm_add


def ref_post_norm_add(rms: RMSNorm, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return rms(x) + residual


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    for hidden in (1024, 4096, 8192):
        for T in (1, 8, 1024):
            rms = RMSNorm(hidden, eps=1e-6).to(device=device, dtype=dtype)
            rms._forward_method = rms.forward_native  # force native for ref consistency
            x = torch.randn(T, hidden, device=device, dtype=dtype)
            r = torch.randn(T, hidden, device=device, dtype=dtype)

            out_ref = ref_post_norm_add(rms, x, r)
            out_fused = fused_post_norm_add(x, r, rms.weight, rms.variance_epsilon)

            max_err = (out_ref - out_fused).abs().max().item()
            rel_err = ((out_ref - out_fused).abs() / out_ref.abs().clamp(min=1e-3)).mean().item()
            ok = "OK" if max_err < 0.05 else "FAIL"
            print(
                f"hidden={hidden:5d} T={T:5d} max_err={max_err:.4f} rel_err={rel_err:.4f} {ok}"
            )


if __name__ == "__main__":
    main()
