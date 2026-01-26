import os

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import (
    compute_decode_positions_landmark,
    compute_position,
)


def _vprint(msg: str):
    if os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") not in ("", "0", "false", "False"):
        print(msg, flush=True)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)


def test_flashhsa_extend_positions_landmark_cuda():
    # page_size == chunk_size in FlashHSA
    page_size = 4
    # One request: prefix_len=0, extend_len=5 (engine-visible, includes LMK at index 3)
    prefix_lens = torch.tensor([0], device="cuda", dtype=torch.int32)
    extend_lens = torch.tensor([5], device="cuda", dtype=torch.int32)
    positions, _ = compute_position(
        "hsa",
        prefix_lens,
        extend_lens,
        int(extend_lens.sum().item()),
        page_size=page_size,
        enable_landmark_positions=True,
    )
    got = positions.to("cpu").tolist()
    # indices g=0..4 => pos = g - floor((g+1)/4) => [0,1,2,2,3]
    assert got == [0, 1, 2, 2, 3]
    _vprint(f"extend positions (page_size={page_size}) = {got}")


def test_flashhsa_decode_positions_landmark_cuda():
    page_size = 4
    # seq_len=4 means the last token is LMK at g=3 => pos=2
    seq_lens = torch.tensor([4], device="cuda", dtype=torch.int32)
    pos = compute_decode_positions_landmark(seq_lens, page_size=page_size)
    got = int(pos.item())
    assert got == 2
    _vprint(f"decode position (seq_len=4, page_size={page_size}) = {got}")


