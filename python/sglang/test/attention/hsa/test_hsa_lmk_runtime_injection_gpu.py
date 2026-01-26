import os
import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)

_HSA_VERBOSE = os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") == "1"


def _vprint(*args):
    if _HSA_VERBOSE:
        print(*args, flush=True)


def test_hsa_insert_lmk_prompt_matches_flashhsa_semantics_cuda():
    # FlashHSA semantics: insert LMK after every (page_size-1) tokens, keep tail unchanged.
    from sglang.srt.managers.schedule_batch import Req

    page_size = 4
    lmk_id = 9999
    # chunk = page_size-1 = 3
    # 7 tokens -> full chunk 3 + full chunk 3 + tail 1
    origin = [1, 2, 3, 4, 5, 6, 7]
    out = Req._hsa_insert_lmk_prompt(origin, page_size=page_size, lmk_id=lmk_id)
    _vprint("### test_hsa_insert_lmk_prompt_matches_flashhsa_semantics_cuda")
    _vprint(f"- page_size={page_size} lmk_id={lmk_id}")
    _vprint(f"- origin={origin}")
    _vprint(f"- with_lmk={out}")
    _vprint("=> Conclusion: prompt LMK insertion matches FlashHSA chunking semantics.")
    assert out == [1, 2, 3, lmk_id, 4, 5, 6, lmk_id, 7]


def test_hsa_decode_forces_lmk_and_is_invisible_cuda():
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    # We don't run a real forward; we validate the pure runtime contract:
    # - LMK appended into fill_ids when next slot is LMK
    # - LMK not appended into output_ids (user-visible)
    page_size = 4
    lmk_id = 9999

    req = Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=[],
        sampling_params=SamplingParams(max_new_tokens=128),
    )
    req.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)

    # Simulate a partially-filled page: len(fill_ids)=3 => next slot is LMK.
    req.fill_ids = [10, 11, 12]
    req.output_ids = [10, 11, 12]  # what user has seen so far (no LMK)

    # Model "samples" 42, but runtime should force LMK instead.
    visible = req.hsa_append_next_token_or_lmk(42)
    _vprint("### test_hsa_decode_forces_lmk_and_is_invisible_cuda")
    _vprint(f"- step1 visible={visible} fill_ids={req.fill_ids} output_ids={req.output_ids}")
    assert visible is False
    assert req.fill_ids == [10, 11, 12, lmk_id]
    assert req.output_ids == [10, 11, 12]

    # Next step should accept a visible token again.
    visible2 = req.hsa_append_next_token_or_lmk(43)
    _vprint(f"- step2 visible={visible2} fill_ids={req.fill_ids} output_ids={req.output_ids}")
    _vprint("=> Conclusion: LMK is appended internally (fill_ids) and never emitted (output_ids).")
    assert visible2 is True
    assert req.fill_ids == [10, 11, 12, lmk_id, 43]
    assert req.output_ids == [10, 11, 12, 43]


