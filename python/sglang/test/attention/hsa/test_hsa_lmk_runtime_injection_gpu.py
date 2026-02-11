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
    # - LMK occupies an internal decode step and is appended into fill_ids
    # - LMK step's sampled token is discarded (user-invisible)
    # - User-visible token chain is not broken: the sampled visible token is consumed after LMK
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

    # Step A (normal decode): model samples 42 (user-visible). Since next position is LMK slot,
    # we schedule an internal LMK step as the next input and stash 42 to be consumed after LMK.
    visible, next_input_override, skip_checks = req.hsa_decode_postprocess_sampled_token(42)
    _vprint("### test_hsa_decode_forces_lmk_and_is_invisible_cuda")
    _vprint(
        f"- step1 visible={visible} next_input_override={next_input_override} skip={skip_checks} "
        f"fill_ids={req.fill_ids} output_ids={req.output_ids}"
    )
    assert visible is True
    assert skip_checks is False
    assert next_input_override == lmk_id
    assert req.fill_ids == [10, 11, 12, lmk_id]
    assert req.output_ids == [10, 11, 12, 42]
    assert req.hsa_waiting_for_lmk_step is True
    assert req.hsa_pending_visible_token_id == 42

    # Step B (internal LMK decode): model samples 43 but it is discarded.
    visible2, next_input_override2, skip_checks2 = req.hsa_decode_postprocess_sampled_token(43)
    _vprint(
        f"- step2 visible={visible2} next_input_override={next_input_override2} skip={skip_checks2} "
        f"fill_ids={req.fill_ids} output_ids={req.output_ids}"
    )
    assert visible2 is False
    assert skip_checks2 is True
    assert next_input_override2 == 42
    # Pending visible token is appended into engine-visible sequence after LMK.
    assert req.fill_ids == [10, 11, 12, lmk_id, 42]
    assert req.output_ids == [10, 11, 12, 42]
    assert req.hsa_waiting_for_lmk_step is False
    assert req.hsa_pending_visible_token_id is None

    # Step C (next normal decode): model samples 44; now next input is 44 (no LMK scheduling).
    visible3, next_input_override3, skip_checks3 = req.hsa_decode_postprocess_sampled_token(44)
    _vprint(
        f"- step3 visible={visible3} next_input_override={next_input_override3} skip={skip_checks3} "
        f"fill_ids={req.fill_ids} output_ids={req.output_ids}"
    )
    _vprint("=> Conclusion: LMK step is internal; user-visible token chain is preserved across LMK.")
    assert visible3 is True
    assert skip_checks3 is False
    assert next_input_override3 is None
    assert req.fill_ids == [10, 11, 12, lmk_id, 42, 44]
    assert req.output_ids == [10, 11, 12, 42, 44]


