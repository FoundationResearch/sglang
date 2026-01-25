import pytest
import torch


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA device required for this test")
def test_hsa_build_active_page_candidates_excludes_swa_window_pages_cuda():
    from sglang.srt.layers.attention.hsa.selector import build_active_page_candidates

    device = "cuda"
    page_size = 4

    # One request, 12 tokens mapped to locs 0..11 => pages {0,1,2}
    page_table_1 = torch.arange(12, device=device, dtype=torch.int32)[None, :]
    seq_lens = torch.tensor([12], device=device, dtype=torch.int32)

    # Exclude last 5 tokens (7..11) => pages {1,2} excluded => only page 0 remains
    cand_page_ids, cand_mask = build_active_page_candidates(
        page_table_1=page_table_1,
        seq_lens=seq_lens,
        page_size=page_size,
        window_size=5,
    )

    assert cand_page_ids.is_cuda
    assert cand_mask.is_cuda
    assert cand_page_ids.shape == (1, 1)
    assert cand_mask.shape == (1, 1)
    assert cand_mask[0, 0].item() is True
    assert cand_page_ids[0, 0].item() == 0


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA device required for this test")
def test_hsa_selector_decode_head_strategy_deterministic_topk_cuda():
    from sglang.srt.layers.attention.hsa.selector import select_topk_pages_decode

    device = "cuda"
    dtype = torch.float16

    B = 1
    H = 2
    G = 2
    D = 4
    HQ = H * G
    K = 2

    # Candidates: pages [0,1,2]
    cand_page_ids = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
    cand_mask = torch.ones((B, 3), device=device, dtype=torch.bool)
    cand_valid = torch.ones((B, 3), device=device, dtype=torch.bool)

    # q: [B, HQ, D] -> [B, H, G, D]
    q = torch.zeros((B, HQ, D), device=device, dtype=dtype)
    # kv head 0 uses dim0, kv head 1 uses dim1
    q[:, 0, 0] = 1.0  # h0 g0
    q[:, 2, 1] = 1.0  # h1 g0

    # repr: [B, C, H, D]
    cand_repr = torch.zeros((B, 3, H, D), device=device, dtype=dtype)
    # For head0: increasing on dim0 by page id
    cand_repr[0, 0, 0, 0] = 1.0
    cand_repr[0, 1, 0, 0] = 2.0
    cand_repr[0, 2, 0, 0] = 3.0
    # For head1: increasing on dim1 by page id
    cand_repr[0, 0, 1, 1] = 1.0
    cand_repr[0, 1, 1, 1] = 2.0
    cand_repr[0, 2, 1, 1] = 3.0

    sel = select_topk_pages_decode(
        q=q,
        cand_page_ids=cand_page_ids,
        cand_mask=cand_mask,
        cand_chunk_repr=cand_repr,
        cand_chunk_repr_valid=cand_valid,
        topk=K,
        selection_strategy="head",
        sm_scale=1.0,  # keep arithmetic simple for assertions
    )

    # top-2 should be pages {1,2}. Order is sorted by candidate index (deterministic).
    assert sel.selected_page_ids.shape == (B, H, K)
    assert sel.selected_page_ids[0, 0].tolist() == [1, 2]
    assert sel.selected_page_ids[0, 1].tolist() == [1, 2]


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA device required for this test")
def test_hsa_selector_decode_masks_invalid_candidates_by_valid_mask_cuda():
    from sglang.srt.layers.attention.hsa.selector import (
        build_active_page_candidates,
        select_topk_pages_decode,
    )

    device = "cuda"
    dtype = torch.float16

    page_size = 4
    B = 1
    H = 2
    D = 4
    G = 2
    HQ = H * G
    topk = 2

    # Token locs 0..11 => pages 0,1,2; no window exclusion here.
    page_table_1 = torch.arange(12, device=device, dtype=torch.int32)[None, :]
    seq_lens = torch.tensor([12], device=device, dtype=torch.int32)
    cand_page_ids, cand_mask = build_active_page_candidates(
        page_table_1=page_table_1,
        seq_lens=seq_lens,
        page_size=page_size,
        window_size=None,
    )

    # Make candidate 2 have huge repr so it would win if valid.
    C = int(cand_page_ids.shape[1])
    assert C == 3
    cand_repr = torch.zeros((B, C, H, D), device=device, dtype=dtype)
    cand_repr[0, 2, :, 0] = 100.0  # page 2, all kv heads

    # Mark candidate 2 as invalid via explicit valid-mask.
    cand_valid = torch.ones((B, C), device=device, dtype=torch.bool)
    cand_valid[0, 2] = False

    q = torch.zeros((B, HQ, D), device=device, dtype=dtype)
    q[:, 0, 0] = 1.0  # should favor dim0

    sel = select_topk_pages_decode(
        q=q,
        cand_page_ids=cand_page_ids,
        cand_mask=cand_mask,
        cand_chunk_repr=cand_repr,
        cand_chunk_repr_valid=cand_valid,
        topk=topk,
        selection_strategy="group",
        sm_scale=1.0,
    )

    # Candidate page 2 is invalid -> should not appear in selection.
    assert 2 not in sel.selected_page_ids[0, 0].tolist()


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA device required for this test")
def test_hsa_selector_decode_fixed_k_padding_cuda():
    from sglang.srt.layers.attention.hsa.selector import select_topk_pages_decode

    device = "cuda"
    dtype = torch.float16

    B = 1
    H = 2
    G = 2
    D = 4
    HQ = H * G
    K = 3

    cand_page_ids = torch.tensor([[5]], device=device, dtype=torch.int32)
    cand_mask = torch.tensor([[True]], device=device, dtype=torch.bool)
    cand_valid = torch.tensor([[True]], device=device, dtype=torch.bool)
    cand_repr = torch.zeros((B, 1, H, D), device=device, dtype=dtype)

    q = torch.zeros((B, HQ, D), device=device, dtype=dtype)
    q[:, 0, 0] = 1.0

    sel = select_topk_pages_decode(
        q=q,
        cand_page_ids=cand_page_ids,
        cand_mask=cand_mask,
        cand_chunk_repr=cand_repr,
        cand_chunk_repr_valid=cand_valid,
        topk=K,
        selection_strategy="group",
        sm_scale=1.0,
    )

    assert sel.selected_page_ids.shape == (B, H, K)
    # First entry is the only candidate; rest padded with -1.
    assert sel.selected_page_ids[0, 0].tolist() == [5, -1, -1]
    assert sel.selected_page_ids[0, 1].tolist() == [5, -1, -1]

