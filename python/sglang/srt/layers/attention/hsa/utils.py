from __future__ import annotations

import torch


def transform_page_table_1_to_real(page_table_1: torch.Tensor, page_size: int) -> torch.Tensor:
    """
    Convert token->slot mapping (page_size=1 semantics) to a page_id table.

    For page_size>1, we take the first token_loc of each page and integer-divide by page_size:
      page_id = token_loc // page_size

    This mirrors the logic used by NSA backend (`NativeSparseAttnBackend._transform_table_1_to_real`).
    """
    if page_size == 1:
        return page_table_1

    max_seqlen_k = page_table_1.shape[1]
    strided_indices = torch.arange(
        0, max_seqlen_k, page_size, device=page_table_1.device, dtype=torch.int32
    )
    return page_table_1[:, strided_indices] // page_size


