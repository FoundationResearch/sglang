import torch


def insert_special_tokens(input_ids, fill_id, chunk_size):
    N, L = input_ids.shape
    full_chunks = L // (chunk_size - 1)
    remainder = L % (chunk_size - 1)
    
    parts = []
    if full_chunks > 0:
        chunk_part = input_ids[:, :full_chunks * (chunk_size - 1)].view(N, full_chunks, chunk_size - 1)
        fill_tokens = torch.full((N, full_chunks, 1), fill_id, device=input_ids.device, dtype=input_ids.dtype)
        parts.append(torch.cat([chunk_part, fill_tokens], dim=2).view(N, -1))
    
    if remainder > 0:
        parts.append(input_ids[:, full_chunks * (chunk_size - 1):])
    
    return torch.cat(parts, dim=1)


def create_position_ids_with_landmarks(seq_length, chunk_size, device):
    L = seq_length
    position_ids = torch.arange(0, seq_length, device=device)

    full_chunks = L // (chunk_size - 1)
    remainder = L % (chunk_size - 1)

    result_parts = []

    if full_chunks > 0:
        full_part = position_ids[:full_chunks * (chunk_size - 1)]

        # repeat lmk pos
        full_part = full_part.view(-1, chunk_size - 1)
        last_pos = full_part[:, -1:] + 1 
        full_part = torch.cat([full_part, last_pos], dim=-1) 
        full_part = full_part.view(-1)
        result_parts.append(full_part)
    
    if remainder > 0:
        remainder_part = position_ids[full_chunks * (chunk_size - 1):]  # (N, remainder)
        result_parts.append(remainder_part)
    
    pos = torch.cat(result_parts, dim=0)
    return pos.unsqueeze(0)


import pytest
@pytest.mark.parametrize("ids, chunk_size", [
    ([1,2,3,4,5,6,7], 4),
    ([1,2,3,4,5,6], 4),
    [[i for i in range(8192)], 64]
])
def test_insert_lmk_tokens(ids, chunk_size) -> None:
    lmk_id = -100
    new_ids = [0] * (len(ids) // (chunk_size - 1) + len(ids))
    old_i = 0
    for i in range(len(new_ids)):
        if (i + 1) % chunk_size == 0:
            new_ids[i] = lmk_id
        else:
            new_ids[i] = ids[old_i]
            old_i += 1

    new_ids_torch = torch.tensor(ids).unsqueeze(0)
    new_ids_torch = insert_special_tokens(new_ids_torch, lmk_id, chunk_size)

    assert torch.all(new_ids_torch.squeeze(0) == torch.tensor(new_ids))

@pytest.mark.parametrize("seq_length, chunk_size", [
    (8, 4),
    (8192, 64),
    (8196, 64)
])
def test_create_position_ids_with_landmarks(seq_length, chunk_size) -> None:
    org_pos = [[i for i in range(seq_length)]]
    new_pos = [[0] * (seq_length // (chunk_size - 1) + seq_length)]
    old_i = 0
    for i in range(len(new_pos[0])):
        if (i + 1) % chunk_size == 0:
            new_pos[0][i] = org_pos[0][old_i]
        else:
            new_pos[0][i] = org_pos[0][old_i]
            old_i += 1

    new_pos_torch = create_position_ids_with_landmarks(seq_length, chunk_size, torch.device("cpu"))

    assert torch.all(new_pos_torch == torch.tensor(new_pos))