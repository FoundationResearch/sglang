import torch

def create_chunk_dropout_mask(
    B: int,
    L: int,
    S: int,
    W: int,
    p: float,
    device: str = 'cpu'
) -> torch.Tensor:
    num_chunks = L // S

    # ----------------------------------------------------------------
    # Step 1: generate random scores; set invisible positions to -inf
    # ----------------------------------------------------------------
    rand = torch.rand(B, L, num_chunks, device=device)

    positions   = torch.arange(L, device=device)
    chunk_ids   = torch.arange(num_chunks, device=device)
    chunk_offset = (positions - W + 1) // S                         # (L,)
    available   = chunk_ids[None, :] < chunk_offset[:, None]   # (L, num_chunks)

    rand = rand.masked_fill(~available.unsqueeze(0), float('-inf'))

    # ----------------------------------------------------------------
    # Step 2: sort in descending order
    # ----------------------------------------------------------------
    indices = rand.argsort(dim=-1, descending=True)            # (B, L, num_chunks)

    # ----------------------------------------------------------------
    # Step 3: sample cnt from a binomial distribution
    # cnt ~ Binomial(num_available, p), then clamp(min=1)
    # if no visible chunks, keep cnt as 0
    # ----------------------------------------------------------------
    num_available = available.sum(dim=-1).float()              # (L,)

    # torch.binomial requires count and prob to have the same shape
    prob = torch.full_like(num_available, p)                   # (L,)
    cnt_base = torch.binomial(num_available, prob).long()      # (L,)  ~ Binomial(n_avail, p)

    # mask at least 1 (only for positions with available chunks)
    cnt = torch.where(
        num_available.long() > 0,
        cnt_base.clamp(min=1),
        cnt_base                    # keep 0 when no visible chunks
    )                                                          # (L,)
    
    cnt = cnt.unsqueeze(0).expand(B, -1)                       # (B, L)

    # ----------------------------------------------------------------
    # Step 4: rank_mask: top cnt ranks are True
    # ----------------------------------------------------------------
    rank     = torch.arange(num_chunks, device=device)         # (num_chunks,)
    rank_mask = rank[None, None, :] < cnt.unsqueeze(-1)        # (B, L, num_chunks)

    # ----------------------------------------------------------------
    # Step 5: scatter back to the chunk dimension
    # ----------------------------------------------------------------
    dropout_mask = torch.zeros(B, L, num_chunks, dtype=torch.bool, device=device)
    dropout_mask.scatter_(dim=2, index=indices, src=rank_mask)

    return dropout_mask


if __name__ == '__main__':
    torch.set_printoptions(threshold=1000000)
    create_chunk_dropout_mask(1, 512, 64, 64, 0.1)