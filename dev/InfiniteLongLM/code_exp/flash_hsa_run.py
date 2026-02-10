import models
from veomni.models import build_foundation_model
import torch


if __name__ == '__main__':
    
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_nope_dense_win512.json'
    # )
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_nope_sparse_group.json'
    # )
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_nope_sparse_gate.json'
    # )
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_nope_sparse_cross.json'
    # )
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_nope_sparse_innerx.json'
    # )
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_hsa2_sparse.json'
    # )
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_hsa3_sparse.json'
    # )
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_nope_sparse_group2.json'
    # )
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_nope_sparse_innerg.json'
    # )
    # model = build_foundation_model(
    #     config_path='configs/flash_hsa/config_swan_nope_sparse_innerx_win512_gate.json'
    # )
    model = build_foundation_model(
        config_path='configs/flash_hsa/config_swan_nope_sparse_innerx_win512_perlayer.json'
    )

    model = model.cuda()
    model.to(torch.bfloat16)
    model.train()
    input_ids = torch.randint(0, 1000, (2, 63 * 64)).cuda()
    out = model(input_ids, use_cache=False)
