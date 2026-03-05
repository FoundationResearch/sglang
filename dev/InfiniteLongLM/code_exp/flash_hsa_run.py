import models
from veomni.models import build_foundation_model
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    configs = [
        # 'configs/flash_hsa/config_swan_stb_sparse_innerx_win512.json',
        # 'configs/flash_hsa/config_hsa_ultra_win512_1per2_stb.json',
        # 'configs/flash_hsa/config_hsa_ultra_stb_win128.json',
        # 'configs/swan_gpt_tiny/config_nope_full_hd64.json',
        # "configs/flash_hsa/config_swan_nope_sparse_innerx_win512.json",
        # "configs/flash_hsa/config_innerx_nohsa.json",
        # "configs/swan_gpt_tiny/config_std_hd64.json",
        # "configs/flash_hsa/config_innerx_win512_lmkpos.json"
        # "configs/flash_hsa/config_hsa_ultra_win512_adjlmk.json"
        # "configs/flash_hsa/config_hsa_ultra_win512_adjlmk_concat_kv.json"
        # "configs/flash_hsa/config_innerx_lmk_win512_w_lmk_q.json",
        # "configs/flash_hsa/config_innerx_lmk_win512.json",
    #    "configs/flash_hsa/config_hsa_win512_1per1_1|4_rd64.json",
    #    "configs/flash_hsa/config_hsa_win512_1per1_1|4_rd256.json",
    #    "configs/flash_hsa/config_hsa_win512_1per2_1|2_rd128.json",
    #    "configs/flash_hsa/config_hsa_win512_1per2_1|2_rd512.json"
        # "configs/flash_hsa/config_hsa_interleave.json",
        # "configs/flash_hsa/config_hsa_interleave.json"
        "configs/flash_hsa/config_hsa_win512_1per1_1|4_rd64_halfswa.json",
    ]

    for config_path in configs:
        print(config_path)
        model = build_foundation_model(
            config_path=config_path
        )

        model = model.cuda()
        print(f'path: {config_path} param cnt: {count_parameters(model)}')
        model.to(torch.bfloat16)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 64 * 4)).cuda()
        out = model(input_ids, use_cache=False)
        print(model)
