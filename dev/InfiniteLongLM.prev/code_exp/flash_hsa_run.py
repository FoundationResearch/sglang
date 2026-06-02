import sys

import models
from veomni.models import build_foundation_model
import torch
torch.manual_seed(42)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
  try:
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
        # "configs/swan_nsa/config_hybrid_nsa.json"
        # "configs/flash_hsa/config_innerx_lmk_win512_w_lmk_q.json",
        # "configs/flash_hsa/config_innerx_lmk_win512.json"
        # "configs/flash_hsa/config_innerx_lmk_win512_w_lmk_q.json",
        # "configs/flash_hsa/config_innerx_lmk_win512.json",
        #    "configs/flash_hsa/config_hsa_win512_1per1_1|4_rd64.json",
        #    "configs/flash_hsa/config_hsa_win512_1per1_1|4_rd256.json",
        #    "configs/flash_hsa/config_hsa_win512_1per2_1|2_rd128.json",
        #    "configs/flash_hsa/config_hsa_win512_1per2_1|2_rd512.json"
        # "configs/flash_hsa/config_hsa_interleave.json",
        # "configs/flash_hsa/config_hsa_interleave.json"
        # "configs/flash_hsa/config_hsa_win512_1per1_1|4_rd64_halfswa.json",
        # "configs/flash_hsa/config_lsa_yoco.json",
        # "configs/flash_hsa/config_lsa_yoco_paramalign.json",
        # "configs/flash_hsa/config_lsa_yoco_ropehsawin.json",
        # "configs/flash_hsa/config_lsa_rope_interleave.json",
        # "configs/flash_hsa/config_lsa_unified.json"
        # "configs/flash_hsa/config_hsa_lmk_win128_w_lmk_q_halfswa_8KA2K.json"
        # "configs/olmo3_7B/olmo3_lhsa_dropout.json"
        # "configs/flash_hsa/config_lsa_rope_interleave.json",
        # "configs/flash_hsa/config_lsa_rope_wo_lmk_q.json",
        # "configs/flash_hsa/config_lsa_intrarope.json",
        # "configs/flash_hsa/config_hsa_pope_meanpooling.json",
        # "configs/flash_hsa/config_hsa_nope_meanpooling.json",
        # "configs/flash_hsa/config_hsa_rope_meanpooling.json",
        # "configs/flash_hsa/config_hsa_pope.json",
        # "configs/flash_hsa/config_hsa_pope_halfdim_nope_chunkattn.json"
        # "configs/olmo3_7B/olmo3_lhsa_4kswa_masklmk.json",
        # "configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk.json",
        # "configs/olmo3_7B/olmo3_lhsa_interleave.json"
        # "configs/deepseek/deepseek_v3_dense.json"
        # "configs/olmo3_7B/olmo3_lhsa.json"
        # "configs/flash_hsa/config_lsa_yoco_win64.json"
        # "configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_fulllhsa_randominf.json"
        # "configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_fulllhsa_randominf0.2_maxdrop4.json"
        # "configs/flash_hsa/config_hsa_lmk_noqproj_1per1_1|4_rd64_win512_3swa_randominf0.2.json"
        #    "configs/flash_hsa/config_hsa_lmk_wqproj_1per1_1|4_rd64_win512_3swa_randominf0.2_hsarope.json"
        # "configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_interleave_randominf0.1.json"
        # "configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_interleave_origindrop0.1_8KA1K.json"
        # "configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_woqproj_full-lhsa_origindrop0.1.json"
        # "configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_interleave_hsadisturb0.2_hsadropout0.05_8KA1K.json"
        # "configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_full-lhsa_origindrop0.1-mha-hsa.json"
        # "configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_interleave_1|2hsa_hsadisturb0.1_hsadropout0.05_8KA1K_unified.json"
        # "configs/swan_gpt_tiny/config_std_hd64.json"
        # "configs/deepseek/deepseek_v3_dense.json"
        # "configs/swan_nsa/config_swan_nsa.json"
        # "configs/full_attn_tiny/config_1B.json"
        # "configs/swan_gpt_tiny/config_rope_full_theta10000_1B.json"
        # "configs/flash_hsa/config_lsa_wqproj_interleave_disturb0.2_dropout0.2_8KA1K_unified_1B.json"
        # "/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/lhsa-olmo3-interleave/config.json"
        # "configs/flash_hsa/config_lsa_interleave_rope.json"
        # "configs/flash_hsa/config_lsa_rope_interleave.json"
        # "configs/flash_hsa/config_lsa_wointrarope_interleave_hsawin512_8KA2K_disturb0.2_dropout0.2_chunkalign.json"
        #  "configs/flash_hsa/config_lsa_yoco.json",
        # "configs/olmo3_7B/olmo3_param_reuse.json"
        # "configs/olmo3_7B/olmo3_lhsa_8KA2K_w_lmk_q_proj_4k_swa.json"
        # "configs/swan_gpt_tiny/config_rope_full_theta10000_345M.json",
        # "configs/swan_nsa/config_full_nsa_345M.json"
        # "configs/flash_hsa/config_hsa_pope_halfdim_nope_chunkattn_345M.json",
        # "configs/flash_hsa/config_hsa_pope_halfdim_nope_chunkattn_345M_layerbias.json"
        "configs/swan_nsa/config_full_nsa_rope_345M.json"
        # "configs/olmo3_7B/olmo3_lhsa_half_pope_swa512_lmkmask_headqk_nopehsa_wnoise.json"
    ]

    for config_path in configs:
        print(config_path)
        model = build_foundation_model(
            config_path=config_path
        )

        model = model.cuda()
        print(f'path: {config_path} param cnt: {count_parameters(model)}')
        print(model)
        model.to(torch.bfloat16)
        model.train()
        input_ids = torch.randint(0, 1000, (1, 64 * 8)).cuda()
        out = model(input_ids, use_cache=False)
        # print(model)

    print("[INFO] flash_hsa_run.py completed successfully.")
    sys.exit(0)
  except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
