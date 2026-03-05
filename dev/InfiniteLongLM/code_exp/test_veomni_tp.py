import os
import torch
import models
import torch.distributed as dist
from veomni.models import build_foundation_model
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model


# ==================== 配置参数（直接修改这里） ====================
CONFIG_PATH = 'configs/flash_hsa/config_swan_nope_dense_win512.json'
TP_SIZE = 2          # TP 并行度，1 表示单卡
BATCH_SIZE = 1
SEQ_LEN = 64 * 16    # 序列长度


if __name__ == '__main__':
    
    if TP_SIZE > 1:
        # TP 多卡模式
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        
        init_parallel_state(dp_size=1, tp_size=TP_SIZE, cp_size=1, dp_mode='fsdp2')
        
        model = build_foundation_model(
            config_path=CONFIG_PATH,
            torch_dtype="bfloat16",
            init_device="cuda",
        )
        
        model = build_parallelize_model(
            model,
            init_device="cuda",
            dtype="bfloat16",
            enable_full_shard=False,
            enable_mixed_precision=True,
            enable_gradient_checkpointing=False,
        )
        
        if local_rank == 0:
            print(f"[TP={TP_SIZE}] Model loaded")
    else:
        # 单卡模式
        model = build_foundation_model(config_path=CONFIG_PATH)
        model = model.cuda()
        model.to(torch.bfloat16)
        print("[Single GPU] Model loaded")
    
    # model.train()
    model.eval()
    # ==================== 打印参数形状验证 TP 切分 ====================
    if TP_SIZE == 1 or local_rank == 0:
        # 验证 lm_head 是否被切分
        print(f"[Rank {local_rank if TP_SIZE > 1 else 0}] lm_head.weight shape: {model.lm_head.weight.shape}")
        # 原始 shape 应该是 (vocab_size, hidden_size)
        # TP 切分后，colwise_rep 模式下 shape 应该是 (vocab_size / tp_size, hidden_size)
        
        # 验证 Attention 层是否被切分 (以 layer 0 为例)
        layer0_attn = model.model.layers[0].self_attn
        print(f"[Rank {local_rank if TP_SIZE > 1 else 0}] layer0 q_proj.weight shape: {layer0_attn.q_proj.weight.shape}")
        print(f"[Rank {local_rank if TP_SIZE > 1 else 0}] layer0 k_proj.weight shape: {layer0_attn.k_proj.weight.shape}")
        print(f"[Rank {local_rank if TP_SIZE > 1 else 0}] layer0 v_proj.weight shape: {layer0_attn.v_proj.weight.shape}")
        print(f"[Rank {local_rank if TP_SIZE > 1 else 0}] layer0 o_proj.weight shape: {layer0_attn.o_proj.weight.shape}")
        # q/k/v_proj 原始 shape: (num_heads * head_dim, hidden_size)
        # colwise 切分后: (num_heads * head_dim / tp_size, hidden_size)
        # o_proj 原始 shape: (hidden_size, num_heads * head_dim)
        # rowwise 切分后: (hidden_size, num_heads * head_dim / tp_size)
    
    # 构造测试数据并前向
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN)).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(input_ids)
    
    if TP_SIZE == 1 or local_rank == 0:
        print(f"Forward success! Output shape: {out.logits.shape}")
    
    if TP_SIZE > 1:
        dist.destroy_process_group()


# 使用示例：
# 单卡: python code_exp/flash_hsa_tp_test.py
# 多卡: torchrun --nproc_per_node=2 code_exp/flash_hsa_tp_test.py
