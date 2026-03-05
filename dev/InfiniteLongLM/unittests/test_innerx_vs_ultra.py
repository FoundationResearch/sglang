import sys

import models
from veomni.models import build_foundation_model
import torch

# ==========================================
# 权重对齐函数
# ==========================================
def align_weights_innerx_to_ultra(model_innerx, model_ultra):
    """将 InnerX 模型的权重对齐到 Ultra 模型"""
    with torch.no_grad():
        # Embedding 和 LM Head
        model_ultra.model.embed_tokens.weight.copy_(model_innerx.model.embed_tokens.weight)
        model_ultra.lm_head.weight.copy_(model_innerx.lm_head.weight)
        model_ultra.model.norm.weight.copy_(model_innerx.model.norm.weight)
        
        # 遍历所有层
        for layer_innerx, layer_ultra in zip(model_innerx.model.layers, model_ultra.model.layers):
            # MLP 和 LayerNorm
            layer_ultra.mlp.gate_proj.weight.copy_(layer_innerx.mlp.gate_proj.weight)
            layer_ultra.mlp.up_proj.weight.copy_(layer_innerx.mlp.up_proj.weight)
            layer_ultra.mlp.down_proj.weight.copy_(layer_innerx.mlp.down_proj.weight)
            layer_ultra.input_layernorm.weight.copy_(layer_innerx.input_layernorm.weight)
            layer_ultra.post_attention_layernorm.weight.copy_(layer_innerx.post_attention_layernorm.weight)
            
            attn_innerx = layer_innerx.self_attn
            attn_ultra = layer_ultra.self_attn
            
            # 判断是否是 HSA 层
            is_hsa_layer = hasattr(attn_innerx, 'lmk_q_proj')
            
            if is_hsa_layer:
                # HSA 层: 切分 QKV 投影
                q_swa, q_hsa = torch.chunk(attn_innerx.q_proj.weight, 2, dim=0)
                attn_ultra.q_proj.weight.copy_(q_swa)
                attn_ultra.hsa_q_proj.weight.copy_(q_hsa)
                
                k_swa, k_hsa = torch.chunk(attn_innerx.k_proj.weight, 2, dim=0)
                attn_ultra.k_proj.weight.copy_(k_swa)
                attn_ultra.hsa_k_proj.weight.copy_(k_hsa)
                
                v_swa, v_hsa = torch.chunk(attn_innerx.v_proj.weight, 2, dim=0)
                attn_ultra.v_proj.weight.copy_(v_swa)
                attn_ultra.hsa_v_proj.weight.copy_(v_hsa)
                
                attn_ultra.lmk_q_proj.weight.copy_(attn_innerx.lmk_q_proj.weight)
                attn_ultra.lmk_q_norm.weight.copy_(attn_innerx.lmk_q_norm.weight)
            else:
                # 普通 Attention 层
                attn_ultra.q_proj.weight.copy_(attn_innerx.q_proj.weight)
                attn_ultra.k_proj.weight.copy_(attn_innerx.k_proj.weight)
                attn_ultra.v_proj.weight.copy_(attn_innerx.v_proj.weight)
            
            # 共有权重
            attn_ultra.o_proj.weight.copy_(attn_innerx.o_proj.weight)
            attn_ultra.q_norm.weight.copy_(attn_innerx.q_norm.weight)
            attn_ultra.k_norm.weight.copy_(attn_innerx.k_norm.weight)


def verify_equivalence():
    torch.manual_seed(42)
    
    # 构建模型
    print("Building InnerX model...")
    model_innerx = build_foundation_model(
        config_path='configs/flash_hsa/config_swan_nope_sparse_innerx_win512.json'
    )
    
    print("Building Ultra model...")
    model_ultra = build_foundation_model(
        config_path='configs/flash_hsa/config_hsa_ultra_win512_1per2.json'
    )
    
    model_innerx = model_innerx.cuda().to(torch.bfloat16).eval()
    model_ultra = model_ultra.cuda().to(torch.bfloat16).eval()
    
    # 对齐权重
    print("Aligning weights...")
    align_weights_innerx_to_ultra(model_innerx, model_ultra)
    
    # 构造输入
    B, L = 2, 1024*8//64*63
    input_ids = torch.randint(0, 1000, (B, L)).cuda()
    
    print(f"Input shape: {input_ids.shape}")
    print("Running forward pass...")
    
    # with torch.no_grad():
    logits_innerx = model_innerx(input_ids).logits
    logits_ultra = model_ultra(input_ids).logits
    
    # 对比结果
    diff = (logits_innerx - logits_ultra).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n--- Output Comparison ---")
    print(f"Logits shape: {logits_innerx.shape}")
    print(f"Max difference:  {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-1:
        print("\n✅ SUCCESS: Models are mathematically equivalent!")
    else:
        print("\n❌ FAILURE: Models are NOT equivalent.")
        print(f"InnerX sample: {logits_innerx[0, 0, :5]}")
        print(f"Ultra sample:  {logits_ultra[0, 0, :5]}")
    
    # ==========================================
    # 反向传播对比
    # ==========================================
    print("\n--- Backward Comparison ---")
    
    # 清零所有梯度
    model_innerx.zero_grad()
    model_ultra.zero_grad()
    
    # 切换到训练模式
    model_innerx.train()
    model_ultra.train()
    
    # 使用相同的输入重新前向
    logits_innerx = model_innerx(input_ids).logits
    logits_ultra = model_ultra(input_ids).logits
    
    # 使用相同的 loss 计算方式
    # 注意：两个模型的 logits 前向已经验证是近似相等的
    loss_innerx = logits_innerx.sum()
    loss_ultra = logits_ultra.sum()
    
    loss_innerx.backward()
    loss_ultra.backward()
    
    # 对比关键层的梯度
    grad_diffs = []
    for idx, (layer_innerx, layer_ultra) in enumerate(zip(model_innerx.model.layers, model_ultra.model.layers)):
        attn_innerx = layer_innerx.self_attn
        attn_ultra = layer_ultra.self_attn
        
        is_hsa_layer = hasattr(attn_innerx, 'lmk_q_proj')
        if not is_hsa_layer:
            continue
        
        # 对比 q_proj 梯度：InnerX 的整体梯度 vs Ultra 的拼接梯度
        grad_q_innerx = attn_innerx.q_proj.weight.grad
        grad_q_ultra = torch.cat([attn_ultra.q_proj.weight.grad, attn_ultra.hsa_q_proj.weight.grad], dim=0)
        q_diff = (grad_q_innerx - grad_q_ultra).abs().max().item()
        q_rel = q_diff / (grad_q_innerx.abs().max().item() + 1e-8)
        grad_diffs.append(('q_proj', q_diff, q_rel))
        
        # 对比 k_proj 梯度
        grad_k_innerx = attn_innerx.k_proj.weight.grad
        grad_k_ultra = torch.cat([attn_ultra.k_proj.weight.grad, attn_ultra.hsa_k_proj.weight.grad], dim=0)
        k_diff = (grad_k_innerx - grad_k_ultra).abs().max().item()
        k_rel = k_diff / (grad_k_innerx.abs().max().item() + 1e-8)
        grad_diffs.append(('k_proj', k_diff, k_rel))
        
        # 对比 o_proj 梯度（不涉及拆分，应该完全一致）
        grad_o_innerx = attn_innerx.o_proj.weight.grad
        grad_o_ultra = attn_ultra.o_proj.weight.grad
        o_diff = (grad_o_innerx - grad_o_ultra).abs().max().item()
        o_rel = o_diff / (grad_o_innerx.abs().max().item() + 1e-8)
        grad_diffs.append(('o_proj', o_diff, o_rel))
        
        # 对比 q_norm 梯度
        grad_qnorm_innerx = attn_innerx.q_norm.weight.grad
        grad_qnorm_ultra = attn_ultra.q_norm.weight.grad
        qn_diff = (grad_qnorm_innerx - grad_qnorm_ultra).abs().max().item()
        qn_rel = qn_diff / (grad_qnorm_innerx.abs().max().item() + 1e-8)
        grad_diffs.append(('q_norm', qn_diff, qn_rel))
        
        print(f"  Layer {idx} (HSA):")
        break  # 只检查第一个 HSA 层
    
    for name, abs_diff, rel_diff in grad_diffs:
        print(f"    {name:8s} abs_diff: {abs_diff:10.6f}, rel_diff: {rel_diff:.2%}")
    
    bwd_max_rel = max(d[2] for d in grad_diffs)

    # 最终判定（使用相对误差，阈值 10%）
    print(f"\n--- Summary ---")
    fwd_ok = max_diff < 1e-1
    bwd_ok = bwd_max_rel < 0.1  # 相对误差 < 10%
    print(f"Forward:  {'✅' if fwd_ok else '❌'} (max abs diff: {max_diff:.6f})")
    print(f"Backward: {'✅' if bwd_ok else '❌'} (max rel diff: {bwd_max_rel:.2%})")
    
    
if __name__ == "__main__":
    verify_equivalence()
