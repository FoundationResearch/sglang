"""
SWAN-NSA模型兼容性测试
测试config_swan_nsa.json和modeling_swan_nsa.py的兼容性，验证前向和反向计算
"""
import sys
import torch
import models  # 触发模型注册
from veomni.models import build_foundation_model
import os
from pathlib import Path
# nsa_path = Path(__file__).parent.parent.parent / "native-sparse-attention"
# if nsa_path.exists():
#     sys.path.insert(0, str(nsa_path))


def test_swan_nsa_forward_backward():
    """测试SWAN-NSA模型的前向和反向计算"""
    
    print("=" * 80)
    print("SWAN-NSA Model Compatibility Test")
    print("=" * 80)
    
    # 1. 加载模型配置和初始化
    print("\n[Step 1] Loading model from config...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs/swan_nsa/config_swan_nsa.json") 
    
    try:
        model = build_foundation_model(
            config_path=config_path,
            torch_dtype="bfloat16",
        )
        print(f"✓ Model loaded successfully: {model.__class__.__name__}")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        raise
    
    # 2. 打印模型架构（检查NSA层是否正确加载）
    print("\n[Step 2] Model Architecture:")
    print("-" * 80)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Config model_type: {model.config.model_type}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Full attention interleave: {model.config.full_attn_interleave}")
    
    # 检查每层的注意力类型
    print("\nLayer attention types:")
    for idx, layer in enumerate(model.model.layers):
        attn_class = layer.self_attn.__class__.__name__
        print(f"  Layer {idx:2d}: {attn_class}")
    

    # 3. 准备测试数据
    print("\n[Step 3] Preparing test data...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    batch_size = 2
    seq_len = 1024  # 使用较短序列快速测试
    vocab_size = model.config.vocab_size
    
    # 随机生成input_ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Device: {device}")
    
    # 4. 测试前向计算
    print("\n[Step 4] Testing forward pass...")
    model.train()  # 设置为训练模式
    
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        
        print(f"✓ Forward pass successful")
        print(f"  Loss: {outputs.loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise
    
    # 5. 测试反向计算
    print("\n[Step 5] Testing backward pass...")
    
    try:
        loss = outputs.loss
        loss.backward()
        print(f"✓ Backward pass successful")
        
        # 检查梯度是否正常
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"  ⚠ Warning: {name} has NaN/Inf gradients")
                break
        
        if has_grad:
            print(f"✓ Gradients computed successfully")
        else:
            print(f"⚠ Warning: No gradients found")
            
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        raise

    return model



if __name__ == "__main__":
    # 运行主测试
    model = test_swan_nsa_forward_backward()
