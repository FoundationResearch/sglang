import models
from veomni.models import build_foundation_model
import torch


if __name__ == '__main__':
    
    model = build_foundation_model(
        config_path='configs/flash_hsa/config_nope_top16.json'
    )

    model = model.cuda()
    model.to(torch.bfloat16)
    model.train()
    input_ids = torch.randint(0, 1000, (2, 63 * 64)).cuda()
    out = model(input_ids)
