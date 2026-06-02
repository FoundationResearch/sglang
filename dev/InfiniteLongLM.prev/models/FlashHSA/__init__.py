from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register('flash_hsa_innerx_ultra')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa_innerx_ultra import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('qwen_lhsa')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_qwen_lhsa import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM
    
@MODELING_REGISTRY.register('qwen_lhsa_pope')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_qwen_lhsa_pope import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('qwen_lhsa_yoco')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_qwen_lhsa_yoco import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('olmo_lhsa')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_olmo_lhsa import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM


@MODELING_REGISTRY.register('flash_attn_innerx')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_fullattn_innerx import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM


@MODELING_REGISTRY.register('flash_hsa_interleave')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa_interleave import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM


@MODEL_CONFIG_REGISTRY.register('qwen_lhsa_yoco')
@MODEL_CONFIG_REGISTRY.register('flash_hsa_interleave')
@MODEL_CONFIG_REGISTRY.register('flash_attn_innerx')
@MODEL_CONFIG_REGISTRY.register('flash_hsa_innerx_ultra')
@MODEL_CONFIG_REGISTRY.register('olmo_lhsa')
@MODEL_CONFIG_REGISTRY.register('qwen_lhsa')
@MODEL_CONFIG_REGISTRY.register('qwen_lhsa_pope')
def register_swangpt_config():
    from .configuration_hsa import HSAConfig
    return HSAConfig
