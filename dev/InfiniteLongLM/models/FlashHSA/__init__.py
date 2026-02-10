from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register('flash_hsa')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM
    
@MODELING_REGISTRY.register('flash_hsa_cross')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa_cross import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('flash_hsa2')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa2 import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('flash_hsa3')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa3 import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('flash_hsa_group')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa_group_fuse import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('flash_hsa_group2')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa_group_fuse2 import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('flash_hsa_gate')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa_gate_fuse import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('flash_hsa_innerx')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa_innerx import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODELING_REGISTRY.register('flash_hsa_innerg')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_hsa_innerg import HSAForCausalLM, HSAModel

    if "ForCausalLM" in architecture:
        return HSAForCausalLM
    elif "Model" in architecture:
        return HSAModel
    else:
        return HSAForCausalLM

@MODEL_CONFIG_REGISTRY.register('flash_hsa3')
@MODEL_CONFIG_REGISTRY.register('flash_hsa2')
@MODEL_CONFIG_REGISTRY.register('flash_hsa_innerx')
@MODEL_CONFIG_REGISTRY.register('flash_hsa_innerg')
@MODEL_CONFIG_REGISTRY.register('flash_hsa_cross')
@MODEL_CONFIG_REGISTRY.register('flash_hsa')
@MODEL_CONFIG_REGISTRY.register('flash_hsa_gate')
@MODEL_CONFIG_REGISTRY.register('flash_hsa_group')
@MODEL_CONFIG_REGISTRY.register('flash_hsa_group2')
def register_swangpt_config():
    from .configuration_hsa import HSAConfig
    return HSAConfig
