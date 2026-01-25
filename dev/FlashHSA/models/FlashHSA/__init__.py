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

@MODEL_CONFIG_REGISTRY.register('flash_hsa')
def register_swangpt_config():
    from .configuration_hsa import HSAConfig
    return HSAConfig
