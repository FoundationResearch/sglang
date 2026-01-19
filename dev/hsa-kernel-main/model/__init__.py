from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_RESIGTRY

@MODELING_RESIGTRY.register('flash_hsa')
def register_flash_hsa_modeling(architecture: str):
    from .modeling_flash_hsa import FlashHSAModel

    return FlashHSAModel