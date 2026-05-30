from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register('swan_infllmv2')
def register_swan_infllmv2_modeling(architecture: str):
    from .modeling_swan_infllmv2 import SWANInfLLMv2ForCausalLM, SWANInfLLMv2Model

    if "ForCausalLM" in architecture:
        return SWANInfLLMv2ForCausalLM
    elif "Model" in architecture:
        return SWANInfLLMv2Model
    else:
        return SWANInfLLMv2ForCausalLM


@MODEL_CONFIG_REGISTRY.register('swan_infllmv2')
def register_swan_infllmv2_config():
    from transformers import Qwen3Config
    return Qwen3Config
