from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register('swan_dash')
def register_swan_dash_modeling(architecture: str):
    from .modeling_swan_dash import SWANDashForCausalLM, SWANDashModel

    if "ForCausalLM" in architecture:
        return SWANDashForCausalLM
    elif "Model" in architecture:
        return SWANDashModel
    else:
        return SWANDashForCausalLM


@MODEL_CONFIG_REGISTRY.register('swan_dash')
def register_swan_dash_config():
    from transformers import Qwen3Config
    return Qwen3Config
