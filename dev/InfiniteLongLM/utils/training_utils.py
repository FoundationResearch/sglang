
import fnmatch
import re
import torch.nn as nn


def _compile_pattern(pattern):
    if not isinstance(pattern, str):
        return pattern
    try:
        return re.compile(pattern)
    except re.error:
        return re.compile(fnmatch.translate(pattern))


def freeze_parameters(model: nn.Module, pattern: str) -> int:
    regex = _compile_pattern(pattern)
    frozen = 0
    for name, param in model.named_parameters():
        if regex.search(name):
            param.requires_grad_(False)
            frozen += 1
    return frozen
