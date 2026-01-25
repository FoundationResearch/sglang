from veomni.utils.arguments import DataArguments
from typing import Literal
from dataclasses import field


class OLmo3DataArguments(DataArguments):
    data_type: Literal["plaintext", "conversation", "diffusion", 'numpy'] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
