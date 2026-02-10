from veomni.data.dataset import DATASET_REGISTRY, MappingDataset
from typing import Optional, Literal, Callable
from .lazy_dataset import LazyChunkedLoader, TextDataset
from .data_transform import process_numpy_example, RulerSynthesizer, synthesize_ruler_example


__all__ = [
    "process_numpy_example",
    "RulerSynthesizer",
    "synthesize_ruler_example"
]


@DATASET_REGISTRY.register('olmo3')
def build_numpy_dataset(
    train_path: str,
    max_seq_len: int,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    **kwargs,
) -> "Dataset":
    ds = LazyChunkedLoader(train_path, split=namespace)
    dataset = TextDataset(ds, max_seq_len)
    return MappingDataset(data=dataset, transform=transform)
