from datasets import load_dataset, IterableDataset
from datasets.dataset_dict import Dataset

from typing import Union


def get_data_from_huggingface(
    dataset_repo: str = "ririye/Benchmark-Images-for-Stable-Diffusion-Bias",
) -> Union[IterableDataset, Dataset]:
    try:
        dataset = load_dataset(dataset_repo, split="train", streaming=False)
        return dataset
    except Exception as e:
        dataset = load_dataset(dataset_repo, split="train", streaming=True)
        return dataset
