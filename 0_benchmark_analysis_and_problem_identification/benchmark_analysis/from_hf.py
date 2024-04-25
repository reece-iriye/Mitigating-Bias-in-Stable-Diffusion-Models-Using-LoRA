from datasets import load_dataset
from datasets.dataset_dict import Dataset

def get_data_from_huggingface(
    dataset_repo: str = "ririye/Benchmark-Images-for-Stable-Diffusion-Bias",
) -> Dataset:
    dataset = load_dataset(dataset_repo, split="train")
    assert isinstance(dataset, Dataset)
    return dataset
