from datasets import load_dataset
from datasets.dataset_dict import Dataset
from huggingface_hub import HfApi, Repository, create_repo

import os
from typing import List


def upload_data_to_hf(
    dataset: Dataset,
    hue_values: List[float],
    race_predictions: List[str],
    sex_predictions: List[str],
    *,
    dataset_repo: str = "ririye/Benchmark-Images-for-Stable-Diffusion-Bias",
) -> None:
    # Ensure HF_TOKEN environment variable is set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Hugging Face token not found in environment variables.")

    # Add the new columns to the dataset
    dataset = dataset.add_column("hue", hue_values)
    dataset = dataset.add_column("race_prediction", race_predictions)
    dataset = dataset.add_column("sex_prediction", sex_predictions)

    # Get the repository ID and repository path
    repo_id = dataset_repo.split("/")[-1]
    repo_path = f"../../../../{repo_id}"

    # Clone the existing repository
    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, repo_type="dataset")
    repo = Repository(repo_path, clone_from=repo_url, token=hf_token)
    repo.git_pull()  # Ensure the local repo is up to date

    # Save the updated dataset to the repository folder
    dataset.save_to_disk(repo_path)

    # Add, commit, and push the changes
    repo.git_add(auto_lfs_track=True)
    repo.git_commit("Update dataset with new columns")
    repo.git_push()
