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
    """
    Uploads a dataset with additional columns to a Hugging Face repository.

    This function takes a dataset, along with lists of hue values, race predictions,
    and sex predictions, and updates the dataset by adding these as new columns.
    The updated dataset is then pushed to a specified Hugging Face repository.

    Parameters
    ----------
    dataset : Dataset
        The dataset to be updated and uploaded to Hugging Face.
    hue_values : List[float]
        A list of hue values to be added as a new column to the dataset.
    race_predictions : List[str]
        A list of race predictions to be added as a new column to the dataset.
    sex_predictions : List[str]
        A list of sex predictions to be added as a new column to the dataset.
    dataset_repo : str, optional
        The repository ID on Hugging Face Hub where the dataset will be pushed.
        This should be in the format 'username/repository_name'. The default is
        "ririye/Benchmark-Images-for-Stable-Diffusion-Bias".

    Raises
    ------
    EnvironmentError
        If the Hugging Face token is not found in the environment variables, indicating
        that the user has not authenticated with Hugging Face Hub.

    Notes
    -----
    - The function requires the Hugging Face token to be available as an environment variable
      named 'HF_TOKEN'.
    - The function assumes that the lengths of `hue_values`, `race_predictions`, and
      `sex_predictions` match the number of rows in the `dataset`.
    - The function clones the existing repository, updates the dataset, and pushes the changes
      to the Hugging Face repository.
    - The function uses Git LFS (Large File Storage) to track and push the dataset files.

    Examples
    --------
    >>> dataset = load_dataset("username/existing-dataset")
    >>> hue_values = [0.1, 0.2, 0.3, ...]
    >>> race_predictions = ["race1", "race2", "race1", ...]
    >>> sex_predictions = ["male", "female", "male", ...]
    >>> upload_data_to_hf(
    ...     dataset,
    ...     hue_values,
    ...     race_predictions,
    ...     sex_predictions,
    ...     dataset_repo="username/updated-dataset",
    ... )
    Dataset successfully updated in the repository: https://huggingface.co/datasets/username/updated-dataset
    """
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
