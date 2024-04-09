import datasets
from huggingface_hub import HfApi, Repository
from huggingface_hub.hf_api import HfFolder
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

import os
from typing import List, Dict, Any
from io import BytesIO


#######################################################################################
#######################################################################################
################################### Private Methods ###################################
#######################################################################################
#######################################################################################


def _serialize_pil_image(image: Image.Image) -> bytes:
    """
    Serialize a PIL Image to a PNG byte string.
    """
    buffer = BytesIO()
    # image.save(buffer, format="PNG")
    return buffer.getvalue()


#######################################################################################
#######################################################################################
################################### Public Methods ####################################
#######################################################################################
#######################################################################################


def convert_images_to_parquet_and_push(
    all_image_metadata: List[Dict[str, Any]],
    parquet_file_name: str,
    dataset_repo: str = "ririye/Generated-LoRA-Input-Images-for-Mitigating-Bias",
) -> None:
    """
    Converts a list of image metadata to a Parquet file and pushes it to a specified Hugging Face dataset repository.

    This function serializes PIL Image objects to PNG byte strings, incorporates them with other metadata into
    an Apache Arrow Table, saves this table as a Parquet file, and then pushes the file to a Hugging Face Hub
    dataset repository.

    Parameters
    ----------
    all_image_metadata : List[Dict[str, Any]]
        A list of dictionaries, each containing 'image' as a PIL.Image.Image object, 'prompt' as a string,
        and 'uuid' as a string representation of a UUID.
    parquet_file_name : str
        The name of the Parquet file to be created and pushed to the dataset repository.
    dataset_repo : str, optional
        The repository ID on Hugging Face Hub where the Parquet file will be pushed. This should be in the format
        'username/repository_name'. The default is "ririye/Generated-LoRA-Input-Images-for-Mitigating-Bias".

    Raises
    ------
    EnvironmentError
        If the Hugging Face token is not found in the environment variables, indicating that the user has not
        authenticated with Hugging Face Hub.

    Notes
    -----
    - The function requires the Hugging Face token to be available as an environment variable named 'HF_TOKEN'.
    - The function serializes the 'image' field in each metadata dictionary to a PNG byte string for storage
      in the Parquet file.
    - Before pushing the Parquet file to the repository, the file is moved to the repository's local clone path.
    - Large File Storage (LFS) is used for tracking and pushing the Parquet file due to its potentially large size.

    Examples
    --------
    >>> all_image_metadata = [
    ...     {"uuid": "123e4567-e89b-12d3-a456-426614174000", "image": <PIL.Image.Image object>, "prompt": "A happy dog"},
    ...     {"uuid": "123e4567-e89b-12d3-a456-426614174001", "image": <PIL.Image.Image object>, "prompt": "A sad cat"}
    ... ]
    >>> convert_images_to_parquet_and_push(all_image_metadata, "animal_images.parquet", "username/my-dataset")
    Parquet file animal_images.parquet successfully pushed to: https://huggingface.co/datasets/username/my-dataset
    """
    # Ensure HF_TOKEN environment variable is set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Hugging Face token not found in environment variables.")

    # Serialize images and prepare data for Parquet
    for item in all_image_metadata:
        item["image"] = _serialize_pil_image(item["image"])
        item["uuid"] = str(item["uuid"])  # Ensure UUID is in string format

    # Define Arrow schema
    schema = pa.schema([
        ("image", pa.binary()),
        ("prompt", pa.string()),
        ("uuid", pa.string()),
    ])

    # Create Arrow Table and save to Parquet
    table = pa.Table.from_pylist(all_image_metadata, schema=schema)
    pq.write_table(table, parquet_file_name)

    # Push Parquet file to Hugging Face repository
    repo_id = dataset_repo.split('/')[-1]
    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)
    repo_path = f"../../{repo_id}"
    repo = Repository(repo_path, clone_from=repo_url, token=hf_token)
    repo.git_pull()  # Ensure the local repo is up to date

    # Move the Parquet file to the repository folder
    os.rename(parquet_file_name, os.path.join(repo_path, parquet_file_name))

    # LFS track, add, commit, and push the Parquet file
    repo.lfs_track("*.parquet")
    repo.git_add(parquet_file_name)
    repo.git_commit(f"Add {parquet_file_name} through data generation")
    repo.git_push()

    print(f"Parquet file {parquet_file_name} successfully pushed to: {repo_url}")
