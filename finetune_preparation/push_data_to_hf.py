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
