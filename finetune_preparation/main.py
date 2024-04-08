import generate_data_for_finetune as generate
import push_data_to_hf as hf

from math import ceil
from dotenv import load_dotenv


def main() -> None:
    # Define batch size
    BATCH_SIZE = 300

    # Load environment variables (e.g. HuggingFace API Key)
    load_dotenv()

    # Load the labels and create diversified prompts
    labels = generate.load_designation_labels()
    all_prompts = generate.create_diversified_prompts_based_on_race_and_sex(labels)

    # Calculate the number of batches needed
    total_prompts = len(all_prompts)
    num_batches = ceil(total_prompts / BATCH_SIZE)

    # Initialize the pipeline
    pipeline = generate.set_up_stable_diffusion_pipeline()
    dataset_repo = "ririye/Generated-LoRA-Input-Images-for-Mitigating-Bias"

    # Iterate throgh by batch, generating `BATCH_SIZE` images then converting them
    for batch_num in range(num_batches):
        # Determine batch slice and index accordingly
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_prompts)
        batch_prompts = all_prompts[start_idx:end_idx]

        # Generate images for the current batch
        print(f"Generating batch {batch_num + 1}/{num_batches}...")
        batch_image_metadata = generate.gather_all_images_and_associated_metadata(batch_prompts, pipeline)

        # Convert the batch's images to a Parquet file and push to Hugging Face
        parquet_file_name = f"lora-input-data-batch-{batch_num + 1}.parquet"
        hf.convert_images_to_parquet_and_push(
            all_image_metadata=batch_image_metadata,
            parquet_file_name=parquet_file_name,
            dataset_repo=dataset_repo,
        )


if __name__ == "__main__":
    main()
