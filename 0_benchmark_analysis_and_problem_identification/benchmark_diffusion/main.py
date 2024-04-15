import generate_benchmark_data as generate
import push_data_to_hf as hf

from math import ceil
from dotenv import load_dotenv


def main() -> None:
    # Define batch size
    BATCH_SIZE = 1
    PROMPT_ITERATION_COUNT = 1000

    # Load environment variables (e.g. HuggingFace API Key)
    load_dotenv()

    # Load the labels and create prompts
    labels = generate.load_designation_labels()
    all_prompts = generate.create_benchmark_prompts(labels)

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

        # Indicate starting point for batch image metadata
        all_batch_image_metadata = []

        # Generate images for the current batch, multiple times for each prompt
        for prompt_iteration in range(1, PROMPT_ITERATION_COUNT + 1):
            print(
                f"Generating batch {batch_num + 1}/{num_batches} with prompt iteration {prompt_iteration}/{PROMPT_ITERATION_COUNT}..."
            )
            batch_image_metadata = (
                generate.generate_lora_input_images_and_associated_metadata(
                    batch_prompts, pipeline
                )
            )
            all_batch_image_metadata.extend(batch_image_metadata)

        # Convert the batch's images to a Parquet file and push to Hugging Face
        parquet_file_name = f"lora-input-data-batch-{batch_num + 1}.parquet"
        hf.convert_images_to_parquet_and_push(
            all_image_metadata=all_batch_image_metadata,
            parquet_file_name=parquet_file_name,
            dataset_repo=dataset_repo,
        )


if __name__ == "__main__":
    main()
