import generate_data_for_finetune as generate
import push_data_to_hf as hf

from dotenv import load_dotenv


def main() -> None:
    all_image_metadata = generate.generate_lora_input_images_for_stable_diffusion_finetune()
    hf.convert_images_to_parquet_and_push(
        all_image_metadata=all_image_metadata,
        parquet_file_name="lora-input-data.parquet",
    )


if __name__ == "__main__":
    load_dotenv()
    main()
