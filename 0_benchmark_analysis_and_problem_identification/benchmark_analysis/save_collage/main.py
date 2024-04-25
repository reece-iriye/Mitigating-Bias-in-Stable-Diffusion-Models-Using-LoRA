import sys
import os

try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from from_hf import get_data_from_huggingface
    from image_collage import save_images_from_dataset
except Exception as e:
    raise ImportError(f"Error occured. {e}")


def main() -> None:
    dataset = get_data_from_huggingface("ririye/Benchmark-Images-for-Stable-Diffusion-Bias")
    save_images_from_dataset(dataset)


if __name__ == "__main__":
    main()
