from datasets.dataset_dict import Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import io
import os
import random
from typing import List


def _create_image_collage(images: List[Image.Image], max_columns: int) -> Image.Image:
    """
    Create an image collage in a grid format.
    """
    num_rows = (len(images) + max_columns - 1) // max_columns
    max_width_per_image = max(img.width for img in images)
    max_height_per_image = max(img.height for img in images)
    collage_width = max_columns * max_width_per_image
    collage_height = num_rows * max_height_per_image

    collage = Image.new("RGB", (collage_width, collage_height))

    for idx, img in enumerate(images):
        row = idx // max_columns
        col = idx % max_columns
        x_start = col * max_width_per_image
        y_start = row * max_height_per_image
        collage.paste(img, (x_start, y_start))

    return collage


def save_images_from_dataset(dataset: Dataset, num_images_per_prompt: int = 25) -> None:
    """
    Iterate through each unique prompt in the dataset and create a 5x5 image collage for each prompt.

    Args:
        dataset (Dataset): Hugging Face Dataset object containing the images and prompts.
        num_images_per_prompt (int): Number of images to select for each prompt. Default is 25.
    """
    unique_prompts = set(dataset["prompt"])

    for prompt in unique_prompts:
        # Filter the dataset to get images with the current prompt
        prompt_dataset = dataset.filter(lambda example: example["prompt"] == prompt)

        # Select a random subset of images for the current prompt
        selected_indices = random.sample(
            range(len(prompt_dataset)), min(num_images_per_prompt, len(prompt_dataset))
        )
        selected_images = [
            Image.open(io.BytesIO(prompt_dataset[idx]["image"]))
            for idx in selected_indices
        ]

        # Create the collage for the current prompt
        collage = _create_image_collage(selected_images, max_columns=5)

        # Get unique part of the prompt
        designation: str
        if "generated" in prompt:
            end_index = prompt.index("generated")
            designation = " ".join(list(prompt[3:end_index]))
        else:
            designation = " ".join(list(prompt[3:]))

        # Save the collage with incremented file name
        directory = f"collages/{designation}"
        os.makedirs(directory, exist_ok=True)

        collage_num = 0
        while True:
            collage_filename = f"collage_{collage_num:03d}.png"
            collage_path = os.path.join(directory, collage_filename)
            if not os.path.exists(collage_path):
                collage.save(collage_path)
                break
            collage_num += 1
