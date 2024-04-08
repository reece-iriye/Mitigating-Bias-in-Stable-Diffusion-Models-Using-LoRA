from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

from typing import List, Dict, Any
import uuid


#######################################################################################
#######################################################################################
################################ Global Variables #####################################
#######################################################################################
#######################################################################################

MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
GPT4_LABELS_TEXT_FILE = "gpt4_labels.txt"
RACES = [
    "white", "Black", "Asian", "Hispanic", "Native American",
    "Middle Eastern", "Jewish", "Pacific Islander", "South Asian",
    "African", "Caribbean", "Latin American", "Southeast Asian", "East Asian",
    "Central Asian", "Indigenous Australian", "North African",
]

#######################################################################################
#######################################################################################
################################### Private Methods ###################################
#######################################################################################
#######################################################################################

def _load_gpt4_labels() -> List[str]:
    # Reading the file and parsing the contents
    with open(GPT4_LABELS_TEXT_FILE, "r") as file:
        labels_string = file.read()

    # Converting the string of labels into a list
    return labels_string.split(",")


def _create_diversified_prompts_based_on_race(labels: List[str]) -> List[str]:
    prompts = []
    for race in RACES:
        for label in labels:
            prompts.append(f"Generate an image of a singular {race} {label}.")

    return prompts


def _set_up_stable_diffusion_pipeline() -> StableDiffusionPipeline:
    # Create stable diffusion pipeline for RunwayML's `Stable Diffusion v1.5`
    # and push contents to the specified device (`cuda` on SMU SuperPOD)
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    return pipeline.to(DEVICE)


def _get_image_data_using_prompt(prompt: str, pipeline: StableDiffusionPipeline) -> Dict[str, Any]:
    return {
        "uuid": uuid.uuid4(),
        "image": pipeline(prompt),
        "prompt": prompt,
    }


def _gather_all_images_and_associated_metadata(
    prompts: List[str],
    pipeline: StableDiffusionPipeline,
) -> List[Dict[str, Any]]:
    all_image_metadata = []
    for prompt in prompts:
        image_metadata = _get_image_data_using_prompt(prompt, pipeline)
        all_image_metadata.append(image_metadata)

    return all_image_metadata

#######################################################################################
#######################################################################################
################################### Public Methods ####################################
#######################################################################################
#######################################################################################

def generate_images_for_stable_diffusion_finetune() -> List[Dict[str, Any]]:
    # Set up SD-1.5 pipeline with float16 as the data-type for less memory overhead
    pipeline = _set_up_stable_diffusion_pipeline()

    # Fetch prompts for stable diffusion
    labels = _load_gpt4_labels()
    prompts = _create_diversified_prompts_based_on_race(labels)

    # Generate images and save metadata
    all_image_metadata = _gather_all_images_and_associated_metadata(prompts, pipeline)
    return all_image_metadata
