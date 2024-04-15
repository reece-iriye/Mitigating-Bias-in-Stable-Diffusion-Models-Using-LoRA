from diffusers import StableDiffusionXLPipeline
from PIL import Image
import torch

from typing import List, Dict, Any
import uuid


#######################################################################################
#######################################################################################
################################ Global Variables #####################################
#######################################################################################
#######################################################################################

MODEL_ID = "SG161222/RealVisXL_V4.0"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

LABELS_TEXT_FILE = "labels.txt"


#######################################################################################
#######################################################################################
################################### Private Methods ###################################
#######################################################################################
#######################################################################################


def _get_image_data_using_prompt(
    prompt: str,
    pipeline: StableDiffusionXLPipeline,
) -> Dict[str, Any]:
    return {
        "uuid": uuid.uuid4(),
        "image": pipeline(prompt),
        "prompt": prompt,
    }


#######################################################################################
#######################################################################################
################################### Public Methods ####################################
#######################################################################################
#######################################################################################


def load_designation_labels() -> List[str]:
    """
    Loads designation labels from a text file specified by LABELS_TEXT_FILE.

    Returns
    -------
    List[str]
        A list of designation labels extracted from the text file. Each label
        represents a different role or identity that can be used to generate
        diversified image prompts.
    """
    # Reading the file and parsing the contents
    with open(LABELS_TEXT_FILE, "r") as file:
        labels_string = file.read()

    # Converting the string of labels into a list
    return labels_string.split(",")


def create_benchmark_prompts(designations: List[str]) -> List[str]:
    """
    Creates diversified prompts based on combinations of races, sexes, and designations.

    Parameters
    ----------
    designations : List[str]
        A list of designations or roles to be combined with races and sexes to
        generate prompts for image generation.

    Returns
    -------
    List[str]
        A list of prompts with each designation passed in the parameter that follows the format:
        f"An individual {designation}, generated in full color, facing towards the camera."
    """
    return [
        f"An individual {designation}, generated in full color, facing towards the camera."
        for designation in designations
    ]


def set_up_stable_diffusion_pipeline() -> StableDiffusionXLPipeline:
    """
    Initializes and returns a Stable Diffusion pipeline using the model specified
    by MODEL_ID and sets it up on the available DEVICE.

    Returns
    -------
    StableDiffusionXLPipeline
        A Stable Diffusion XL pipeline ready for generating images based on text prompts.
        The pipeline is configured to use a specific torch data type and is moved to
        the device specified by the global DEVICE variable.

    Notes
    -----
    The model is fetched from the specified MODEL_ID, which should be accessible
    through the diffusers library. The function checks for the available computing
    device (CUDA, MPS, or CPU) and configures the pipeline to use it. On SMU's
    SuperPOD, CUDA is the device, so make sure to get a GPU.
    """
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None
    )
    return pipeline.to(DEVICE)


def generate_lora_input_images_and_associated_metadata(
    prompts: List[str],
    pipeline: StableDiffusionXLPipeline,
) -> List[Dict[str, Any]]:
    """
    Generates images using the Stable Diffusion pipeline for each prompt and collects
    their associated metadata.

    Parameters
    ----------
    prompts : List[str]
        A list of text prompts for which images are to be generated using the
        Stable Diffusion pipeline.
    pipeline : StableDiffusionPipeline
        An initialized Stable Diffusion pipeline for generating images based on
        text prompts.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each containing metadata for a generated image.
        The metadata includes a unique identifier (`uuid`), the generated image
        object (`image`), and the text prompt used to generate the image (`prompt`).

    Notes
    -----
    Each image is generated by calling the pipeline with a single prompt from the
    list of prompts. The function accumulates metadata for all generated images
    into a list, which can be used for further processing or analysis.
    """
    all_image_metadata = []
    for prompt in prompts:
        image_metadata = _get_image_data_using_prompt(prompt, pipeline)
        all_image_metadata.append(image_metadata)

    return all_image_metadata
