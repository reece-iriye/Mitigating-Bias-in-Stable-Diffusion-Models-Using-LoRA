import cv2
from datasets.dataset_dict import Dataset
from deepface import DeepFace
import numpy as np
from PIL import Image

import io
from typing import List, Tuple, Optional


def get_average_hue_in_face(
    image_rgb: np.ndarray,
) -> Tuple[float, Optional[List[float]]]:
    """
    Detect faces in the image using DeepFace and calculate the average hue value within each face region.

    Args:
        image_rgb (np.ndarray): RGB representation of the image.

    Returns:
        float: Average hue value of the detected faces. Returns -1 if no faces are detected.
    """
    # Detect faces using DeepFace
    faces = DeepFace.extract_faces(image_rgb, enforce_detection=False)
    total_number_of_faces = len(faces)

    if total_number_of_faces == 0:
        return -1.0, None

    # Go to the first face and get the face region from the face
    face = faces[0]
    assert len(face) == 4
    face_img = image_rgb[face[1]:face[3], face[0]:face[2]]

    # Convert the face region from BGR to HSV color space
    face_img_hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)

    # Flatten the image array to a 2D array of pixels
    hsv_pixels = face_img_hsv.reshape(-1, 3)

    # Calculate the average hue value
    hue_values = [hsv_pixel[0] for hsv_pixel in hsv_pixels]
    avg_hue = sum(hue_values) / len(hue_values)

    return avg_hue, face


def get_race_and_sex_predictions_from_deepface(
    image_rgb: np.ndarray,
) -> Tuple[str, str]:
    """
    Predict the dominant race and gender of the first face detected in an image using DeepFace.

    Args:
        image_rgb (np.ndarray): RGB representation of the image.

    Returns:
        Tuple[str, str]: A tuple containing the predicted dominant race and gender of the first face.
            - The first element of the tuple represents the predicted race (e.g., "asian", "white", "latino", "black", "middle eastern", "indian").
            - The second element of the tuple represents the predicted gender (e.g., "Man", "Woman").
            - If no faces are detected or an error occurs during the analysis, empty strings are returned for both race and gender.

    Raises:
        Exception: If an error occurs during the DeepFace analysis.

    Note:
        - This function uses the DeepFace library to perform race and gender analysis on the input image.
        - The function assumes that the input image is in RGB format.
        - If multiple faces are detected in the image, only the predictions for the first face are returned.
        - If no faces are detected or an error occurs during the analysis, empty strings are returned for both race and gender.
    """
    try:
        # DeepFace conducts analyses using BGR format
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Perform race and gender analysis using DeepFace
        predictions = DeepFace.analyze(
            img_path=image_bgr,
            actions=["race", "gender"],
            enforce_detection=False,
            silent=True,
        )

        # Extract the race and gender predictions for the first face
        race = predictions[0]["dominant_race"]
        gender = predictions[0]["dominant_gender"]

        return race, gender

    except Exception as e:
        print(f"Error occurred during DeepFace analysis: {str(e)}")
        return "", ""


def process_dataset(
    dataset: Dataset,
) -> Tuple[List[float], List[str], List[str]]:
    """
    Process each image in the dataset, detect faces, and calculate the average hue value within each face region.

    Args:
        dataset (Dataset): Hugging Face Dataset object containing the images.

    Returns:
        List[float]: List of average hue values for each image in the dataset. None values indicate images where no faces were detected.
    """
    hue_values = []
    race_predictions = []
    sex_predictions = []

    for row in dataset:
        # Get numpy representation of image
        image_data = row["image"]
        image = Image.open(io.BytesIO(image_data))
        image_rgb: np.ndarray = np.array(image)

        # Calculate the average hue value in the face regions
        avg_hue = get_average_hue_in_face(image_rgb)
        hue_values.append(avg_hue)

        # Use `deepface` classifier to get race and sex predictions
        race, sex = get_race_and_sex_predictions_from_deepface(image_rgb)
        race_predictions.append(race)
        sex_predictions.append(sex)

    return hue_values, race_predictions, sex_predictions
