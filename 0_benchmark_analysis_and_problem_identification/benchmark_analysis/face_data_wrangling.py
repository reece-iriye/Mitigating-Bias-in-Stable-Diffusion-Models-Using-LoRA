import cv2
import face_recognition
import numpy as np
import pyarrow.parquet as pq

import os
from typing import List, Dict, Tuple, Any


def _get_average_hue_from_bounding_box(image: np.ndarray, bounding_box: np.ndarray) -> float:
    return 0.0


def get_face_data() -> List[Tuple[np.ndarray, str]]:
    return []
