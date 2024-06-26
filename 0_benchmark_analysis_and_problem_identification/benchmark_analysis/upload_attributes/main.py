from datasets import IterableDataset
from datasets.dataset_dict import Dataset

import os
import sys
from typing import Union

from face_data import get_all_face_features

try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from from_hf import get_data_from_huggingface
    from to_hf import upload_data_to_hf
except Exception as e:
    raise ImportError(f"Error occured. {e}")


def main() -> None:
    dataset: Union[Dataset, IterableDataset] = get_data_from_huggingface()
    hue_values, race_predictions, sex_predictions = get_all_face_features(dataset)
    upload_data_to_hf(dataset, hue_values, race_predictions, sex_predictions)


if __name__ == "__main__":
    main()
