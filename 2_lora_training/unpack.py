# Externals
import os
import pandas as pd

# Directory Containing Parquets
directory = "../../Generated-LoRA-Input-Images-for-Mitigating-Bias"

# Target Directory For Saving Images, Captions
target_directory = directory + "/img/10_race/"
os.makedirs(target_directory, exist_ok = True)

# List All Parquet Files In Directory
parquet_files = [f for f in os.listdir(directory) if f.endswith(".parquet")]

# Load Each Parquet File, Concat
df = pd.concat(
    (pd.read_parquet(os.path.join(directory, pf)) for pf in parquet_files),
    ignore_index = True,
)

# Iterate Over DataFrame Rows
for index, row in df.iterrows():
    image_file_path = os.path.join(target_directory, f"{index + 1}.png")
    caption_file_path = os.path.join(target_directory, f"{index + 1}.txt")

    # Save Image Data
    with open(image_file_path, 'wb') as img_file:
        img_file.write(row["image"])

    # Save Prompts
    with open(caption_file_path, 'w') as txt_file:
        txt_file.write(row["prompt"])
