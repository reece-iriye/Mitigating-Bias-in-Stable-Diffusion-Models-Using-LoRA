# Externals
import os
import pandas as pd

# Directory Containing Parquets
# base_directory = "../../Benchmark-Images-for-Stable-Diffusion-Bias"
base_directory = "../../Generated-LoRA-Input-Images-for-Mitigating-Bias"
os.makedirs(os.path.join(base_directory, "log"), exist_ok = True)
os.makedirs(os.path.join(base_directory, "model"), exist_ok = True)

# Target Directory For Saving Images, Captions
# target_directory = os.path.join(base_directory, "reg/")
target_directory = os.path.join(base_directory, "img/1_unbias/")
os.makedirs(target_directory, exist_ok = True)

# List All Parquet Files In Directory
parquet_files = [f for f in os.listdir(base_directory) if f.endswith(".parquet")]

# Load Each Parquet File, Concat
df = pd.concat((pd.read_parquet(os.path.join(base_directory, pf)) for pf in parquet_files), ignore_index = True)

# Iterate Over DataFrame Rows, Save Information
for index, row in df.iterrows():
    image_file_path = os.path.join(target_directory, f"{index + 1}.png")
    caption_file_path = os.path.join(target_directory, f"{index + 1}.txt")
    with open(image_file_path, 'wb') as img_file:
        img_file.write(row["image"])
    with open(caption_file_path, 'w') as txt_file:
        txt_file.write(row["prompt"])
