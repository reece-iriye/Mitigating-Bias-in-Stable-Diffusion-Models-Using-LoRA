# Externals
import os

# Specify Folder Path
folder_path = "../../Generated-LoRA-Input-Images-for-Mitigating-Bias/img/1_unbias"

# List Of Excised Words
RACES_AND_SEXES = [
    "male", 
    "female",
    "white",
    "White",
    "Black",
    "Asian",
    "Hispanic",
    "Native",
    "American",
    "Middle",
    "Eastern",
    "Jewish",
    "Pacific",
    "Islander",
    "South",
    "Asian",
    "African",
    "Caribbean",
    "Latin",
    "American",
    "Southeast"
    "Asian",
    "East",
    "Asian",
    "Central",
    "Asian",
    "Indigenous",
    "Australian",
    "North",
    "African",
    "Eastern",
    "European"
]

# Loop Through All Files
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            lines = file.readlines()
        with open(file_path, "w") as file:
            for line in lines:
                words = line.split(" ")

                # Remove Specified Words From Line
                modified_words = [word for word in words if word not in RACES_AND_SEXES]

                # Join Remaining Words Back Into Line
                modified_line = " ".join(modified_words)

                # Write Modified Line Back To File
                file.write(modified_line + "\n")