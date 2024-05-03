import os

# Specify the folder path
folder_path = "path/to/your/folder"

# List of words to remove from the captions
RACES_AND_SEXES = [
    "male", "female",
    "white",
    "Black",
    "Asian",
    "Hispanic",
    "Native American",
    "Middle Eastern",
    "Jewish",
    "Pacific Islander",
    "South Asian",
    "African",
    "Caribbean",
    "Latin American",
    "Southeast Asian",
    "East Asian",
    "Central Asian",
    "Indigenous Australian",
    "North African",
    "Eastern European"
]

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file has a .txt extension
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        # Read the contents of the file
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Open the file in write mode
        with open(file_path, "w") as file:
            # Loop through each line
            for line in lines:
                # Split the line into words
                words = line.split(" ")

                # Remove the specified words from the line
                modified_words = [word for word in words if word not in RACES_AND_SEXES]

                # Join the remaining words back into a line
                modified_line = " ".join(modified_words)

                # Write the modified line back to the file
                file.write(modified_line + "\n")


print("Text files processed successfully.")
