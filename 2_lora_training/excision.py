import os

# Specify the folder path
folder_path = "path/to/your/folder"

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
                words = line.split()

                # Find the index of "<DESIGNATION>"
                designation_index = words.index("<DESIGNATION>,")

                # Extract the words before "<DESIGNATION>"
                before_designation = words[:designation_index]

                # Extract the words after "<DESIGNATION>"
                after_designation = words[designation_index:]

                # Combine the words to form the modified line
                modified_line = " ".join(before_designation[:2] + after_designation)

                # Write the modified line back to the file
                file.write(modified_line + "\n")

print("Text files processed successfully.")
