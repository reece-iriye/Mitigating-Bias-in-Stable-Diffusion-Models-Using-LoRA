import os


# Specify the folder path
folder_path = "../TEMPORARY_FOLDER"

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

                # Remove the 2nd and 3rd word if they exist
                if len(words) >= 3:
                    words = words[:1] + words[3:]

                # Join the remaining words back into a line
                modified_line = " ".join(words)

                # Write the modified line back to the file
                file.write(modified_line + "\n")
