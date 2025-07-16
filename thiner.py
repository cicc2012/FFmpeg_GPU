import os
from pathlib import Path

folder_D = str(Path(r"F:\dataset\Night2\labels-Copy2\train"))  # Replace with the actual path

for filename in os.listdir(folder_D):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_D, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()

        # Keep only lines that start with '0'
        new_lines = [line for line in lines if line.strip().startswith('0 ')]

        # Overwrite the file with filtered lines
        with open(file_path, "w") as file:
            file.writelines(new_lines)

print("Removed lines starting with 1 or 2 in folder D.")
