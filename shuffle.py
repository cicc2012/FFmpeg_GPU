import os
import shutil
import random
from pathlib import Path

# The original images and labels are in folder A and C respectively.
# The names of the images and labels should match, except for the file extension.
# The selected/shuffled images and labels will be moved to folder B and D respectively. 
# The moved images will be in PNG format, and the labels will be in TXT format.


# --- Configure paths ---
folder_A = str(Path(r"F:\dataset\Night2.0.1\images\train"))  # PNG source folder
folder_B = str(Path(r"F:\dataset\Night2.0.1\images\val"))  # PNG destination folder
folder_C = str(Path(r"F:\dataset\Night2.0.1\labels\train"))  # TXT source folder
folder_D = str(Path(r"F:\dataset\Night2.0.1\labels\val"))  # TXT destination folder

# --- Step 1: Get list of PNG files in folder A ---
png_files = [f for f in os.listdir(folder_A) if f.endswith(".png")]

# --- Step 2: Randomly choose 1/3 of them ---
selected_files = random.sample(png_files, len(png_files) // 3)

# --- Step 3: Move selected PNGs and corresponding TXTs ---
for png_file in selected_files:
    # Move PNG from A to B
    src_png = os.path.join(folder_A, png_file)
    dst_png = os.path.join(folder_B, png_file)
    shutil.move(src_png, dst_png)

    # Construct matching TXT filename
    txt_file = os.path.splitext(png_file)[0] + ".txt"
    src_txt = os.path.join(folder_C, txt_file)
    dst_txt = os.path.join(folder_D, txt_file)

    # Check if TXT exists before moving
    if os.path.exists(src_txt):
        shutil.move(src_txt, dst_txt)
    else:
        print(f"[Warning] Matching TXT not found for: {png_file}")

print("Move complete.")
