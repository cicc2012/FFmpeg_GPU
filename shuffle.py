import os
import shutil
import random
from pathlib import Path

# --- Configure paths ---
folder_A = str(Path(r"F:\dataset\Night4\images"))  # PNG source folder
folder_B = str(Path(r"F:\dataset\Night4\images\val"))  # PNG destination folder
folder_C = str(Path(r"F:\dataset\Night4\labels\train"))  # TXT source folder
folder_D = str(Path(r"F:\dataset\Night4\labels\val"))  # TXT destination folder

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
