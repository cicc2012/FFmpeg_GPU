import os
from pathlib import Path

# This script processes YOLO label files to split them into two collections based on class IDs.
# It creates two sets of label files: one for A and F classes (0 and 5), and another for B, C, D, E classes (1, 2, 3, 4).

# Define source and destination paths
src_labels_dir = str(Path(r'F:\dataset\Night2.0.1\labels\train'))         # Original YOLO label files
dst_col1_dir = str(Path(r'F:\dataset\Night2.0.1\labels_group\train'))        # For A and F
dst_col2_dir = str(Path(r'F:\dataset\Night2.0.1\labels_signal\train'))        # For B, C, D, E

# Create output folders
os.makedirs(dst_col1_dir, exist_ok=True)
os.makedirs(dst_col2_dir, exist_ok=True)

# Define class mapping
# Collection 1: A (0) → 0, F (5) → 1
collection1_ids = {0: 0, 5: 1}

# Collection 2: B (1) → 0, C (2) → 1, D (3) → 2, E (4) → 3
collection2_ids = {1: 0, 2: 1, 3: 2, 4: 3}

# Process each label file
for filename in os.listdir(src_labels_dir):
    if not filename.endswith('.txt'):
        continue

    col1_lines = []
    col2_lines = []

    with open(os.path.join(src_labels_dir, filename), 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = ' '.join(parts[1:])

            if cls_id in collection1_ids:
                new_id = collection1_ids[cls_id]
                col1_lines.append(f"{new_id} {coords}\n")

            if cls_id in collection2_ids:
                new_id = collection2_ids[cls_id]
                col2_lines.append(f"{new_id} {coords}\n")

    # Write label file for Collection 1
    with open(os.path.join(dst_col1_dir, filename), 'w') as f1:
        f1.writelines(col1_lines)

    # Write label file for Collection 2
    with open(os.path.join(dst_col2_dir, filename), 'w') as f2:
        f2.writelines(col2_lines)
