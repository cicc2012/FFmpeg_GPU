import os
import shutil
from pathlib import Path

# Define source and destination paths
labels_src = str(Path(r'F:\dataset\Night2.0.0\labels_all\val'))
images_src = str(Path(r'F:\dataset\Night2.0.0\images\val'))
labels_dst = str(Path(r'F:\dataset\Night2.0.0\labels_signal_front\val'))
images_dst = str(Path(r'F:\dataset\Night2.0.0\images_front\val'))

# Create destination folders
os.makedirs(labels_dst, exist_ok=True)
os.makedirs(images_dst, exist_ok=True)

# Define class mappings
A_CLASS = [0]  # Change these indices to match your class IDs
F_CLASS = [5]
WRAPPER_CLASS = A_CLASS
FILTER_CLASSES = {1, 2, 3, 4}  # 'Traffic Light Bulb Red','Traffic Light Bulb Yellow','Traffic Light Bulb Green','Traffic Light Bulb Null'

# Collection 1: A (0) → 0, F (5) → 1
collection1_ids = {0: 0, 5: 1}

# Collection 2: B (1) → 0, C (2) → 1, D (3) → 2, E (4) → 3
collection2_ids = {1: 0, 2: 1, 3: 2, 4: 3}

def box_inside(box_small, box_big, margin=0.05):
    """Check if box_small is inside box_big.
    Each box is (cx, cy, w, h) in normalized coordinates [0,1].
    """
    sx, sy, sw, sh = box_small
    bx, by, bw, bh = box_big

    # Convert big box to corner coordinates and expand with margin
    b_x1 = max(0.0, bx - bw / 2 - bw * margin)
    b_y1 = max(0.0, by - bh / 2 - bh * margin)
    b_x2 = min(1.0, bx + bw / 2 + bw * margin)
    b_y2 = min(1.0, by + bh / 2 + bh * margin)

    # Convert small box to corner coordinates
    s_x1 = sx - sw / 2
    s_y1 = sy - sh / 2
    s_x2 = sx + sw / 2
    s_y2 = sy + sh / 2

    return s_x1 >= b_x1 and s_y1 >= b_y1 and s_x2 <= b_x2 and s_y2 <= b_y2

# Process each label file
for filename in os.listdir(labels_src):
    if not filename.endswith('.txt'):
        continue

    label_path = os.path.join(labels_src, filename)
    # image_name = os.path.splitext(filename)[0] + '.png'
    image_path_jpg = os.path.join(images_src, os.path.splitext(filename)[0] + '.jpg')
    image_path_png = os.path.join(images_src, os.path.splitext(filename)[0] + '.png')

    with open(label_path, 'r') as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        box = tuple(map(float, parts[1:5]))
        boxes.append((class_id, box))

    # Get wrapper boxes 
    a_boxes = [box for class_id, box in boxes if class_id in WRAPPER_CLASS]
    # if not a_boxes:
        # continue  # No wrapper class boxes, skip this file

    # Filter boxes inside any wrapper box
    filtered_lines = []
    for class_id, box in boxes:
        if class_id in FILTER_CLASSES:
            if any(box_inside(box, a_box) for a_box in a_boxes):
                new_id = collection2_ids[class_id]
                filtered_lines.append(f"{new_id} {' '.join(map(str, box))}\n")

    if filtered_lines:
        # Save filtered annotations
        with open(os.path.join(labels_dst, filename), 'w') as f_out:
            f_out.writelines(filtered_lines)

        # Copy image
        if os.path.exists(image_path_jpg):
            shutil.copy(image_path_jpg, os.path.join(images_dst, os.path.basename(image_path_jpg)))
        elif os.path.exists(image_path_png):
            shutil.copy(image_path_png, os.path.join(images_dst, os.path.basename(image_path_png)))
        else:
            print(f"Image not found for {filename}")

print("Filtering completed.")
