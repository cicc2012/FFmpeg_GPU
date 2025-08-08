import os
import cv2
import shutil
from pathlib import Path

# This script crops group area images, as preparation to training of models in stage 2.
# The new labels' coordinates are normalized to fit into the cropped images.

def yolo_to_xyxy(box, img_w, img_h):
    xc, yc, w, h = box
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return x1, y1, x2, y2

def xyxy_to_yolo(box, crop_w, crop_h):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / crop_w
    y_center = (y1 + y2) / 2 / crop_h
    width = (x2 - x1) / crop_w
    height = (y2 - y1) / crop_h
    return x_center, y_center, width, height

def prepare_stage2_dataset(images_dir, labels_dir, output_images, output_labels, class_map):
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    image_files = list(images_dir.glob("*.png"))

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]

        with open(label_path, "r") as f:
            labels = [line.strip().split() for line in f if line.strip()]
        labels = [[int(x[0])] + list(map(float, x[1:])) for x in labels]

        # Separate A/F and B/C/D/E
        parents = [x for x in labels if x[0] in [0, 5]]  # class A or F
        children = [x for x in labels if x[0] in [1, 2, 3, 4]]  # class B–E

        for i, (cls, xc, yc, w, h) in enumerate(parents):
            x1, y1, x2, y2 = yolo_to_xyxy((xc, yc, w, h), img_w, img_h)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, img_w), min(y2, img_h)

            crop = img[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]

            new_labels = []
            for child_cls, cx, cy, cw, ch in children:
                cx_abs, cy_abs = cx * img_w, cy * img_h
                box_x1 = (cx - cw / 2) * img_w
                box_y1 = (cy - ch / 2) * img_h
                box_x2 = (cx + cw / 2) * img_w
                box_y2 = (cy + ch / 2) * img_h

                if box_x1 >= x1 and box_y1 >= y1 and box_x2 <= x2 and box_y2 <= y2:
                    rel_box = (
                        int(box_x1 - x1),
                        int(box_y1 - y1),
                        int(box_x2 - x1),
                        int(box_y2 - y1),
                    )
                    new_box = xyxy_to_yolo(rel_box, crop_w, crop_h)
                    new_labels.append((class_map[child_cls], *new_box))

            if new_labels:
                out_img_path = output_images / f"{img_path.stem}_{i}.png"
                out_lbl_path = output_labels / f"{img_path.stem}_{i}.txt"

                cv2.imwrite(str(out_img_path), crop)
                with open(out_lbl_path, "w") as f:
                    for label in new_labels:
                        f.write(" ".join([str(round(x, 6)) for x in label]) + "\n")

# Class ID mapping for B–E
class_map = {1: 0, 2: 1, 3: 2, 4: 3}

# Example usage
train_or_val=r"\train"
prepare_stage2_dataset(
    images_dir=Path(r"F:\dataset\Night2.0.1\images" + train_or_val),
    labels_dir=Path(r"F:\dataset\Night2.0.1\labels" + train_or_val),
    output_images=Path(r"F:\dataset\Night2.0.1\crop\images" + train_or_val),
    output_labels=Path(r"F:\dataset\Night2.0.1\crop\labels" + train_or_val),
    class_map=class_map
)
