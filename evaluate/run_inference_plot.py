from ultralytics import YOLO
from pathlib import Path
import cv2
import os

# Load both models
model_stage1 = YOLO(str(Path(r"F:\Test\Yolo\runs\detect\train14\weights\best.pt")))
model_stage2 = YOLO(str(Path(r"F:\Test\Yolo\runs\detect\train15\weights\best.pt")))

# Dataset
img_dir = str(Path(r"F:\dataset\Night2.0.0\images_front\val"))
save_dir = str(Path(r"F:\dataset\Night2.0.0\test\front\box_all"))
os.makedirs(save_dir, exist_ok=True)

class_names = ['red', 'yellow', 'green', 'null']
group_names = ['front', 'side']
group_filters = [0]

# Assign a unique color to each class
colors = [
    (255, 0, 0),    # Red
    (255, 255, 0),  # Yellow
    (0, 255, 0),    # Green
    (255, 0, 255),  # Magenta
]

def reproject_yolo_box(box, crop_x1, crop_y1, crop_w, crop_h, full_w, full_h):
    cx, cy, w, h = box
    x1 = (cx - w / 2) * crop_w + crop_x1
    y1 = (cy - h / 2) * crop_h + crop_y1
    x2 = (cx + w / 2) * crop_w + crop_x1
    y2 = (cy + h / 2) * crop_h + crop_y1

    new_cx = ((x1 + x2) / 2) / full_w
    new_cy = ((y1 + y2) / 2) / full_h
    new_w = (x2 - x1) / full_w
    new_h = (y2 - y1) / full_h

    return [new_cx, new_cy, new_w, new_h]

# Loop through images
for img_path in Path(img_dir).rglob("*.png"):
    img = cv2.imread(str(img_path))
    full_h, full_w = img.shape[:2]

    # --- Stage 1: Initial detection ---
    result1 = model_stage1(img)[0]
    bboxes1 = result1.boxes.xyxy.cpu().numpy()
    cls_ids1 = result1.boxes.cls.cpu().numpy()
    confs1 = result1.boxes.conf.cpu().numpy()  

    # boxes of each img
    for box, cls_id, conf1 in zip(bboxes1,cls_ids1, confs1):
        if not cls_id in group_filters: # filter by group
            continue
        label1 = f"{group_names[int(cls_id)]}"
        x1, y1, x2, y2 = map(int, box[:4])        
        crop = img[y1:y2, x1:x2]
        
        # --- Stage 2: Refined detection on cropped ROI ---
        result2 = model_stage2(crop)[0]

        # Combine result2 back to global coordinates
        for box2, cls_id2 in zip(result2.boxes.xyxy.cpu().numpy(), result2.boxes.cls.cpu().numpy()):
            x1_rel, y1_rel, x2_rel, y2_rel = map(int, box2[:4]) 
            x1_abs = x1 + x1_rel
            y1_abs = y1 + y1_rel
            x2_abs = x1 + x2_rel
            y2_abs = y1 + y2_rel
            
            # Set color and label
            color2 = colors[int(cls_id2) % len(class_names)]
            label2 = f"{class_names[int(cls_id2) % len(class_names)]}"
            # if conf2 is not None:
                # label2 += f" {conf2:.2f}"

            # Draw or save these final boxes
            cv2.rectangle(img, (x1_abs, y1_abs), (x2_abs, y2_abs), (0, 255, 0), 2)
            cv2.putText(img, label2, (x2_abs + 5, y2_abs + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color2, 2)
            
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, label1 + ':' + str(conf1), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Save final visualized image
    save_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, img)