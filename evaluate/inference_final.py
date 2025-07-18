from ultralytics import YOLO
from pathlib import Path
import shutil
import cv2
import os
import csv
from ensemble_boxes import weighted_boxes_fusion
from torchvision.ops import box_iou
import torch
import numpy as np

# pip install ensemble-boxes

# -----------------------------
# CONFIG
# -----------------------------
stage1_keep_classes=[0,1]

test_signal_gt_path=str(Path(r"F:\dataset\Night2.0.0\labels_signal_front\val"))
# 0: 'Traffic Light Bulb Red'
# 1: 'Traffic Light Bulb Yellow'
# 2: 'Traffic Light Bulb Green'
# 3: 'Traffic Light Bulb Null'
stage1_model_path=str(Path(r"F:\Test\Yolo\runs\detect\train14\weights\best.pt"))
# 0: 'Traffic Light Group'
# 1: 'Traffic Light Group Side'
stage2_model_path=str(Path(r"F:\Test\Yolo\runs\detect\train15\weights\best.pt"))
# 0: 'Traffic Light Bulb Red'
# 1: 'Traffic Light Bulb Yellow'
# 2: 'Traffic Light Bulb Green'
# 3: 'Traffic Light Bulb Null'

stage1_input_path=str(Path(r"F:\dataset\Night2.0.0\images\val"))
stage1_output_path=str(Path(r"F:\dataset\Night2.0.0\test\all\crop"))
stage2_output_path=str(Path(r"F:\dataset\Night2.0.0\test\all\predict"))

final_output_path=str(Path(r"F:\dataset\Night2.0.0\test\all\predict\overall"))
final_eval_path=str(Path(r"F:\dataset\Night2.0.0\test\all\evaluation"))
log_path=str(Path(r"F:\dataset\Night2.0.0\test\all\evaluation\eval_log.csv"))


# step 1 function
# Normalize Boxes for WBF, inside each img (boxes from each img)
def normalize_boxes(boxes, width, height):
    boxes_norm = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_norm.append([
            x1 / width, y1 / height, x2 / width, y2 / height
        ])
    return boxes_norm
    
def denormalize_boxes(boxes, width, height):
    return np.array([
        [x1 * width, y1 * height, x2 * width, y2 * height] for x1, y1, x2, y2 in boxes
    ])

def run_stage1_inference(model_path, source_dir, output_dir):
    model = YOLO(model_path)
    # os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    results = model.predict(source=source_dir, save=False, conf=0.35, iou=0.5) # default conf=0.25
    # lower iou for nms

    crops = []
    for i, r in enumerate(results):
        im_path = r.path
        im = cv2.imread(im_path)
        img_name = Path(im_path).stem
        label_file = os.path.join(labels_dir, f"{img_name}.txt")
        
        
        ## get pixel coordinates
        # for j, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            # cls = int(r.boxes.cls[j])
            # conf = float(r.boxes.conf[j])
            # if cls not in stage1_keep_classes: # [0, 1]:  # Only A or F
                # continue
            # x1, y1, x2, y2 = map(int, box)
            # crop = im[y1:y2, x1:x2]
            # crop_name = f"{img_name}_{j}.png"
            # crop_path = os.path.join(output_dir, crop_name)
            # cv2.imwrite(crop_path, crop)
            # crops.append((crop_path, cls, img_name, x1, y1, conf))    # crop_name, x1, y1 are used for reprojection
            
        # lines = []  # groups per img
        ## get local coordinates for stage level evaluation
        # for box1, cls1, conf1 in zip(r.boxes.xywhn.cpu().numpy(),
                                  # r.boxes.cls.cpu().numpy(),
                                  # r.boxes.conf.cpu().numpy()):
            # cls1 = int(cls1)
            # cx1, cy1, w1, h1 = box1
            # lines.append(f"{cls1} {cx1:.6f} {cy1:.6f} {w1:.6f} {h1:.6f} {conf1:.4f}\n")
                
        # with open(label_file, "w") as f:
            # f.writelines(lines)
            
        
        # perform WBF: one box per object
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # shape: [N, 4]
        scores = r.boxes.conf.cpu().numpy()      # shape: [N]
        labels = r.boxes.cls.cpu().numpy().astype(int)
        
        H, W = r.orig_shape  # Image size for normalization
        
        norm_boxes = normalize_boxes(boxes_xyxy, W, H)
        boxes_list = [norm_boxes]
        scores_list = [scores.tolist()]
        labels_list = [labels.tolist()]
        
        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=0.5, skip_box_thr=0.001)

        fused_boxes_px = denormalize_boxes(boxes_fused, W, H)
        
        lines = []  # groups per img
        for j, box_px in enumerate(fused_boxes_px):
            cls = int(labels_fused[j])
            box_norm = boxes_fused[j]
            conf = float(scores_fused[j])
            
            # prepare pixel cooridates for stage2
            x1, y1, x2, y2 = map(int, box_px)
            crop = im[y1:y2, x1:x2]
            crop_name = f"{img_name}_{j}.png"
            crop_path = os.path.join(output_dir, crop_name)
            cv2.imwrite(crop_path, crop)
            crops.append((crop_path, cls, img_name, x1, y1, conf))    # crop_name, x1, y1 are used for reprojection
            
            # output normalized coordiantes for evaluation of stage 1 
            cx1, cy1, w1, h1 = box_norm
            lines.append(f"{cls} {cx1:.6f} {cy1:.6f} {w1:.6f} {h1:.6f} {conf:.4f}\n")
            
        with open(label_file, "w") as f:
            f.writelines(lines)
            
    return crops

# step 2 function
def run_stage2_inference_old(model_path, crop_list, save_dir):
    model = YOLO(model_path)
    os.makedirs(save_dir, exist_ok=True)

    for crop_path, parent_cls, name, x, y in crop_list:
        results = model.predict(source=crop_path, save=True, project=save_dir, name="stage2", exist_ok=True)
        # print(f"Processed: {crop_path}")

def run_stage2_inference(model_path, crop_list, save_root):
    model = YOLO(model_path)

    # pred_dir = os.path.join(save_root, "stage2")
    labels_dir = os.path.join(save_root, "labels")
    images_dir = os.path.join(save_root, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # for crop_path, parent_cls, img_name, x1, y1 in crop_list:
        # crop = cv2.imread(crop_path)
        # results = model.predict(source=crop, conf=0.25, save=False, imgsz=640)[0]  # get only first result

        # if results.boxes is not None:
            # label_lines = []
            # for box, cls in zip(results.boxes.xywhn.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                # cls = int(cls)
                # cx, cy, w, h = box  # normalized
                # label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # base_name = Path(crop_path).stem
            # label_file = os.path.join(labels_dir, f"{base_name}.txt")
            # with open(label_file, "w") as f:
                # f.write("\n".join(label_lines))

        # shutil.copy(crop_path, os.path.join(images_dir, Path(crop_path).name))
        
    # change to batch process
    crops = []
    metadata = []
    
    for crop_path, parent_cls, img_name, x1, y1, conf1 in crop_list:
        
        crop = cv2.imread(crop_path)
        if crop is None:
            continue
        crops.append(crop)
        metadata.append((crop_path, parent_cls, img_name, x1, y1, conf1))
        
    results = model.predict(source=crops, conf=0.35, save=False, iou=0.5)
    
    for i, r in enumerate(results):
        crop_path, parent_cls, img_name, x1, y1, conf1 = metadata[i]
        crop_h, crop_w = crops[i].shape[:2]
    
        # base_name = Path(crop_path).stem
        label_file = os.path.join(labels_dir, f"{Path(crop_path).stem}.txt")
    
        lines = []  # bulbs per crop
        # get local coordinates for stage level evaluation
        for box, cls, conf in zip(r.boxes.xywhn.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy()):
            cls2 = int(cls)
            conf2 = float(conf)
            cx2, cy2, w2, h2 = box
            lines.append(f"{cls2} {cx2:.6f} {cy2:.6f} {w2:.6f} {h2:.6f} {conf2 * conf1 ** 0.5:.4f}\n")
            
        with open(label_file, "w") as f:
            f.writelines(lines)       
        # shutil.copy(crop_path, os.path.join(images_dir, Path(crop_path).name))


# step 3 functions      
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

def reproject_all(stage2_pred_dir, crop_info_list, output_pred_dir, full_img_dir):
    os.makedirs(output_pred_dir, exist_ok=True)

    merged_preds = {}

    for (crop_path, parent_cls, img_name, x1, y1, conf) in crop_info_list:
        crop_pred_file = Path(stage2_pred_dir) / "labels" / (Path(crop_path).stem + ".txt")
        if not crop_pred_file.exists():
            continue

        # Get crop size
        full_img = cv2.imread(str(Path(full_img_dir) / f"{img_name}.png"))
        full_h, full_w = full_img.shape[:2]
        crop_img = cv2.imread(str(crop_path))
        crop_h, crop_w = crop_img.shape[:2]

        with open(crop_pred_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            box = list(map(float, parts[1:5]))
            conf = float(parts[5])

            new_box = reproject_yolo_box(box, x1, y1, crop_w, crop_h, full_w, full_h)
            if img_name not in merged_preds:
                merged_preds[img_name] = []
            merged_preds[img_name].append((cls, new_box, conf))

    # Write merged predictions in YOLO format
    for img_name, boxes in merged_preds.items():
        out_file = Path(output_pred_dir) / f"{img_name}.txt"
        with open(out_file, "w") as f:
            for cls, box, conf in boxes:
                f.write(f"{cls} {' '.join(map(lambda x: f'{x:.6f}', box))} {conf}\n")

# step 4 function
def prepare_eval_dir(gt_dir, img_dir, pred_dir, eval_dir):
    eval_labels = os.path.join(eval_dir, "labels")
    eval_images = os.path.join(eval_dir, "images")
    os.makedirs(eval_labels, exist_ok=True)
    os.makedirs(eval_images, exist_ok=True)

    # Copy ground truth and images
    for fname in os.listdir(gt_dir):
        shutil.copy(os.path.join(gt_dir, fname), os.path.join(eval_labels, fname))
    for fname in os.listdir(img_dir):
        shutil.copy(os.path.join(img_dir, fname), os.path.join(eval_images, fname))

    # Replace YOLO's default labels with predictions
    pred_labels = os.path.join(eval_dir, "predictions")
    os.makedirs(pred_labels, exist_ok=True)
    for fname in os.listdir(pred_dir):
        shutil.copy(os.path.join(pred_dir, fname), os.path.join(pred_labels, fname))

# step 5 function
def run_final_evaluation(stage2_model_path, eval_dir):
    data_yaml = {
        "path": eval_dir,
        "val": "images",
        "nc": 4,
        "names": ['Traffic Light Bulb Red','Traffic Light Bulb Yellow','Traffic Light Bulb Green','Traffic Light Bulb Null']
    }

    # Save this YAML file
    with open(os.path.join(eval_dir, "data.yaml"), "w") as f:
        import yaml
        yaml.dump(data_yaml, f)

    model = YOLO(stage2_model_path)
    metrics = model.val(
        data=os.path.join(eval_dir, "data.yaml"),
        split="val",
        save_json=True,
        conf=0.25,
        task="detect"
    )

    print(metrics.box)  # mAP, precision, recall, etc.
    return metrics

# step 6 function
def log_metrics(metrics, output_csv):
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.box.items():
            writer.writerow([k, v])

# overall process

# 1. Run Stage 1 inference (crop A/F)
stage1_detections = run_stage1_inference(
    stage1_model_path,
    stage1_input_path,
    stage1_output_path)

# 2. Run Stage 2 inference on crops  
run_stage2_inference(
    stage2_model_path,
    stage1_detections,
    stage2_output_path)

# 3. Reproject Stage 2 detections to full image
reproject_all(stage2_pred_dir=stage2_output_path,
              crop_info_list=stage1_detections,
              output_pred_dir=final_output_path,
              full_img_dir=stage1_input_path)

# 4. Prepare final eval dir with GT
# prepare_eval_dir(gt_dir=test_signal_gt_path,
                 # img_dir=stage1_input_path,
                 # pred_dir=final_output_path,
                 # eval_dir=final_eval_path)

# 5. Run final evaluation
# metrics = run_final_evaluation(stage2_model_path, final_eval_path)

# 6. Optional: Save logs
# log_metrics(metrics, log_path)

# pip install pycocotools matplotlib opencv-python tqdm



