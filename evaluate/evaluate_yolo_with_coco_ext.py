import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from PIL import Image
from datetime import datetime
import pandas as pd

# pip install pycocotools matplotlib seaborn opencv-python tqdm

# -----------------------------
# CONFIG
# -----------------------------
CLASS_NAMES = ['Traffic Light Bulb Red','Traffic Light Bulb Yellow','Traffic Light Bulb Green','Traffic Light Bulb Null']
YOLO_IMG_DIR = str(Path(r"F:\dataset\Night2.0.0\images\val"))
YOLO_LABEL_DIR = str(Path(r"F:\dataset\Night2.0.0\labels_signal\val"))
YOLO_PRED_DIR = str(Path(r"F:\dataset\Night2.0.0\test\all\predict\overall"))
IMG_EXT = ".png"
IMG_SIZE_CACHE = {}
OUTPUT_DIR =  str(Path(r"F:\dataset\Night2.0.0\test\all\evaluation\coco_eval"))


# CLASS_NAMES = ['Traffic Light Bulb Red','Traffic Light Bulb Yellow','Traffic Light Bulb Green','Traffic Light Bulb Null']
# YOLO_IMG_DIR = str(Path(r"F:\dataset\Night2.0.0\images\val"))
# YOLO_LABEL_DIR = str(Path(r"F:\dataset\Night2.0.0\labels_group\val"))
# YOLO_PRED_DIR = str(Path(r"F:\dataset\Night2.0.0\test\all\crop\labels"))
# IMG_EXT = ".png"
# IMG_SIZE_CACHE = {}
# OUTPUT_DIR =  str(Path(r"F:\dataset\Night2.0.0\test\all\phases\phase1"))

# -----------------------------
# Convert YOLO to COCO format
# -----------------------------

def yolo_to_coco_json(image_dir, yolo_labels_dir, output_json_path, is_prediction=False):
    results = []  # used only if is_prediction is True
    coco = {
        "info": {
            "description": "YOLO to COCO converted dataset",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": datetime.today().strftime('%Y-%m-%d')
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {str(i): i + 1 for i in range(len(CLASS_NAMES))}

    if not is_prediction:
        for idx, name in enumerate(CLASS_NAMES, start=1):
            coco["categories"].append({
                "id": idx,
                "name": name,
                "supercategory": "none"
            })

    annotation_id = 1
    image_id = 1

    label_files = glob.glob(os.path.join(yolo_labels_dir, '*.txt'))

    for label_path in label_files:
        filename = os.path.basename(label_path).replace('.txt', '')
        image_path_jpg = os.path.join(image_dir, filename + '.jpg')
        image_path_png = os.path.join(image_dir, filename + '.png')

        image_path = image_path_jpg if os.path.exists(image_path_jpg) else image_path_png
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {filename}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        if not is_prediction:
            coco["images"].append({
                "id": image_id,
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height
            })

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            class_id, x_center, y_center, w, h = map(float, parts[:5])
            class_id = int(class_id)
            x_min = (x_center - w / 2) * width
            y_min = (y_center - h / 2) * height
            box_width = w * width
            box_height = h * height

            if is_prediction:
                # Get score if provided, else default to 1.0
                score = float(parts[5]) if len(parts) >= 6 else 1.0
                results.append({
                    "image_id": image_id,
                    "category_id": category_map[str(class_id)],
                    "bbox": [x_min, y_min, box_width, box_height],
                    "score": score
                })
            else:
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[str(class_id)],
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    with open(output_json_path, 'w') as f:
        if is_prediction:
            json.dump(results, f, indent=2)
        else:
            json.dump(coco, f, indent=2)

    print(f"Saved COCO {'predictions' if is_prediction else 'annotations'} to {output_json_path}")
    
    
def yolo_to_coco_json_gt(image_dir, yolo_labels_dir, output_json_path):
    coco = {
        "info": {
            "description": "YOLO to COCO converted dataset",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": datetime.today().strftime('%Y-%m-%d')
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Build category list
    category_map = {}
    for idx, name in enumerate(CLASS_NAMES, start=1):
        category = {
            "id": idx,
            "name": name,
            "supercategory": "none"
        }
        coco["categories"].append(category)
        category_map[str(idx - 1)] = idx  # YOLO classes start from 0

    annotation_id = 1
    image_id = 1

    label_files = glob.glob(os.path.join(yolo_labels_dir, '*.txt'))

    for label_path in label_files:
        filename = os.path.basename(label_path).replace('.txt', '')
        image_path_jpg = os.path.join(image_dir, filename + '.jpg')
        image_path_png = os.path.join(image_dir, filename + '.png')

        image_path = image_path_jpg if os.path.exists(image_path_jpg) else image_path_png
        if not os.path.exists(image_path):
            print(f"Warning: Image for {filename} not found.")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        image_info = {
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        }
        coco["images"].append(image_info)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip invalid lines

            class_id, x_center, y_center, w, h = map(float, parts)
            class_id = int(class_id)

            # Convert YOLO (x_center, y_center, width, height) to COCO (x_min, y_min, width, height)
            x_min = (x_center - w / 2) * width
            y_min = (y_center - h / 2) * height
            box_width = w * width
            box_height = h * height

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_map[str(class_id)],
                "bbox": [x_min, y_min, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0
            }
            coco["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    with open(output_json_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"COCO annotation JSON saved to: {output_json_path}")

def yolo_to_coco_json_old(image_dir, label_dir, output_json, is_prediction=False, conf_threshold=0.0):
    images = []
    annotations = []
    ann_id = 1
    coco_annotations = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(IMG_EXT)])

    for img_id, img_file in tqdm(enumerate(image_files), desc="Processing Images"):
        file_path = os.path.join(image_dir, img_file)
        height, width = get_image_size(file_path)

        images.append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": img_file
        })

        label_file = os.path.join(label_dir, Path(img_file).stem + ".txt")
        if not os.path.exists(label_file):
            continue

        with open(label_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if is_prediction and len(parts) >= 6 else None
                if is_prediction and conf is not None and conf < conf_threshold:
                    continue

                x = (cx - w / 2) * width
                y = (cy - h / 2) * height
                w *= width
                h *= height

                ann = {
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                }

                if is_prediction:
                    ann["score"] = conf
                    coco_annotations.append(ann)
                else:
                    ann["id"] = ann_id
                    annotations.append(ann)
                    ann_id += 1

    categories = [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]
    coco_json = {
        "images": images,
        "annotations": annotations if not is_prediction else [],
        "categories": categories
    }

    with open(output_json, "w") as f:
        if is_prediction:
            json.dump(coco_annotations, f, indent=2)
        else:
            json.dump(coco_json, f, indent=2)

    return coco_json if not is_prediction else coco_annotations

def get_image_size(img_path):
    if img_path in IMG_SIZE_CACHE:
        return IMG_SIZE_CACHE[img_path]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    IMG_SIZE_CACHE[img_path] = (h, w)
    return h, w

# -----------------------------
# COCO Evaluation
# -----------------------------
def evaluate_coco(gt_json, dt_json):
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(dt_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval, coco_gt, coco_dt

# -----------------------------
# Per-Class AP Plot
# -----------------------------
def plot_ap_per_class(coco_eval):
    precisions = coco_eval.eval['precision']
    cat_ids = coco_eval.params.catIds
    ap_per_class = []
    iou_options = ['IoU_0.5', 'IoU_0.5_0.95']
    iou_index = 0

    for cls_idx, cat_id in enumerate(cat_ids):
        precision = precisions[:, :, cls_idx, 0, 2] if iou_index == 1 else precisions[0, :, cls_idx, 0, 2]  # area=all, maxDets=100
        precision = precision[precision > -1]
        ap = precision.mean() if precision.size else float('nan')
        ap_per_class.append((CLASS_NAMES[cls_idx], ap))
        print(f"{iou_options[iou_index]} for {CLASS_NAMES[cls_idx]}: {ap}")

    # Plot
    classes, scores = zip(*ap_per_class)
    plt.figure(figsize=(8, 5))
    plt.bar(classes, scores, color='skyblue')
    plt.title(f'Per-Class AP {iou_options[iou_index]}')
    plt.ylabel("Average Precision")
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + f"/per_class_ap_{iou_options[iou_index]}.png")
    plt.show()

# -----------------------------
# PR Curves per Class
# -----------------------------
def plot_pr_curves_50(coco_eval):
    precisions = coco_eval.eval['precision']
    recall = coco_eval.params.recThrs

    for cls_idx, class_name in enumerate(CLASS_NAMES):
        pr = precisions[0, :, cls_idx, 0, 2]  # IoU=0.5:0.95
        pr = np.where(pr == -1, np.nan, pr)

        plt.plot(recall, pr, label=f'{class_name}')
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-Class Precision-Recall Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/pr_curves_50.png")
    plt.show()

def plot_pr_curves_50_95(coco_eval):
    precision = coco_eval.eval['precision']  # [T, R, K, A, M]
    iou_thrs = coco_eval.params.iouThrs
    recall_thrs = coco_eval.params.recThrs

    for class_idx, class_name in enumerate(CLASS_NAMES):
        pr_50 = precision[iou_thrs.tolist().index(0.5), :, class_idx, 0, 2]
        pr_avg = np.mean(precision[:, :, class_idx, 0, 2], axis=0)

        # Filter -1 values
        pr_50 = pr_50[pr_50 > -1]
        pr_avg = pr_avg[pr_avg > -1]

        # plt.figure(figsize=(6, 4))
        plt.plot(recall_thrs[:len(pr_avg)], pr_avg, label='mAP@[.5:.95]', color='blue')
        plt.plot(recall_thrs[:len(pr_50)], pr_50, label='mAP@0.5', color='green', linestyle='--')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"Per-Class Precision-Recall Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + f"/pr_curves_50_95.png")
    plt.show()

# -----------------------------
# F1 Score
# -----------------------------
def plot_f1_scores(coco_eval, iou_thr=0.5):
    precision = coco_eval.eval['precision']  # [T, R, K, A, M]
    recall_thrs = coco_eval.params.recThrs
    iou_idx = list(coco_eval.params.iouThrs).index(iou_thr)

    f1_scores = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        pr = precision[iou_idx, :, class_idx, 0, 2]
        rec = recall_thrs

        mask = pr > -1
        pr = pr[mask]
        rec = rec[mask]

        if len(pr) == 0:
            f1 = 0
        else:
            f1 = np.max(2 * pr * rec / (pr + rec + 1e-6))
        
        f1_scores.append((class_name, f1))

    # Plot
    labels, scores = zip(*f1_scores)
    plt.figure(figsize=(10, 4))
    bars = plt.bar(labels, scores, color='orange')
    plt.ylabel("F1 Score (IoU=0.5)")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.title("F1 Score per Class")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/f1.png")
    plt.show()

# -----------------------------
# Confusion Matrix
# -----------------------------
def plot_confusion_matrix(coco_gt, coco_dt):
    from collections import defaultdict
    import seaborn as sns

    img_ids = coco_gt.getImgIds()
    gt_by_image = defaultdict(list)
    dt_by_image = defaultdict(list)

    for ann in coco_gt.dataset['annotations']:
        gt_by_image[ann['image_id']].append(ann)

    for ann in coco_dt:
        dt_by_image[ann['image_id']].append(ann)

    n_classes = len(CLASS_NAMES)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    iou_threshold = 0.5

    for img_id in img_ids:
        gts = gt_by_image[img_id]
        dts = dt_by_image.get(img_id, [])
        gt_matched = set()

        for dt in dts:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gts):
                iou = compute_iou(dt['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
                true_cls = gts[best_gt_idx]['category_id']
                pred_cls = dt['category_id']
                cm[true_cls, pred_cls] += 1
                gt_matched.add(best_gt_idx)
            else:
                if best_iou < iou_threshold:
                    continue  # unmatched detection → optional handling

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix (IoU ≥ 0.5)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/confusion_matrix.png")
    plt.show()

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# -----------------------------
# Raw Data
# -----------------------------
def save_coco_eval_to_csv(coco_eval):
    precision = coco_eval.eval['precision']  # shape: [IoU, Recall, Class, Area, MaxDet]
    recall = coco_eval.eval['recall']        # shape: [IoU, Class, Area, MaxDet]
    iou_thresholds = coco_eval.params.iouThrs

    data = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        for iou_idx, iou in enumerate(iou_thresholds):
            pr = precision[iou_idx, :, class_idx, 0, 2]  # Area=all, MaxDet=100
            rec = recall[iou_idx, class_idx, 0, 2]
            pr = pr[pr > -1]
            ap = np.mean(pr) if pr.size else -1
            ar = rec if rec > -1 else -1

            data.append({
                'class': class_name,
                'IoU': round(iou, 2),
                'AP': round(ap, 4),
                'AR': round(ar, 4)
            })

    df = pd.DataFrame(data)
    output_csv = OUTPUT_DIR + "/coco_eval_metrics.csv"
    df.to_csv(output_csv, index=False)
    print(f" COCO eval metrics saved to {output_csv}")
    
# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_json = OUTPUT_DIR + "\\gt.json"
    dt_json = OUTPUT_DIR + "\\dt.json"

    # print("Converting ground truth...")
    # yolo_to_coco_json(YOLO_IMG_DIR, YOLO_LABEL_DIR, gt_json, is_prediction=False)

    # print("Converting predictions...")
    # yolo_to_coco_json(YOLO_IMG_DIR, YOLO_PRED_DIR, dt_json, is_prediction=True)

    print(" Running COCO evaluation...")
    coco_eval, coco_gt, coco_dt = evaluate_coco(gt_json, dt_json)

    print("Plotting metrics...")
    plot_ap_per_class(coco_eval)
    # plot_pr_curves_50_95(coco_eval)
    plot_pr_curves_50(coco_eval)
    plot_f1_scores(coco_eval)
    # plot_confusion_matrix(coco_gt, json.load(open(dt_json)))
    # save_coco_eval_to_csv(coco_eval)

if __name__ == "__main__":
    main()
