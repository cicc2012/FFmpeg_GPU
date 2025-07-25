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
import json

# pip install pycocotools matplotlib seaborn opencv-python tqdm

# first run the model and get the prediction, then save the prediction result in YOLO_PRED_DIR

# -----------------------------
# CONFIG
# -----------------------------
# for stage 1
CLASS_NAMES = ['Traffic Light Group','Traffic Light Group Side'] 
YOLO_IMG_DIR = str(Path(r"F:\dataset\Night2.0.0\images\val"))
YOLO_LABEL_DIR = str(Path(r"F:\dataset\Night2.0.0\labels\val"))
YOLO_PRED_DIR = str(Path(r"F:\dataset\Night2.0.0\test\all\crop\labels_wbf"))
IMG_EXT = ".png"
IMG_SIZE_CACHE = {}
OUTPUT_DIR =  str(Path(r"F:\dataset\Night2.0.0\test\all\crop\evaluation\coco_eval"))

# for stage 2
# CLASS_NAMES = ['Traffic Light Bulb Red','Traffic Light Bulb Yellow','Traffic Light Bulb Green','Traffic Light Bulb Null']
# YOLO_IMG_DIR = str(Path(r"F:\dataset\Night2.0.0\images\val"))
# YOLO_LABEL_DIR = str(Path(r"F:\dataset\Night2.0.0\labels_group\val"))
# YOLO_PRED_DIR = str(Path(r"F:\dataset\Night2.0.0\test\all\crop\labels"))
# IMG_EXT = ".png"
# IMG_SIZE_CACHE = {}
# OUTPUT_DIR =  str(Path(r"F:\dataset\Night2.0.0\test\all\phases\phase1"))

# for overall process


# -----------------------------
# Convert YOLO to COCO format
# -----------------------------

def yolo_to_coco_json(image_dir, yolo_labels_dir, output_json_path, is_prediction=False, conf_threshold=0.0):
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
                if score < conf_threshold:
                    continue
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
    # coco_gt = COCO(gt_json)
    # coco_dt = coco_gt.loadRes(dt_json)
    result = safe_load_detections(gt_json, dt_json)
    if result is None:
        return  # Exit early if no detections
    coco_gt, coco_dt = result
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    # coco_eval.params.iouThrs = [0.5]  # Evaluate only at IoU=0.5
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval, coco_gt, coco_dt

# -----------------------------
# PR Curves per Class
# -----------------------------

def safe_load_detections(gt_json_path, dt_json_path):
    with open(dt_json_path, "r") as f:
        dt_data = json.load(f)
    if not dt_data:  # Empty list
        print(f"Warning: No detections found in {dt_json_path}. Skipping evaluation.")
        return None
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dt_data)
    return coco_gt, coco_dt


def plot_pr_curves_50(coco_eval):
    precisions = coco_eval.eval['precision']
    recall = coco_eval.params.recThrs

    mean_precision = np.zeros(101, dtype=np.float64)  # Interpolated to 101 recall points
    valid_class_count = 0
    recall_interp = np.linspace(0, 1, 101)  # 101-point recall levels

    for cls_idx, class_name in enumerate(CLASS_NAMES):
        pr = precisions[0, :, cls_idx, 0, 2]  # IoU=0.5:0.95
        pr = np.where(pr == -1, np.nan, pr)

        # Step 1: Interpolate precision to be monotonically decreasing
        pr_interp = np.maximum.accumulate(pr[::-1])[::-1]

        # Step 2: Interpolate to 101-point PR curve
        pr_101 = np.interp(recall_interp, recall, pr_interp)

        # Step 3: Compute AP = area under interpolated precision-recall curve
        ap = np.mean(pr_101)
        mean_precision += pr_101
        valid_class_count += 1

        plt.plot(recall_interp, pr_101, label=f'{class_name} (AP={ap:.2f})')

    # Calculate and plot mean precision curve
    if valid_class_count > 0:
        mean_precision /= valid_class_count
        mean_ap = np.mean(mean_precision)
        plt.plot(recall_interp, mean_precision, label=f'Mean (mAP@50={mean_ap:.2f})', color='black', linestyle='--')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-Class Precision-Recall Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/pr_curves_50.png")
    plt.show()

def plot_pr_curves_50_coco(coco_eval):
    precisions = coco_eval.eval['precision']
    recall = coco_eval.params.recThrs  # np.linspace(0, 1, 101)
    
    mean_precision = np.zeros(101, dtype=np.float64)
    valid_class_count = 0
    recall_interp = np.linspace(0, 1, 101)

    for cls_idx, class_name in enumerate(CLASS_NAMES):
        pr = precisions[0, :, cls_idx, 0, 2]  # IoU=0.5, all area, maxDets=100
        pr = np.where(pr == -1, np.nan, pr)  # Replace undefined precision with NaN

        if np.all(np.isnan(pr)):
            continue  # No data for this class

        # Step 1: Interpolate precision to be monotonic decreasing
        valid_mask = ~np.isnan(pr)
        pr_valid = pr.copy()
        pr_valid[~valid_mask] = 0  # Temporarily fill NaNs with 0 for accumulation
        pr_interp = np.maximum.accumulate(pr_valid[::-1])[::-1]

        # Step 2: Limit interpolation only up to max valid recall
        max_valid_recall = recall[valid_mask][-1]  # Highest recall where precision is valid
        recall_mask = recall_interp <= max_valid_recall

        # Step 3: Interpolate over the valid recall range only
        pr_101 = np.zeros_like(recall_interp)
        pr_101[recall_mask] = np.interp(
            recall_interp[recall_mask],
            recall[valid_mask],
            pr_interp[valid_mask]
        )
        # Beyond max recall, precision is undefined — leave at 0 or NaN (optional)

        # Step 4: Compute AP only over valid range
        ap = np.mean(pr_101[recall_mask])
        mean_precision += pr_101
        valid_class_count += 1

        plt.plot(recall_interp, pr_101, label=f'{class_name} (AP={ap:.4f})')

    # Mean PR curve
    if valid_class_count > 0:
        mean_precision /= valid_class_count
        mean_ap = np.mean(mean_precision)
        plt.plot(recall_interp, mean_precision, label=f'Mean (mAP@50={mean_ap:.4f})', color='black', linestyle='--')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-Class Precision-Recall Curves (IoU=0.5)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/pr_curves_50_coco.png")
    plt.show()


def plot_pr_curves_50_95(coco_eval):
    precision = coco_eval.eval['precision']  # [T, R, K, A, M]
    recall = coco_eval.params.recThrs
    iou_thrs = coco_eval.params.iouThrs

    mean_precision = np.zeros(101, dtype=np.float64)  # Interpolated to 101 recall points
    valid_class_count = 0
    recall_interp = np.linspace(0, 1, 101)  # 101-point recall levels

    for class_idx, class_name in enumerate(CLASS_NAMES):
        pr_avg = np.zeros(101, dtype=np.float64)

        # Aggregate precision across IoU thresholds
        for iou_idx, iou_thr in enumerate(iou_thrs):
            pr = precision[iou_idx, :, class_idx, 0, 2]
            pr = np.where(pr == -1, np.nan, pr)

            # Step 1: Interpolate precision to be monotonically decreasing
            pr_interp = np.maximum.accumulate(pr[::-1])[::-1]

            # Step 2: Interpolate to 101-point PR curve
            pr_101 = np.interp(recall_interp, recall, pr_interp)

            pr_avg += pr_101

        # Average precision across IoU thresholds
        pr_avg /= len(iou_thrs)
        ap = np.mean(pr_avg)
        mean_precision += pr_avg
        valid_class_count += 1

        plt.plot(recall_interp, pr_avg, label=f'{class_name} (AP@50-95={ap:.2f})')

    # Calculate and plot mean precision curve
    if valid_class_count > 0:
        mean_precision /= valid_class_count
        mean_ap = np.mean(mean_precision)
        plt.plot(recall_interp, mean_precision, label=f'Mean (mAP@50-95={mean_ap:.2f})', color='black', linestyle='--')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-Class Precision-Recall Curves (IoU=0.5:0.95)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/pr_curves_50_95.png")
    plt.show()

def plot_pr_curves_50_95_coco(coco_eval):
    precision = coco_eval.eval['precision']  # shape: [T, R, K, A, M]
    recall = coco_eval.params.recThrs         # 101 recall thresholds
    iou_thrs = coco_eval.params.iouThrs       # default: np.linspace(0.5, 0.95, 10)

    mean_precision = np.zeros(101, dtype=np.float64)
    valid_class_count = 0
    recall_interp = np.linspace(0, 1, 101)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        pr_avg = np.zeros(101, dtype=np.float64)
        iou_valid_count = 0  # To normalize only over valid IoU thresholds

        for iou_idx, iou_thr in enumerate(iou_thrs):
            pr = precision[iou_idx, :, class_idx, 0, 2]  # shape: [101]
            pr = np.where(pr == -1, np.nan, pr)

            if np.all(np.isnan(pr)):
                continue  # skip if no data for this IoU threshold

            # Step 1: Fill NaNs with 0 temporarily to enable monotonic interpolation
            valid_mask = ~np.isnan(pr)
            pr_valid = pr.copy()
            pr_valid[~valid_mask] = 0
            pr_interp = np.maximum.accumulate(pr_valid[::-1])[::-1]

            # Step 2: Only interpolate up to max valid recall
            max_valid_recall = recall[valid_mask][-1]
            recall_mask = recall_interp <= max_valid_recall

            pr_101 = np.zeros_like(recall_interp)
            pr_101[recall_mask] = np.interp(
                recall_interp[recall_mask],
                recall[valid_mask],
                pr_interp[valid_mask]
            )

            pr_avg += pr_101
            iou_valid_count += 1

        if iou_valid_count == 0:
            continue  # skip this class if no valid iou thresholds

        # Average over valid IoU thresholds
        pr_avg /= iou_valid_count
        ap = np.mean(pr_avg)
        mean_precision += pr_avg
        valid_class_count += 1

        plt.plot(recall_interp, pr_avg, label=f'{class_name} (AP@50-95={ap:.2f})')

    # Finalize mean precision curve
    if valid_class_count > 0:
        mean_precision /= valid_class_count
        mean_ap = np.mean(mean_precision)
        plt.plot(
            recall_interp, mean_precision,
            label=f'Mean (mAP@50-95={mean_ap:.2f})',
            color='black', linestyle='--'
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-Class Precision-Recall Curves (IoU=0.5:0.95)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/pr_curves_50_95.png")
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
        pr = np.where(pr == -1, np.nan, pr)

        # Step 1: Interpolate precision to be monotonically decreasing
        pr_interp = np.maximum.accumulate(pr[::-1])[::-1]

        # Step 2: Compute F1 scores for each recall threshold
        rec = recall_thrs
        f1 = 2 * pr_interp * rec / (pr_interp + rec + 1e-6)
        f1[np.isnan(f1)] = 0  # Replace NaN values with 0

        f1_scores.append((class_name, f1))

        plt.plot(rec, f1, label=f'{class_name}')

    # Plot
    plt.xlabel("Recall")
    plt.ylabel(f"F1 Score (IoU={iou_thr})")
    plt.title("F1 Score vs Recall per Class")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/f1_vs_recall.png")
    plt.show()

def evaluate_f1_conf(gt_json, dt_json, output_csv_name="prf_conf_curve.csv", plot=True):
    
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(dt_json)

    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = np.array([0.5])  # only AP@0.5
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Load detection scores
    with open(dt_json, "r") as f:
        detections = json.load(f)

    scores = np.array([d['score'] for d in detections])
    thresholds = np.sort(np.unique(scores))[::-1]  # descending

    # Gather image IDs
    img_ids = coco_gt.getImgIds()

    # Storage for metrics
    rows = []

    for conf_thresh in thresholds:
        # Filter detections by threshold
        filtered_dets = [d for d in detections if d['score'] >= conf_thresh]
        if len(filtered_dets) == 0:
            continue

        # Evaluate filtered detections
        coco_dt_thresh = coco_gt.loadRes(filtered_dets)
        coco_eval_thresh = COCOeval(coco_gt, coco_dt_thresh, iouType='bbox')
        coco_eval_thresh.params.iouThrs = np.array([0.5])
        coco_eval_thresh.params.imgIds = img_ids
        coco_eval_thresh.evaluate()
        coco_eval_thresh.accumulate()

        # TP and FP for maxDets=100, area=all, class=all
        precision = coco_eval_thresh.eval['precision'][0, :, 0, 0, 2]
        recall = coco_eval_thresh.eval['recall'][0, 0, 0, 2]
        
        valid = precision > -1
        if not np.any(valid):
            continue

        p = np.mean(precision[valid])
        r = recall
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0

        rows.append({'confidence': conf_thresh, 'precision': p, 'recall': r, 'f1': f1})

    # Dump to CSV
    df = pd.DataFrame(rows)
    df = df.sort_values('confidence', ascending=False)
    output_csv = OUTPUT_DIR + "/" + output_csv_name
    
    df.to_csv(output_csv, index=False)
    print(f"Saved curve data to: {output_csv}")

    # Plot
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(df['confidence'], df['precision'], label='Precision')
        plt.plot(df['confidence'], df['recall'], label='Recall')
        plt.plot(df['confidence'], df['f1'], label='F1')
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Score")
        plt.title("P/R/F1 vs Confidence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + "/f1_conf.png")
        plt.show()

    return df

def compute_f1_confidence_curve(gt_json, dt_json, iou_thresh=0.5):
    
    # Load annotations
    coco_gt = COCO(gt_json)
    with open(dt_json) as f:
        dt_data = json.load(f)
    
    # Sort detections by descending score
    dt_data_sorted = sorted(dt_data, key=lambda x: -x['score'])

    # Confidence thresholds to evaluate at
    thresholds = np.linspace(0.0, 1.0, 101)
    f1_scores = []

    for t in thresholds:
        dt_filtered = [d for d in dt_data_sorted if d['score'] >= t]

        if len(dt_filtered) == 0:
            f1_scores.append(last_valid_f1)  # Interpolation: reuse previous value
            continue

        coco_dt = coco_gt.loadRes(dt_filtered)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.iouThrs = [iou_thresh]
        coco_eval.evaluate()
        coco_eval.accumulate()

        precision = coco_eval.eval['precision'][0, :, 0, 0, 2]
        recall = coco_eval.params.recThrs

        f1 = []
        for p, r in zip(precision, recall):
            if (p + r) > 0:
                f1.append(2 * p * r / (p + r))
            else:
                f1.append(0)

        max_f1 = np.max(f1)
        last_valid_f1 = max_f1  # Update tracker
        f1_scores.append(max_f1)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Confidence Threshold')
    plt.grid(True)
    plt.legend()
    plt.savefig(OUTPUT_DIR + F"/f1_conf_iou{iou_thresh}.png")
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
# Quick Test
# -----------------------------
def quick_test(gt_json, dt_json):
    # Load COCO formatted data
    cocoGt = COCO(gt_json)
    cocoDt = cocoGt.loadRes(dt_json)
    
    # Get list of all category IDs
    catIds = cocoGt.getCatIds()
    imgIds = cocoGt.getImgIds()
    
    # Dictionary to store class-wise AP
    class_ap = {}
    
    for cid in catIds:
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.params.catIds = [cid]
        cocoEval.params.imgIds = imgIds
        cocoEval.params.iouThrs = [0.5]  # For AP@0.5
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        # AP@0.5 is first result after summarize() at IoU 0.5
        ap = cocoEval.stats[0]  # AP IoU=0.50:0.95
        class_name = cocoGt.loadCats([cid])[0]['name']
        class_ap[class_name] = ap
        print(f"Class: {class_name}, AP@0.5: {ap:.4f}")
    
    # Print results for all classes
    print("\nClass-wise AP@0.5:")
    for cls, ap in class_ap.items():
        print(f"{cls}: {ap:.4f}")


def quick_test2(gt_json, dt_json):
    cocoGt = COCO(gt_json)
    cocoDt = cocoGt.loadRes(dt_json)
    
    # Initialize COCOeval object for bbox evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.iouThrs = np.array([0.5])  # Use IoU=0.5 for AP@0.5
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # Extract precision, recall, and confidence scores for plotting
    precision = cocoEval.eval['precision'][0, :, 0, 0, -1]  # iou=0.5, area=all, maxDets=100
    recall = cocoEval.params.recThrs
    scores = cocoEval.eval['scores'][0, :, 0, 0, -1]        # confidence scores
    
    # Compute F1 score for each recall threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-16)
    
    # Plot precision-recall curve
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve @IoU=0.5')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT_DIR + "/pr_curves_50.png")
    plt.show()
    
    # Plot F1 score vs. confidence threshold (approximate using recall as proxy)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, f1_scores, label='F1 Score')
    plt.xlabel('Recall / Confidence (approx.)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Confidence')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT_DIR + "/f1.png")
    plt.show()

# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_json = OUTPUT_DIR + "\\gt.json"
    dt_json = OUTPUT_DIR + "\\dt.json"

    print("Converting ground truth...")
    yolo_to_coco_json(YOLO_IMG_DIR, YOLO_LABEL_DIR, gt_json, is_prediction=False)

    print("Converting predictions...")
    yolo_to_coco_json(YOLO_IMG_DIR, YOLO_PRED_DIR, dt_json, is_prediction=True)

    print(" Running COCO evaluation...")
    # quick_test2(gt_json, dt_json)
    coco_eval, coco_gt, coco_dt = evaluate_coco(gt_json, dt_json)

    print("Plotting metrics...")
    # plot_ap_per_class(coco_eval)
    # plot_pr_curves_50_95(coco_eval)
    plot_pr_curves_50(coco_eval)
    plot_pr_curves_50_95_coco(coco_eval)
    # plot_pr_curves_50_coco(coco_eval)
    # plot_f1_scores(coco_eval)
    # evaluate_f1_conf(gt_json, dt_json)
    # compute_f1_confidence_curve(gt_json, dt_json)
    # plot_confusion_matrix(coco_gt, json.load(open(dt_json)))
    # save_coco_eval_to_csv(coco_eval)

if __name__ == "__main__":
    main()
