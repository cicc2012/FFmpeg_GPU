import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

# pip install numpy matplotlib scikit-learn tqdm

IOU_THRESHOLD = 0.5
class_names = ['red', 'yellow', 'green', 'null']

def load_yolo_labels(file_path, with_conf=False):
    boxes = []
    if not os.path.exists(file_path):
        return boxes
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if with_conf:
                cls, x, y, w, h, conf = map(float, parts)
                boxes.append((int(cls), x, y, w, h, conf))
            else:
                cls, x, y, w, h = map(float, parts)
                boxes.append((int(cls), x, y, w, h))
    return boxes

def yolo_to_bbox(box, img_w, img_h):
    # Support both (cls, x, y, w, h) and (cls, x, y, w, h, conf)
    if len(box) == 6:
        cls, x, y, w, h, conf = box
    else:
        cls, x, y, w, h = box
        conf = None

    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2, cls, conf]


def compute_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0

def evaluate_detections(gt_dir, pred_dir, img_shape, class_names):
    num_classes = len(class_names)
    all_precisions = []
    all_recalls = []
    all_f1s = []
    aps = []

    for cls_id in range(num_classes):
        y_true = []
        y_scores = []

        for file in tqdm(os.listdir(gt_dir)):
            if not file.endswith(".txt"):
                continue
            image_id = file[:-4]
            gt_boxes = load_yolo_labels(os.path.join(gt_dir, file))
            pred_file = os.path.join(pred_dir, file)
            pred_boxes_raw = load_yolo_labels(pred_file, True)

            # Filter current class
            gt_cls_boxes = [yolo_to_bbox(b, *img_shape) for b in gt_boxes if b[0] == cls_id]
            pred_cls_boxes = [yolo_to_bbox(b, *img_shape) for b in pred_boxes_raw if b[0] == cls_id]

            matched_gt = set()
            for pred in pred_cls_boxes:
                pred_bbox = pred[:4]
                conf = pred[5] if pred[5] is not None else 1.0  # fallback if missing
                best_iou = 0
                matched = -1
                for i, gt in enumerate(gt_cls_boxes):
                    if i in matched_gt:
                        continue
                    iou = compute_iou(pred_bbox, gt[:4])
                    if iou > best_iou:
                        best_iou = iou
                        matched = i
                if best_iou >= IOU_THRESHOLD:
                    y_true.append(1)
                    matched_gt.add(matched)
                else:
                    y_true.append(0)
                y_scores.append(conf)  # or use predicted confidence if available

            # Add false negatives
            y_true.extend([1] * (len(gt_cls_boxes) - len(matched_gt)))
            y_scores.extend([0.0] * (len(gt_cls_boxes) - len(matched_gt)))

        if len(y_true) == 0:
            continue

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-6)

        ap = average_precision_score(y_true, y_scores)

        all_precisions.append((precision, recall, cls_id))
        all_f1s.append((thresholds, f1, cls_id))
        aps.append((class_names[cls_id], ap))

    return all_precisions, all_f1s, aps

def plot_precision_recall(all_precisions, aps, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for precision, recall, cls_id in all_precisions:
        name = class_names[cls_id]
        ap = next((a[1] for a in aps if a[0] == name), 0.0)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")
        # plt.plot(recall, precision, label=f"{cls_id} ({class_names[cls_id]}) AP = {ap:.3f}")
    # for name, ap in aps:
    #     plt.text(0.6, 0.1 - aps.index((name, ap))*0.05, f"{name} AP = {ap:.3f}", fontsize=10)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir + "/pr_curves_50.png")
    plt.show()

def plot_f1_curve(all_f1s, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for thresholds, f1, cls_id in all_f1s:
        plt.plot(thresholds, f1, label=f"{class_names[cls_id]}")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 vs Confidence Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir + "/f1_conf.png")
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
def main():
    gt_dir = str(Path(r"F:\dataset\Night2.0.0\labels_signal_front\val"))
    pred_dir = str(Path(r"F:\dataset\Night2.0.0\test\front\predict\overall\labels"))
    output_dir =  str(Path(r"F:\dataset\Night2.0.0\test\front\predict\evaluation\quick_eval"))
    img_shape = (1280, 720)  # width, height
    class_names = ['person', 'car', 'bike']  # replace with your classes

    precisions, f1s, aps = evaluate_detections(gt_dir, pred_dir, img_shape, class_names)
    plot_precision_recall(precisions, aps, output_dir)
    plot_f1_curve(f1s, output_dir)

if __name__ == "__main__":
    main()