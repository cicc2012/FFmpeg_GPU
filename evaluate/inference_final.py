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
# pip install pycocotools matplotlib opencv-python tqdm

# -----------------------------
# CONFIG
# -----------------------------
stage1_keep_classes=[0,1]
stage2_keep_classes=[0,1,2,3]

pipeline_mode = True  # True for full pipeline, False for single stage

conf_threshold = 0.25  # Confidence threshold for YOLO predictions
iou_threshold = 0.5  # IoU threshold for NMS

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
stage2_input_path=stage1_output_path if pipeline_mode else str(Path(r"F:\dataset\Night2.0.0\crop\images\val"))
stage2_output_path=str(Path(r"F:\dataset\Night2.0.0\test\all\predict"))

final_output_path=str(Path(r"F:\dataset\Night2.0.0\test\all\predict\overall"))
final_eval_path=str(Path(r"F:\dataset\Night2.0.0\test\all\evaluation"))
log_path=str(Path(r"F:\dataset\Night2.0.0\test\all\evaluation\eval_log.csv"))

use_wbf = True  # Whether to use WBF for each stage

# Assign a unique color to each class
colors = [
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (0, 255, 0),    # Green
    (255, 0, 255),  # Blue for 'null' class
]

class_names = ['red', 'yellow', 'green', 'null']
group_names = ['front', 'side']
group_filters = [0]

dirs_map = {'images': 'images',
            'images_wbf': 'images_wbf',
            'previews': 'previews',
            'previews_wbf': 'previews_wbf',
            'labels': 'labels',
            'labels_wbf': 'labels_wbf'
            }

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

def write_images_previews_in_image(im_path, boxes_px, classes, scores, image_dir, preview_dir, obj_names):
    im = cv2.imread(im_path)
    img_name = Path(im_path).stem
    crops = []  # crops per img, as the return
    preview_path = os.path.join(preview_dir, f"{img_name}.png")

    for i, box in enumerate(boxes_px):
        x1, y1, x2, y2 = map(int, box)
        cls = int(classes[i])
        conf = float(scores[i])
        
        # crop the object image
        crop = im[y1:y2, x1:x2]
        crop_name = f"{img_name}_{i}.png"
        image_path = os.path.join(image_dir, crop_name)
        cv2.imwrite(image_path, crop)
        crops.append((image_path, cls, img_name, x1, y1, conf))

        # Draw the object preview
        color = colors[int(cls) if int(cls) < len(colors) else 0]
        label = f"{obj_names[int(cls) if int(cls) < len(obj_names) else 0]}: {conf:.4f}"
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 255), 2) # write rectangle: one color for each stage
        cv2.putText(im, label, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        # TODO: [minor] text coordinates can be adjusted to avoid overlap with the nearby boxes (this will affect the evaluation results)
    
    # Save the previews of this image
    cv2.imwrite(preview_path, im)
    
    return crops


def write_labels_in_image(boxes_norm, classes, scores, label_path):
    lines = []  # groups per img
    # get norm coordinates for stage level evaluation
    for box, cls, conf in zip(boxes_norm, classes, scores):
        cls = int(cls)
        conf = float(conf)
        cx, cy, w, h = box
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")
    
    with open(label_path, "w") as f:
        f.writelines(lines)


def convert_normalized_xyxy_to_yolo_xywhn(boxes_xyxy):
    """
    Convert normalized [x_min, y_min, x_max, y_max] format to YOLO [x_center, y_center, width, height] format.

    Args:
        boxes_fused (numpy.ndarray): Array of boxes in normalized [x_min, y_min, x_max, y_max] format.

    Returns:
        numpy.ndarray: Array of boxes in YOLO [x_center, y_center, width, height] format.
    """
    yolo_boxes = []
    for box in boxes_xyxy:
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        yolo_boxes.append([x_center, y_center, width, height])
    return np.array(yolo_boxes)


# export the cropped objects in an image
# also plot the overall boxes in the image
# return the information of crops, as a list of tuples for further stage
def export_obj_in_image(result, output_parent_dirs, obj_names, is_wbf=False):
    imgage_dir = output_parent_dirs["imgage_dir"]
    wbf_image_dir = output_parent_dirs["wbf_image_dir"]
    preview_dir = output_parent_dirs["preview_dir"]
    wbf_preview_dir = output_parent_dirs["wbf_preview_dir"]
    labels_dir = output_parent_dirs["labels_dir"]
    wbf_labels_dir = output_parent_dirs["wbf_labels_dir"]

    im_path = result.path
    im = cv2.imread(im_path)
    img_name = Path(im_path).stem
    full_h, full_w = im.shape[:2]
    # preview_path = os.path.join(preview_dir, f"{img_name}.png")
    # wbf_preview_path = os.path.join(wbf_preview_dir, f"{img_name}.png")
    label_path = os.path.join(labels_dir, f"{img_name}.txt")
    wbf_label_path = os.path.join(wbf_labels_dir, f"{img_name}.txt")

    crops = []  # crops per img, as the return

    # ---- get images, labels, and previews before WBF ----

    # get pixel coordinates: for images and previews
    # for j, boxes_xyxy in enumerate(result.boxes.xyxy.cpu().numpy()):
    #     cls = int(result.boxes.cls[j])
    #     conf = float(result.boxes.conf[j])
    #     x1, y1, x2, y2 = map(int, boxes_xyxy)

    #     # crop the object image
    #     crop = im[y1:y2, x1:x2]
    #     crop_name = f"{img_name}_{j}.png"
    #     image_path = os.path.join(imgage_dir, crop_name)
    #     cv2.imwrite(image_path, crop)
    #     if not is_wbf:
    #         crops.append((image_path, cls, img_name, x1, y1, conf))

    #     # Draw the object preview
    #     color = colors[int(cls) if int(cls) < len(colors) else 0]
    #     label = f"{obj_names[int(cls) if int(cls) < len(obj_names) else 0]}: {conf:.4f}"
    #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(im, label, (x2 + 5, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # # Save the previews of this image
    # cv2.imwrite(preview_path, im)

    crops = write_images_previews_in_image(
        im_path, result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy(), imgage_dir, preview_dir, obj_names
    )

    # # Save the labels of this image
    # lines = []  # groups per img
    # # get norm coordinates for stage level evaluation
    # for box, cls, conf in zip(result.boxes.xywhn.cpu().numpy(),
    #                           result.boxes.cls.cpu().numpy(),
    #                           result.boxes.conf.cpu().numpy()):
    #     cls = int(cls)
    #     conf = float(conf)
    #     cx, cy, w, h = box
    #     lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")
    
    # with open(label_path, "w") as f:
    #     f.writelines(lines)

    write_labels_in_image(
        result.boxes.xywhn.cpu().numpy(), result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy(), label_path
    )

    # ---- get images, labels, and previews after WBF ----
    if is_wbf:
        # prepare for WBF
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # shape: [N, 4]
        scores = result.boxes.conf.cpu().numpy()      # shape: [N]
        labels = result.boxes.cls.cpu().numpy().astype(int)

        boxes_norm = normalize_boxes(boxes_xyxy, full_w, full_h)
        boxes_list = [boxes_norm]
        scores_list = [scores.tolist()]
        labels_list = [labels.tolist()]

        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=0.1, skip_box_thr=0.001)

        fused_boxes_px = denormalize_boxes(boxes_fused, full_w, full_h)

        crops = []
        crops = write_images_previews_in_image(
            im_path, fused_boxes_px, labels_fused, scores_fused, 
            wbf_image_dir, wbf_preview_dir, obj_names
            )
        
        write_labels_in_image(
            convert_normalized_xyxy_to_yolo_xywhn(boxes_fused),
            labels_fused, scores_fused, wbf_label_path
            )

    return crops


# export labels of all detected objects in an image
def export_label_in_image(result, output_dir):
    im_path = result.path
    im = cv2.imread(im_path)
    img_name = Path(im_path).stem
    label_file = os.path.join(output_dir, f"{img_name}.txt")

    lines = []  # groups per img
    # get norm coordinates for stage level evaluation
    for box, cls, conf in zip(result.boxes.xywhn.cpu().numpy(),
                              result.boxes.cls.cpu().numpy(),
                              result.boxes.conf.cpu().numpy()):
        cls = int(cls)
        conf = float(conf)
        cx, cy, w, h = box
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")  
            
    with open(label_file, "w") as f:
        f.writelines(lines)


def run_stage1_inference(model_path, source_dir, output_dir):
    imgage_dir = os.path.join(output_dir, dirs_map["images"])
    wbf_image_dir = os.path.join(output_dir, dirs_map["images_wbf"])
    preview_dir = os.path.join(output_dir, dirs_map["previews"])
    wbf_preview_dir = os.path.join(output_dir, dirs_map["previews_wbf"])
    labels_dir = os.path.join(output_dir, dirs_map["labels"])
    wbf_labels_dir = os.path.join(output_dir, dirs_map["labels_wbf"])

    for folder in [imgage_dir, wbf_image_dir, preview_dir, wbf_preview_dir, labels_dir, wbf_labels_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    output_dirs = {"imgage_dir":imgage_dir,
                    "wbf_image_dir":wbf_image_dir,
                    "preview_dir":preview_dir,
                    "wbf_preview_dir":wbf_preview_dir,
                    "labels_dir":labels_dir,
                    "wbf_labels_dir":wbf_labels_dir}

    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    # labels_dir = os.path.join(output_dir, "labels")
    # os.makedirs(labels_dir, exist_ok=True)
    # labels_dir_wbf = os.path.join(output_dir, "labels_wbf")
    # os.makedirs(labels_dir_wbf, exist_ok=True)
    
    # results = model.predict(source=source_dir, save=False, conf=0.35, iou=0.5) # default conf=0.25, iou=0.7
    results = model.predict(source=source_dir, save=False, conf=conf_threshold)  # default conf=0.25, iou=0.7
    # lower iou for nms

    crops = []
    for i, r in enumerate(results):
        crops.extend(export_obj_in_image(r, output_dirs, group_names, use_wbf))

        # im_path = r.path
        # im = cv2.imread(im_path)
        # img_name = Path(im_path).stem
        # label_file = os.path.join(labels_dir, f"{img_name}.txt")
        # label_file_wbf = os.path.join(labels_dir_wbf, f"{img_name}.txt")
        
        
        # ## get pixel coordinates
        # # for j, box in enumerate(r.boxes.xyxy.cpu().numpy()):
        #     # cls = int(r.boxes.cls[j])
        #     # conf = float(r.boxes.conf[j])
        #     # if cls not in stage1_keep_classes: # [0, 1]:  # Only A or F
        #         # continue
        #     # x1, y1, x2, y2 = map(int, box)
        #     # crop = im[y1:y2, x1:x2]
        #     # crop_name = f"{img_name}_{j}.png"
        #     # crop_path = os.path.join(output_dir, crop_name)
        #     # cv2.imwrite(crop_path, crop)
        #     # crops.append((crop_path, cls, img_name, x1, y1, conf))    # crop_name, x1, y1 are used for reprojection
            
        # lines = []  # groups per img
        # # get local coordinates for stage level evaluation
        # for box1, cls1, conf1 in zip(r.boxes.xywhn.cpu().numpy(),
        #                           r.boxes.cls.cpu().numpy(),
        #                           r.boxes.conf.cpu().numpy()):
        #     cls1 = int(cls1)
        #     cx1, cy1, w1, h1 = box1
        #     lines.append(f"{cls1} {cx1:.6f} {cy1:.6f} {w1:.6f} {h1:.6f} {conf1:.4f}\n")
                
        # with open(label_file, "w") as f:
        #     f.writelines(lines)
            
        
        # # perform WBF: one box per object
        # # boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # shape: [N, 4]
        # norm_boxes = r.boxes.xywhn.cpu().numpy()  # shape: [N, 4] normalized [cx, cy, w, h]
        # scores = r.boxes.conf.cpu().numpy()      # shape: [N]
        # labels = r.boxes.cls.cpu().numpy().astype(int)
        
        # H, W = r.orig_shape  # Image size for normalization
        
        # # norm_boxes = normalize_boxes(boxes_xyxy, W, H)
        # boxes_list = [norm_boxes]
        # scores_list = [scores.tolist()]
        # labels_list = [labels.tolist()]
        
        # boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
        #     boxes_list, scores_list, labels_list,
        #     iou_thr=0.5, skip_box_thr=0.001)

        # fused_boxes_px = denormalize_boxes(boxes_fused, W, H)
        
        # lines = []  # groups per img
        # for j, box_px in enumerate(fused_boxes_px):
        #     cls = int(labels_fused[j])
        #     box_norm = boxes_fused[j]
        #     conf = float(scores_fused[j])
            
        #     # prepare pixel cooridates for stage2
        #     x1, y1, x2, y2 = map(int, box_px)
        #     crop = im[y1:y2, x1:x2]
        #     crop_name = f"{img_name}_{j}.png"
        #     crop_path = os.path.join(output_dir, crop_name)
        #     cv2.imwrite(crop_path, crop)
        #     crops.append((crop_path, cls, img_name, x1, y1, conf))    # crop_name, x1, y1 are used for reprojection
            
        #     # output normalized coordiantes for evaluation of stage 1 
        #     cx1, cy1, w1, h1 = box_norm
        #     lines.append(f"{cls} {cx1:.6f} {cy1:.6f} {w1:.6f} {h1:.6f} {conf:.4f}\n")
            
        # with open(label_file, "w") as f:
        #     f.writelines(lines)
            
    return crops

# step 2 function
def run_stage2_inference_old(model_path, crop_list, save_dir):
    model = YOLO(model_path)
    os.makedirs(save_dir, exist_ok=True)

    for crop_path, parent_cls, name, x, y in crop_list:
        results = model.predict(source=crop_path, save=True, project=save_dir, name="stage2", exist_ok=True)
        # print(f"Processed: {crop_path}")

def run_stage2_inference(model_path, source_dir, output_dir):

    imgage_dir = os.path.join(output_dir, dirs_map["images"])
    wbf_image_dir = os.path.join(output_dir, dirs_map["images_wbf"])
    preview_dir = os.path.join(output_dir, dirs_map["previews"])
    wbf_preview_dir = os.path.join(output_dir, dirs_map["previews_wbf"])
    labels_dir = os.path.join(output_dir, dirs_map["labels"])
    wbf_labels_dir = os.path.join(output_dir, dirs_map["labels_wbf"])

    os.makedirs(output_dir, exist_ok=True)
    for folder in [imgage_dir, wbf_image_dir, preview_dir, wbf_preview_dir, labels_dir, wbf_labels_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    output_dirs = {"imgage_dir":imgage_dir,
                    "wbf_image_dir":wbf_image_dir,
                    "preview_dir":preview_dir,
                    "wbf_preview_dir":wbf_preview_dir,
                    "labels_dir":labels_dir,
                    "wbf_labels_dir":wbf_labels_dir}

    model = YOLO(model_path)
    
    # crops = []  
    # # information about crops in the previous stage is included in crop_list
    # # need to extract the images for inference: saved in crops
    # for crop_path, parent_cls, img_name, x1, y1, conf1 in crop_list:
        
    #     crop = cv2.imread(crop_path)
    #     if crop is None:
    #         continue
    #     crops.append(crop)

    subfolder = dirs_map["images_wbf"] if use_wbf else dirs_map["images"]
    crop_path = source_dir if not pipeline_mode else os.path.join(source_dir, subfolder)
    print(f'Applying stage 2 inference on crops from {crop_path}')

    results = model.predict(source=crop_path, conf=conf_threshold, save=False)

    objects = []
    for i, r in enumerate(results):
        objects.extend(export_obj_in_image(r, output_dirs, class_names, use_wbf))

    # model = YOLO(model_path)

    # # pred_dir = os.path.join(save_root, "stage2")
    # labels_dir = os.path.join(save_root, "labels")
    # images_dir = os.path.join(save_root, "images")
    # os.makedirs(labels_dir, exist_ok=True)
    # os.makedirs(images_dir, exist_ok=True)

    # # for crop_path, parent_cls, img_name, x1, y1 in crop_list:
    #     # crop = cv2.imread(crop_path)
    #     # results = model.predict(source=crop, conf=0.25, save=False, imgsz=640)[0]  # get only first result

    #     # if results.boxes is not None:
    #         # label_lines = []
    #         # for box, cls in zip(results.boxes.xywhn.cpu().numpy(), results.boxes.cls.cpu().numpy()):
    #             # cls = int(cls)
    #             # cx, cy, w, h = box  # normalized
    #             # label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    #         # base_name = Path(crop_path).stem
    #         # label_file = os.path.join(labels_dir, f"{base_name}.txt")
    #         # with open(label_file, "w") as f:
    #             # f.write("\n".join(label_lines))

    #     # shutil.copy(crop_path, os.path.join(images_dir, Path(crop_path).name))
        
    # # change to batch process
    # crops = []
    # metadata = []
    
    # for crop_path, parent_cls, img_name, x1, y1, conf1 in crop_list:
        
    #     crop = cv2.imread(crop_path)
    #     if crop is None:
    #         continue
    #     crops.append(crop)
    #     metadata.append((crop_path, parent_cls, img_name, x1, y1, conf1))
        
    # results = model.predict(source=crops, conf=0.35, save=False, iou=0.5)
    
    # for i, r in enumerate(results):
    #     crop_path, parent_cls, img_name, x1, y1, conf1 = metadata[i]
    #     crop_h, crop_w = crops[i].shape[:2]
    
    #     # base_name = Path(crop_path).stem
    #     label_file = os.path.join(labels_dir, f"{Path(crop_path).stem}.txt")
    
    #     lines = []  # bulbs per crop
    #     # get local coordinates for stage level evaluation
    #     for box, cls, conf in zip(r.boxes.xywhn.cpu().numpy(),
    #                               r.boxes.cls.cpu().numpy(),
    #                               r.boxes.conf.cpu().numpy()):
    #         cls2 = int(cls)
    #         conf2 = float(conf)
    #         cx2, cy2, w2, h2 = box
    #         lines.append(f"{cls2} {cx2:.6f} {cy2:.6f} {w2:.6f} {h2:.6f} {conf2 * conf1 ** 0.5:.4f}\n")
            
    #     with open(label_file, "w") as f:
    #         f.writelines(lines)       
    #     # shutil.copy(crop_path, os.path.join(images_dir, Path(crop_path).name))


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

# Reproject all boxes from cropped images to full images
# results in stage2_pred_dir are normalized local coordinates, and need to have the results of stage1
# to map local to global coordinates: full_img_dir is used to get the full image size
def reproject_all(stage2_pred_dir, crop_info_list, output_pred_dir, full_img_dir):
    preview_dir = os.path.join(output_pred_dir, dirs_map["previews"])
    labels_dir = os.path.join(output_pred_dir, dirs_map["labels"])

    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    # os.makedirs(output_pred_dir, exist_ok=True)

    reprojected_preds = {} # to store reprojected predictions for each image
    parent_boxes = {}  # to store parent box for each image

    # iterate through each crop info: img_name is the full/parent image name
    for (crop_path, parent_cls, img_name, x1, y1, parent_conf) in crop_info_list:
        subfolder = dirs_map["labels_wbf"] if use_wbf else dirs_map["labels"]
        crop_pred_file = Path(stage2_pred_dir) / subfolder / (Path(crop_path).stem + ".txt")
        if not crop_pred_file.exists():
            continue

        # Get crop size
        full_img = cv2.imread(str(Path(full_img_dir) / f"{img_name}.png"))
        full_h, full_w = full_img.shape[:2]
        crop_img = cv2.imread(str(crop_path))
        crop_h, crop_w = crop_img.shape[:2]
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        parent_box_px = (x1, y1, x2, y2)
        if img_name not in parent_boxes:
            parent_boxes[img_name] = []
        parent_boxes[img_name].append((parent_cls, parent_box_px, parent_conf))

        with open(crop_pred_file, "r") as f:
            lines = f.readlines()

        # for each stage 2 object, get its local coordinate and reproject to global coordinate
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            box = list(map(float, parts[1:5]))
            conf = float(parts[5])

            # get px global coordinates
            x_center, y_center, w, h = map(float, parts[1:5])
            x_min = (x_center - w / 2) * crop_w + x1
            y_min = (y_center - h / 2) * crop_h + y1
            x_max = x_min + w * crop_w
            y_max = y_min + h * crop_h
            box_global_px = (x_min, y_min, x_max, y_max)
            
            # get normalized global coordinates
            box_global_norm = reproject_yolo_box(box, x1, y1, crop_w, crop_h, full_w, full_h)
            if img_name not in reprojected_preds:
                reprojected_preds[img_name] = []
            if pipeline_mode:
                conf = conf * parent_conf ** 0.5  # combine confidence with parent confidence
            reprojected_preds[img_name].append((cls, box_global_norm, box_global_px, conf))

    # Write labels as normalizedglobal coordinates
    for img_name, boxes in reprojected_preds.items():
        
        label_file = Path(labels_dir) / f"{img_name}.txt"
        
        with open(label_file, "w") as f:
            for cls, box_norm, box_px, conf in boxes:
                f.write(f"{cls} {' '.join(map(lambda x: f'{x:.6f}', box_norm))} {conf}\n")
        
    # Write previews with reprojected boxes: parent boxes and reprojected boxes

    # Draw parent boxes on the full image
    for img_name, parent_boxes in parent_boxes.items():
        
        full_img = cv2.imread(str(Path(full_img_dir) / f"{img_name}.png"))
        preview_file = Path(preview_dir) / f"{img_name}.png"

        # Draw parent boxes on the full image
        for parent_cls, parent_box_px, parent_conf in parent_boxes:
            x1, y1, x2, y2 = map(int, parent_box_px)
            cv2.rectangle(full_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label1 = f"{group_names[int(parent_cls)]}"
            color = colors[int(parent_cls) if int(parent_cls) < len(colors) else 0]
            cv2.putText(full_img, label1 + ':' + str(parent_conf), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Draw reprojected boxes on the full image
        # parent_boxes and reprojected_preds are both dicts with img_name as keys
        boxes = reprojected_preds.get(img_name, [])
        for cls, box_norm, box_px, conf in boxes:
            x1, y1, x2, y2 = map(int, box_px)
            cv2.rectangle(full_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label2 = f"{class_names[int(cls)]}"
            color = colors[int(cls) if int(cls) < len(colors) else 0]
            cv2.putText(full_img, label2 + ':' + str(conf), (x1 + 5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.imwrite(str(preview_file), full_img)

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

# -----------------------------
# MAIN
# -----------------------------
def main():
    # 1. Run Stage 1 inference (crop A/F)
    stage1_detections = run_stage1_inference(
        stage1_model_path,
        stage1_input_path,
        stage1_output_path)

    # 2. Run Stage 2 inference on crops  
    run_stage2_inference(
        stage2_model_path,
        stage2_input_path,
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

if __name__ == "__main__":
    main()





