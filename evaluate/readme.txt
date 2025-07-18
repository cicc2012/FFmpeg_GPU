split.py
split group labels and bulb labels
used to include all annotations without filter


filter.py
filter by weather (clear, rain, snow) or view (front, side), in images and labels
split labels with filter: e.g., select bulbs within front groups


prepare_stage2_data.py
crop group area images, as preparation to training of models in stage 2


run_inference_plot.py
visually check the detected objects


inference_final.py
stage1: find group
stage2: crop group area, and find bulb inside group area
projection: local box mapped to global box


evaluate_yolo_with_coco_ext.py
yolo-to-coco annotation/label conversion
evluate prediction results: compare with ground truth
plot graphs


