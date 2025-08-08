shuffle.py
Once the annotations are downloaded, this program can be used to divide the dataset into train and val, for both images and labels.


split.py
split group labels and bulb labels
used to include all annotations without filter
It's likely, you need to run this script two times: first for train, and then for val.


filter.py
filter by weather (clear, rain, snow: TODO) or view (front, side), in images and labels
split labels with filter: e.g., select bulbs within front groups
It's likely that you need to run this script two times: first for train, and then for val. 


prepare_stage2_data.py
crop group area images, as preparation to training of models in stage 2
The new labels' coordiantes are re-calculated, to fit into the cropped images.
It's likely that you need to run this script two times: first for train, and then for val. 


run_inference_plot.py
visually check the detected objects
Now this program is optional: this function has been integrated into inference_final.py


inference_final.py
stage1: find group
stage2: crop group area, and find bulb inside group area
projection: local box mapped to global box


evaluate_yolo_with_coco_ext.py
yolo-to-coco annotation/label conversion
evluate prediction results: compare with ground truth
plot graphs


