"""This module contains all configuration settings for the ReID module"""

import os
import sys


data_pat = os.getcwd()
MODEL_WEIGHT_ALIGNED = os.path.join(data_pat, "models", "alignedReID", "model_weight.pth")
MODEL_YOLOV3_WEIGHTS = os.path.join(data_pat, "models", "yolov3", "yolov3.weights")
MODEL_YOLOV3_CFG = os.path.join(data_pat, "models",  "yolov3", "yolov3.cfg")
PATH_CLASSES_COCO = os.path.join(data_pat, "models",  "yolov3", "coco.names")

# classes from Microsoft coco, required for YOLOV3
CLASSES_COCO = None
with open(PATH_CLASSES_COCO, 'r') as f:
    CLASSES_COCO = [line.strip() for line in f.readlines()]

# Params for qualitative analysis
CONFIDENCE = 0.7  # confidence value for cv2 object detector
FRAME_DROP = 60  # process every FRAME_DROP'th to make predictions less noisy

# Params for quantitattive analysis
NO_SHOTS = 50
UNKNOWNS = 100
