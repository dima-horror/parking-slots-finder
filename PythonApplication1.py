import os
import cv2

import mrcnn.config
import mrcnn
from mrcnn.model import MaskRCNN
import mrcnn.utils

# import tensorflow as tf 
# print(tf.__version__)cl

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8 # минимальный процент отображения прямоугольника
    NUM_CLASSES = 81

DATASET_FILE = "mask_rcnn_coco.h5"
if not os.path.exists(DATASET_FILE):
    mrcnn.utils.download_trained_weights(DATASET_FILE)

model = MaskRCNN(mode="inference", model_dir="logs", config=MaskRCNNConfig())
model.load_weights(DATASET_FILE, by_name=True)

print('hello world')