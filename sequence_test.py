"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from generator2 import YoloSequence
import albumentations as A

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def _main():
    annotation_path = r'C:\Users\q\raccoon\annotations\raccoon_anno.csv'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw



    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    batch_size = 32
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    augmentor = A.Compose([
        A.HorizontalFlip(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc'))
    #gen_result = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
    # result1, result2 = next(gen_result)
    train_set = YoloSequence(lines[:num_train], input_shape, batch_size, num_classes, anchors, max_boxes=20, augmentor=augmentor)
    valid_set = YoloSequence(lines[num_train:], input_shape, batch_size, num_classes, anchors, max_boxes=20, augmentor=None)

    image_batch, batch_size_array = next(iter(train_set))
    print(image_batch, batch_size_array)
_main()
