from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
from yolo3.model import preprocess_true_boxes


'''https://github.com/david8862/keras-YOLOv3-model-set/blob/475953c7da123c2c3e1b24b83676dbcda7576e7c/common/data_utils.py#L111
에서 reshape_box() 함수 참조하여 box filtering 적용 여부  검토 할것. '''

class YoloSequence(Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, anchors, max_boxes=20, augmentor=None):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.anchors = anchors
        self.max_boxes = max_boxes
        self.augmentor = augmentor

    def __len__(self):
        return int(np.ceil(len(self.annotation_lines) / self.batch_size))

    def __getitem__(self, index):
        annotation_batch = self.annotation_lines[index * self.batch_size:(index + 1) * self.batch_size]
        # scale/augmentation 변환된 image와 개별 image별로 scale된 box 정보를 배치건수 만큼 받을 수 있는 배열 image_batch, box_batch 생성.
        image_batch = np.zeros((len(annotation_batch), self.input_shape[0], self.input_shape[1], 3), dtype='float32')
        box_batch = np.zeros((len(annotation_batch), self.max_boxes, 5))
        # 배치건수만큼 반복하면서 scale/augmented 된 image와 box정보를 image_batch, box_batch에 담음.
        for batch_index in range(len(annotation_batch)):
            annotation_line = annotation_batch[batch_index]
            image_data, box_data = self.get_image_box(annotation_line, input_shape=self.input_shape,
                                                      max_boxes=self.max_boxes, augmentor=self.augmentor)
            image_batch[batch_index] = image_data
            box_batch[batch_index] = box_data

        y_true = preprocess_true_boxes(box_batch, self.input_shape, self.anchors, self.num_classes)

        return [image_batch, *y_true], np.zeros(self.batch_size)

    # 개별 scale/augmented image및 scale적용된 box 정보 반환'
    # annotation 좌표는 (xmin, ymin, xmax, ymax) 형식으로 입력됨. yolo는 xcenter, ycenter, width, height 형태임.'
    def get_image_box(self, annotation_line, input_shape, max_boxes=20, augmentor=None):
        line = annotation_line.split()
        image_filename = line[0]
        # annotation에 있는 파일절대경로를 참조하여 image 배열로 로드하고 이미지의 사이즈 추출.
        image = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[0:2]
        input_height, input_width = input_shape

        # 하나의 image에 여러개의 box 정보가 있음.
        line = annotation_line.split()
        boxes_list = [box.split(',') for box in [box for box in line[1:]]]
        # boxes는 개별 box의 xmin, ymin, xmax, ymax, class값을 1차원 배열로 여러개를 가지는 2차원 배열.
        boxes = np.array(boxes_list, dtype=np.int32)

        # 만일 augmentor가 입력되면 원본 image를 augmentation적용하고 bbox정보도 변환.
        if augmentor is not None:
            transformed = augmentor(image=image, bboxes=boxes)
            image = transformed['image']
            # albumentation의 bbox 변환 시 tuple로 값이 반환됨. 이를 array로 변환.
            boxes = np.array(transformed['bboxes'], np.int32)

        # model의 input_size에 원본 image의 scale을 맞추기 위한 width/height scale값 구함.
        scale_w = input_width / image_width
        scale_h = input_height / image_height

        # 입력된 input_size로 원본 image size를 재 조정.
        image_resized = cv2.resize(image, input_shape)

        # 하나에 image에 여러개의 box 정보를 받을 수 있는 array 생성. image size를 scale하였으므로 scale을 반영한 좌표 정보를 저장.
        box_data = np.zeros((max_boxes, 5))
        # 적어도 하나의 box 좌표가 있을 경우
        if len(boxes) > 0:
            if len(boxes) > max_boxes:
                boxes = boxes[:max_boxes]
            # boxes[:, [0, 2]]는 x_min, x_max이고 boxes[:, [1, 3]]은 y_min, y_max임.
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
            box_data[:len(boxes)] = boxes
            # scaling된 좌표값을 int 형으로 일괄 변경.
            box_data = box_data.astype(np.int32)

        return image_resized, box_data