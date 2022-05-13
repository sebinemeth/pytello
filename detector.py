from time import time

import cv2
import seaborn as sns
import torch

# yolo (coco) classes
classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush')
colors = [(int(c[0] * 256), int(c[1] * 256), int(c[2] * 256)) for c in sns.color_palette("bright", 80)]


class Detection(object):
    def __init__(self, x=0, y=0, w=0, h=0, conf=0, cls=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.conf = conf
        self.cls = cls

    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2, conf, n_cls, label):
        return Detection(x1, y1, x2 - x1, y2 - y1, conf, n_cls)

    @property
    def label(self):
        if self.cls is None or self.cls < 0 or self.cls >= len(classes):
            return None
        return classes[self.cls]

    @property
    def corners(self):
        return (int(self.x), int(self.y)), (int(self.x + self.w), int(self.y + self.h))

    @classmethod
    def print_to_frame(cls, detections, frame):
        for d in detections:

            # overlay = frame.copy()
            # cv2.rectangle(overlay, d.corners[0], d.corners[1], colors[d.cls], -1)
            # c = d.conf / 6
            # frame = cv2.addWeighted(overlay, c, frame, 1 - c, 0)

            # overlay = frame.copy()
            # cv2.rectangle(overlay, d.corners[0], d.corners[1], colors[d.cls], 2)
            # cv2.putText(overlay, d.label, (int(d.x) + 8, int(d.y) + 20), 0, 0.6, colors[d.cls])
            # cv2.putText(overlay, f"{round(d.conf, 2)}", (int(d.x) + 8, int(d.y) + 40), 0, 0.6, colors[d.cls])
            # frame = cv2.addWeighted(overlay, d.conf, frame, 1 - d.conf, 0)

            overlay = frame.copy()
            cv2.rectangle(overlay, d.corners[0], d.corners[1], colors[d.cls], -1)
            c = d.conf / 6
            frame = cv2.addWeighted(overlay, c, frame, 1 - c, 0)

            overlay = frame.copy()
            cv2.rectangle(overlay, d.corners[0], d.corners[1], colors[d.cls], 2)
            cv2.rectangle(overlay, (int(d.x), int(d.y + d.h) - 26), (int(d.x + d.w), int(d.y + d.h)), colors[d.cls], -1)
            cv2.putText(overlay, d.label, (int(d.x) + 8, int(d.y + d.h) - 8), 0, 0.6, (255,255,255))
            cv2.putText(overlay, f"{round(d.conf, 2)}", (int(d.x) + 8, int(d.y) + 20), 0, 0.6, colors[d.cls])
            frame = cv2.addWeighted(overlay, d.conf, frame, 1 - d.conf, 0)

        return frame

    def __str__(self):
        return f"[{self.x} {self.x}] {self.w}x{self.h} {round(self.conf, 3)} {self.label} ({self.cls})"


class Detector(object):
    def __init__(self, adaptive=False):
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
        self._model = torch.hub.load('yolov5', 'custom', path='models/yolov5n.pt', source='local')
        self._adaptive = adaptive
        self._frame_time = 1
        self._counter = -1
        self._detection_freq = 1

    @property
    def fps(self):
        return 1 / self._frame_time

    @property
    def det_freq(self):
        return self._detection_freq

    def detect(self, frame):
        self._counter += 1
        if self._counter % self._detection_freq != 0:
            return []

        t = time()
        results = self._model(frame)
        self._frame_time = time() - t
        detections = [Detection.from_xyxy(*box) for box in results.pandas().xyxy[0].values]
        self._detection_freq = int(max(1, min(10, 21 - self.fps)))
        return detections
