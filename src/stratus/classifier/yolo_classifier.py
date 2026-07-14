from __future__ import annotations
import cv2
import numpy as np
import logging

from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand, DetectedObject
from ultralytics import YOLO

logger = logging.getLogger(__name__)

COCO_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
              5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
              10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
              14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
              20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
              25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
              30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
              35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
              39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
              44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
              49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
              54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
              59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
              64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
              68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
              73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
              78: 'hair drier', 79: 'toothbrush'}

GRADE_A_NAMES = {"cup", "bottle", "cell phone", "keyboard", "mouse",
                 "book", "scissors", "laptop", "remote", "clock",
                 "fork", "spoon", "knife", "bowl", "vase",
                 "teddy bear", "tv", "refrigerator", "microwave",
                 "toaster", "oven", "sink", "chair", "couch",
                 "potted plant", "dining table", "donut", "cake",
                 "banana", "apple", "sandwich", "orange", "broccoli",
                 "carrot", "hot dog", "pizza", "wine glass",
                 "baseball bat", "sports ball", "frisbee", "kite",
                 "umbrella", "backpack", "suitcase", "handbag",
                 "tie", "bird", "cat", "dog", "horse",
                 "car", "bicycle", "motorcycle", "truck", "bus",
                 "train", "boat", "airplane"}
GRADE_C_NAMES = set()

DROP_JOINTS = {
    "A": [45, -30, 30, 0, 0, 0],
    "B": [-45, -30, 30, 0, 0, 0],
    "C": [0, -50, 60, 0, 0, 0],
}


class YOLOClassifier:
    def __init__(self, model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.3):
        self._model = YOLO(str(model_path))
        self._conf = conf_threshold
        self._bg_captured = False
        logger.info("YOLO loaded (%d classes)", len(COCO_NAMES))

    def set_background(self, frame: CameraFrame) -> None:
        self._bg_captured = True

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        sharp = cv2.addWeighted(enhanced, 1.5,
                                cv2.GaussianBlur(enhanced, (0, 0), 2.0), -0.5, 0)
        return sharp

    def classify(self, frame: CameraFrame) -> TriageCommand:
        h, w = frame.image.shape[:2]
        margin_x, margin_y = int(w * 0.30), int(h * 0.30)
        ws_x1, ws_y1 = margin_x, margin_y
        ws_x2, ws_y2 = w - margin_x, h - margin_y

        if not self._bg_captured:
            return TriageCommand(
                action="pick_and_place", target_bin="B",
                label="", detected_labels=[], detected_objects=[],
                pickup_pose={"x": 0.25, "y": 0.0, "z": 0.15,
                             "roll": 0, "pitch": 0.4, "yaw": 0},
                drop_joints=DROP_JOINTS["B"],
            )

        # Run YOLO on the center crop at native resolution
        workspace = frame.image[ws_y1:ws_y2, ws_x1:ws_x2]
        enhanced = self._enhance(workspace)
        results = self._model(enhanced, conf=self._conf, verbose=False, imgsz=640)[0]

        labels = []
        objects = []
        for b in results.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            label = COCO_NAMES.get(cls_id, f"id_{cls_id}")
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            labels.append({"label": label, "confidence": conf})
            objects.append(DetectedObject(
                name=label, confidence=conf,
                left=(ws_x1 + x1) / w,
                top=(ws_y1 + y1) / h,
                width=(x2 - x1) / w,
                height=(y2 - y1) / h,
            ))

        unique = list(dict.fromkeys(l["label"] for l in labels))
        logger.info(f"Detected: {unique}")

        if not unique:
            logger.info("No objects detected by YOLO")
            return TriageCommand(
                action="none", target_bin="", label="",
                detected_labels=[], detected_objects=[],
                pickup_pose=None, drop_joints=None,
            )

        lower = {l.lower() for l in unique}
        if lower & GRADE_C_NAMES:
            grade, text, target = "C", "Scrap/Recycle", "C"
        elif lower & GRADE_A_NAMES:
            grade, text, target = "A", "Refurbishable", "A"
        else:
            grade, text, target = "B", "Needs Repair", "B"

        top = unique[:5]

        # Use the first detected box for pickup pose
        obj = objects[0]
        cx = (obj.left + obj.width / 2)
        cy = (obj.top + obj.height / 2)
        map_x = 0.08 + cx * 0.34
        map_y = -0.15 + cy * 0.30

        logger.info(f"Pick {top[0]} at ({map_x:.3f}, {map_y:.3f}) -> bin_{target.lower()} (Grade {grade})")

        return TriageCommand(
            action="pick_and_place", target_bin=f"bin_{target.lower()}",
            label=f"Grade {grade} - {text}",
            detected_labels=top,
            detected_objects=objects,
            pickup_pose={"x": map_x, "y": map_y, "z": 0.12,
                         "roll": 0, "pitch": 0.4, "yaw": 0},
            drop_joints=DROP_JOINTS[grade],
        )
