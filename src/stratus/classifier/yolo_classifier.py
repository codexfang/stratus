from __future__ import annotations
import cv2
import numpy as np
import logging

from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand, DetectedObject
from stratus.core.detector import LocalDetector
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs for common ITAD-relevant objects
GRADE_A_IDS = {41: "cup", 39: "bottle", 47: "apple", 46: "banana",
               77: "cell phone", 76: "keyboard", 75: "mouse",
               73: "book", 76: "remote", 74: "clock",
               72: "tv", 70: "laptop", 62: "chair",
               84: "book", 67: "cell phone"}
GRADE_C_IDS = {}

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

DROP_JOINTS = {
    "A": [45, -30, 30, 0, 0, 0],
    "B": [-45, -30, 30, 0, 0, 0],
    "C": [0, -50, 60, 0, 0, 0],
}


class YOLOClassifier:
    def __init__(self, model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.5):
        self._model = YOLO(str(model_path))
        self._conf = conf_threshold
        self._detector = LocalDetector(min_area=800)
        self._bg_captured = False
        n = self._model.names
        logger.info("YOLO loaded (%d classes)", len(n))

    def set_background(self, frame: CameraFrame) -> None:
        self._detector.set_background(frame.image)
        self._bg_captured = True

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        sharp = cv2.addWeighted(enhanced, 1.5,
                                cv2.GaussianBlur(enhanced, (0, 0), 2.0), -0.5, 0)
        return sharp

    def classify(self, frame: CameraFrame) -> TriageCommand:
        h, w = frame.image.shape[:2]
        margin_x, margin_y = int(w * 0.35), int(h * 0.35)
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

        candidates = self._detector.detect(frame.image)
        if not candidates:
            logger.info("No objects detected")
            return TriageCommand(
                action="none", target_bin="", label="",
                detected_labels=[], detected_objects=[],
                pickup_pose=None, drop_joints=None,
            )

        box = candidates[0]
        crop = self._detector.crop_object(frame.image, box, margin=0.4)
        if crop.shape[0] < 64 or crop.shape[1] < 64:
            return TriageCommand(
                action="none", target_bin="", label="",
                detected_labels=[], detected_objects=[],
                pickup_pose=None, drop_joints=None,
            )

        crop_hires = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
        enhanced = self._enhance(crop_hires)

        results = self._model(enhanced, conf=self._conf, verbose=False)[0]
        labels = []
        for b in results.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            label = COCO_NAMES.get(cls_id, f"id_{cls_id}")
            labels.append({"label": label, "confidence": conf})

        unique = list(dict.fromkeys(l["label"] for l in labels))
        logger.info(f"Object @({box.cx:.2f},{box.cy:.2f}) {box.w}x{box.h}: {unique[:6]}")

        if not unique:
            unique = ["unknown"]

        lower = {l.lower() for l in unique}
        if lower & {"broken", "cracked", "damaged", "scratch", "rust", "burn", "bent", "dented"}:
            grade, text, target = "C", "Scrap/Recycle", "C"
        elif lower & {"cup", "bottle", "cell phone", "keyboard", "mouse",
                       "book", "scissors", "laptop", "remote", "clock",
                       "fork", "spoon", "knife", "bowl", "vase",
                       "teddy bear", "tv", "refrigerator", "microwave"}:
            grade, text, target = "A", "Refurbishable", "A"
        else:
            grade, text, target = "B", "Needs Repair", "B"

        top = unique[:5]

        ws_cx = ws_x1 + box.x + box.w / 2
        ws_cy = ws_y1 + box.y + box.h / 2
        norm_x = ws_cx / w
        norm_y = ws_cy / h
        map_x = 0.08 + norm_x * 0.34
        map_y = -0.15 + norm_y * 0.30

        logger.info(f"Pick {top[0]} at ({map_x:.3f}, {map_y:.3f}) -> bin_{target.lower()} (Grade {grade})")

        return TriageCommand(
            action="pick_and_place", target_bin=f"bin_{target.lower()}",
            label=f"Grade {grade} - {text}",
            detected_labels=top,
            detected_objects=[DetectedObject(
                name=top[0], confidence=80.0,
                left=box.x / w, top=box.y / h,
                width=box.w / w, height=box.h / h,
            )],
            pickup_pose={"x": map_x, "y": map_y, "z": 0.12,
                         "roll": 0, "pitch": 0.4, "yaw": 0},
            drop_joints=DROP_JOINTS[grade],
        )
