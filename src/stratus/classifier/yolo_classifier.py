from __future__ import annotations
import cv2
import numpy as np
import logging
from pathlib import Path

from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand, DetectedObject
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CLASSES = ["marker", "cup", "pen", "pencil", "cell phone", "phone",
           "bottle", "remote", "keyboard", "mouse", "scissors",
           "book", "laptop", "watch", "coin", "card", "cable",
           "charger", "adapter", "battery", "toy", "eraser",
           "stapler", "tape", "glue", "ruler", "clip"]

DROP_JOINTS = {
    "A": [45, -30, 30, 0, 0, 0],
    "B": [-45, -30, 30, 0, 0, 0],
    "C": [0, -50, 60, 0, 0, 0],
}


class YOLOClassifier:
    def __init__(self, model_path: str = "models/yolov8s-world.pt",
                 conf_threshold: float = 0.1,
                 map_offset_x: float = 0.15, map_scale_x: float = 0.50,
                 map_offset_y: float = -0.20, map_scale_y: float = 0.40,
                 pickup_z: float = 0.15, pitch: float = 0.2):
        path = Path(model_path)
        if not path.exists():
            logger.info("Downloading YOLO-World model (first run)...")
        self._model = YOLO(str(model_path))
        self._model.set_classes(CLASSES)
        self._conf = conf_threshold
        self._map_offset_x = map_offset_x
        self._map_scale_x = map_scale_x
        self._map_offset_y = map_offset_y
        self._map_scale_y = map_scale_y
        self._pickup_z = pickup_z
        self._pitch = pitch
        self._bg_captured = False
        logger.info("YOLO-World loaded (%d custom classes)", len(CLASSES))

    def set_background(self, frame: CameraFrame) -> None:
        self._bg_captured = True

    def classify(self, frame: CameraFrame) -> TriageCommand:
        h, w = frame.image.shape[:2]

        if not self._bg_captured:
            return TriageCommand(
                action="none", target_bin="", label="",
                detected_labels=[], detected_objects=[],
                pickup_pose=None, drop_joints=DROP_JOINTS["B"],
            )

        results = self._model(frame.image, conf=self._conf, verbose=False, iou=0.5)[0]

        labels = []
        objects = []
        for b in results.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            label = results.names[cls_id]
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            labels.append({"label": label, "confidence": conf})
            objects.append(DetectedObject(
                name=label, confidence=conf,
                left=x1 / w, top=y1 / h,
                width=(x2 - x1) / w, height=(y2 - y1) / h,
            ))

        unique = list(dict.fromkeys(l["label"] for l in labels))
        logger.info(f"Detected: {unique}")

        if not unique:
            logger.info("No objects detected")
            return TriageCommand(
                action="none", target_bin="", label="",
                detected_labels=[], detected_objects=[],
                pickup_pose=None, drop_joints=None,
            )

        grade = "A"
        target = "A"
        top = unique[:5]

        obj = objects[0]
        cx = (obj.left + obj.width / 2)
        cy = (obj.top + obj.height / 2)
        map_x = self._map_offset_x + cx * self._map_scale_x
        map_y = self._map_offset_y + cy * self._map_scale_y

        logger.info(f"Pick {top[0]} at ({map_x:.3f}, {map_y:.3f}) -> bin_{target.lower()}")

        return TriageCommand(
            action="pick_and_place", target_bin=f"bin_{target.lower()}",
            label=f"Grade {grade} - Refurbishable",
            detected_labels=top,
            detected_objects=objects,
            pickup_pose={"x": map_x, "y": map_y, "z": self._pickup_z,
                         "roll": 0, "pitch": self._pitch, "yaw": 0},
            drop_joints=DROP_JOINTS[target],
        )
