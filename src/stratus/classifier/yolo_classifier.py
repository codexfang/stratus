from __future__ import annotations
import cv2
import numpy as np
import logging
from pathlib import Path

from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand, DetectedObject
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CLASSES = [
    "marker", "cup, coffee cup, mug, glass", "pen", "pencil",
    "cell phone", "phone", "bottle, water bottle", "remote",
    "keyboard", "mouse, computer mouse", "scissors",
    "book", "laptop", "watch", "coin", "card", "cable",
    "charger", "adapter", "battery", "toy", "eraser",
    "stapler", "tape", "glue", "ruler", "clip",
]

DROP_JOINTS = {
    "A": [60, -10, 40, 0, 10, 0],
    "B": [-60, -10, 40, 0, 10, 0],
    "C": [10, -10, 70, 0, 10, 0],
}

CALIBRATION_PATH = Path.home() / "stratus/calibration/workspace_cal.json"


class YOLOClassifier:
    def __init__(self, model_path: str = "models/yolov8s-world.pt",
                 conf_threshold: float = 0.15,
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

        # Load calibration if available
        self._homography = None
        self._use_calibration = False
        self._load_calibration()

        logger.info("YOLO-World loaded (%d custom classes)", len(CLASSES))

    def _load_calibration(self) -> None:
        if not CALIBRATION_PATH.exists():
            logger.info("No calibration file at %s, using linear map", CALIBRATION_PATH)
            return
        try:
            import json
            with open(CALIBRATION_PATH) as f:
                cal = json.load(f)
            # Support both "matrix" and "homography" keys for backward compat
            H = np.array(cal.get("matrix") or cal.get("homography", []), dtype=np.float32)
            if H.shape == (3, 3):
                # Only use homography if calibration error is reasonable (< 20mm)
                mean_err = cal.get("mean_error_mm", 999)
                if mean_err < 20:
                    self._homography = H
                    self._use_calibration = True
                    logger.info("Loaded workspace calibration from %s (type=%s, err=%.1fmm)",
                                CALIBRATION_PATH, cal.get("type", "homography"),
                                cal.get("mean_error_mm", 0))
                else:
                    logger.warning("Calibration error %.1fmm too high, ignoring homography", mean_err)
            else:
                logger.warning("Calibration matrix invalid shape: %s", H.shape)
        except Exception as e:
            logger.warning("Failed to load calibration: %s", e)

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

        # Select highest confidence object
        obj = max(objects, key=lambda o: o.confidence)
        cx = (obj.left + obj.width / 2)
        cy = (obj.top + obj.height / 2)

        # Determine target bin based on object class
        bin_map = {
            "cup, coffee cup, mug, glass": "A",
            "bottle, water bottle": "B",
            "book": "C",
        }
        target = bin_map.get(obj.name, "A")
        grade = "A"

        # Use calibration homography if available, else linear map
        if self._use_calibration and self._homography is not None:
            # H maps pixel (x,y,1) -> world (x,y,w); use H directly
            pt = np.array([cx * w, cy * h, 1.0], dtype=np.float32)
            proj = self._homography @ pt
            map_x, map_y = proj[0] / proj[2], proj[1] / proj[2]
            logger.info(f"Calibrated pick: ({map_x:.3f}, {map_y:.3f}) from pixel ({cx*w:.1f},{cy*h:.1f})")
        else:
            map_x = self._map_offset_x + cx * self._map_scale_x
            map_y = self._map_offset_y + cy * self._map_scale_y
            logger.info(f"Linear map pick: ({map_x:.3f}, {map_y:.3f}) from pixel ({cx*w:.1f},{cy*h:.1f})")

        logger.info(f"Pick {obj.name} at ({map_x:.3f}, {map_y:.3f}) -> bin_{target.lower()}")

        return TriageCommand(
            action="pick_and_place", target_bin=f"bin_{target.lower()}",
            label=f"Grade {grade} - Refurbishable",
            detected_labels=unique[:5],
            detected_objects=objects,
            pickup_pose={"x": map_x, "y": map_y, "z": self._pickup_z,
                         "roll": 0, "pitch": self._pitch, "yaw": 0},
            drop_joints=DROP_JOINTS[target],
        )
