from __future__ import annotations
import cv2
import numpy as np
import logging
from pathlib import Path

from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand, DetectedObject
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# YOLO-World matches class names by text embedding similarity.
# Listing short common names alongside verbose ones improves recall —
# "cup" catches mugs/tumblers that the long form misses.
CLASSES = [
    "cup",
    "mug",
    "coffee cup",
    "cup, coffee cup, mug, glass",
    "pen",
    "pencil",
    "marker",
    "cell phone",
    "phone",
    "bottle",
    "water bottle",
    "bottle, water bottle",
    "remote",
    "keyboard",
    "mouse",
    "computer mouse",
    "mouse, computer mouse",
    "scissors",
    "book",
    "laptop",
    "watch",
    "coin",
    "card",
    "cable",
    "charger",
    "adapter",
    "battery",
    "toy",
    "eraser",
    "stapler",
    "tape",
    "glue",
    "ruler",
    "clip",
]

# Drop joint angles (degrees) for each bin.
# IMPORTANT: joint3 (idx2) faults in testing — keep idx2=0 for all bins.
# Use only idx0 (base rotation) to swing left/right to each bin,
# and idx1 (shoulder) to lower the arm down to drop height.
# idx2=0 avoids the motor fault. idx3,4,5 stay at 0.
#
# Bin layout (facing forward from arm base):
#   A = refurbishable items  → swing RIGHT  (idx0 positive)
#   B = recyclable/scrap     → swing LEFT   (idx0 negative)
#   C = books/media          → straight ahead, lowered (idx0=0)
DROP_JOINTS = {
    "A": [45,  -20, 0, 0, 0, 0],   # swing 45° right, shoulder lowers to drop height
    "B": [-45, -20, 0, 0, 0, 0],   # swing 45° left
    "C": [0,   -20, 0, 0, 0, 0],   # straight ahead, lowered
}

# Which bin each detected class maps to.
# Short aliases map to the same bin as their verbose counterparts.
BIN_MAP: dict[str, str] = {
    "cup":                          "A",
    "mug":                          "A",
    "coffee cup":                   "A",
    "cup, coffee cup, mug, glass":  "A",
    "bottle":                       "B",
    "water bottle":                 "B",
    "bottle, water bottle":         "B",
    "book":                         "C",
    "laptop":                       "A",
    "cell phone":                   "A",
    "phone":                        "A",
    "keyboard":                     "A",
    "mouse":                        "A",
    "computer mouse":               "A",
    "mouse, computer mouse":        "A",
}

CALIBRATION_PATH = Path.home() / "Projects/stratus/calibration/workspace_cal.json"


def _box_iou(a: DetectedObject, b: DetectedObject) -> float:
    """Compute IoU between two normalised DetectedObject boxes."""
    ax1, ay1 = a.left, a.top
    ax2, ay2 = a.left + a.width, a.top + a.height
    bx1, by1 = b.left, b.top
    bx2, by2 = b.left + b.width, b.top + b.height
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a.width * a.height) + (b.width * b.height) - inter
    return inter / union if union > 0 else 0.0


def _nms_merge(objects: list[DetectedObject],
               iou_threshold: float = 0.4) -> list[DetectedObject]:
    """Greedy NMS: for overlapping boxes keep only the highest-confidence one.
    Handles the case where YOLO-World fires both "cup" and
    "cup, coffee cup, mug, glass" on the same physical object."""
    objects = sorted(objects, key=lambda o: o.confidence, reverse=True)
    kept: list[DetectedObject] = []
    suppressed = set()
    for i, obj in enumerate(objects):
        if i in suppressed:
            continue
        kept.append(obj)
        for j in range(i + 1, len(objects)):
            if j not in suppressed and _box_iou(obj, objects[j]) > iou_threshold:
                suppressed.add(j)
    return kept


class YOLOClassifier:
    def __init__(
        self,
        model_path: str = "models/yolov8s-world.pt",
        conf_threshold: float = 0.10,   # lowered from 0.15 — catches cups more reliably
        # Linear map: arm_x = map_offset_x + cx_norm * map_scale_x
        map_offset_x: float = 0.15,
        map_scale_x: float = 0.50,
        map_offset_y: float = -0.20,
        map_scale_y: float = 0.40,
        pickup_z: float = 0.10,
        pitch: float = 0.4,
    ):
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

        # Workspace homography (loaded from calibration file if present)
        self._homography: np.ndarray | None = None
        self._use_calibration = False
        self._load_calibration()

        logger.info("YOLO-World loaded (%d custom classes)", len(CLASSES))

    def _load_calibration(self) -> None:
        if not CALIBRATION_PATH.exists():
            logger.info("No calibration file at %s — using linear map", CALIBRATION_PATH)
            return
        try:
            import json
            with open(CALIBRATION_PATH) as f:
                cal = json.load(f)
            # Support both "matrix" and "homography" keys
            H = np.array(cal.get("homography") or cal.get("matrix", []), dtype=np.float32)
            if H.shape == (3, 3):
                mean_err = cal.get("mean_error_mm", 999)
                if mean_err < 20:
                    self._homography = H
                    self._use_calibration = True
                    logger.info(
                        "Loaded workspace calibration (err=%.1f mm)", mean_err
                    )
                else:
                    logger.warning(
                        "Calibration error %.1f mm too high — ignoring", mean_err
                    )
            else:
                logger.warning("Calibration matrix bad shape: %s", H.shape)
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

        results = self._model(frame.image, conf=self._conf, verbose=False, iou=0.45)[0]

        objects: list[DetectedObject] = []
        for b in results.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            label = results.names[cls_id]
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            objects.append(DetectedObject(
                name=label,
                confidence=conf,
                left=x1 / w,
                top=y1 / h,
                width=(x2 - x1) / w,
                height=(y2 - y1) / h,
            ))

        if not objects:
            return TriageCommand(
                action="none", target_bin="", label="",
                detected_labels=[], detected_objects=[],
                pickup_pose=None, drop_joints=None,
            )

        # Deduplicate: when both "cup" and "cup, coffee cup, mug, glass" fire,
        # keep only the highest-confidence one for each spatial location.
        # We do a simple IoU merge — boxes with IoU > 0.4 keep only the top conf.
        objects = _nms_merge(objects, iou_threshold=0.4)

        unique_labels = list(dict.fromkeys(o.name for o in objects))
        logger.info("Detected (post-NMS): %s", unique_labels)

        # Pick the primary pick target from the LARGEST-area bounding box (area
        # first, confidence as tiebreak). The biggest box is the actual object —
        # not a spurious high-labels fragment — so its center is the accurate
        # pick point. High confidence boxes that are tiny fragments (handle,
        # rim, shadow) get ignored this way.
        obj = max(objects, key=lambda o: (o.width * o.height, o.confidence))

        # Normalised centre of the bounding box (0-1)
        cx = obj.left + obj.width / 2
        cy = obj.top + obj.height / 2

        bin_key = BIN_MAP.get(obj.name, "A")
        target_bin = f"bin_{bin_key.lower()}"

        # Pixel → arm-workspace conversion
        if self._use_calibration and self._homography is not None:
            px_abs = cx * w
            py_abs = cy * h
            pt = np.array([[[px_abs, py_abs]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, self._homography)
            map_x = float(out[0, 0, 0])
            map_y = float(out[0, 0, 1])
            logger.info("Calibrated pick: (%.3f, %.3f) from pixel (%.1f, %.1f)",
                        map_x, map_y, px_abs, py_abs)
        else:
            map_x = self._map_offset_x + cx * self._map_scale_x
            map_y = self._map_offset_y + cy * self._map_scale_y
            logger.info("Linear map pick: (%.3f, %.3f) from centre (%.3f, %.3f)",
                        map_x, map_y, cx, cy)

        logger.info("Pick '%s' -> %s at arm (%.3f, %.3f)", obj.name, target_bin, map_x, map_y)

        return TriageCommand(
            action="pick_and_place",
            target_bin=target_bin,
            label="Grade A - Refurbishable",
            detected_labels=unique_labels[:5],
            detected_objects=objects,
            pickup_pose={
                "x": map_x,
                "y": map_y,
                "z": self._pickup_z,
                "roll": 0,
                "pitch": self._pitch,
                "yaw": 0,
            },
            drop_joints=DROP_JOINTS[bin_key],
        )
