from __future__ import annotations
import cv2
import numpy as np
import logging

from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand, DetectedObject
from stratus.core.detector import LocalDetector

try:
    import boto3
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)

# Drop joint angles (degrees) keyed by bin name
DROP_JOINTS = {
    "bin_a": [45, -30, 30, 0, 0, 0],
    "bin_b": [-45, -30, 30, 0, 0, 0],
    "bin_c": [0, -50, 60, 0, 0, 0],
}

GRADE_C = {
    "damage", "scratch", "crack", "dent", "rust", "corrosion",
    "broken", "fracture", "worn", "defect", "stain",
    "crumpled", "torn", "bent", "burn",
    "ripped", "faded", "discolored",
    "cracked", "shattered", "chipped",
    "scratched", "dented", "rusted", "corroded", "fractured",
}

GRADE_A = {
    "electronics", "circuit board", "computer component", "cpu",
    "server", "hardware", "chip", "processor", "memory", "ram",
    "network equipment", "router", "switch", "modem",
    "hard drive", "ssd", "storage device",
    "camera", "lens", "sensor", "optics",
    "adapter", "connector", "cable", "wire",
    "keyboard", "mouse", "peripheral",
    "cell phone", "smartphone", "tablet computer", "laptop",
    "computer", "desktop", "monitor", "screen",
    "battery", "power supply", "charger",
    "microchip", "integrated circuit", "pcb", "motherboard",
    "drive", "disk", "flash drive", "memory card",
    "network card", "graphics card", "video card", "gpu",
    "heat sink", "fan", "cooler", "controller",
    "electronic device", "device", "tool",
    "equipment", "machine", "appliance",
    "toy", "figure", "miniature",
    "office supply", "stationery", "writing instrument",
    "pen", "pencil", "marker", "highlighter",
    "eraser", "ruler", "scissors", "tape", "glue",
    "book", "document", "notebook", "folder",
    "bottle", "container", "cup", "mug",
    "can", "box", "package", "bag", "wrap",
    "key", "lock", "padlock", "badge", "id card",
    "coin", "money", "currency", "card",
    "jewelry", "ring", "watch", "bracelet", "necklace",
    "clothing", "hat", "cap", "glove", "shoe",
    "food", "snack", "fruit", "vegetable",
    "utensil", "spoon", "fork", "knife",
    "plant", "flower", "leaf", "branch",
}


class RekognitionClassifier:
    def __init__(self, region: str = "us-east-2", min_confidence: float = 20.0):
        if boto3 is None:
            raise ImportError("boto3 not installed. Run: pip install boto3")
        self._client = boto3.client("rekognition", region_name=region)
        self._min_conf = min_confidence
        self._detector = LocalDetector(min_area=800)
        self._bg_captured = False

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
        sharp = cv2.addWeighted(
            enhanced, 1.5, cv2.GaussianBlur(enhanced, (0, 0), 2.0), -0.5, 0
        )
        return sharp

    def _rekognize(self, img: np.ndarray) -> list[str]:
        enhanced = self._enhance(img)
        _, buffer = cv2.imencode(
            ".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        response = self._client.detect_labels(
            Image={"Bytes": buffer.tobytes()},
            MaxLabels=30,
            MinConfidence=self._min_conf,
        )
        return [lbl["Name"] for lbl in response["Labels"]]

    def _grade(self, labels: list[str]) -> tuple[str, str, str]:
        lower = {lbl.lower() for lbl in labels}
        if lower & GRADE_C:
            return "C", "Scrap/Recycle", "bin_c"
        if lower & GRADE_A:
            return "A", "Refurbishable", "bin_a"
        return "B", "Needs Repair", "bin_b"

    def classify(self, frame: CameraFrame) -> TriageCommand:
        h, w = frame.image.shape[:2]

        # If no background has been set yet, scan the full centre crop
        if not self._bg_captured:
            margin_x = int(w * 0.35)
            margin_y = int(h * 0.35)
            crop = frame.image[margin_y: h - margin_y, margin_x: w - margin_x]
            upscaled = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
            names = self._rekognize(upscaled)
            top = names[:5] if names else ["scanning..."]
            grade, text, target_bin = self._grade(names)
            logger.info("No background. Labels: %s", top)
            # Default pickup in the centre of the arm workspace
            return TriageCommand(
                action="pick_and_place",
                target_bin=target_bin,
                label=f"Grade {grade} - {text}",
                detected_labels=top,
                detected_objects=[],
                pickup_pose={"x": 0.35, "y": 0.0, "z": 0.10, "roll": 0, "pitch": 0.4, "yaw": 0},
                drop_joints=DROP_JOINTS[target_bin],
            )

        # Detect object via background subtraction
        candidates = self._detector.detect(frame.image)
        if not candidates:
            logger.info("No objects detected")
            return TriageCommand(
                action="none", target_bin="", label="",
                detected_labels=[], detected_objects=[],
                pickup_pose=None, drop_joints=None,
            )

        box = candidates[0]

        # Crop the detected region (with margin) for Rekognition
        crop = self._detector.crop_object(frame.image, box, margin=0.4)
        if crop.shape[0] < 64 or crop.shape[1] < 64:
            return TriageCommand(
                action="none", target_bin="", label="",
                detected_labels=[], detected_objects=[],
                pickup_pose=None, drop_joints=None,
            )

        crop_hires = cv2.resize(crop, (832, 832), interpolation=cv2.INTER_CUBIC)
        names = self._rekognize(crop_hires)
        unique = list(dict.fromkeys(names))
        logger.info("Object @(%.2f,%.2f) %dx%d: %s", box.cx, box.cy, box.w, box.h, unique[:6])

        if not unique:
            unique = ["unknown"]

        grade, text, target_bin = self._grade(unique)
        top = unique[:5]

        # ── Coordinate mapping ─────────────────────────────────────────────
        # box.cx / box.cy are already normalised to [0, 1] relative to the
        # FULL frame (LocalDetector stores cx = (x + w/2) / frame_w).
        # Map directly through the linear arm-workspace transform.
        #
        # The workspace is roughly the centre 30% of the frame on each axis,
        # so a centred cup at cx~0.5, cy~0.5 should map to ~(0.35, 0.0) in
        # arm coordinates — the centre of the reachable workspace.
        #
        # arm_x = 0.20 + cx_norm * 0.30   →  range [0.20, 0.50] m
        # arm_y = -0.15 + cy_norm * 0.30  →  range [-0.15, +0.15] m
        map_x = 0.20 + box.cx * 0.30
        map_y = -0.15 + box.cy * 0.30

        logger.info(
            "Pick '%s' at arm (%.3f, %.3f) -> %s (Grade %s)",
            top[0], map_x, map_y, target_bin, grade,
        )

        return TriageCommand(
            action="pick_and_place",
            target_bin=target_bin,
            label=f"Grade {grade} - {text}",
            detected_labels=top,
            detected_objects=[DetectedObject(
                name=top[0],
                confidence=80.0,
                left=box.x / w,
                top=box.y / h,
                width=box.w / w,
                height=box.h / h,
            )],
            pickup_pose={
                "x": map_x,
                "y": map_y,
                "z": 0.10,
                "roll": 0,
                "pitch": 0.4,
                "yaw": 0,
            },
            drop_joints=DROP_JOINTS[target_bin],
        )
