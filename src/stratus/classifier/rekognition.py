from __future__ import annotations
import cv2
import numpy as np
import logging
import time

from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand, DetectedObject
from stratus.core.detector import LocalDetector

try:
    import boto3
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)

GRADE_A = {"Electronics", "Circuit Board", "Computer Component", "CPU",
           "Server", "Hardware", "Chip", "Processor", "Memory", "RAM",
           "Network Equipment", "Router", "Switch", "Modem",
           "Hard Drive", "SSD", "Storage Device",
           "Camera", "Lens", "Sensor", "Optics",
           "Adapter", "Connector", "Cable", "Wire",
           "Keyboard", "Mouse", "Peripheral",
           "Cell Phone", "Smartphone", "Tablet Computer", "Laptop",
           "Computer", "Desktop", "Monitor", "Screen",
           "Battery", "Power Supply", "Charger",
           "Microchip", "Integrated Circuit", "PCB", "Motherboard",
           "Drive", "Disk", "Flash Drive", "Memory Card",
           "Network Card", "Graphics Card", "Video Card", "GPU",
           "Heat Sink", "Fan", "Cooler", "Controller",
           "Electronic Device", "Device", "Tool",
           "Equipment", "Machine", "Appliance",
           "Toy", "Figure", "Miniature",
           "Office Supply", "Stationery", "Writing Instrument",
           "Pen", "Pencil", "Marker", "Highlighter",
           "Eraser", "Ruler", "Scissors", "Tape", "Glue",
           "Book", "Document", "Notebook", "Folder",
           "Bottle", "Container", "Cup", "Mug",
           "Can", "Box", "Package", "Bag", "Wrap",
           "Key", "Lock", "Padlock", "Badge", "ID Card",
           "Coin", "Money", "Currency", "Card",
           "Jewelry", "Ring", "Watch", "Bracelet", "Necklace",
           "Clothing", "Hat", "Cap", "Glove", "Shoe",
           "Food", "Snack", "Fruit", "Vegetable",
           "Utensil", "Spoon", "Fork", "Knife",
           "Plant", "Flower", "Leaf", "Branch"}

GRADE_C = {"Damage", "Scratch", "Crack", "Dent", "Rust", "Corrosion",
           "Broken", "Fracture", "Worn", "Defect", "Stain",
           "Crumpled", "Torn", "Bent", "Burn",
           "Ripped", "Faded", "Discolored",
           "Cracked", "Shattered", "Chipped",
           "Scratched", "Dented", "Rusted", "Corroded",
           "Broken", "Fractured"}

GRADE_C_LABELS = {l.lower() for l in GRADE_C}
GRADE_A_LABELS = {l.lower() for l in GRADE_A}


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
        sharp = cv2.addWeighted(enhanced, 1.5, cv2.GaussianBlur(enhanced, (0, 0), 2.0), -0.5, 0)
        return sharp

    def _rekognize(self, img: np.ndarray) -> list[str]:
        enhanced = self._enhance(img)
        _, buffer = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        response = self._client.detect_labels(
            Image={"Bytes": buffer.tobytes()},
            MaxLabels=30, MinConfidence=self._min_conf,
        )
        names = [l["Name"] for l in response["Labels"]]
        return names

    def _grade(self, labels: list[str]) -> tuple[str, str, str]:
        lower = {l.lower() for l in labels}
        if lower & GRADE_C_LABELS:
            return "C", "Scrap/Recycle", "bin_c"
        if lower & GRADE_A_LABELS:
            return "A", "Refurbishable", "bin_a"
        return "B", "Needs Repair", "bin_b"

    def classify(self, frame: CameraFrame) -> TriageCommand:
        h, w = frame.image.shape[:2]
        margin_x, margin_y = int(w * 0.35), int(h * 0.35)
        ws_x1, ws_y1 = margin_x, margin_y
        ws_x2, ws_y2 = w - margin_x, h - margin_y

        if not self._bg_captured:
            crop = frame.image[ws_y1:ws_y2, ws_x1:ws_x2]
            upscaled = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
            names = self._rekognize(upscaled)
            top = names[:5] if names else ["scanning..."]
            grade, text, target = self._grade(names)
            logger.info(f"No background. Labels: {top}")
            return TriageCommand(
                action="pick_and_place", target_bin=target,
                label=f"Grade {grade} - {text}",
                detected_labels=top, detected_objects=[],
                pickup_pose={"x": 0.25, "y": 0.0, "z": 0.15, "roll": 0, "pitch": 0.4, "yaw": 0},
                drop_joints={"bin_a": [45, -30, 30, 0, 0, 0],
                             "bin_b": [-45, -30, 30, 0, 0, 0],
                             "bin_c": [0, -50, 60, 0, 0, 0]}[target],
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

        crop_hires = cv2.resize(crop, (832, 832), interpolation=cv2.INTER_CUBIC)
        names = self._rekognize(crop_hires)
        unique = list(dict.fromkeys(names))
        logger.info(f"Object @({box.cx:.2f},{box.cy:.2f}) {box.w}x{box.h}: {unique[:6]}")

        if not unique:
            unique = ["unknown"]
        grade, text, target = self._grade(unique)
        top = unique[:5]

        ws_cx = ws_x1 + box.x + box.w / 2
        ws_cy = ws_y1 + box.y + box.h / 2
        norm_x = ws_cx / w
        norm_y = ws_cy / h

        map_x = 0.08 + norm_x * 0.34
        map_y = -0.15 + norm_y * 0.30

        logger.info(f"Pick {top[0]} at ({map_x:.3f}, {map_y:.3f}) -> {target} (Grade {grade})")

        return TriageCommand(
            action="pick_and_place", target_bin=target,
            label=f"Grade {grade} - {text}",
            detected_labels=top,
            detected_objects=[DetectedObject(
                name=top[0], confidence=80.0,
                left=box.x / w, top=box.y / h,
                width=box.w / w, height=box.h / h,
            )],
            pickup_pose={"x": map_x, "y": map_y, "z": 0.12,
                         "roll": 0, "pitch": 0.4, "yaw": 0},
            drop_joints={"bin_a": [45, -30, 30, 0, 0, 0],
                         "bin_b": [-45, -30, 30, 0, 0, 0],
                         "bin_c": [0, -50, 60, 0, 0, 0]}[target],
        )
