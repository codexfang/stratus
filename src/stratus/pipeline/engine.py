from __future__ import annotations
import cv2
import logging
import time
import numpy as np
from typing import Optional

from stratus.core.arm_driver import ArmDriver, TriageCommand, DetectedObject
from stratus.core.vision import Camera
from stratus.core.classifier import Classifier
from stratus.core.telemetry import TelemetryBridge, TelemetryEvent

logger = logging.getLogger(__name__)

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
BIN_NAMES = {"bin_a": "Bin A", "bin_b": "Bin B", "bin_c": "Bin C"}


class StratusPipeline:
    def __init__(
        self,
        arm: ArmDriver | None,
        camera: Camera,
        classifier: Classifier,
        telemetry: Optional[TelemetryBridge] = None,
        classify_every: int = 3,
        show_preview: bool = True,
    ):
        self._arm = arm
        self._camera = camera
        self._classifier = classifier
        self._telemetry = telemetry
        self._classify_every = classify_every
        self._show_preview = show_preview
        self._frame_count = 0
        self._last_h = 480
        self._last_w = 640
        self._bg_captured = False
        self._current_objects: list[DetectedObject] = []
        self._selected_idx: int = 0
        cv2.namedWindow("Stratus – ITAD Sorting")
        cv2.setMouseCallback("Stratus – ITAD Sorting", self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        h, w = self._last_h, self._last_w
        if h == 0 or w == 0:
            return
        self._selected_idx = -1
        for i, obj in enumerate(self._current_objects):
            x1 = int(obj.left * w)
            y1 = int(obj.top * h)
            x2 = int((obj.left + obj.width) * w)
            y2 = int((obj.top + obj.height) * h)
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._selected_idx = i
                logger.info("Selected object %d: %s", i, obj.name)
                break

    def _draw_boxes(self, display: np.ndarray, objects: list[DetectedObject],
                    highlight: int = -1) -> None:
        h, w = display.shape[:2]
        for i, obj in enumerate(objects):
            x1 = int(obj.left * w)
            y1 = int(obj.top * h)
            x2 = int((obj.left + obj.width) * w)
            y2 = int((obj.top + obj.height) * h)
            color = (0, 255, 255) if i == highlight else GREEN
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            if i == highlight:
                cv2.putText(display, f"[{i}] {obj.name}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_workspace(self, display: np.ndarray) -> None:
        h, w = display.shape[:2]
        mx, my = int(w * 0.35), int(h * 0.35)
        cv2.rectangle(display, (mx, my), (w - mx, h - my), (100, 100, 255), 2)

    def _bottom_bar(self, display: np.ndarray, text: str, color=WHITE) -> None:
        h, w = display.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        x = (w - tw) // 2
        y = h - 18
        overlay = display.copy()
        cv2.rectangle(overlay, (x - 10, y - th - 8), (x + tw + 10, y + 6), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        cv2.putText(display, text, (x, y), font, 0.7, color, 2)

    def _classify(self, frame) -> TriageCommand | None:
        logger.info("Classifying...")
        cmd = self._classifier.classify(frame)
        self._current_objects = cmd.detected_objects
        self._selected_idx = 0
        logger.info(f"-> {cmd.label}  labels={cmd.detected_labels[:3]}")
        return cmd

    def _exec_and_telemetry(self, cmd: TriageCommand) -> None:
        if self._arm:
            self._arm.execute_triage(cmd)
        if self._telemetry:
            drop = cmd.drop_joints or cmd.drop_pose
            self._telemetry.publish(TelemetryEvent(event_type="classification", payload={
                "action": cmd.action, "target_bin": cmd.target_bin,
                "frame": self._frame_count, "pickup": cmd.pickup_pose,
                "drop": drop, "grade": cmd.label,
            }))

    def _confirm(self, cmd: TriageCommand) -> bool:
        idx = self._selected_idx
        if idx < 0 or idx >= len(cmd.detected_objects):
            idx = 0
        obj = cmd.detected_objects[idx]
        name = obj.name
        for _ in range(300):
            frame = self._camera.read()
            if frame is None:
                continue
            display = frame.image.copy()
            self._draw_workspace(display)
            self._draw_boxes(display, cmd.detected_objects, highlight=idx)
            bin_name = BIN_NAMES.get(cmd.target_bin, cmd.target_bin)
            self._bottom_bar(display, f"[{idx}] {name}  ->  {bin_name}", GREEN)
            cv2.imshow("Stratus – ITAD Sorting", display)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('y'):
                cx = (obj.left + obj.width / 2)
                cy = (obj.top + obj.height / 2)
                cmd.pickup_pose = {"x": 0.15 + cx * 0.50, "y": -0.20 + cy * 0.40,
                                   "z": 0.15, "roll": 0, "pitch": 0.2, "yaw": 0}
                cmd.detected_labels = [obj.name]
                cmd.detected_objects = [obj]
                return True
            if key == ord('n'):
                return False
            if key == ord('q'):
                raise KeyboardInterrupt()
        return False

    def run_loop(self) -> None:
        logger.info("Stratus running. Y=pick, N=skip, Q=quit.\n")

        for _ in range(20):
            frame = self._camera.read()
            if frame is not None:
                break
            time.sleep(0.2)
        if frame is None:
            logger.error("No camera frame")
            return

        if self._show_preview:
            for i in range(30, 0, -1):
                frame = self._camera.read()
                if frame is None:
                    continue
                display = frame.image.copy()
                self._draw_workspace(display)
                self._bottom_bar(display, f"Clear workspace... {i}", (0, 255, 255))
                cv2.imshow("Stratus – ITAD Sorting", display)
                cv2.waitKey(1)
                time.sleep(0.05)
            cv2.waitKey(500)

        self._classifier.set_background(frame)
        self._bg_captured = True
        logger.info("Background captured")

        while True:
            frame = self._camera.read()
            if frame is None:
                continue
            self._frame_count += 1
            self._last_h, self._last_w = frame.image.shape[:2]
            display = frame.image.copy()

            cmd = None
            if self._frame_count % self._classify_every == 0:
                cmd = self._classify(frame)

            if self._show_preview:
                self._draw_workspace(display)
                if cmd:
                    self._draw_boxes(display, cmd.detected_objects)
                    if cmd.detected_labels:
                        name = cmd.detected_labels[0]
                        bin_name = BIN_NAMES.get(cmd.target_bin, cmd.target_bin)
                        self._bottom_bar(display, f"{name}  ->  {bin_name}", GREEN)
                    else:
                        self._bottom_bar(display, "scanning...", GRAY)
                else:
                    self._bottom_bar(display, "scanning...", GRAY)
                cv2.imshow("Stratus – ITAD Sorting", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt()

            if cmd is not None and cmd.action == "pick_and_place" and cmd.detected_labels:
                if self._confirm(cmd):
                    self._exec_and_telemetry(cmd)
                    logger.info("Picked.\n")
                else:
                    logger.info("Skipped.\n")
