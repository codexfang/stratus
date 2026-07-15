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
        arm_camera: Camera | None = None,
        arm_cam_fov: float = 60.0,
        classify_every: int = 3,
        show_preview: bool = True,
    ):
        self._arm = arm
        self._camera = camera
        self._arm_camera = arm_camera
        self._arm_cam_fov = arm_cam_fov
        self._classifier = classifier
        self._telemetry = telemetry
        self._classify_every = classify_every
        self._show_preview = show_preview
        self._frame_count = 0
        self._last_h = 480
        self._last_w = 640
        self._bg_captured = False
        self._bg_frame = None
        self._current_objects: list[DetectedObject] = []
        self._selected_idx: int = 0
        self._last_arm_frame = None
        self._arm_frame_counter = 0
        self._update_preview_counter = 0
        cv2.namedWindow("Stratus")
        cv2.setMouseCallback("Stratus", self._on_mouse)

    def _update_preview(self) -> None:
        """Read both cameras and refresh the combined display.
        Call this during arm movement to keep preview live."""
        self._update_preview_counter += 1
        frame = self._camera.read()
        if frame is None:
            return
        display = frame.image.copy()
        self._last_h, self._last_w = display.shape[:2]
        self._draw_workspace(display)
        if self._arm_camera is not None and self._update_preview_counter % 5 == 0:
            self._last_arm_frame = self._arm_camera.read()
        self._show_both(display, self._last_arm_frame)
        cv2.waitKey(1)

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

    def _show_both(self, workspace: np.ndarray, arm_frame=None) -> None:
        if arm_frame is not None:
            h, w = workspace.shape[:2]
            arm = arm_frame.image.copy()
            ah = int(h * arm.shape[1] / arm.shape[0])
            arm = cv2.resize(arm, (h, ah))
            if ah < h:
                pad = np.zeros((h - ah, arm.shape[1], 3), dtype=np.uint8)
                arm = np.vstack([arm, pad])
            elif ah > h:
                arm = cv2.resize(arm, (h, h))
            cv2.putText(workspace, "workspace", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            cv2.putText(arm, "arm cam", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            combined = np.hstack([workspace, arm])
        else:
            combined = workspace
        max_w = 1600
        if combined.shape[1] > max_w:
            scale = max_w / combined.shape[1]
            new_w = int(combined.shape[1] * scale)
            new_h = int(combined.shape[0] * scale)
            combined = cv2.resize(combined, (new_w, new_h))
        cv2.imshow("Stratus", combined)

    def _classify(self, frame) -> TriageCommand | None:
        logger.info("Classifying...")
        cmd = self._classifier.classify(frame)
        self._current_objects = cmd.detected_objects
        self._selected_idx = 0
        logger.info(f"-> {cmd.label}  labels={cmd.detected_labels[:3]}")
        return cmd

    def _gripper_pixel(self, frame, estimated_gx=None, estimated_gy=None):
        """Find gripper centroid via background subtraction.
        Returns (norm_x, norm_y) or None."""
        if self._bg_frame is None:
            return None
        diff = cv2.absdiff(frame.image, self._bg_frame.image)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h, w = frame.image.shape[:2]
        candidates = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 300:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"] / w
            cy = M["m01"] / M["m00"] / h
            candidates.append((cx, cy, cv2.contourArea(cnt)))
        if not candidates:
            return None
        if estimated_gx is not None:
            candidates.sort(key=lambda c: (c[0] - estimated_gx)**2 + (c[1] - estimated_gy)**2)
            cx, cy, _ = candidates[0]
        else:
            cx, cy, _ = max(candidates, key=lambda c: c[2])
        return (cx, cy)



    def _visual_servo(self, cmd: TriageCommand) -> bool:
        """Move to pre-approach height at the stationary-camera estimated position.
        Returns False if unreachable."""
        if not self._arm or not cmd.pickup_pose:
            return True
        pu = cmd.pickup_pose
        pre_z = max(pu["z"] + 0.20, 0.30)
        pitch = pu.get("pitch", 0)

        ok = self._arm.move_to_pose(pu["x"], pu["y"], pre_z, pitch=pitch, duration=5.0,
                                    frame_cb=self._update_preview)
        if not ok:
            logger.warning("[servo] pre_z IK failed (%.3f, %.3f) — unreachable", pu["x"], pu["y"])
            return False
        cmd.pickup_refined = True
        return True

    def _micro_adjust(self, cmd: TriageCommand) -> None:
        """Open gripper, descend to 10 cm above object, read arm camera ONCE,
        apply a dampened XY correction, then set pickup_refined so execute_triage
        goes straight to pickup z."""
        if not self._arm or not cmd.pickup_pose:
            return
        if self._arm_camera is None or not self._arm_camera.is_connected:
            return
        pu = cmd.pickup_pose
        pitch = pu.get("pitch", 0)
        pz = pu.get("z", 0.15)
        approach_z = pz + 0.15
        target_name = cmd.detected_labels[0] if cmd.detected_labels else ""

        self._arm.gripper_open()
        cmd.gripper_open_done = True
        ok = self._arm.move_to_pose(pu["x"], pu["y"], approach_z, pitch=pitch, duration=3.0,
                                    frame_cb=self._update_preview)
        if not ok:
            logger.warning("[micro] approach_z IK failed (%.3f, %.3f) — unreachable", pu["x"], pu["y"])
            return

        frame = self._arm_camera.read()
        if frame is None:
            logger.warning("[micro] arm cam read failed, using workspace position")
            return

        refined = self._classifier.classify(frame)
        self._update_preview()

        best = None
        for obj in refined.detected_objects:
            if obj.name == target_name:
                best = obj
                break
        if best is None and refined.detected_objects:
            best = min(refined.detected_objects,
                       key=lambda o: abs(o.left + o.width / 2 - 0.5)
                                    + abs(o.top + o.height / 2 - 0.5))
            logger.warning("[micro] target '%s' not found, using closest '%s'",
                           target_name, best.name)
        if best is None:
            logger.warning("[micro] no objects in arm cam")
            return

        hfov_rad = np.deg2rad(self._arm_cam_fov)
        ox = best.left + best.width / 2
        oy = best.top + best.height / 2
        ih, iw = frame.image.shape[:2]
        vfov_rad = hfov_rad * ih / iw
        cam_h = approach_z - pz
        if cam_h <= 0.01:
            cam_h = 0.10
        scale_x = cam_h * 2 * np.tan(hfov_rad / 2)
        scale_y = cam_h * 2 * np.tan(vfov_rad / 2)
        dx = (ox - 0.5) * scale_x
        dy = (oy - 0.5) * scale_y
        damping = 0.3
        dx *= damping
        dy *= damping

        obj_w_m = best.width * scale_x
        pu["inset"] = min(max(obj_w_m * 0.4, 0.01), 0.08)

        logger.info("[micro] obj=(%.3f,%.3f) corr=(%.3f,%.3f)m damp=%.1f size=%.3fm inset=%.3f",
                    ox, oy, dx, dy, damping, obj_w_m, pu["inset"])

        if abs(dx) > 0.002 or abs(dy) > 0.002:
            pu["x"] += dx
            pu["y"] += dy
            pu["x"] = max(0.18, min(0.60, pu["x"]))
            pu["y"] = max(-0.28, min(0.28, pu["y"]))
            self._arm.move_to_pose(pu["x"], pu["y"], approach_z, pitch=pitch,
                                   duration=1.5, frame_cb=self._update_preview)

    def _exec_and_telemetry(self, cmd: TriageCommand) -> None:
        if self._arm:
            if not self._visual_servo(cmd):
                logger.warning("[exec] abort — unreachable position")
                return
            self._micro_adjust(cmd)
            if not self._arm.execute_triage(cmd, frame_cb=self._update_preview):
                logger.warning("[exec] triage failed")
        if self._telemetry:
            drop = cmd.drop_joints or cmd.drop_pose
            self._telemetry.publish(TelemetryEvent(event_type="classification", payload={
                "action": cmd.action, "target_bin": cmd.target_bin,
                "frame": self._frame_count, "pickup": cmd.pickup_pose,
                "drop": drop, "grade": cmd.label,
            }))

    def _confirm(self, cmd: TriageCommand) -> bool:
        if not cmd.detected_objects:
            logger.warning("[confirm] no detected objects")
            return False
        for _ in range(300):
            frame = self._camera.read()
            if frame is None:
                continue
            idx = self._selected_idx
            if idx < 0 or idx >= len(cmd.detected_objects):
                idx = 0
            obj = cmd.detected_objects[idx]
            name = obj.name
            display = frame.image.copy()
            self._draw_workspace(display)
            self._draw_boxes(display, cmd.detected_objects, highlight=idx)
            bin_name = BIN_NAMES.get(cmd.target_bin, cmd.target_bin)
            self._bottom_bar(display, f"[{idx}] {name}  ->  {bin_name} — Click object, Y=pick N=skip", GREEN)
            if self._arm_camera is not None and self._arm_frame_counter % 10 == 0:
                self._last_arm_frame = self._arm_camera.read()
            self._show_both(display, self._last_arm_frame)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('y'):
                cx = (obj.left + obj.width / 2)
                cy = (obj.top + obj.height / 2)
                mo = self._classifier._map_offset_x if hasattr(self._classifier, '_map_offset_x') else 0.15
                ms = self._classifier._map_scale_x if hasattr(self._classifier, '_map_scale_x') else 0.50
                mox = self._classifier._map_offset_y if hasattr(self._classifier, '_map_offset_y') else -0.20
                msy = self._classifier._map_scale_y if hasattr(self._classifier, '_map_scale_y') else 0.40
                pz = self._classifier._pickup_z if hasattr(self._classifier, '_pickup_z') else 0.15
                pt = self._classifier._pitch if hasattr(self._classifier, '_pitch') else 0.2
                cmd.pickup_pose = {"x": mo + cx * ms, "y": mox + cy * msy,
                                   "z": pz, "roll": 0, "pitch": pt, "yaw": 0}
                cmd.pickup_pose["x"] = max(0.18, min(0.60, cmd.pickup_pose["x"]))
                cmd.pickup_pose["y"] = max(-0.28, min(0.28, cmd.pickup_pose["y"]))
                logger.info("[confirm] pick %s at (%.3f, %.3f, %.3f)",
                            obj.name, cmd.pickup_pose["x"], cmd.pickup_pose["y"], pz)
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

        if self._arm_camera is not None:
            for _ in range(5):
                self._last_arm_frame = self._arm_camera.read()
                if self._last_arm_frame is not None:
                    break
                time.sleep(0.2)
        if self._show_preview:
            for i in range(30, 0, -1):
                frame = self._camera.read()
                if frame is None:
                    continue
                display = frame.image.copy()
                self._draw_workspace(display)
                self._bottom_bar(display, f"Clear workspace... {i}", (0, 255, 255))
                self._show_both(display, self._last_arm_frame)
                cv2.waitKey(1)
                time.sleep(0.05)
            cv2.waitKey(500)

        self._classifier.set_background(frame)
        self._bg_frame = frame
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
                self._arm_frame_counter += 1
                if self._arm_camera is not None and self._arm_frame_counter % 10 == 0:
                    self._last_arm_frame = self._arm_camera.read()
                self._show_both(display, self._last_arm_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt()

            if cmd is not None and cmd.action == "pick_and_place" and cmd.detected_labels and cmd.detected_objects:
                if self._confirm(cmd):
                    self._exec_and_telemetry(cmd)
                    logger.info("Picked.\n")
                else:
                    logger.info("Skipped.\n")
