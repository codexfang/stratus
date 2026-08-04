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
GRAY  = (150, 150, 150)
CYAN  = (0, 255, 255)
BIN_NAMES = {"bin_a": "Bin A", "bin_b": "Bin B", "bin_c": "Bin C"}

# Z heights used throughout the pick sequence
HOVER_Z  = 0.28   # arm hovers here above the object before descending
PICKUP_Z = 0.10   # final grip height (top of object surface)

# How many consecutive frames must agree on the same object class before
# the pipeline treats it as a real detection and prompts the user to confirm.
STABLE_FRAMES_REQUIRED = 3


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
        map_offset_x: float = 0.15,
        map_scale_x: float = 0.50,
        map_offset_y: float = -0.20,
        map_scale_y: float = 0.40,
        scan_joints: list[float] | None = None,
    ):
        self._arm = arm
        self._camera = camera
        self._arm_camera = arm_camera
        self._arm_cam_fov = arm_cam_fov
        self._classifier = classifier
        self._telemetry = telemetry
        self._classify_every = classify_every
        self._show_preview = show_preview
        self._map_off_x = map_offset_x
        self._map_scl_x = map_scale_x
        self._map_off_y = map_offset_y
        self._map_scl_y = map_scale_y
        # Custom scan-pose joint angles (radians).  None → arm driver default.
        self._scan_joints = scan_joints
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
        self._display_scale = 1.0
        # Stable-detection state
        self._stable_label: str = ""
        self._stable_count: int = 0
        self._stable_cmd: TriageCommand | None = None
        cv2.namedWindow("Stratus")
        cv2.setMouseCallback("Stratus", self._on_mouse)

    # ──────────────────────────────────────────────────────────────
    # Preview helpers
    # ──────────────────────────────────────────────────────────────

    def _update_preview(self) -> None:
        """Refresh the display window. Called as frame_cb during arm movement."""
        self._update_preview_counter += 1
        frame = self._camera.read()
        if frame is None:
            return
        display = frame.image.copy()
        self._last_h, self._last_w = display.shape[:2]
        self._draw_workspace(display)
        if self._arm_camera is not None:
            self._last_arm_frame = self._arm_camera.read()
        self._show_both(display, self._last_arm_frame)
        cv2.waitKey(1)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        h, w = self._last_h, self._last_w
        if h == 0 or w == 0:
            return
        x_img = int(x / self._display_scale)
        y_img = int(y / self._display_scale)
        if x_img >= w:
            return
        self._selected_idx = -1
        for i, obj in enumerate(self._current_objects):
            x1 = int(obj.left * w)
            y1 = int(obj.top * h)
            x2 = int((obj.left + obj.width) * w)
            y2 = int((obj.top + obj.height) * h)
            if x1 <= x_img <= x2 and y1 <= y_img <= y2:
                self._selected_idx = i
                logger.info("Selected object %d: %s", i, obj.name)
                break

    def _draw_boxes(self, display: np.ndarray, objects: list[DetectedObject],
                    highlight: int = -1) -> None:
        h, w = display.shape[:2]
        class_colors = {
            "cup, coffee cup, mug, glass": (0, 140, 255),
            "book": (255, 100, 0),
            "phone": (255, 255, 0),
            "cell phone": (255, 255, 0),
            "laptop": (255, 0, 255),
            "mouse, computer mouse": (0, 255, 100),
            "keyboard": (255, 200, 0),
            "remote": (200, 0, 200),
            "scissors": (255, 150, 0),
            "pen": (0, 255, 150),
            "pencil": (0, 255, 150),
            "bottle, water bottle": (0, 100, 255),
            "watch": (255, 200, 0),
            "cup": (0, 140, 255),
            "marker": (100, 100, 255),
        }
        default_color = (0, 255, 0)

        for i, obj in enumerate(objects):
            x1 = int(obj.left * w)
            y1 = int(obj.top * h)
            x2 = int((obj.left + obj.width) * w)
            y2 = int((obj.top + obj.height) * h)

            color = class_colors.get(obj.name.lower(), default_color)
            if i == highlight:
                color = (0, 255, 255)
                thickness = 3
            else:
                thickness = 2

            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            label = f"{obj.name} {obj.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if i == highlight:
                cv2.putText(display, f"[{i}] SELECTED", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _draw_workspace(self, display: np.ndarray) -> None:
        h, w = display.shape[:2]
        mx, my = int(w * 0.35), int(h * 0.35)
        cv2.rectangle(display, (mx, my), (w - mx, h - my), (60, 60, 60), 1)

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
            self._display_scale = scale
        else:
            self._display_scale = 1.0
        cv2.imshow("Stratus", combined)

    # ──────────────────────────────────────────────────────────────
    # Scan pose
    # ──────────────────────────────────────────────────────────────

    def _go_to_scan_pose(self) -> None:
        """Move arm to scan pose so the top-mounted camera sees the full
        workspace.  No-op when running without a physical arm."""
        if self._arm is None:
            return
        # move_to_scan_pose is defined on VectorBH6ArmDriver; guard with
        # hasattr so the Protocol stub (no-arm mode) doesn't raise.
        if not hasattr(self._arm, "move_to_scan_pose"):
            return
        self._arm.move_to_scan_pose(
            scan_joints=self._scan_joints,
            duration=4.0,
            frame_cb=self._update_preview,
        )

    # ──────────────────────────────────────────────────────────────
    # Classification
    # ──────────────────────────────────────────────────────────────

    def _classify(self, frame) -> TriageCommand | None:
        """Run the classifier and return a command only once the same object
        class has been detected for STABLE_FRAMES_REQUIRED consecutive frames.
        This prevents single-frame noise from triggering a pick."""
        cmd = self._classifier.classify(frame)
        self._current_objects = cmd.detected_objects

        if cmd.action != "pick_and_place" or not cmd.detected_labels:
            # Reset stability counter on miss
            self._stable_label = ""
            self._stable_count = 0
            self._stable_cmd = None
            return cmd   # still return so preview can show "scanning..."

        top_label = cmd.detected_labels[0]
        if top_label == self._stable_label:
            self._stable_count += 1
        else:
            self._stable_label = top_label
            self._stable_count = 1
            self._stable_cmd = cmd

        # Always update the stored command with latest bounding-box position
        self._stable_cmd = cmd
        self._selected_idx = 0

        if self._stable_count >= STABLE_FRAMES_REQUIRED:
            logger.info("[classify] stable detection (%d frames): %s -> %s",
                        self._stable_count, top_label, cmd.target_bin)
            return cmd

        logger.info("[classify] unstable (%d/%d): %s",
                    self._stable_count, STABLE_FRAMES_REQUIRED, top_label)
        # Return the cmd so bounding boxes appear in preview, but mark as
        # not-yet-stable so the main loop won't prompt for a pick yet.
        cmd.action = "detecting"
        return cmd

    # ──────────────────────────────────────────────────────────────
    # Visual servoing helpers (arm-camera only, optional)
    # ──────────────────────────────────────────────────────────────

    def _servo_center_armcam(self, cmd: TriageCommand) -> bool:
        """One-shot centering using arm-mounted camera + object detector.
        Probes two joints to learn the correct correction direction, applies
        a single clamped correction, then marks cmd.pickup_refined = True."""
        if not self._arm or not self._arm_camera or not self._arm_camera.is_connected:
            cmd.pickup_refined = True
            return True

        obj_name = cmd.detected_labels[0] if cmd.detected_labels else ""
        GAIN = 0.2

        def _detect_arm() -> tuple[float, float] | None:
            frame = self._arm_camera.read()
            if frame is None:
                return None
            det = self._classifier.classify(frame)
            for obj in det.detected_objects:
                if obj.name == obj_name:
                    return (obj.left + obj.width / 2, obj.top + obj.height / 2)
            return None

        q0, _, _ = self._arm.get_state()

        pos0 = _detect_arm()
        if pos0 is None:
            logger.info("[servo] object not visible in arm camera — skipping centering")
            cmd.pickup_refined = True
            return True

        acx0, acy0 = pos0
        logger.info("[servo] object at arm-cam (%.3f, %.3f)", acx0, acy0)

        # Probe j1 to get correction sign
        qj1 = q0.copy()
        qj1[0] += 0.015
        self._arm.move_to_joints(qj1, duration=0.8, frame_cb=self._update_preview)
        time.sleep(0.3)
        pos1 = _detect_arm()
        self._arm.move_to_joints(q0, duration=0.8, frame_cb=self._update_preview)
        time.sleep(0.3)

        de_dj1 = (pos1[0] - acx0) if pos1 is not None else -1.0

        # Probe j2 to get correction sign
        qj2 = q0.copy()
        qj2[1] += 0.015
        self._arm.move_to_joints(qj2, duration=0.8, frame_cb=self._update_preview)
        time.sleep(0.3)
        pos2 = _detect_arm()
        self._arm.move_to_joints(q0, duration=0.8, frame_cb=self._update_preview)
        time.sleep(0.3)

        de_dj2 = (pos2[1] - acy0) if pos2 is not None else -1.0

        corr_j1 = (acx0 - 0.5) * GAIN * (-1 if de_dj1 > 0 else 1)
        corr_j2 = (acy0 - 0.5) * GAIN * (-1 if de_dj2 > 0 else 1)
        corr_j1 = float(np.clip(corr_j1, -0.15, 0.15))
        corr_j2 = float(np.clip(corr_j2, -0.15, 0.15))

        if abs(corr_j1) < 0.005 and abs(corr_j2) < 0.005:
            logger.info("[servo] already centered — no correction needed")
            cmd.pickup_refined = True
            return True

        q_corr = q0.copy()
        q_corr[0] += corr_j1
        q_corr[1] += corr_j2
        logger.info("[servo] correcting j1=%+.4f j2=%+.4f", corr_j1, corr_j2)
        self._arm.move_to_joints(q_corr, duration=2.0, frame_cb=self._update_preview)

        pu = cmd.pickup_pose
        pu["x"] = pu.get("x", 0.25) + corr_j1 * 0.15
        pu["y"] = pu.get("y", 0.0) + corr_j2 * 0.15

        cmd.pickup_refined = True
        return True

    # ──────────────────────────────────────────────────────────────
    # Main pick sequence — called once per confirmed pick
    # ──────────────────────────────────────────────────────────────

    def _approach_and_pick(self, cmd: TriageCommand) -> bool:
        """
        The complete pick sequence in one place:

            1. Move above object (HOVER_Z) with gripper OPEN
            2. Optional arm-camera visual servo correction
            3. Hand off to execute_triage which descends → grips → lifts → drops
        """
        if not self._arm:
            return False

        pu = cmd.pickup_pose
        px = pu.get("x", 0.25)
        py = pu.get("y", 0.0)
        pitch = pu.get("pitch", 0.4)
        hover_z = HOVER_Z

        logger.info("[approach] moving above object (%.3f, %.3f, z=%.3f pitch=%.2f)",
                    px, py, hover_z, pitch)

        # ── Step 1: Move to hover position above the object ──────────────
        ik_ok = self._arm.move_to_pose(px, py, hover_z,
                                       roll=0, pitch=pitch, yaw=0,
                                       duration=5.0, frame_cb=self._update_preview)
        if not ik_ok:
            logger.warning("[approach] IK failed for hover — trying joint-space fallback")
            q_approx = self._xy_to_joints(px, py)
            self._arm.move_to_joints(q_approx, duration=4.0, frame_cb=self._update_preview)

        # ── Step 2: Open gripper NOW (arm is above object) ────────────────
        logger.info("[approach] opening gripper above object")
        self._arm.gripper_open()

        # ── Step 3: Optional arm-camera servo correction ──────────────────
        if self._arm_camera is not None and self._arm_camera.is_connected:
            self._servo_center_armcam(cmd)

        # Refresh pickup_pose z to our standard value so execute_triage
        # knows exactly where to descend to
        pu["z"] = PICKUP_Z
        pu["pitch"] = pitch

        # ── Step 4: execute_triage handles descend → grip → lift → drop ──
        return self._arm.execute_triage(cmd, frame_cb=self._update_preview)

    def _xy_to_joints(self, x: float, y: float) -> np.ndarray:
        """Rough joint-space fallback when IK is unavailable."""
        j1 = float(np.clip((x - 0.25) / 0.1, -0.6, 0.6))
        return np.array([j1, 0.3, 0.8, 0.0, 1.0, 0.0])

    # ──────────────────────────────────────────────────────────────
    # Confirm dialog
    # ──────────────────────────────────────────────────────────────

    def _confirm(self, cmd: TriageCommand) -> bool:
        """Show preview loop, wait for Y/N key press. Returns True if user confirms pick."""
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
            display = frame.image.copy()
            self._draw_workspace(display)
            self._draw_boxes(display, cmd.detected_objects, highlight=idx)
            bin_name = BIN_NAMES.get(cmd.target_bin, cmd.target_bin)
            self._bottom_bar(display,
                             f"[{idx}] {obj.name}  ->  {bin_name} — Click object, Y=pick N=skip",
                             GREEN)
            if self._arm_camera is not None and self._arm_frame_counter % 10 == 0:
                self._last_arm_frame = self._arm_camera.read()
            self._show_both(display, self._last_arm_frame)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('y'):
                cx = obj.left + obj.width / 2   # normalized 0-1
                cy = obj.top + obj.height / 2

                # Map normalized pixel center → arm workspace coordinates
                map_x = self._map_off_x + cx * self._map_scl_x
                map_y = self._map_off_y + cy * self._map_scl_y

                h, w = display.shape[:2]
                logger.info("[confirm] pick %s at pixel (%.1f, %.1f) -> world (%.3f, %.3f)",
                            obj.name, cx * w, cy * h, map_x, map_y)

                # Overwrite pickup_pose with confirmed coordinates; keep pitch from classifier
                pitch = cmd.pickup_pose.get("pitch", 0.4) if cmd.pickup_pose else 0.4
                cmd.pickup_pose = {"x": map_x, "y": map_y, "z": PICKUP_Z, "pitch": pitch}
                cmd.detected_labels = [obj.name]
                cmd.detected_objects = [obj]
                return True

            if key == ord('n'):
                return False
            if key == ord('q'):
                raise KeyboardInterrupt()

        return False

    # ──────────────────────────────────────────────────────────────
    # Execution + telemetry
    # ──────────────────────────────────────────────────────────────

    def _exec_and_telemetry(self, cmd: TriageCommand) -> None:
        """Run the pick sequence, log telemetry, then return to scan pose."""
        success = False
        if self._arm:
            success = self._approach_and_pick(cmd)
            if not success:
                logger.warning("[exec] pick sequence failed")

        # Reset stability so next scan starts fresh
        self._stable_label = ""
        self._stable_count = 0
        self._stable_cmd = None

        if self._telemetry:
            drop = cmd.drop_joints or cmd.drop_pose
            self._telemetry.publish(TelemetryEvent(event_type="classification", payload={
                "action": cmd.action,
                "target_bin": cmd.target_bin,
                "frame": self._frame_count,
                "pickup": cmd.pickup_pose,
                "drop": drop,
                "grade": cmd.label,
                "success": success,
            }))

    # ──────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────

    def run_loop(self) -> None:
        logger.info("Stratus running. Y=pick, N=skip, Q=quit.\n")

        # ── Warm up camera ────────────────────────────────────────────────
        frame = None
        for _ in range(20):
            frame = self._camera.read()
            if frame is not None:
                break
            time.sleep(0.2)
        if frame is None:
            logger.error("No camera frame — check camera connection")
            return

        if self._arm_camera is not None:
            for _ in range(5):
                self._last_arm_frame = self._arm_camera.read()
                if self._last_arm_frame is not None:
                    break
                time.sleep(0.2)

        # ── Move arm to scan pose so camera sees full workspace ───────────
        # Do this before the background countdown so the background frame
        # is captured with the arm already in scanning position.
        if self._arm is not None:
            logger.info("Moving arm to scan pose...")
            self._go_to_scan_pose()
            # Brief settle — let camera exposure stabilise after arm stops
            time.sleep(0.5)

        # ── Countdown + background capture ───────────────────────────────
        if self._show_preview:
            for i in range(30, 0, -1):
                frame = self._camera.read()
                if frame is None:
                    continue
                display = frame.image.copy()
                self._draw_workspace(display)
                self._bottom_bar(display, f"Clear workspace… {i}", CYAN)
                self._show_both(display, self._last_arm_frame)
                cv2.waitKey(1)
                time.sleep(0.05)
            cv2.waitKey(500)

        # Grab the background frame with arm in scan pose
        frame = self._camera.read() or frame
        self._classifier.set_background(frame)
        self._bg_frame = frame
        self._bg_captured = True
        logger.info("Background captured (arm in scan pose)")

        # ── Main detection + pick loop ────────────────────────────────────
        while True:
            frame = self._camera.read()
            if frame is None:
                continue

            self._frame_count += 1
            self._last_h, self._last_w = frame.image.shape[:2]

            cmd = None
            if self._frame_count % self._classify_every == 0:
                cmd = self._classify(frame)

            if self._show_preview:
                display = frame.image.copy()
                self._draw_workspace(display)
                if self._arm_camera is not None:
                    self._last_arm_frame = self._arm_camera.read()

                if cmd and cmd.detected_objects:
                    self._draw_boxes(display, cmd.detected_objects)

                if cmd and cmd.detected_labels:
                    top = cmd.detected_labels[0]
                    bin_name = BIN_NAMES.get(cmd.target_bin, cmd.target_bin)
                    if cmd.action == "pick_and_place":
                        self._bottom_bar(display, f"{top}  ->  {bin_name}", GREEN)
                    elif cmd.action == "detecting":
                        self._bottom_bar(
                            display,
                            f"detecting: {top} ({self._stable_count}/{STABLE_FRAMES_REQUIRED})",
                            CYAN,
                        )
                    else:
                        self._bottom_bar(display, "scanning…", GRAY)
                else:
                    if self._current_objects:
                        self._draw_boxes(display, self._current_objects)
                    self._bottom_bar(display, "scanning…", GRAY)

                self._show_both(display, self._last_arm_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt()

            # Only prompt for confirmed stable pick_and_place detections
            if (cmd is not None
                    and cmd.action == "pick_and_place"
                    and cmd.detected_labels
                    and cmd.detected_objects):
                if self._confirm(cmd):
                    self._exec_and_telemetry(cmd)
                    logger.info("Picked — returning to scan pose\n")
                    # Arm is already at scan pose (execute_triage ends there);
                    # re-capture background so new objects are detected cleanly.
                    time.sleep(0.8)
                    frame = self._camera.read() or frame
                    self._classifier.set_background(frame)
                    self._bg_frame = frame
                    logger.info("Background refreshed")
                else:
                    logger.info("Skipped.\n")
