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
        self._display_scale = 1.0  # Track display scaling for click coords
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
        # Convert click from displayed window coords to original image coords
        x_img = int(x / self._display_scale)
        y_img = int(y / self._display_scale)
        # Check if click is in workspace portion (left side of combined window)
        # Combined window has workspace (width w) + arm_cam (width h) = total width w + h
        if x_img >= w:  # Click in arm camera portion
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
        """Draw thin bounding box outlines with class labels and confidence.
        Click inside a box to select it for picking."""
        h, w = display.shape[:2]
        class_colors = {
            "cup, coffee cup, mug, glass": (0, 140, 255),   # orange
            "book": (255, 100, 0),                           # blue
            "phone": (255, 255, 0),                          # cyan
            "cell phone": (255, 255, 0),
            "laptop": (255, 0, 255),                         # magenta
            "mouse, computer mouse": (0, 255, 100),          # green
            "keyboard": (255, 200, 0),                       # teal
            "remote": (200, 0, 200),                         # purple
            "scissors": (255, 150, 0),                       # light blue
            "pen": (0, 255, 150),                            # spring green
            "pencil": (0, 255, 150),
            "bottle, water bottle": (0, 100, 255),           # orange-red
            "watch": (255, 200, 0),                          # teal
            "cup": (0, 140, 255),
            "marker": (100, 100, 255),                       # light blue
        }
        default_color = (0, 255, 0)  # green

        for i, obj in enumerate(objects):
            x1 = int(obj.left * w)
            y1 = int(obj.top * h)
            x2 = int((obj.left + obj.width) * w)
            y2 = int((obj.top + obj.height) * h)

            color = class_colors.get(obj.name.lower(), default_color)
            if i == highlight:
                color = (0, 255, 255)  # yellow highlight
                thickness = 3
            else:
                thickness = 2

            # Thin rectangular outline (no fill)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            # Label with confidence
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
            self._display_scale = scale
        else:
            self._display_scale = 1.0
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
        """Move to home joint position (known safe), then let _micro_adjust do joint-space visual servoing."""
        if not self._arm or not self._arm_camera or not self._arm_camera.is_connected:
            logger.error("[servo] arm camera not available")
            return False
            
        # Move to home/zero joint position (known safe - all joints at 0)
        logger.info("[servo] moving to home joint position")
        home_joints = np.zeros(6, dtype=np.float64)
        self._arm.stop_control_loop()
        self._slew_mit(home_joints, duration=5.0, frame_cb=self._update_preview)
        self._arm.start_control_loop(self._endpos._loop_cb, rate=10)
        
        cmd.pickup_refined = True
        # Start from home joint position
        cmd.pickup_pose = {"x": 0.25, "y": 0.0, "z": 0.15, "roll": 0, "pitch": 0, "yaw": 0}
        return True

    def _micro_adjust(self, cmd: TriageCommand) -> None:
        """Pure visual servoing: start at search position, scan with arm camera,
        find object, center it, then descend and pick."""
        if not self._arm or not self._arm_camera or not self._arm_camera.is_connected:
            logger.warning("[micro] arm camera not available")
            return
        
        pu = cmd.pickup_pose
        pitch = pu.get("pitch", 0)
        pz = pu.get("z", 0.15)
        approach_z = pz + 0.15
        target_name = cmd.detected_labels[0] if cmd.detected_labels else ""
        
        # Ensure gripper is open
        self._arm.gripper_open()
        cmd.gripper_open_done = True
        
        # Start at known safe search position
        search_x, search_y, search_z = 0.25, 0.0, 0.35
        
        # Move to search position (should already be there from _visual_servo)
        ok = self._arm.move_to_pose(search_x, search_y, search_z, pitch=0.0, duration=3.0,
                                    frame_cb=self._update_preview)
        if not ok:
            logger.error("[micro] cannot reach search position")
            return
        
        # Scan in a grid pattern at search height to find object
        logger.info("[micro] starting visual search for '%s'", target_name)
        found = False
        
        # Search grid: 3x3 grid around search position
        grid_offsets = [
            (0.0, 0.0),      # center
            (0.05, 0.0),     # right
            (-0.05, 0.0),    # left
            (0.0, 0.05),     # forward
            (0.0, -0.05),    # back
            (0.04, 0.04),    # diagonals
            (-0.04, 0.04),
            (-0.04, -0.04),
            (0.04, -0.04),
        ]
        
        best = None
        found_at = None
        
        for dx, dy in grid_offsets:
            gx = 0.25 + dx
            gy = 0.0 + dy
            
            # Skip if out of reachable bounds
            if gx < 0.20 or gx > 0.35 or abs(gy) > 0.15:
                continue
                
            ok = self._arm.move_to_pose(gx, gy, 0.35, pitch=0.0, duration=2.0,
                                        frame_cb=self._update_preview)
            if not ok:
                continue
                
            frame = self._arm_camera.read()
            if frame is None:
                continue
                
            result = self._classifier.classify(frame)
            self._update_preview()
            
            # Check for target
            for obj in result.detected_objects:
                if obj.name == target_name:
                    logger.info("[micro] found '%s' at (%.3f, %.3f) in arm cam", target_name, 0.25 + dx, 0.0 + dy)
                    best = obj
                    found = True
                    break
            
            if best is not None:
                break
        
        if best is None:
            logger.error("[micro] target '%s' not found in search grid", target_name)
            return
        
        # Now visually servo to center the object
        logger.info("[micro] starting visual servo to center object")
        max_iters = 8
        
        for i in range(max_iters):
            frame = self._arm_camera.read()
            if frame is None:
                logger.warning("[micro] arm cam read failed")
                break
            
            result = self._classifier.classify(frame)
            self._update_preview()
            
            # Find target object
            best = None
            for obj in result.detected_objects:
                if obj.name == target_name:
                    best = obj
                    break
            if best is None and result.detected_objects:
                best = min(result.detected_objects,
                           key=lambda o: abs(o.left + o.width/2 - 0.5) + abs(o.top + o.height/2 - 0.5))
            
            if best is None:
                logger.warning("[micro] target lost during servo")
                break
            
            # Compute offset from image center
            cx = best.left + best.width / 2
            cy = best.top + best.height / 2
            dx_px = (cx - 0.5)
            dy_px = (cy - 0.5)
            
            # If well centered, break
            if abs(dx_px) < 0.015 and abs(dy_px) < 0.015:
                logger.info("[micro] centered in arm cam (dx=%.3f, dy=%.3f)", dx_px, dy_px)
                break
            
            # Convert pixel offset to world offset
            cam_height = 0.35 - 0.15  # search_z - pz = 0.20
            if cam_height < 0.02:
                cam_height = 0.20
            
            hfov = np.deg2rad(60.0)
            fh, fw = frame.image.shape[:2]
            vfov = hfov * fh / fw
            scale_x = cam_height * 2 * np.tan(hfov / 2)
            scale_y = cam_height * 2 * np.tan(vfov / 2)
            
            dx = dx_px * scale_x
            dy = dy_px * scale_y
            
            # Apply correction (camera coords to arm coords)
            pu["x"] -= dy
            pu["y"] -= dx
            
            # Clamp to reachable bounds
            pu["x"] = max(0.20, min(0.35, pu["x"]))
            pu["y"] = max(-0.15, min(0.15, pu["y"]))
            
            logger.info("[servo] iter %d: pixel_offset=(%.3f,%.3f) world_corr=(%.3f,%.3f) pos=(%.3f,%.3f)",
                        i, dx_px, dy_px, -dy, -dx, pu["x"], pu["y"])
            
            # Move to corrected position
            ok = self._arm.move_to_pose(pu["x"], pu["y"], 0.35, pitch=0.0, duration=1.5,
                                        frame_cb=self._update_preview)
            if not ok:
                logger.warning("[servo] correction IK failed")
                break
        
        # Now centered - descend to pickup height
        logger.info("[micro] descending to pickup height at (%.3f, %.3f)", pu["x"], pu["y"])
        ok = self._arm.move_to_pose(pu["x"], pu["y"], pz, pitch=0.0, duration=3.0,
                                    frame_cb=self._update_preview)
        if not ok:
            logger.warning("[micro] descent IK failed")
            return
        
        # Small forward move for grip
        ok = self._arm.move_to_pose(pu["x"] + 0.02, pu["y"], pz, pitch=0.0, duration=2.0,
                                    frame_cb=self._update_preview)
        if not ok:
            logger.warning("[micro] final approach failed")
            return
            
        logger.info("[micro] visual servo complete at (%.3f, %.3f, %.3f)", pu["x"], pu["y"], pz)
    
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

    def _try_pickup_pose(self, x: float, y: float, z: float, pitch: float) -> bool:
        """Try to move to a pickup pose using joint-space movement, return True if succeeds."""
        # Use joint-space movement from home position
        # This bypasses Cartesian IK entirely
        try:
            # Go to home first
            home_joints = np.zeros(6)
            self._arm.stop_control_loop()
            self._slew_mit(np.zeros(6), duration=2.0, frame_cb=self._update_preview)
            
            # Simple approach: move to pickup area using joint space
            # Joint 1: base rotation (x position)
            # Joint 2: shoulder (y position / height)
            # Joint 3: elbow (reach)
            # Joint 4: wrist roll
            # Joint 5: wrist pitch
            # Joint 6: wrist yaw
            
            # Map x to joint 1 (base rotation): x=0.25 -> joint 0, x=0.35 -> joint 1.0
            j1 = np.clip((x - 0.25) / 0.1, -0.5, 0.5)
            # Joint 2: shoulder - higher z = more negative (up), lower z = more positive (down)
            j2 = np.clip((0.3 - z) * 2.0, -0.5, 1.0)
            # Joint 3: elbow - compensates shoulder
            j3 = np.clip(-j2 * 0.8, -1.0, 0.5)
            
            q_target = np.array([j1, 0.5, -0.3, 0.0, 0.0, 0.0])
            self._arm.stop_control_loop()
            self._slew_mit(q_target, duration=3.0, frame_cb=self._update_preview)
            self._arm.start_control_loop(self._endpos._loop_cb, rate=10)
            
            # Check if we reached close to target
            q_actual, _, _ = self._arm.get_state()
            return np.linalg.norm(q_actual[:3] - q_target[:3]) < 0.5
        except Exception:
            return False

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
                h, w = display.shape[:2]
                
                # Don't use coordinate mapping - just use visual servoing from home
                logger.info("[confirm] pick %s at pixel (%.1f, %.1f) - using visual servoing", 
                            obj.name, cx * w, cy * h)
                
                # Start from home position, visual servo will find and center object
                cmd.pickup_pose = {"x": 0.25, "y": 0.0, "z": 0.15, "roll": 0, "pitch": 0, "yaw": 0}
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
                # Always read arm camera every frame for smooth preview
                if self._arm_camera is not None:
                    self._last_arm_frame = self._arm_camera.read()
                if cmd:
                    self._draw_boxes(display, cmd.detected_objects)
                    if cmd.detected_labels:
                        name = cmd.detected_labels[0]
                        bin_name = BIN_NAMES.get(cmd.target_bin, cmd.target_bin)
                        self._bottom_bar(display, f"{name}  ->  {bin_name}", GREEN)
                    else:
                        self._bottom_bar(display, "scanning...", GRAY)
                else:
                    # Persist last detection boxes between classifications
                    if self._current_objects:
                        self._draw_boxes(display, self._current_objects)
                    self._bottom_bar(display, "scanning...", GRAY)
                self._show_both(display, self._last_arm_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt()

            if cmd is not None and cmd.action == "pick_and_place" and cmd.detected_labels and cmd.detected_objects:
                if self._confirm(cmd):
                    self._exec_and_telemetry(cmd)
                    logger.info("Picked.\n")
                else:
                    logger.info("Skipped.\n")
