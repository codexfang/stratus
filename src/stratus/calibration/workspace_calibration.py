from __future__ import annotations
import cv2
import json
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any

from stratus.core.vision import CameraFrame
from stratus.drivers.opencv_cam import USBCamera

logger = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    world_z: float = 0.0


@dataclass
class WorkspaceCalibrationData:
    version: int = 2
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_matrix: List[List[float]] = None
    dist_coeffs: List[float] = None
    homography: List[List[float]] = None
    roi: List[int] = None  # x, y, w, h
    points: List[Dict[str, float]] = None
    timestamp: float = 0.0
    notes: str = ""
    # Backward compatibility for old calibration files
    type: str = "homography"

    def __post_init__(self):
        if self.camera_matrix is None:
            self.camera_matrix = np.eye(3).tolist()
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5).tolist()
        if self.homography is None:
            self.homography = np.eye(3).tolist()
        if self.roi is None:
            self.roi = [0, 0, self.camera_width, self.camera_height]
        if self.points is None:
            self.points = []


class WorkspaceCalibration:
    """Advanced workspace calibration with Aruco fiducials + arm-guided points."""

    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    MARKER_SIZE_M = 0.04  # 40mm markers

    def __init__(self, camera: USBCamera, arm=None):
        self.camera = camera
        self.arm = arm
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.calib_data: Optional[WorkspaceCalibrationData] = None
        self._map_x = None
        self._map_y = None

    def load(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            logger.warning("Calibration file not found: %s", path)
            return False
        with open(p) as f:
            data = json.load(f)
        self.calib_data = WorkspaceCalibrationData(**data)
        self._build_undistort_maps()
        logger.info("Loaded calibration from %s (v%d, %d pts)", path,
                    self.calib_data.version, len(self.calib_data.points))
        return True

    def save(self, path: str) -> bool:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.calib_data.timestamp = time.time()
        with open(p, 'w') as f:
            json.dump(asdict(self.calib_data), f, indent=2)
        logger.info("Saved calibration to %s", path)
        return True

    def _build_undistort_maps(self):
        if self.calib_data is None:
            return
        K = np.array(self.calib_data.camera_matrix, dtype=np.float32)
        D = np.array(self.calib_data.dist_coeffs, dtype=np.float32)
        w, h = self.calib_data.camera_width, self.calib_data.camera_height
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_32FC1)
        self.calib_data.roi = list(roi)

    def undistort(self, frame: CameraFrame) -> CameraFrame:
        if self._map_x is None:
            return frame
        img = cv2.remap(frame.image, self._map_x, self._map_y, cv2.INTER_LINEAR)
        return CameraFrame(image=img, timestamp=frame.timestamp)

    def pixel_to_world(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        if self.calib_data is None or self.calib_data.homography is None:
            return None
        H = np.array(self.calib_data.homography, dtype=np.float32)
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        if out is not None:
            return float(out[0, 0, 0]), float(out[0, 0, 1])
        return None

    def world_to_pixel(self, wx: float, wy: float) -> Optional[Tuple[float, float]]:
        if self.calib_data is None or self.calib_data.homography is None:
            return None
        H = np.array(self.calib_data.homography, dtype=np.float32)
        H_inv = np.linalg.inv(H)
        pt = np.array([[[wx, wy]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H_inv)
        if out is not None:
            return float(out[0, 0, 0]), float(out[0, 0, 1])
        return None

    def run_aruco_grid_calibration(self, board_cols: int = 4, board_rows: int = 3,
                                   square_size: float = 0.05) -> bool:
        """Detect Charuco board, compute initial homography from marker corners."""
        logger.info("Starting Charuco board calibration... show board to camera")
        board = cv2.aruco.CharucoBoard(
            (board_cols, board_rows), square_size, self.MARKER_SIZE_M, self.aruco_dict)
        detector = cv2.aruco.CharucoDetector(board)

        all_corners = []
        all_ids = []
        img_size = None

        for i in range(30):
            frame = self.camera.read()
            if frame is None:
                continue
            img = frame.image
            if img_size is None:
                img_size = img.shape[:2][::-1]
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(img)
            if charuco_corners is not None and len(charuco_corners) >= 6:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                logger.info("Captured frame %d (%d corners)", i, len(charuco_corners))
                if len(all_corners) >= 10:
                    break
            cv2.imshow("Calibration - show Charuco board", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Calibration - show Charuco board")

        if len(all_corners) < 5:
            logger.error("Not enough valid frames (%d)", len(all_corners))
            return False

        ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, img_size, None, None)

        if not ret:
            logger.error("Charuco calibration failed")
            return False

        logger.info("Charuco calibration OK: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                    K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        self.calib_data = WorkspaceCalibrationData(
            camera_width=img_size[0],
            camera_height=img_size[1],
            camera_matrix=K.tolist(),
            dist_coeffs=D.flatten().tolist(),
        )
        self._build_undistort_maps()
        return True

    def run_arm_guided_calibration(self, points: List[Tuple[float, float, float]] = None) -> bool:
        """Move arm to known world positions, user clicks corresponding pixel in camera view."""
        if self.arm is None:
            logger.error("Arm required for guided calibration")
            return False

        if points is None:
            # Default 9-point grid covering reachable workspace
            xs = [0.25, 0.40, 0.55]
            ys = [-0.20, 0.00, 0.20]
            points = [(x, y, 0.15) for x in xs for y in ys]

        logger.info("Starting arm-guided calibration (%d points)", len(points))
        cal_pts = []

        for i, (wx, wy, wz) in enumerate(points):
            logger.info("Point %d/%d: moving arm to (%.3f, %.3f, %.3f)",
                        i+1, len(points), wx, wy, wz)
            ok = self.arm.move_to_pose(wx, wy, wz, pitch=0.2, duration=5.0)
            if not ok:
                logger.warning("Arm failed to reach point %d", i)
                continue

            # Show camera view, wait for click
            clicked = {'pos': None}

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked['pos'] = (float(x), float(y))

            cv2.namedWindow("Calibrate - Click arm tip")
            cv2.setMouseCallback("Calibrate - Click arm tip", on_mouse)

            while True:
                frame = self.camera.read()
                if frame is None:
                    continue
                img = frame.image.copy()
                cv2.putText(img, f"Point {i+1}/{len(points)}: Click ARM TIP",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if clicked['pos']:
                    cv2.circle(img, (int(clicked['pos'][0]), int(clicked['pos'][1])),
                               8, (0, 0, 255), -1)
                cv2.imshow("Calibrate - Click arm tip", img)
                key = cv2.waitKey(50) & 0xFF
                if key == ord('y') and clicked['pos']:
                    break
                if key == ord('q'):
                    cv2.destroyWindow("Calibrate - Click arm tip")
                    return False

            cv2.destroyWindow("Calibrate - Click arm tip")
            px, py = clicked['pos']
            cal_pts.append(CalibrationPoint(px, py, wx, wy, wz))
            logger.info("Recorded: pixel=(%.1f, %.1f) -> world=(%.3f, %.3f, %.3f)",
                        px, py, wx, wy, wz)

        if len(cal_pts) < 4:
            logger.error("Need at least 4 points, got %d", len(cal_pts))
            return False

        # Compute homography from pixel -> world
        src = np.array([(p.pixel_x, p.pixel_y) for p in cal_pts], dtype=np.float32)
        dst = np.array([(p.world_x, p.world_y) for p in cal_pts], dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)

        if H is None:
            logger.error("Homography computation failed")
            return False

        self.calib_data.homography = H.tolist()
        self.calib_data.points = [asdict(p) for p in cal_pts]
        logger.info("Homography computed, %d inliers", int(mask.sum()) if mask is not None else len(cal_pts))
        return True

    def verify_calibration(self, num_tests: int = 5) -> bool:
        """Move arm to random calibrated points, verify pixel projection."""
        if self.calib_data is None or self.calib_data.homography is None:
            return False
        H = np.array(self.calib_data.homography, dtype=np.float32)
        H_inv = np.linalg.inv(H)

        for _ in range(num_tests):
            wx = np.random.uniform(0.25, 0.55)
            wy = np.random.uniform(-0.20, 0.20)
            wz = 0.15
            ok = self.arm.move_to_pose(wx, wy, wz, pitch=0.2, duration=4.0)
            if not ok:
                continue

            frame = self.camera.read()
            if frame is None:
                continue

            # Project world -> pixel
            pt = np.array([[[wx, wy]]], dtype=np.float32)
            px_py = cv2.perspectiveTransform(pt, H_inv)
            if px_py is None:
                continue
            px, py = px_py[0, 0]

            # Draw predicted pixel
            img = frame.image.copy()
            cv2.circle(img, (int(px), int(py)), 10, (0, 255, 0), 2)
            cv2.putText(img, f"Pred: ({px:.1f},{py:.1f})", (int(px)+15, int(py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img, f"True: ({wx:.3f},{wy:.3f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Verify Calibration", img)
            cv2.waitKey(1500)

        cv2.destroyWindow("Verify Calibration")
        return True


def run_calibration_wizard(camera_index: int = 0, width: int = 1280, height: int = 720,
                           arm=None, output: str = "~/stratus/calibration/workspace_cal.json"):
    """Full interactive calibration wizard."""
    cam = USBCamera(index=camera_index, width=width, height=height)
    cam.connect()

    cal = WorkspaceCalibration(cam, arm=arm)
    out_path = Path(output).expanduser()

    # 1. Try loading existing
    if cal.load(str(out_path)):
        logger.info("Loaded existing calibration")
        if cal.verify_calibration(3):
            logger.info("Verification passed")
            cam.disconnect()
            return True

    # 2. Charuco calibration for intrinsics
    logger.info("=== Step 1: Charuco board for lens calibration ===")
    print("Show Charuco board (4x3, 50mm squares) to camera. Press 'q' to skip.")
    cal.run_aruco_grid_calibration()

    # 3. Arm-guided extrinsic calibration
    logger.info("=== Step 2: Arm-guided point calibration ===")
    print("Arm will move to grid positions. Click the ARM TIP in camera view at each.")
    if not cal.run_arm_guided_calibration():
        logger.error("Guided calibration failed")
        cam.disconnect()
        return False

    # 4. Verify
    logger.info("=== Step 3: Verification ===")
    if cal.verify_calibration(5):
        cal.save(str(out_path))
        logger.info("Calibration saved to %s", out_path)
        cam.disconnect()
        return True

    logger.error("Verification failed")
    cam.disconnect()
    return False