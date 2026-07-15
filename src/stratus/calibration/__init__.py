from __future__ import annotations
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

from stratus.drivers.opencv_cam import USBCamera

logger = logging.getLogger(__name__)


def run_calibration_wizard(camera_index: int = 0, width: int = 1280, height: int = 720,
                           arm=None, output: str = "") -> bool:
    """Interactive workspace camera calibration using ARM-GUIDED method.
    
    The arm moves to known world positions. At each position, you click the
    ARM TIP in the camera view. This builds a pixel->world homography.
    """
    if arm is None:
        logger.error("Arm required for guided calibration")
        return False

    from stratus.calibration.workspace_calibration import WorkspaceCalibration
    from stratus.drivers.opencv_cam import USBCamera

    cam = USBCamera(index=camera_index, width=width, height=height)
    cam.connect()

    cal = WorkspaceCalibration(cam, arm=arm)
    out_path = Path(output).expanduser() if output else Path.home() / "stratus/calibration/workspace_cal.json"

    # 1. Try loading existing
    if cal.load(str(out_path)):
        logger.info("Loaded existing calibration")
        if cal.verify_calibration(3):
            logger.info("Verification passed")
            cam.disconnect()
            return True

    # 2. Charuco calibration for lens intrinsics + undistortion
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


def load_calibration(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def apply_calibration(cal: dict, px: float, py: float) -> Tuple[float, float]:
    H = np.array(cal["matrix"], dtype=np.float32)
    pt = np.array([px, py, 1.0], dtype=np.float32)
    proj = H @ pt
    return (proj[0] / proj[2], proj[1] / proj[2])