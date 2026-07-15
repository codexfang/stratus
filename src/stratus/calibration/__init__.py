from __future__ import annotations
import json
import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def run_calibration_wizard(camera_index: int, width: int, height: int,
                           arm=None, output: str = "") -> bool:
    """Interactive workspace camera calibration.

    Steps:
    1. Capture background frame
    2. User places object at known positions, clicks to mark
    3. Solve homography / linear map from pixel -> world
    4. Save calibration JSON
    """
    import cv2
    import numpy as np
    from stratus.drivers.opencv_cam import USBCamera

    cam = USBCamera(index=camera_index, width=width, height=height)
    cam.connect()

    print("\n=== Workspace Camera Calibration ===")
    print("1. Clear the workspace completely")
    print("2. Press SPACE when ready for background capture")

    # Background capture
    for _ in range(10):
        bg = cam.read()
        if bg is not None:
            break
    cv2.waitKey(500)
    while True:
        frame = cam.read()
        if frame is None:
            continue
        disp = frame.image.copy()
        cv2.putText(disp, "Clear workspace, press SPACE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Calibrate", disp)
        if cv2.waitKey(50) & 0xFF == ord(' '):
            bg_frame = frame.image.copy()
            break
    print("Background captured")

    # Calibration points - user places object and clicks
    points = []  # list of (pixel_x, pixel_y, world_x, world_y)
    print("\nPlace object at known positions, click center, press ENTER when done")
    print("Suggested grid: (0.20, -0.20), (0.20, 0.20), (0.50, -0.20), (0.50, 0.20), (0.35, 0.00)")

    while True:
        frame = cam.read()
        if frame is None:
            continue
        diff = cv2.absdiff(frame.image, bg_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        disp = frame.image.copy()
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(disp, (cx, cy), 10, (0, 255, 0), 2)

        for i, (px, py, wx, wy) in enumerate(points):
            cv2.putText(disp, f"{i}: ({wx:.2f},{wy:.2f})", (px+15, py-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.putText(disp, "Click object center, ENTER=done, ESC=cancel", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibrate", disp)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
        if key == 13:  # ENTER
            if len(points) >= 4:
                break
            print(f"Need at least 4 points, have {len(points)}")
            continue

    if len(points) < 4:
        print("Not enough points, calibration cancelled")
        cv2.destroyAllWindows()
        cam.disconnect()
        return False

    # Solve homography
    src = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
    dst = np.array([(p[2], p[3]) for p in points], dtype=np.float32)

    # Try affine first, fallback to perspective
    if len(points) >= 4:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        map_type = "homography"
    else:
        H = cv2.getAffineTransform(src[:3], dst[:3])
        H = np.vstack([H, [0, 0, 1]])
        map_type = "affine"

    # Validate
    errors = []
    for i, (px, py, wx, wy) in enumerate(points):
        pt = np.array([px, py, 1.0], dtype=np.float32)
        proj = H @ pt
        proj = proj[:2] / proj[2]
        err = np.linalg.norm(proj - np.array([wx, wy]))
        errors.append(err)
        print(f"  Point {i}: target=({wx:.3f},{wy:.3f}) proj=({proj[0]:.3f},{proj[1]:.3f}) err={err*1000:.1f}mm")

    print(f"Mean error: {np.mean(errors)*1000:.1f}mm, Max: {np.max(errors)*1000:.1f}mm")

    # Save calibration
    cal = {
        "type": map_type,
        "matrix": H.tolist(),
        "camera": {"index": camera_index, "width": width, "height": height},
        "points": points,
        "mean_error_mm": float(np.mean(errors) * 1000),
        "max_error_mm": float(np.max(errors) * 1000),
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(cal, f, indent=2)
        print(f"Calibration saved to {output}")

    cv2.destroyAllWindows()
    cam.disconnect()
    return True


def load_calibration(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def apply_calibration(cal: dict, px: float, py: float) -> tuple:
    H = np.array(cal["matrix"], dtype=np.float32)
    pt = np.array([px, py, 1.0], dtype=np.float32)
    proj = H @ pt
    return (proj[0] / proj[2], proj[1] / proj[2])