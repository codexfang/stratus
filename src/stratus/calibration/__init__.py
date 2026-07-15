from __future__ import annotations
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

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


def run_calibration_wizard(camera_index: int, width: int, height: int,
                           arm=None, output: str = "") -> bool:
    """Interactive workspace calibration using main preview window.
    
    1. Place object at known world positions
    2. Click object center in preview window
    3. Enter world coordinates (or use defaults)
    4. Minimum 4 points for homography
    """
    cam = USBCamera(index=camera_index, width=width, height=height)
    cam.connect()

    print("\n=== Workspace Camera Calibration ===")
    print("Steps:")
    print("  1. Place an object at a KNOWN world position (e.g., x=0.40, y=0.00)")
    print("  2. In the preview window, click the OBJECT CENTER")
    print("  3. Enter the world X Y (meters) when prompted")
    print("  4. Repeat for at least 4 positions covering the workspace")
    print("  5. Press ENTER in terminal when done")

    points: List[Tuple[float, float, float, float]] = []  # px, py, wx, wy

    # Default grid positions to guide user
    default_grid = [
        (0.25, -0.20), (0.25, 0.00), (0.25, 0.20),
        (0.40, -0.20), (0.40, 0.00), (0.40, 0.20),
        (0.55, -0.20), (0.55, 0.00), (0.55, 0.20),
    ]
    grid_idx = 0

    cv2.namedWindow("Stratus")
    clicked = {'pos': None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked['pos'] = (float(x), float(y))

    cv2.setMouseCallback("Stratus", on_mouse)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            img = frame.image.copy()
            h, w = img.shape[:2]

            # Draw existing points
            for i, (px, py, wx, wy) in enumerate(points):
                cv2.circle(img, (int(px), int(py)), 8, (0, 255, 0), 2)
                cv2.putText(img, f"{i}: ({wx:.2f},{wy:.2f})", (int(px)+10, int(py)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Next suggested position
            if grid_idx < len(default_grid):
                nx, ny = default_grid[grid_idx]
                cv2.putText(img, f"Next: place object at x={nx:.2f} y={ny:.2f}  -> click center",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(img, f"Have {len(points)} pts. Press ENTER to compute, ESC to cancel",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(img, f"Points: {len(points)}/9", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Stratus", img)
            key = cv2.waitKey(50) & 0xFF

            if key == 27:  # ESC
                print("Cancelled")
                return False

            if key == 13:  # ENTER
                if len(points) >= 4:
                    break
                print(f"Need at least 4 points, have {len(points)}")
                continue

            if clicked['pos'] is not None:
                px, py = clicked['pos']
                clicked['pos'] = None

                if grid_idx < len(default_grid):
                    wx, wy = default_grid[grid_idx]
                    print(f"Clicked ({px:.1f}, {py:.1f}) -> world ({wx:.2f}, {wy:.2f})")
                    points.append((px, py, wx, wy))
                    grid_idx += 1
                    print(f"  Saved point {len(points)}. Next: {default_grid[grid_idx] if grid_idx < len(default_grid) else 'Done'}")
                else:
                    # Manual entry
                    try:
                        wx = float(input(f"  World X for click ({px:.1f},{py:.1f}): "))
                        wy = float(input(f"  World Y for click ({px:.1f},{py:.1f}): "))
                        points.append((px, py, wx, wy))
                    except ValueError:
                        print("Invalid input, click again")

    finally:
        cv2.destroyWindow("Stratus")
        cam.disconnect()

    if len(points) < 4:
        print(f"Need at least 4 points, got {len(points)}")
        return False

    # Compute homography
    src = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
    dst = np.array([(p[2], p[3]) for p in points], dtype=np.float32)

    if len(points) >= 4:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        map_type = "homography"
    else:
        H = cv2.getAffineTransform(src[:3], dst[:3])
        H = np.vstack([H, [0, 0, 1]])
        map_type = "affine"

    if H is None:
        print("Homography computation failed")
        return False

    # Validate
    errors = []
    for i, (px, py, wx, wy) in enumerate(points):
        pt = np.array([px, py, 1.0], dtype=np.float32)
        proj = H @ pt
        proj = proj[:2] / proj[2]
        err = np.linalg.norm(proj - np.array([wx, wy]))
        errors.append(err)
        print(f"  Pt {i}: target=({wx:.3f},{wy:.3f}) proj=({proj[0]:.3f},{proj[1]:.3f}) err={err*1000:.1f}mm")

    print(f"Mean error: {np.mean(errors)*1000:.1f}mm, Max: {np.max(errors)*1000:.1f}mm")

    # Save
    cal = {
        "type": map_type,
        "matrix": H.tolist(),
        "camera": {"index": camera_index, "width": width, "height": height},
        "points": [{"pixel_x": p[0], "pixel_y": p[1], "world_x": p[2], "world_y": p[3]} for p in points],
        "mean_error_mm": float(np.mean(errors) * 1000),
        "max_error_mm": float(np.max(errors) * 1000),
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(cal, f, indent=2)
        print(f"Calibration saved to {output}")

    cam.disconnect()
    return True


def load_calibration(path: str) -> Optional[dict]:
    with open(path, 'r') as f:
        return json.load(f)


def apply_calibration(cal: dict, px: float, py: float) -> Tuple[float, float]:
    H = np.array(cal["matrix"], dtype=np.float32)
    pt = np.array([px, py, 1.0], dtype=np.float32)
    proj = H @ pt
    return (proj[0] / proj[2], proj[1] / proj[2])