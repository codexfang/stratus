#!/usr/bin/env python3
"""Snapshot every working camera to /tmp so you can identify which index
is your USB workspace camera."""
import cv2
import time

for i in range(8):
    cap = cv2.VideoCapture(i)
    time.sleep(0.5)
    ret, frame = cap.read()
    if ret and frame is not None:
        out = f"/tmp/cam_{i}.jpg"
        cv2.imwrite(out, frame)
        print(f"Camera {i}: OK -> {out} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        print(f"Camera {i}: no frame")
    cap.release()
print("Done. Open /tmp/cam_N.jpg files to identify your camera.")