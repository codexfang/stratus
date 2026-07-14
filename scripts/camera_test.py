#!/usr/bin/env python3
import cv2, time
cap = cv2.VideoCapture(0)
time.sleep(1)
ret, frame = cap.read()
if ret:
    print(f"OK: {frame.shape[1]}x{frame.shape[0]}")
    cv2.imwrite("/tmp/camera_test.jpg", frame)
else:
    print("NO FRAME")
cap.release()
