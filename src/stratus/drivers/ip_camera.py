"""Phone camera via WiFi IP Webcam with flash control.

Setup:
  1. Install "IP Webcam" on Android (Play Store) or "ReCamera" on iPhone
  2. Connect phone and Mac to same WiFi
  3. Start the server on the app (it shows a URL like http://192.168.1.X:8080)
  4. Pass that URL to this class

Flash requires the app to support HTTP flash toggle (IP Webcam for Android does).
"""
from __future__ import annotations
import time
import cv2
import numpy as np
import urllib.request
from stratus.core.vision import Camera, CameraFrame


class PhoneCamera:
    """Wireless phone camera with flash control via IP Webcam."""

    def __init__(self, stream_url: str = "http://192.168.1.100:8080"):
        self._base = stream_url.rstrip("/")
        self._cap = None

    def connect(self) -> None:
        video_url = f"{self._base}/video"
        self._cap = cv2.VideoCapture(video_url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot connect to {video_url}")

    def flash_on(self) -> None:
        try:
            urllib.request.urlopen(f"{self._base}/enabletorch", timeout=2)
        except Exception:
            pass  # ignore if flash not supported

    def flash_off(self) -> None:
        try:
            urllib.request.urlopen(f"{self._base}/disabletorch", timeout=2)
        except Exception:
            pass

    def read(self, use_flash: bool = False) -> CameraFrame:
        if self._cap is None:
            raise RuntimeError("Camera not connected")
        if use_flash:
            self.flash_on()
            time.sleep(0.3)  # let flash stabilize
        ret, frame = self._cap.read()
        if use_flash:
            self.flash_off()
        if not ret:
            raise RuntimeError("Failed to read frame")
        return CameraFrame(
            image=frame, timestamp=time.time(),
            width=frame.shape[1], height=frame.shape[0],
        )

    def disconnect(self) -> None:
        if self._cap:
            self._cap.release()
        self.flash_off()

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
