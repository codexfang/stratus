from __future__ import annotations
import time
import cv2
import numpy as np
from stratus.core.vision import Camera, CameraFrame


class USBCamera:
    def __init__(self, index: int | str = 0, width: int = 1920, height: int = 1080):
        self._index = index
        self._width = width
        self._height = height
        self._cap = None

    def connect(self) -> None:
        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._index}")
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger = __import__('logging').getLogger(__name__)
        logger.info("Camera %s: %dx%d", self._index, actual_w, actual_h)

    def read(self, use_flash: bool = False) -> CameraFrame:
        if self._cap is None:
            raise RuntimeError("Camera not connected")
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame")
        return CameraFrame(image=frame, timestamp=time.time(), width=frame.shape[1], height=frame.shape[0])

    def disconnect(self) -> None:
        if self._cap:
            self._cap.release()

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
