from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class ObjectCandidate:
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float


class LocalDetector:
    def __init__(self, min_area: int = 800):
        self._min_area = min_area
        self._bg: np.ndarray | None = None
        self._bg_h, self._bg_w = 0, 0

    def set_background(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (15, 15), 0)
        self._bg = enhanced.astype(np.float32)
        self._bg_h, self._bg_w = frame.shape[:2]

    def detect(self, frame: np.ndarray) -> list[ObjectCandidate]:
        if self._bg is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (15, 15), 0).astype(np.float32)

        diff = cv2.absdiff(enhanced, self._bg)

        blocks = 8
        bh = self._bg_h // blocks
        bw = self._bg_w // blocks
        thresh = np.zeros_like(diff, dtype=np.uint8)
        for by in range(blocks):
            y1 = by * bh
            y2 = self._bg_h if by == blocks - 1 else (by + 1) * bh
            for bx in range(blocks):
                x1 = bx * bw
                x2 = self._bg_w if bx == blocks - 1 else (bx + 1) * bw
                block = diff[y1:y2, x1:x2]
                if block.size == 0:
                    continue
                mean = float(np.mean(block))
                std = float(np.std(block))
                t = max(12, mean + std * 1.5)
                _, block_th = cv2.threshold(block, t, 255, cv2.THRESH_BINARY)
                thresh[y1:y2, x1:x2] = block_th

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = self._bg_h * self._bg_w
        max_area = total_area * 0.4
        results = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            results.append(ObjectCandidate(
                x=x, y=y, w=w, h=h,
                cx=(x + w / 2) / self._bg_w,
                cy=(y + h / 2) / self._bg_h,
            ))
        results.sort(key=lambda o: (o.cx - 0.5) ** 2 + (o.cy - 0.5) ** 2)
        return results

    def crop_object(self, frame: np.ndarray, obj: ObjectCandidate, margin: float = 0.2) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(obj.x - obj.w * margin))
        y1 = max(0, int(obj.y - obj.h * margin))
        x2 = min(w, int(obj.x + obj.w * (1 + margin)))
        y2 = min(h, int(obj.y + obj.h * (1 + margin)))
        return frame[y1:y2, x1:x2]
