from __future__ import annotations
from typing import Protocol, Optional
from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand


class Classifier(Protocol):
    def classify(self, frame: CameraFrame) -> TriageCommand: ...
    def set_background(self, frame: CameraFrame) -> None: ...
