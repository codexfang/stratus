from __future__ import annotations
import numpy as np
from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand


class DummyClassifier:
    def __init__(self):
        self._count = 0
        self._bins = {
            "bin_a": {"angles": [45, -30, 30, 0, 0, 0], "label": "Left bin"},
            "bin_b": {"angles": [-45, -30, 30, 0, 0, 0], "label": "Right bin"},
            "bin_c": {"angles": [0, -50, 60, 0, 0, 0], "label": "Forward bin"},
        }

    def set_background(self, frame: CameraFrame) -> None:
        pass

    def classify(self, frame: CameraFrame) -> TriageCommand:
        grade = ["bin_a", "bin_b", "bin_c"][self._count % 3]
        self._count += 1
        bin_cfg = self._bins[grade]
        return TriageCommand(
            action="pick_and_place",
            target_bin=grade,
            pickup_pose={"x": 0.25, "y": 0.0, "z": 0.12, "roll": 0, "pitch": 0.2, "yaw": 0},
            drop_joints=bin_cfg["angles"],
            label=bin_cfg["label"],
            detected_labels=["test"],
            detected_objects=[],
        )
