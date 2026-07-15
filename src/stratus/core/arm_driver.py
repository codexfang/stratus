from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol
import numpy as np
import numpy.typing as npt


@dataclass
class ArmObservation:
    joint_positions: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(6))
    joint_velocities: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(6))
    joint_torques: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(6))


@dataclass
class DetectedObject:
    name: str
    confidence: float
    left: float
    top: float
    width: float
    height: float


@dataclass
class TriageCommand:
    action: str
    target_bin: Optional[str] = None
    pickup_pose: Optional[dict] = None
    drop_pose: Optional[dict] = None
    drop_joints: Optional[list] = None
    label: str = ""
    detected_labels: list[str] = field(default_factory=list)
    detected_objects: list[DetectedObject] = field(default_factory=list)
    pickup_refined: bool = False
    gripper_open_done: bool = False


class ArmDriver(Protocol):
    def connect(self) -> None: ...
    def get_observation(self) -> ArmObservation: ...
    def send_joint_positions(self, positions: npt.NDArray[np.float64]) -> None: ...
    def move_to_pose(self, x: float, y: float, z: float, roll: float = 0, pitch: float = 0, yaw: float = 0) -> bool: ...
    def execute_triage(self, command: TriageCommand) -> bool: ...
    def disable(self) -> None: ...
    def disconnect(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...
