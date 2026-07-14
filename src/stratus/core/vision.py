from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import numpy.typing as npt


@dataclass
class CameraFrame:
    image: npt.NDArray[np.uint8]
    timestamp: float
    width: int
    height: int


class Camera(Protocol):
    def connect(self) -> None: ...
    def read(self) -> CameraFrame: ...
    def disconnect(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...
