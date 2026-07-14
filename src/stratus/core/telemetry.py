from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TelemetryEvent:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


class TelemetryBridge(Protocol):
    def connect(self) -> None: ...
    def publish(self, event: TelemetryEvent) -> None: ...
    def disconnect(self) -> None: ...
