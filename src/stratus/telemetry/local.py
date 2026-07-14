from __future__ import annotations
import json
import logging
from pathlib import Path
from stratus.core.telemetry import TelemetryBridge, TelemetryEvent

logger = logging.getLogger(__name__)


class LocalTelemetry:
    def __init__(self, log_path: str | Path | None = None):
        self._log_path = Path(log_path) if log_path else None
        self._file = None

    def connect(self) -> None:
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._log_path, "a")

    def publish(self, event: TelemetryEvent) -> None:
        line = json.dumps({"event": event.event_type, **event.payload})
        logger.info(f"[telemetry] {line}")
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    def disconnect(self) -> None:
        if self._file:
            self._file.close()
