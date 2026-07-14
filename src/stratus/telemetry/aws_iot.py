from __future__ import annotations
import json
import logging
from pathlib import Path
from stratus.core.telemetry import TelemetryBridge, TelemetryEvent

try:
    from awscrt.io import EventLoopGroup
    from awsiot import mqtt_connection_builder
    AWSIOT_AVAILABLE = True
except ImportError:
    AWSIOT_AVAILABLE = False

logger = logging.getLogger(__name__)


class AWSIoTTelemetry:
    def __init__(
        self,
        device_id: str = "stratus-dev-01",
        endpoint: str = "",
        cert_path: str = "",
        key_path: str = "",
        root_ca: str = "",
        port: int = 8883,
    ):
        if not AWSIOT_AVAILABLE:
            raise ImportError("Run: pip install awscrt")
        self._device_id = device_id
        self._endpoint = endpoint
        self._cert = Path(cert_path).expanduser()
        self._key = Path(key_path).expanduser()
        self._ca = Path(root_ca).expanduser()
        self._port = port
        self._connection = None

    def connect(self) -> None:
        self._connection = mqtt_connection_builder.mtls_from_path(
            endpoint=self._endpoint, port=self._port,
            cert_filepath=str(self._cert), pri_key_filepath=str(self._key),
            ca_filepath=str(self._ca), client_id=self._device_id,
        )
        self._connection.connect().result()
        logger.info(f"AWS IoT connected as {self._device_id}")

    def publish(self, event: TelemetryEvent) -> None:
        if not self._connection:
            raise RuntimeError("Not connected")
        payload = json.dumps({"type": event.event_type, **event.payload})
        self._connection.publish(
            topic=f"stratus/{self._device_id}/telemetry", payload=payload, qos=1,
        )

    def disconnect(self) -> None:
        if self._connection:
            self._connection.disconnect().result()
