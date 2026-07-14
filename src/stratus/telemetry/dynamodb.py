from __future__ import annotations
import json
import logging
import time
from stratus.core.telemetry import TelemetryBridge, TelemetryEvent

try:
    import boto3
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)


class DynamoDBTelemetry:
    """Logs telemetry events to DynamoDB for portfolio-worthy time-series."""

    def __init__(self, table_name: str = "stratus_classifications", region: str = "us-east-2"):
        if boto3 is None:
            raise ImportError("boto3 not installed. Run: pip install boto3")
        self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        self._count = 0

    def connect(self) -> None:
        logger.info(f"DynamoDB telemetry ready")

    def publish(self, event: TelemetryEvent) -> None:
        self._count += 1
        item = {
            "item_id": f"sort-{int(time.time() * 1000)}-{self._count}",
            "timestamp": int(time.time()),
            "event_type": event.event_type,
            **{k: v for k, v in event.payload.items() if isinstance(v, (str, int, float, bool, list, dict))},
        }
        try:
            self._table.put_item(Item=item)
        except Exception as e:
            logger.warning(f"DynamoDB write failed: {e}")

    def disconnect(self) -> None:
        logger.info(f"DynamoDB done — {self._count} events logged")
