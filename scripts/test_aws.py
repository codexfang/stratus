#!/usr/bin/env python3
"""Test AWS pipeline without arm — camera → Rekognition → DynamoDB → IoT."""
import sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stratus.drivers.opencv_cam import USBCamera
from stratus.classifier.rekognition import RekognitionClassifier
from stratus.telemetry.dynamodb import DynamoDBTelemetry
from stratus.telemetry.aws_iot import AWSIoTTelemetry
from stratus.core.telemetry import TelemetryEvent

logging.basicConfig(level=logging.INFO, format="%(message)s")

CERTS = Path.home() / "stratus/certs"
ENDPOINT = "a1edmkwpjcxhz-ats.iot.us-east-2.amazonaws.com"

INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1

cam = USBCamera(index=INDEX)
classifier = RekognitionClassifier()
db = DynamoDBTelemetry()
iot = AWSIoTTelemetry(
    device_id="stratus-dev-01", endpoint=ENDPOINT,
    cert_path=str(CERTS / "device-certificate.pem.crt"),
    key_path=str(CERTS / "device-private.pem.key"),
    root_ca=str(CERTS / "AmazonRootCA1.pem"),
)

cam.connect()
db.connect()
iot.connect()

try:
    for i in range(3):
        print(f"\n--- Capture {i+1} ---")
        frame = cam.read()
        if frame is None:
            print("No frame")
            continue
        print(f"Captured: {frame.width}x{frame.height}")
        cmd = classifier.classify(frame)
        print(f"→ {cmd.label}  →  {cmd.target_bin}")
        print(f"  Labels: {', '.join(cmd.detected_labels[:5])}")
        db.publish(TelemetryEvent(event_type="classification", payload={
            "action": cmd.action, "target_bin": cmd.target_bin,
            "grade": cmd.label, "frame": i,
        }))
        iot.publish(TelemetryEvent(event_type="classification", payload={
            "action": cmd.action, "target_bin": cmd.target_bin,
            "grade": cmd.label, "frame": i,
        }))
        time.sleep(2)
finally:
    cam.disconnect()
    db.disconnect()
    iot.disconnect()
    print("\nDone. Check DynamoDB: aws dynamodb scan --table-name stratus_classifications")
