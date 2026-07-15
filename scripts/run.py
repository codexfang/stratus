#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stratus.drivers.vectorbh6_arm import VectorBH6ArmDriver, GripperConfig
from stratus.drivers.opencv_cam import USBCamera
from stratus.drivers.ip_camera import PhoneCamera
from stratus.classifier.local import DummyClassifier
from stratus.pipeline.engine import StratusPipeline
from stratus.telemetry.local import LocalTelemetry

logging.basicConfig(level=logging.INFO, format="%(message)s")

CERTS_DIR = Path.home() / "stratus/certs"
AWS_ENDPOINT = "a1edmkwpjcxhz-ats.iot.us-east-2.amazonaws.com"
AWS_REGION = "us-east-2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "aws"], default="local")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--phone-url", default="")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--gripper-id", type=int, default=0)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-close", type=float, default=-5.0)
    parser.add_argument("--gripper-kp", type=float, default=10.0)
    parser.add_argument("--settle-time", type=float, default=4.0)
    parser.add_argument("--model", default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--map-offset-x", type=float, default=0.15)
    parser.add_argument("--map-scale-x", type=float, default=0.50)
    parser.add_argument("--map-offset-y", type=float, default=-0.20)
    parser.add_argument("--map-scale-y", type=float, default=0.40)
    parser.add_argument("--pickup-z", type=float, default=0.15)
    parser.add_argument("--pitch", type=float, default=0.2)
    parser.add_argument("--approach-inset", type=float, default=0.03)
    parser.add_argument("--arm-cam-url", default="")
    parser.add_argument("--arm-cam-index", type=int, default=-1)
    parser.add_argument("--arm-cam-fov", type=float, default=60.0)
    args = parser.parse_args()

    print("=== Stratus Pipeline ===")
    print(f"Mode: {args.mode}{' (no arm)' if args.no_arm else ''}")

    gripper_cfg = None
    if args.gripper_id > 0:
        gripper_cfg = GripperConfig(
            motor_id=args.gripper_id,
            open_pos=args.gripper_open,
            close_pos=args.gripper_close,
            mit_kp=args.gripper_kp,
            settle_time=args.settle_time,
            approach_inset=args.approach_inset,
        )
        print(f"Gripper: motor ID {args.gripper_id}, open={args.gripper_open} close={args.gripper_close} "
              f"kp={args.gripper_kp} settle={args.settle_time}s inset={args.approach_inset}m")

    arm = VectorBH6ArmDriver(gripper=gripper_cfg) if not args.no_arm else None

    if args.phone_url:
        camera = PhoneCamera(stream_url=args.phone_url)
    else:
        camera = USBCamera(index=args.camera)

    arm_camera = None
    if args.arm_cam_url:
        arm_camera = PhoneCamera(stream_url=args.arm_cam_url)
        print(f"Arm camera: WiFi stream {args.arm_cam_url}")
    elif args.arm_cam_index >= 0:
        arm_camera = USBCamera(index=args.arm_cam_index)
        print(f"Arm camera: USB index {args.arm_cam_index}")

    classifier = DummyClassifier()
    telemetry = LocalTelemetry(log_path=Path.home() / "stratus/data/logs/telemetry.jsonl")

    if args.model:
        from stratus.classifier.yolo_classifier import YOLOClassifier
        classifier = YOLOClassifier(
            model_path=args.model, conf_threshold=args.conf,
            map_offset_x=args.map_offset_x, map_scale_x=args.map_scale_x,
            map_offset_y=args.map_offset_y, map_scale_y=args.map_scale_y,
            pickup_z=args.pickup_z, pitch=args.pitch,
        )
        print(f"YOLO model: {args.model} (conf={args.conf})")

    if args.mode == "aws":
        from stratus.classifier.rekognition import RekognitionClassifier
        from stratus.telemetry.aws_iot import AWSIoTTelemetry
        classifier = RekognitionClassifier(region=AWS_REGION)
        telemetry = AWSIoTTelemetry(
            device_id="stratus-dev-01", endpoint=AWS_ENDPOINT,
            cert_path=str(CERTS_DIR / "device-certificate.pem.crt"),
            key_path=str(CERTS_DIR / "device-private.pem.key"),
            root_ca=str(CERTS_DIR / "AmazonRootCA1.pem"),
        )
        print("AWS mode — Rekognition + IoT Core")

    if arm:
        print("Connecting arm...")
        arm.connect()

    print("Connecting camera...")
    camera.connect()
    if arm_camera:
        arm_camera.connect()

    print("Connecting telemetry...")
    telemetry.connect()

    pipeline = StratusPipeline(arm=arm, camera=camera, classifier=classifier, telemetry=telemetry,
                                arm_camera=arm_camera, arm_cam_fov=args.arm_cam_fov,
                                classify_every=1)

    print("Pipeline running. Y=pick N=skip Q=quit.\n")
    try:
        pipeline.run_loop()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if arm:
            arm.disconnect()
        camera.disconnect()
        if arm_camera:
            arm_camera.disconnect()
        telemetry.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
