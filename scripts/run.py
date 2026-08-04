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

CERTS_DIR = Path.home() / "Projects/stratus/certs"
AWS_ENDPOINT = "a1edmkwpjcxhz-ats.iot.us-east-2.amazonaws.com"
AWS_REGION = "us-east-2"


def main():
    parser = argparse.ArgumentParser(description="Stratus ITAD robotic sorting pipeline")
    parser.add_argument("--mode", choices=["local", "aws"], default="local",
                        help="Classification backend (default: local/YOLO)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Workspace camera index (default: 0)")
    parser.add_argument("--phone-url", default="",
                        help="IP Webcam URL for workspace camera, e.g. http://192.168.1.x:8080")

    # Arm / gripper
    parser.add_argument("--no-arm", action="store_true",
                        help="Run without physical arm (preview only)")
    parser.add_argument("--gripper-id", type=int, default=7,
                        help="Damiao gripper motor CAN ID (default: 7)")
    parser.add_argument("--gripper-open", type=float, default=4.0,
                        help="Gripper open position in motor radians (default: 4.0)")
    parser.add_argument("--gripper-close", type=float, default=-5.0,
                        help="Gripper fully closed position (default: -5.0)")
    parser.add_argument("--gripper-grip", type=float, default=-2.0,
                        help="Gripper grip target — stopped by object before this (default: -2.0)")
    parser.add_argument("--gripper-kp", type=float, default=10.0,
                        help="Gripper MIT kp (default: 10.0)")
    parser.add_argument("--gripper-delta", type=float, default=0.5,
                        help="Min position delta to confirm object in gripper (default: 0.5)")
    parser.add_argument("--settle-time", type=float, default=2.0,
                        help="Gripper settle time in seconds (default: 2.0)")

    # YOLO / classifier
    parser.add_argument("--model", default="",
                        help="Path to YOLO-World .pt model file")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="YOLO confidence threshold (default: 0.15)")

    # Coordinate mapping (used when no calibration file is present)
    parser.add_argument("--map-offset-x", type=float, default=0.15,
                        help="Linear map: arm_x = offset_x + cx * scale_x (default: 0.15)")
    parser.add_argument("--map-scale-x", type=float, default=0.50,
                        help="Linear map X scale (default: 0.50)")
    parser.add_argument("--map-offset-y", type=float, default=-0.20,
                        help="Linear map: arm_y = offset_y + cy * scale_y (default: -0.20)")
    parser.add_argument("--map-scale-y", type=float, default=0.40,
                        help="Linear map Y scale (default: 0.40)")
    parser.add_argument("--pitch", type=float, default=0.4,
                        help="Gripper approach pitch angle in radians (default: 0.4)")

    # Cameras
    parser.add_argument("--arm-cam-url", default="",
                        help="IP Webcam URL for arm-mounted camera")
    parser.add_argument("--arm-cam-index", type=int, default=-1,
                        help="USB index for arm-mounted camera (-1 = disabled)")
    parser.add_argument("--arm-cam-fov", type=float, default=60.0,
                        help="Arm camera horizontal FOV in degrees (default: 60)")
    parser.add_argument("--cam-width", type=int, default=1280)
    parser.add_argument("--cam-height", type=int, default=720)
    parser.add_argument("--arm-cam-width", type=int, default=640)
    parser.add_argument("--arm-cam-height", type=int, default=480)
    parser.add_argument("--classify-every", type=int, default=3,
                        help="Run classifier every N frames (default: 3)")

    # Calibration
    parser.add_argument("--calibrate", action="store_true",
                        help="Run interactive workspace calibration wizard then exit")
    parser.add_argument("--scan-joints", nargs=6, type=float, default=None,
                        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help="Override scan-pose joint angles in radians (6 values). "
                             "Default: arm driver built-in scan pose.")

    args = parser.parse_args()

    print("=== Stratus Pipeline ===")
    print(f"Mode: {args.mode}{' (no arm)' if args.no_arm else ''}")

    # ── Gripper config ────────────────────────────────────────────────────
    gripper_cfg = GripperConfig(
        motor_id=args.gripper_id,
        open_pos=args.gripper_open,
        close_pos=args.gripper_close,
        grip_pos=args.gripper_grip,
        mit_kp=args.gripper_kp,
        grip_delta_threshold=args.gripper_delta,
        settle_time=args.settle_time,
    )
    print(
        f"Gripper: motor ID {args.gripper_id}, "
        f"open={args.gripper_open} close={args.gripper_close} "
        f"grip={args.gripper_grip} kp={args.gripper_kp} "
        f"delta={args.gripper_delta} settle={args.settle_time}s"
    )

    # ── Arm ───────────────────────────────────────────────────────────────
    arm = VectorBH6ArmDriver(gripper=gripper_cfg) if not args.no_arm else None

    # ── Workspace camera ──────────────────────────────────────────────────
    if args.phone_url:
        camera = PhoneCamera(stream_url=args.phone_url)
    else:
        camera = USBCamera(index=args.camera, width=args.cam_width, height=args.cam_height)

    # ── Arm-mounted camera (optional) ─────────────────────────────────────
    arm_camera = None
    if args.arm_cam_url:
        arm_camera = PhoneCamera(stream_url=args.arm_cam_url)
        print(f"Arm camera: WiFi stream {args.arm_cam_url}")
    elif args.arm_cam_index >= 0:
        arm_camera = USBCamera(
            index=args.arm_cam_index,
            width=args.arm_cam_width,
            height=args.arm_cam_height,
        )
        print(f"Arm camera: USB index {args.arm_cam_index} "
              f"({args.arm_cam_width}x{args.arm_cam_height})")

    # ── Classifier ────────────────────────────────────────────────────────
    classifier = DummyClassifier()

    if args.model:
        from stratus.classifier.yolo_classifier import YOLOClassifier
        classifier = YOLOClassifier(
            model_path=args.model,
            conf_threshold=args.conf,
            map_offset_x=args.map_offset_x,
            map_scale_x=args.map_scale_x,
            map_offset_y=args.map_offset_y,
            map_scale_y=args.map_scale_y,
            pitch=args.pitch,
        )
        print(f"YOLO model: {args.model} (conf={args.conf})")

    # ── Telemetry ─────────────────────────────────────────────────────────
    telemetry = LocalTelemetry(
        log_path=Path.home() / "Projects/stratus/data/logs/telemetry.jsonl"
    )

    if args.mode == "aws":
        from stratus.classifier.rekognition import RekognitionClassifier
        from stratus.telemetry.aws_iot import AWSIoTTelemetry
        classifier = RekognitionClassifier(region=AWS_REGION)
        telemetry = AWSIoTTelemetry(
            device_id="stratus-dev-01",
            endpoint=AWS_ENDPOINT,
            cert_path=str(CERTS_DIR / "device-certificate.pem.crt"),
            key_path=str(CERTS_DIR / "device-private.pem.key"),
            root_ca=str(CERTS_DIR / "AmazonRootCA1.pem"),
        )
        print("AWS mode — Rekognition + IoT Core")

    # ── Calibration wizard (exits after completion) ───────────────────────
    if args.calibrate:
        from stratus.calibration import run_calibration_wizard
        ok = run_calibration_wizard(
            camera_index=args.camera,
            width=args.cam_width,
            height=args.cam_height,
            arm=arm,
            output=str(Path.home() / "Projects/stratus/calibration/workspace_cal.json"),
        )
        if arm:
            arm.disconnect()
        camera.disconnect()
        if arm_camera:
            arm_camera.disconnect()
        sys.exit(0 if ok else 1)

    # ── Connect everything ────────────────────────────────────────────────
    print("Connecting arm...")
    if arm:
        # Apply CLI scan-joints override before connect so _safe_return_home
        # uses the same pose from the very first move.
        if args.scan_joints:
            arm.set_scan_joints(args.scan_joints)
        arm.connect()
        print("Arm connected")

    print("Connecting camera...")
    camera.connect()
    if arm_camera:
        arm_camera.connect()

    print("Connecting telemetry...")
    telemetry.connect()

    # ── Build and run pipeline ────────────────────────────────────────────
    pipeline = StratusPipeline(
        arm=arm,
        camera=camera,
        classifier=classifier,
        telemetry=telemetry,
        arm_camera=arm_camera,
        arm_cam_fov=args.arm_cam_fov,
        classify_every=args.classify_every,
        map_offset_x=args.map_offset_x,
        map_scale_x=args.map_scale_x,
        map_offset_y=args.map_offset_y,
        map_scale_y=args.map_scale_y,
        scan_joints=args.scan_joints,
    )

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
