"""Standalone gripper diagnostic.

Connects the arm (brings up CAN + control loop), then drives the gripper
open -> grip -> open a few times with direct MIT commands so we can *watch*
physically whether the fingers move. Run with the same CAN setup as run.py:

    conda run -n rebot python scripts/test_gripper.py        # or
    python scripts/test_gripper.py

Flags:
    --open POS     open position (default 2.5)
    --grip POS     grip position (default -2.5)
    --reps N       open/grip cycles (default 4)
    --wait S       seconds between commands (default 0.5)
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

from stratus.drivers.vectorbh6_arm import GripperConfig, VectorBH6ArmDriver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", type=float, default=2.5)
    ap.add_argument("--grip", type=float, default=-2.5)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--wait", type=float, default=0.5)
    ap.add_argument("--gripper-id", type=int, default=7)
    ap.add_argument("--dry", action="store_true", help="don't send commands")
    args = ap.parse_args()

    cfg = GripperConfig(motor_id=args.gripper_id,
                        open_pos=args.open, grip_pos=args.grip)
    handle = VectorBH6ArmDriver(gripper=cfg)

    print(f"== Gripper diagnostic: motor {args.gripper_id}  "
          f"open {args.open}  grip {args.grip}  reps {args.reps}  wait {args.wait}")
    try:
        handle.connect()
    except Exception as e:
        print(f"!! connect failed: {e}")
        return 1

    try:
        mot = handle._gripper_motor
        if mot is None:
            print("!! no gripper motor registered — check motor_id / CAN bus")
            return 1
        print(">> gripper motor registered. Driving cycles — WATCH FINGERS MOVE <<")

        for i in range(args.reps):
            print(f"-> cycle {i + 1}/{args.reps}: OPEN {args.open}")
            handle._gripper_cmd(args.open)
            time.sleep(args.wait)

            print(f"-> cycle {i + 1}/{args.reps}: GRIP {args.grip}")
            handle._gripper_cmd(args.grip)
            time.sleep(args.wait)

        print(f"-> final: OPEN {args.open} (left holding open)")
        handle._gripper_cmd(args.open)
        time.sleep(1.0)
    finally:
        handle.disconnect()
        print("== done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())