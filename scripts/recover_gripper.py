"""Gripper fault recovery + gentle range walk.

IMPORTANT: power-cycle the robot first — unplug the USB-CAN dongle for 5s,
plug it back, THEN run this. A Damiao motor stuck in over-limit fault
(red blinking LED) ignores every command until it is power-cycled.

This script:
  1. retries clear_error+enable several times (fault recovery)
  2. walks position 0 -> 2.0 with LOW gains (gentle, won't re-fault)
  3. holds open, walks gently back to -0.8, then opens again

Usage:
    python scripts/recover_gripper.py            # safe defaults
    python scripts/recover_gripper.py --max 2.2  # try a bit more open
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

from stratus.drivers.vectorbh6_arm import GripperConfig, VectorBH6ArmDriver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper-id", type=int, default=7)
    ap.add_argument("--max", type=float, default=3.6, help="open-side walk target (3.6 real max; above 4 faults)")
    ap.add_argument("--min", type=float, default=-0.8, help="close-side walk target")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--step-dur", type=float, default=0.3)
    ap.add_argument("--clear-reps", type=int, default=6)
    args = ap.parse_args()

    cfg = GripperConfig(motor_id=args.gripper_id,
                        open_pos=args.max, grip_pos=args.min, close_pos=0.0)
    handle = VectorBH6ArmDriver(gripper=cfg)

    print("==", "GRIPPER RECOVERY", "==")
    print("POWER-CYCLED the robot since last run? "
          "If not, unplug the USB-CAN dongle 5s, replug, retry this.")
    try:
        handle.connect()
    except Exception as e:
        print(f"!! connect failed: {e}")
        return 1

    try:
        mot = handle._gripper_motor
        if mot is None:
            print("!! no gripper motor registered")
            return 1

        # 1) Persistent fault recovery: clear_error + enable, several passes
        for i in range(args.clear_reps):
            try:
                mot.clear_error()
            except Exception as e:
                print(f"  clear_error pass {i + 1}: {e}")
            time.sleep(0.3)
            try:
                mot.enable()
            except Exception as e:
                print(f"  enable pass {i + 1}: {e}")
            time.sleep(0.3)
            st = None
            try:
                mot.request_feedback()
                time.sleep(0.05)
                ctrl = handle._arm._ctrl_map.get("damiao")
                if ctrl:
                    ctrl.poll_feedback_once()
                st = mot.get_state()
            except Exception:
                pass
            print(f"  pass {i + 1}: status={getattr(st, 'status_code', None)} "
                  f"pos={getattr(st, 'pos', None)!r}")

        # 2) Gentle walk open 0 -> max
        print(f">> walking OPEN 0 -> {args.max} in steps of {args.step} "
              f"(low gains, {args.step_dur}s each) — WATCH FINGERS")
        for p in _frange(0.0, args.max + 1e-9, args.step):
            handle.gripper_drive(p, args.step_dur, poll_feedback=False)
        print(f">> holding open {args.max} for 2s")
        handle.gripper_drive(args.max, 2.0, poll_feedback=True)

        # 3) Gentle walk close  max -> min
        print(f">> walking CLOSE {args.max} -> {args.min} — WATCH FINGERS")
        for p in _frange(args.max, args.min - 1e-9, -args.step):
            handle.gripper_drive(p, args.step_dur, poll_feedback=False)

        # 4) Back open
        print(f">> walking OPEN {args.min} -> {args.max} — WATCH FINGERS")
        for p in _frange(args.min, args.max + 1e-9, args.step):
            handle.gripper_drive(p, args.step_dur, poll_feedback=False)
        print(">> final: holding open")
        handle.gripper_drive(args.max, 1.5)
    finally:
        handle.disconnect()
        print("== done")
    return 0


def _frange(start: float, stop: float, step: float):
    v = start
    while (step > 0 and v <= stop) or (step < 0 and v >= stop):
        yield round(v, 4)
        v += step


if __name__ == "__main__":
    raise SystemExit(main())