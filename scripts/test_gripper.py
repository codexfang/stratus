"""Standalone gripper diagnostic.

Connects the arm (brings up CAN + control loop), then drives the gripper with
continuous MIT streams so we can *watch* physically how the fingers behave.
Use --cycles for open/grip cycling, or --ramp to sweep position slowly and map
the mechanical range (find exactly how far "open" can go before faulting).

Requires the same CAN setup as run.py:

    python scripts/test_gripper.py                     # cycles, open 2.5 grip -2.5
    python scripts/test_gripper.py --ramp 0.0 3.0      # slow sweep 0 -> 3.0
    python scripts/test_gripper.py --open 2.0 --dur 2  # slower, longer streams
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
    ap.add_argument("--dur", type=float, default=1.5,
                    help="seconds to stream each position (default 1.5)")
    ap.add_argument("--gripper-id", type=int, default=7)
    ap.add_argument("--ramp", nargs=2, type=float, metavar=("MIN", "MAX"),
                    help="slow position sweep from MIN to MAX instead of cycles")
    ap.add_argument("--ramp-steps", type=int, default=80)
    ap.add_argument("--ramp-dt", type=float, default=0.06)
    args = ap.parse_args()

    cfg = GripperConfig(motor_id=args.gripper_id,
                        open_pos=args.open, grip_pos=args.grip)
    handle = VectorBH6ArmDriver(gripper=cfg)

    mode = f"ramp {args.ramp[0]}->{args.ramp[1]}" if args.ramp else (
        f"cycles open {args.open} / grip {args.grip} x{args.reps}")
    print(f"== Gripper diagnostic: motor {args.gripper_id}  mode: {mode}")
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
        print(">> WATCH FINGERS during the streams <<")

        if args.ramp:
            lo, hi = sorted(args.ramp)
            step = (hi - lo) / args.ramp_steps
            print(f">> RAMP {lo:.2f} -> {hi:.2f} in {args.ramp_steps} steps "
                  f"({args.ramp_dt}s each)  [current->open max]")
            for i in range(args.ramp_steps + 1):
                p = lo + step * i
                handle.gripper_drive(p, args.ramp_dt, poll_feedback=False)
            print(">> ramp done; holding at", hi)
            handle.gripper_drive(hi, 1.0)
        else:
            for i in range(args.reps):
                print(f"-> cycle {i + 1}/{args.reps}: OPEN {args.open} "
                      f"(streaming {args.dur}s) — watch fingers", flush=True)
                handle.gripper_drive(args.open, args.dur, poll_feedback=True)
                time.sleep(0.2)
                print(f"-> cycle {i + 1}/{args.reps}: GRIP {args.grip} "
                      f"(streaming {args.dur}s) — watch fingers", flush=True)
                handle.gripper_drive(args.grip, args.dur, poll_feedback=True)
                time.sleep(0.2)
            print(f"-> final: OPEN {args.open} (streaming 1s, left holding open)", flush=True)
            handle.gripper_drive(args.open, 1.0)
    finally:
        handle.disconnect()
        print("== done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())