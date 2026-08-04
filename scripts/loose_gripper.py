"""Loosen the gripper so the fingers move freely by hand.

Streams zero-stiffness MIT (kp=kd=tau=0) — the motor stays powered but applies
no force, so you can open/close the jaws manually.

    python scripts/loose_gripper.py                 # fingers stay loose until Ctrl+C
    python scripts/loose_gripper.py --seconds 10    # loosen for 10s then re-grip hard

To get a firm grip again afterward, run with --tight POS:
    python scripts/loose_gripper.py --tight -0.8    # hold fingers closed at -0.8
    python scripts/loose_gripper.py --open 2.0      # hold fingers open at 2.0
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
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stay loose this long; 0 = forever until Ctrl+C")
    ap.add_argument("--tight", type=float, default=None,
                    help="instead of limping, command a stiff hold at this position")
    ap.add_argument("--open", type=float, default=None,
                    help="instead of limping, command a stiff hold OPEN at this position")
    args = ap.parse_args()

    if args.tight is not None and args.open is not None:
        print("!! use only one of --tight / --open")
        return 2

    cfg = GripperConfig(motor_id=args.gripper_id)
    handle = VectorBH6ArmDriver(gripper=cfg, prime_on_connect=False)

    try:
        handle.connect()
    except Exception as e:
        print(f"!! connect failed: {e}")
        return 1

    try:
        if args.tight is not None:
            print(f">> stiff hold at {args.tight} for 10s ..")
            handle.gripper_drive(args.tight, 10.0, poll_feedback=True)
        elif args.open is not None:
            print(f">> stiff hold OPEN at {args.open} for 10s ..")
            handle.gripper_drive(args.open, 10.0, poll_feedback=True)
        else:
            print(">> fingers are FREE — move the jaws by hand "
                  "(Ctrl+C to finish) ..", flush=True)
            handle.gripper_limp(0.0)
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n>> restoring control (stiff hold/short prime) ..")
                handle.gripper_drive(cfg.open_pos, 1.5)
    finally:
        handle.disconnect()
        print("== done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())