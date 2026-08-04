"""Standalone gripper diagnostic.

Connects the arm (brings up CAN + control loop), probes the gripper motor on
the bus, then drives it with continuous MIT streams so we can *watch* physically
how the fingers behave.

    python scripts/test_gripper.py                       # cycles open 2.0 / grip -1.8
    python scripts/test_gripper.py --ramp -0.5 2.5        # slow sweep to map range
    python scripts/test_gripper.py --open 2.0 --dur 2     # longer streams

Flags:
    --probe       read live motor status/position + damiao registers (default on)
    --open POS    open position (default 2.0 — PROVEN to open on hardware)
    --grip POS    grip position (default -1.8)
    --reps N      open/grip cycles (default 3)
    --dur S       seconds to stream each position (default 2.0)
    --gripper-id  motor CAN id (default 7)
    --ramp MIN MAX  slow position sweep instead of cycles
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

from stratus.drivers.vectorbh6_arm import GripperConfig, VectorBH6ArmDriver


def probe(handle) -> None:
    mot = handle._gripper_motor
    print("-- motor probe: alive? --")
    if mot is None:
        print("  !! no gripper motor registered")
        return
    for label, fn in [
        ("enable()", lambda: mot.enable()),
        ("get_state()", lambda: getattr(mot.get_state(), "pos", None)),
    ]:
        try:
            r = fn()
            print(f"  {label}: OK {r if label.startswith('get_state') else ''}")
        except Exception as e:
            print(f"  {label}: FAIL {e}")
    # Feedback + register reads — do any of them ack?
    try:
        mot.request_feedback()
        time.sleep(0.05)
        ctrl = handle._arm._ctrl_map.get("damiao")
        if ctrl:
            ctrl.poll_feedback_once()
        st = mot.get_state()
        print(f"  feedback get_state: pos={getattr(st, 'pos', None)!r} "
              f"status={getattr(st, 'status_code', None)!r}")
    except Exception as e:
        print(f"  feedback get_state: FAIL {e}")
    for reg in (0, 2, 10, 20, 0x10000):
        try:
            v = mot.get_register_f32(reg, 300)
            print(f"  get_register_f32(0x{reg:x}) = {v}")
        except Exception as e:
            print(f"  get_register_f32(0x{reg:x}) = FAIL ({str(e)[:60]})")
        try:
            v = mot.get_register_u32(reg, 300)
            print(f"  get_register_u32(0x{reg:x}) = {v}")
        except Exception as e:
            print(f"  get_register_u32(0x{reg:x}) = FAIL ({str(e)[:60]})")
    print("-- probe done --")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", type=float, default=6.0,
                    help="open position (default 6.0 = ~3x old 2.0)")
    ap.add_argument("--grip", type=float, default=-0.8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dur", type=float, default=2.0)
    ap.add_argument("--gripper-id", type=int, default=7)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--ramp", nargs=2, type=float, metavar=("MIN", "MAX"))
    ap.add_argument("--ramp-steps", type=int, default=90)
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
        if not args.no_probe:
            probe(handle)
        print(">> WATCH FINGERS during the streams <<")

        if args.ramp:
            lo, hi = sorted(args.ramp)
            step = (hi - lo) / args.ramp_steps
            print(f">> RAMP {lo:.2f} -> {hi:.2f} in {args.ramp_steps} steps "
                  f"({args.ramp_dt}s each)")
            for i in range(args.ramp_steps + 1):
                handle.gripper_drive(lo + step * i, args.ramp_dt, poll_feedback=True)
            print(">> ramp done; holding at", hi)
            handle.gripper_drive(hi, 1.5)
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