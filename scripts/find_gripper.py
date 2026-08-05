#!/usr/bin/env python3
"""find_gripper.py — brute-force search for which motor drives the fingers.

Sweeps every CAN motor id, tries every control mode, and drives each with big
visible swings while printing what it is doing. Watch the fingers the WHOLE
time; when they move, the motor number printed is the gripper.

Usage:
    python scripts/find_gripper.py [--port /dev/tty.usbmodem00000000050C1] [--max-id 0x20]
"""
from __future__ import annotations

import argparse
import time
from motorbridge import Controller, Mode


def model_for(mid: int) -> str:
    return "4340P" if mid < 0x04 else "4310"


def get_state(ctrl: Controller, mot, tries: int = 25) -> object | None:
    mot.request_feedback()
    st = None
    for _ in range(tries):
        ctrl.poll_feedback_once()
        st = mot.get_state()
        if st is not None:
            break
        time.sleep(0.01)
    return st


def ramp(ctrl: Controller, mot, frm: float, to: float, steps: int = 60,
         kp: float = 10.0, kd: float = 1.5, hold: float = 0.0) -> None:
    for i in range(1, steps + 1):
        mot.send_mit(frm + (to - frm) * i / steps, 0.0, kp, kd, 0.0)
        time.sleep(0.02)
    if hold:
        for _ in range(int(hold / 0.02)):
            mot.send_mit(to, 0.0, kp, kd, 0.0)
            time.sleep(0.02)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/tty.usbmodem00000000050C1")
    ap.add_argument("--max-id", default="0x20", help="highest motor id to scan")
    args = ap.parse_args()
    max_id = int(args.max_id, 16)

    ctrl = Controller.from_dm_serial(args.port, 921600)
    mots: dict[int, object] = {}
    try:
        ctrl.enable_all()
        time.sleep(0.5)

        # ---- PHASE 1: find every motor that gives feedback --------------
        live = []
        live_mot = {}
        mots = {}
        for mid in range(0x01, max_id + 1):
            try:
                mot = ctrl.add_damiao_motor(mid, mid + 0x10, model_for(mid))
            except Exception:
                mot = None
            mots[mid] = mot
            if get_state(ctrl, mot) is not None:
                live.append(mid)
                live_mot[mid] = mot
                print(f"[found] motor 0x{mid:X} responds (fid=0x{mid+0x10:X})")
        if not live:
            print("NO MOTORS FOUND on the bus at all!")
            return 1
        print(f"\n=== PHASE 1 done: motors responding: {[hex(m) for m in live]} ===")
        print("(If the gripper is NOT here, its controller is disconnected.)")

        # ---- PHASE 2: big MIT swing on each found motor ------------------
        print("\n=== PHASE 2: big MIT swings, one motor at a time ===")
        print("WATCH THE FINGERS. Note which 'DRIVING MOTOR' line they move with.\n")
        for mid in live:
            mot = live_mot[mid]
            try:
                mot.clear_error()
                mot.enable()
                time.sleep(0.2)
                st = get_state(ctrl, mot)
                if st is None:
                    print(f"--- motor 0x{mid:X}: no feedback, skip ---")
                    continue
                start = st.pos
                print(f"\n*** DRIVING MOTOR 0x{mid:X} ***  start={start:+.2f} rad")
                ramp(ctrl, mot, start, start + 1.0, hold=2.0)
                print(f"    > held +1.0 rad for 2s — WATCH FINGERS <")
                time.sleep(2.0)
                ramp(ctrl, mot, start + 1.0, start, hold=1.0)
                print(f"    back to {start:+.2f}")
            except Exception as e:
                print(f"--- motor 0x{mid:X}: error {e}")
            time.sleep(0.8)

        # ---- PHASE 3: hammer candidate motors in ALL modes ---------------
        print("\n=== PHASE 3: candidate motors (7 and any 0x07-0x0F) in every mode ===")
        cands = [m for m in range(0x07, min(0x10, max_id + 1)) if m not in live]
        for mid in cands:
            mot = mots[mid]
            print(f"\n>>> MOTOR 0x{mid:X}: trying MIT big swing...")
            try:
                mot.clear_error()
                mot.enable()
                time.sleep(0.3)
                ramp(ctrl, mot, 0.0, 9.0, steps=120, kp=12.0, kd=1.5, hold=2.0)
                print("    > MIT pos 9.0 rad held 2s — WATCH FINGERS <")
                time.sleep(2.0)
                ramp(ctrl, mot, 9.0, -1.0, steps=120, kp=12.0, kd=1.5, hold=1.0)
            except Exception as e:
                print(f"    MIT failed: {e}")
            print(f">>> MOTOR 0x{mid:X}: trying POS_VEL mode...")
            try:
                mot.ensure_mode(Mode.POS_VEL, 500)
                for _ in range(80):
                    mot.send_pos_vel(9.0, 3.0)
                    time.sleep(0.02)
                print("    > POS_VEL held pos 9.0 — WATCH FINGERS <")
                time.sleep(2.0)
                for _ in range(80):
                    mot.send_pos_vel(-1.0, 3.0)
                    time.sleep(0.02)
                mot.ensure_mode(Mode.MIT, 500)
            except Exception as e:
                print(f"    POS_VEL failed: {e}")
            time.sleep(1.0)

        print("\n=== DONE ===")
        print("If fingers moved with a 'DRIVING MOTOR 0xN' line, that motor is the gripper.")
        print("If nothing moved, the gripper's own motor/controller is NOT connected.")
        return 0
    finally:
        ctrl.shutdown()
        ctrl.close()


if __name__ == "__main__":
    raise SystemExit(main())
