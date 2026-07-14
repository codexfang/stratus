#!/usr/bin/env python3
from __future__ import annotations
import sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path.home() / "reBotArm_control_py"))
from reBotArm_control_py.actuator import RobotArm as VBArm
from motorbridge import Mode
logging.basicConfig(level=logging.INFO, format="%(message)s")

arm = VBArm()
arm.connect()
ctrl = arm._ctrl_map["damiao"]
mot = ctrl.add_damiao_motor(7, 0x17, "4310")

mot.write_register_f32(2, 150.0)
time.sleep(0.1)
mot.store_parameters()
mot.clear_error()
time.sleep(0.1)
mot.enable()
time.sleep(0.3)

for attempt in range(30):
    mot.request_feedback()
    time.sleep(0.02)
    ctrl.poll_feedback_once()
    st = mot.get_state()
    if st and st.status_code == 1:
        print(f"ENABLED, init pos={st.pos:.4f}")
        break
    if st and st.status_code != 0:
        mot.clear_error()
        time.sleep(0.1)
        mot.enable()
        time.sleep(0.3)
    time.sleep(0.15)
else:
    print("Failed to enable")
    arm.disconnect()
    sys.exit(1)

mot.send_pos_vel(0.0, 3.0)
time.sleep(1.0)

print("\n=== POS_VEL — large position commands ===")
for target in [1.0, 2.0, 3.0, 5.0, 10.0, -1.0, -2.0, -3.0, -5.0, -10.0, 0.0]:
    mot.send_pos_vel(target, 8.0)
    time.sleep(2.0)
    mot.request_feedback()
    time.sleep(0.02)
    ctrl.poll_feedback_once()
    st = mot.get_state()
    if st:
        reached = "YES" if abs(st.pos - target) < 0.1 else "NEAR" if abs(st.pos - target) < 0.5 else "no"
        print(f"  target={target:6.1f} -> pos={st.pos:7.4f} torq={st.torq:7.3f} reached? {reached} status={st.status_code}")

print("\n=== MIT mode — trying to go farther ===")
mot.ensure_mode(Mode.MIT, 1000)
time.sleep(0.3)
for target in [3.0, 5.0, -3.0, -5.0, 8.0, -8.0, 0.0]:
    mot.send_mit(target, 0.0, 2.0, 0.1, 0.0)
    time.sleep(2.0)
    mot.request_feedback()
    time.sleep(0.02)
    ctrl.poll_feedback_once()
    st = mot.get_state()
    if st:
        reached = abs(st.pos - target) < 0.1
        print(f"  MIT target={target:6.1f} -> pos={st.pos:7.4f} torq={st.torq:7.3f} reached={reached} status={st.status_code}")

arm.disconnect()
