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
mot.clear_error()
time.sleep(0.1)
mot.write_register_f32(2, 150.0)
time.sleep(0.1)
mot.store_parameters()
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

mot.ensure_mode(Mode.MIT, 1000)
time.sleep(0.3)
mot.send_mit(0.0, 0.0, 2.0, 0.1, 0.0)
time.sleep(2.0)

print("\n=== MIT mode: open / close cycle ===")
for target, name in [(4.0, "open"), (-5.0, "close"), (4.0, "open"), (-5.0, "close"), (0.0, "center")]:
    mot.send_mit(target, 0.0, 2.0, 0.1, 0.0)
    time.sleep(2.0)
    mot.request_feedback()
    time.sleep(0.02)
    ctrl.poll_feedback_once()
    st = mot.get_state()
    if st:
        reached = abs(st.pos - target) < 0.1
        print(f"  MIT {name:6s} target={target:5.2f} -> pos={st.pos:7.4f} torq={st.torq:7.3f} reached={reached} status={st.status_code}")

arm.disconnect()
