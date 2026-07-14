#!/usr/bin/env python3
"""Gripper test — uses POS_VEL mode with proper PI gains (matching library)."""
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
mot = ctrl.add_damiao_motor(7, 7, "4310")

# Read initial state
mot.request_feedback()
ctrl.poll_feedback_once()
st = mot.get_state()
if st:
    print(f"INIT: pos={st.pos:.4f} status={st.status_code} t_rot={st.t_rotor:.1f}°C")

# Clear error + enable
print("Clearing error...")
mot.clear_error()
time.sleep(0.3)
ctrl.enable_all()
time.sleep(0.3)

for attempt in range(30):
    mot.request_feedback()
    ctrl.poll_feedback_once()
    st = mot.get_state()
    if st:
        print(f"  Attempt {attempt}: status={st.status_code} t_rot={st.t_rotor:.1f}°C")
        if st.status_code == 1:
            print(">>> ENABLED!\n")
            break
        if st.status_code == 12:
            # Still over-temp - increase threshold
            print("  Over-temp, raising OT threshold to 120°C...")
            mot.write_register_f32(2, 120.0)
            time.sleep(0.2)
            mot.clear_error()
            time.sleep(0.2)
            ctrl.enable_all()
            time.sleep(0.3)
    time.sleep(0.15)

if st and st.status_code == 1:
    # Stay in POS_VEL mode (arm already set it)
    arm.mode_pos_vel()
    time.sleep(0.2)

    # Write POS_VEL gains (same as library defaults)
    print("Setting POS_VEL gains (pos_kp=50, pos_ki=1, vel_kp=0.0008, vel_ki=0.002)")
    mot.write_register_f32(25, 0.0008)  # vel_kp
    mot.write_register_f32(26, 0.002)   # vel_ki
    mot.write_register_f32(27, 50.0)    # pos_kp
    mot.write_register_f32(28, 1.0)     # pos_ki
    time.sleep(0.1)

    # Ensure POS_VEL mode on gripper motor
    mot.ensure_mode(Mode.POS_VEL, 1000)
    time.sleep(0.2)

    print("\n=== POS_VEL mode - sweeping positions ===")
    positions = [0.5, 0.0, 0.8, 0.0, 0.3, 0.0, -0.2, 0.0]
    for target in positions:
        mot.send_pos_vel(target, 3.0)
        time.sleep(1.0)
        mot.request_feedback()
        ctrl.poll_feedback_once()
        st = mot.get_state()
        if st:
            print(f"  target={target:5.2f} -> pos={st.pos:.4f} torq={st.torq:.3f} status={st.status_code} t_rot={st.t_rotor:.1f}°C")
        else:
            print(f"  target={target:5.2f} -> NO FEEDBACK")
else:
    print(f"\nStuck at status={st.status_code if st else None}.")

arm.disconnect()
