from __future__ import annotations
import sys
import time
import logging
import cv2
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path.home() / "reBotArm_control_py"))
from reBotArm_control_py.actuator import RobotArm as VBArm
from reBotArm_control_py.controllers import ArmEndPos

from stratus.core.arm_driver import ArmDriver, ArmObservation, TriageCommand

logger = logging.getLogger(__name__)


@dataclass
class GripperConfig:
    motor_id: int = 7
    feedback_id: int = 0x17
    model: str = "4310"
    open_pos: float = 2.0
    close_pos: float = -5.0
    grip_pos: float = -3.0
    mit_kp: float = 10.0
    mit_kd: float = 1.0
    settle_time: float = 4.0
    approach_inset: float = 0.03


class VectorBH6ArmDriver:
    def __init__(self, config_path: str | None = None,
                 gripper: GripperConfig | None = None):
        self._arm = VBArm(config_path)
        self._endpos: ArmEndPos | None = None
        self._gripper_cfg = gripper
        self._gripper_motor = None

    def connect(self) -> None:
        from motorbridge import Mode
        self._arm.connect()
        if self._gripper_cfg is not None:
            self._init_gripper()
        self._arm.mode_mit()
        time.sleep(0.3)
        self._arm.enable()
        time.sleep(0.3)
        if self._gripper_motor is not None:
            try:
                self._gripper_motor.ensure_mode(Mode.MIT, 1000)
            except Exception:
                pass
        self._arm._request_and_poll()
        for jc in self._arm._joints:
            try:
                self._arm._motor_map[jc.name].ensure_mode(Mode.MIT, 1000)
            except Exception:
                pass
            st = self._arm._motor_map[jc.name].get_state()
            if st is not None:
                logger.info("Joint %s: status=%d pos=%.3f", jc.name, st.status_code, st.pos)
        self._endpos = ArmEndPos(self._arm)
        self._mit_kp = np.array([100.0, 100.0, 100.0, 18.0, 18.0, 18.0], dtype=np.float64)
        self._mit_kd = np.array([8.0, 8.0, 8.0, 2.0, 2.0, 2.0], dtype=np.float64)
        self._gripper_hold_target = None
        q_curr, _, _ = self._arm.get_state()
        self._endpos._q_target[:] = q_curr
        self._endpos._loop_cb = lambda ctrl, dt: self._arm_loop(ctrl, dt)
        self._arm.start_control_loop(self._endpos._loop_cb, rate=10)
        self._endpos._running = True

    def _init_gripper(self) -> None:
        cfg = self._gripper_cfg
        ctrl = self._arm._ctrl_map.get("damiao")
        if ctrl is None:
            logger.warning("No damiao controller for gripper")
            return
        try:
            from motorbridge import Mode
            mot = ctrl.add_damiao_motor(cfg.motor_id, cfg.feedback_id, cfg.model)
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
                if st is not None:
                    logger.info("Gripper attempt %d: pos=%.3f status=%d t_rot=%.1f",
                                attempt, st.pos, st.status_code, st.t_rotor)
                    if st.status_code != 0 and st.status_code != 1:
                        mot.clear_error()
                        time.sleep(0.15)
                        mot.enable()
                        time.sleep(0.3)
                    if st.status_code == 1:
                        mot.set_can_timeout_ms(60000)
                        time.sleep(0.1)
                        try:
                            for r in [15, 16]:
                                for _ in range(3):
                                    try:
                                        val = mot.read_register_f32(r, timeout_ms=500)
                                        if r == 15:
                                            logger.info("Gripper P_MIN=%.2f", val)
                                        else:
                                            logger.info("Gripper P_MAX=%.2f", val)
                                            if val < 8.0:
                                                mot.write_register_f32(16, 8.0)
                                                time.sleep(0.05)
                                                mot.store_parameters()
                                                time.sleep(0.1)
                                                logger.info("Gripper P_MAX raised to 8.0")
                                        break
                                    except Exception:
                                        time.sleep(0.1)
                        except Exception as e:
                            logger.warning("Gripper register read/write failed: %s", e)
                        mot.ensure_mode(Mode.MIT, 1000)
                        time.sleep(0.3)
                        self._gripper_motor = mot
                        logger.info("Gripper ID %d enabled in MIT mode (timeout=60s)", cfg.motor_id)
                        return
                time.sleep(0.15)

            logger.warning("Gripper ID %d failed to enable (status != 1 after 30 attempts)", cfg.motor_id)
            self._gripper_motor = mot
        except Exception as e:
            logger.warning("Gripper init failed: %s", e)

    def _arm_loop(self, ctrl, dt) -> None:
        self._arm.mit(self._endpos._q_target,
                      kp=self._mit_kp, kd=self._mit_kd, request_feedback=False)
        if self._gripper_hold_target is not None and self._gripper_motor is not None:
            try:
                for _ in range(2):
                    self._gripper_motor.request_feedback()
                    time.sleep(0.02)
                    if ctrl:
                        ctrl.poll_feedback_once()
                st = self._gripper_motor.get_state()
                if st is not None and st.status_code not in (0, 1):
                    logger.warning("[gripper] loop status=%d, recovering", st.status_code)
                    self._gripper_motor.clear_error()
                    time.sleep(0.15)
                    self._gripper_motor.enable()
                    time.sleep(0.3)
                    from motorbridge import Mode
                    self._gripper_motor.ensure_mode(Mode.MIT, 1000)
                    time.sleep(0.3)
                self._gripper_motor.send_mit(self._gripper_hold_target, 0.0,
                                              self._gripper_cfg.mit_kp,
                                              self._gripper_cfg.mit_kd, 0.0)
            except Exception:
                pass

    def _gripper_cmd(self, pos: float) -> bool:
        if self._gripper_motor is None:
            return False
        from motorbridge import Mode
        cfg = self._gripper_cfg
        ctrl = self._arm._ctrl_map.get("damiao")
        for retry in range(3):
            try:
                st = None
                for _ in range(5):
                    self._gripper_motor.request_feedback()
                    time.sleep(0.02)
                    if ctrl:
                        ctrl.poll_feedback_once()
                    st = self._gripper_motor.get_state()
                    if st is not None:
                        break
                    time.sleep(0.05)
                if st is not None and st.status_code != 1:
                    logger.warning("[gripper] status=%d before cmd, clearing error", st.status_code)
                    self._gripper_motor.clear_error()
                    time.sleep(0.3)
                    self._gripper_motor.enable()
                    time.sleep(0.3)
                    self._gripper_motor.ensure_mode(Mode.MIT, 1000)
                    time.sleep(0.5)
                    st = None
                    for _ in range(5):
                        self._gripper_motor.request_feedback()
                        time.sleep(0.02)
                        if ctrl:
                            ctrl.poll_feedback_once()
                        st = self._gripper_motor.get_state()
                        if st is not None:
                            break
                        time.sleep(0.05)
                if st is None or st.status_code != 1:
                    logger.warning("[gripper] motor not ready (status=%s), retry %d",
                                   st.status_code if st else 'None', retry)
                    continue
                self._gripper_motor.send_mit(pos, 0.0, cfg.mit_kp, cfg.mit_kd, 0.0)
                for _ in range(int(cfg.settle_time / 0.5)):
                    time.sleep(0.5)
                    self._gripper_motor.request_feedback()
                    time.sleep(0.02)
                    if ctrl:
                        ctrl.poll_feedback_once()
                    st = self._gripper_motor.get_state()
                    if st is not None:
                        logger.info("[gripper] retry=%d pos=%.3f status=%d (target=%.1f)",
                                    retry, st.pos, st.status_code, pos)
                        if st.status_code == 1:
                            return True
            except Exception as e:
                logger.warning("[gripper] cmd failed (retry %d): %s", retry, e)
        logger.warning("[gripper] cmd to %.1f failed after 3 retries", pos)
        return False

    def gripper_open(self) -> bool:
        if self._gripper_motor is None:
            logger.info("[gripper] no motor")
            return False
        cfg = self._gripper_cfg
        self._gripper_hold_target = cfg.open_pos
        ok = self._gripper_cmd(cfg.open_pos)
        logger.info("[gripper] open -> %.2f %s", cfg.open_pos,
                    "ok" if ok else "FAILED")
        return ok

    def gripper_close(self) -> None:
        if self._gripper_motor is None:
            logger.info("[gripper] no motor")
            return
        self._gripper_hold_target = self._gripper_cfg.close_pos
        self._gripper_cmd(self._gripper_cfg.close_pos)
        logger.info("[gripper] close -> %.2f", self._gripper_cfg.close_pos)

    def gripper_grip(self) -> bool:
        if self._gripper_motor is None:
            logger.info("[gripper] no motor")
            return False
        cfg = self._gripper_cfg
        ctrl = self._arm._ctrl_map.get("damiao")
        target = cfg.grip_pos

        self._gripper_hold_target = None
        ok = self._gripper_cmd(target)
        if not ok:
            logger.warning("[gripper] grip cmd failed")
            self.gripper_open()
            return False

        st = None
        for _ in range(5):
            self._gripper_motor.request_feedback()
            time.sleep(0.02)
            if ctrl:
                ctrl.poll_feedback_once()
            st = self._gripper_motor.get_state()
            if st is not None:
                break

        if st is None:
            self.gripper_open()
            return False

        actual = st.pos
        delta = abs(actual - target)

        if delta > 0.5:
            logger.info("[gripper] object at pos=%.3f (delta=%.3f) — gripped", actual, delta)
            self._gripper_hold_target = target
            return True
        else:
            logger.info("[gripper] no object pos=%.3f (delta=%.3f) — missed", actual, delta)
            self.gripper_open()
            return False
            return True

        actual = st.pos
        delta = abs(actual - target)

        if delta > 0.5:
            logger.info("[gripper] object at pos=%.3f (delta=%.3f) — gripped", actual, delta)
            return True
        else:
            logger.info("[gripper] no object pos=%.3f (delta=%.3f) — missed", actual, delta)
            return False

    def get_observation(self) -> ArmObservation:
        pos, vel, torq = self._arm.get_state()
        return ArmObservation(
            joint_positions=pos, joint_velocities=vel, joint_torques=torq,
        )

    def send_joint_positions(self, positions: npt.NDArray[np.float64]) -> None:
        self._arm.mit(pos=positions)

    def move_to_pose(self, x: float, y: float, z: float,
                     roll: float = 0, pitch: float = 0, yaw: float = 0,
                     duration: float = 4.0, frame_cb: callable = None) -> bool:
        if self._endpos is None:
            return False
        ok = self._endpos.move_to_ik(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
        if ok:
            q_ik = self._endpos._q_target.copy()
            logger.info("move_to_pose target=(%.3f, %.3f, %.3f, pitch=%.2f) IK ok (joints=%s), "
                        "slewing (%.1fs)", x, y, z, pitch, np.round(q_ik, 3), duration)
            self._arm.stop_control_loop()
            self._slew_mit(q_ik, duration, frame_cb=frame_cb)
            self._endpos._q_target[:] = q_ik
            self._arm.start_control_loop(self._endpos._loop_cb, rate=10)
            self._arm._request_and_poll()
            q, _, _ = self._arm.get_state()
            err = np.max(np.abs(q - q_ik))
            for jc in self._arm._joints:
                st = self._arm._motor_map[jc.name].get_state()
                if st is not None:
                    logger.info("  %s: pos=%.3f status=%d (target=%.3f)",
                                jc.name, st.pos, st.status_code, q_ik[self._arm._joints.index(jc)])
            logger.info("move_to_pose done (max_err=%.3f)", err)
        else:
            logger.warning("move_to_pose target=(%.3f, %.3f, %.3f, pitch=%.2f) IK failed",
                           x, y, z, pitch)
        if frame_cb:
            frame_cb()
        return ok

    def _slew_mit(self, target: npt.NDArray[np.float64], duration: float = 6.0,
                  frame_cb: callable = None) -> None:
        q_start, _, _ = self._arm.get_state()
        n = max(1, int(duration / 0.05))
        dt = duration / n
        for i in range(1, n + 1):
            t = i / n
            alpha = (1 - np.cos(t * np.pi)) / 2
            q = q_start + alpha * (target - q_start)
            self._arm.mit(pos=q, kp=self._mit_kp, kd=self._mit_kd, request_feedback=False)
            if self._gripper_hold_target is not None and self._gripper_motor is not None:
                try:
                    self._gripper_motor.send_mit(self._gripper_hold_target, 0.0,
                                                  self._gripper_cfg.mit_kp,
                                                  self._gripper_cfg.mit_kd, 0.0)
                except Exception:
                    pass
            time.sleep(dt)
            cv2.waitKey(1)
            if frame_cb:
                frame_cb()
        self._arm.mit(pos=target, kp=self._mit_kp, kd=self._mit_kd, request_feedback=False)
        if self._gripper_hold_target is not None and self._gripper_motor is not None:
            try:
                self._gripper_motor.send_mit(self._gripper_hold_target, 0.0,
                                              self._gripper_cfg.mit_kp,
                                              self._gripper_cfg.mit_kd, 0.0)
            except Exception:
                pass
        cv2.waitKey(1)
        if frame_cb:
            frame_cb()

    def execute_triage(self, command: TriageCommand, frame_cb: callable = None) -> bool:
        if not command.pickup_pose:
            return False
        pu = command.pickup_pose
        px = pu.get("x", 0)
        py = pu.get("y", 0)
        pz = pu.get("z", 0)
        pitch = pu.get("pitch", 0)
        inset = pu.get("inset")
        if inset is None:
            inset = self._gripper_cfg.approach_inset if self._gripper_cfg else 0.03

        logger.info("[triage] start: %s", command.detected_labels[:3])

        pre_z = max(pz + 0.20, 0.30)

        if not command.pickup_refined:
            logger.info("[triage] pre-approach up (z=%.3f)", pre_z)
            if not self.move_to_pose(x=px, y=py, z=pre_z, roll=0, pitch=pitch, yaw=0,
                                     duration=8.0, frame_cb=frame_cb):
                logger.warning("pre-approach failed")
                return False

            logger.info("[triage] open gripper (above object)")
            if frame_cb:
                frame_cb()
            self.gripper_open()

            logger.info("[triage] descend to approach (z=%.3f)", pz + 0.05)
            if not self.move_to_pose(x=px, y=py, z=pz + 0.05, roll=0, pitch=pitch, yaw=0,
                                     duration=5.0, frame_cb=frame_cb):
                logger.warning("approach failed")
                return False
        else:
            logger.info("[triage] refined — descend to approach")
            if not command.gripper_open_done:
                if frame_cb:
                    frame_cb()
                self.gripper_open()
            logger.info("[triage] descend to approach (z=%.3f)", pz + 0.05)
            if not self.move_to_pose(x=px, y=py, z=pz + 0.05, roll=0, pitch=pitch, yaw=0,
                                     duration=5.0, frame_cb=frame_cb):
                logger.warning("approach failed")
                return False

        logger.info("[triage] small forward & down for grip")
        if not self.move_to_pose(x=px + 0.02, y=py, z=pz, roll=0, pitch=pitch, yaw=0,
                                 duration=3.0, frame_cb=frame_cb):
            logger.warning("final approach failed")
            return False

        logger.info("[triage] grip object")
        if frame_cb:
            frame_cb()
        if not self.gripper_grip():
            logger.warning("[triage] grip missed — retry at same pose")
            if not self.gripper_grip():
                logger.warning("[triage] second grip also missed")

        logger.info("[triage] lift (z=%.3f)", pre_z)
        self.move_to_pose(x=px + 0.02, y=py, z=pre_z, roll=0, pitch=pitch, yaw=0,
                          duration=6.0, frame_cb=frame_cb)

        if command.drop_joints is not None:
            target = np.deg2rad(command.drop_joints)
            logger.info("[triage] drop to joints %s", target)
            self._arm.stop_control_loop()
            self._slew_mit(target, duration=8.0, frame_cb=frame_cb)
            self._endpos._q_target[:] = target
            self._arm.start_control_loop(self._endpos._loop_cb, rate=10)
            time.sleep(0.5)
            if frame_cb:
                frame_cb()
        elif command.drop_pose:
            self.move_to_pose(**command.drop_pose, duration=8.0, frame_cb=frame_cb)

        logger.info("[triage] release")
        if frame_cb:
            frame_cb()
        self.gripper_open()

        logger.info("[triage] return home")
        self._safe_return_home(frame_cb)
        logger.info("[triage] done")
        return True

    def _safe_return_home(self, frame_cb: callable = None) -> None:
        q, _, _ = self._arm.get_state()
        logger.info("[triage] returning home from joints %s", np.round(q, 3))

        self._arm.stop_control_loop()
        clearance = q.copy()
        clearance[2] = max(q[2], 1.2)
        if np.any(np.abs(clearance - q) > 0.01):
            logger.info("[triage] raise to clearance joints %s", np.round(clearance, 3))
            self._slew_mit(clearance, duration=3.0, frame_cb=frame_cb)
        self._slew_mit(np.zeros(6), duration=5.0, frame_cb=frame_cb)
        self._endpos._q_target[:] = np.zeros(6)
        self._arm.start_control_loop(self._endpos._loop_cb, rate=10)
        time.sleep(0.5)
        if frame_cb:
            frame_cb()
        logger.info("[triage] home")

    def disable(self) -> None:
        self._arm.disable()

    def disconnect(self) -> None:
        if self._endpos:
            self._endpos._running = False
        if self._arm:
            self._arm.disable()
            time.sleep(0.3)
            self._arm.disconnect()
        self._endpos = None
        self._arm = None

    @property
    def is_connected(self) -> bool:
        return True
