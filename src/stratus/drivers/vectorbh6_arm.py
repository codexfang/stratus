from __future__ import annotations
import sys
import time
import logging
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
    open_pos: float = 5.0
    close_pos: float = -5.0
    mit_kp: float = 2.0
    mit_kd: float = 0.1
    settle_time: float = 4.0


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
        gentle_kp = np.array([10.0, 10.0, 10.0, 3.0, 3.0, 3.0], dtype=np.float64)
        gentle_kd = np.array([2.0, 2.0, 2.0, 0.5, 0.5, 0.5], dtype=np.float64)
        q_curr, _, _ = self._arm.get_state()
        self._endpos._q_target[:] = q_curr
        self._endpos._loop_cb = lambda arm, dt: arm.mit(
            self._endpos._q_target,
            kp=gentle_kp, kd=gentle_kd, request_feedback=False,
        )
        self._arm.start_control_loop(self._endpos._loop_cb, rate=50)
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
                        mot.set_can_timeout_ms(5000)
                        time.sleep(0.1)
                        mot.ensure_mode(Mode.MIT, 1000)
                        time.sleep(0.3)
                        self._gripper_motor = mot
                        logger.info("Gripper ID %d enabled in MIT mode (timeout=5s)", cfg.motor_id)
                        return
                time.sleep(0.15)

            logger.warning("Gripper ID %d failed to enable (status != 1 after 30 attempts)", cfg.motor_id)
            self._gripper_motor = mot
        except Exception as e:
            logger.warning("Gripper init failed: %s", e)

    def _gripper_cmd(self, pos: float) -> None:
        if self._gripper_motor is None:
            return
        cfg = self._gripper_cfg
        try:
            self._gripper_motor.send_mit(pos, 0.0, cfg.mit_kp, cfg.mit_kd, 0.0)
            time.sleep(cfg.settle_time)
        except Exception as e:
            logger.warning("gripper mit failed: %s", e)

    def gripper_open(self) -> None:
        if self._gripper_motor is None:
            logger.info("[gripper] no motor")
            return
        self._gripper_cmd(self._gripper_cfg.open_pos)
        logger.info("[gripper] open -> %.2f", self._gripper_cfg.open_pos)

    def gripper_close(self) -> None:
        if self._gripper_motor is None:
            logger.info("[gripper] no motor")
            return
        self._gripper_cmd(self._gripper_cfg.close_pos)
        logger.info("[gripper] close -> %.2f", self._gripper_cfg.close_pos)

    def get_observation(self) -> ArmObservation:
        pos, vel, torq = self._arm.get_state()
        return ArmObservation(
            joint_positions=pos, joint_velocities=vel, joint_torques=torq,
        )

    def send_joint_positions(self, positions: npt.NDArray[np.float64]) -> None:
        self._arm.mit(pos=positions)

    def move_to_pose(self, x: float, y: float, z: float,
                     roll: float = 0, pitch: float = 0, yaw: float = 0,
                     duration: float = 4.0) -> bool:
        if self._endpos is None:
            return False
        ok = self._endpos.move_to_ik(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
        if ok:
            logger.info("move_to_pose target=(%.3f, %.3f, %.3f, pitch=%.2f) IK ok, slewing (%.1fs)",
                        x, y, z, pitch, duration)
            self._slew_to_joints(self._endpos._q_target, duration)
            q, _, _ = self._arm.get_state()
            logger.info("move_to_pose done (joints=%s)", np.round(q, 3))
        else:
            logger.warning("move_to_pose target=(%.3f, %.3f, %.3f, pitch=%.2f) IK failed",
                           x, y, z, pitch)
        return ok

    def _slew_to_joints(self, target: npt.NDArray[np.float64], duration: float = 4.0) -> None:
        q_start, _, _ = self._arm.get_state()
        n = max(1, int(duration / 0.05))
        for i in range(1, n + 1):
            alpha = i / n
            self._endpos._q_target[:] = q_start + alpha * (target - q_start)
            time.sleep(0.05)

    def execute_triage(self, command: TriageCommand) -> bool:
        if not command.pickup_pose:
            return False
        pu = command.pickup_pose
        approach_z = pu.get("z", 0) + 0.10

        logger.info("[triage] start: %s", command.detected_labels[:3])

        logger.info("[triage] approach (z=%.3f)", approach_z)
        if not self.move_to_pose(x=pu.get("x", 0), y=pu.get("y", 0), z=approach_z,
                                 roll=pu.get("roll", 0), pitch=pu.get("pitch", 0), yaw=pu.get("yaw", 0),
                                 duration=4.0):
            logger.warning("approach failed")
            return False

        logger.info("[triage] open gripper")
        self.gripper_open()

        logger.info("[triage] descend (z=%.3f)", pu.get("z", 0))
        self.move_to_pose(**pu, duration=4.0)
        time.sleep(0.3)
        self.gripper_close()
        time.sleep(0.3)

        lift_z = pu.get("z", 0) + 0.10
        logger.info("[triage] lift (z=%.3f)", lift_z)
        self.move_to_pose(x=pu.get("x", 0), y=pu.get("y", 0), z=lift_z,
                          roll=pu.get("roll", 0), pitch=pu.get("pitch", 0), yaw=pu.get("yaw", 0),
                          duration=4.0)
        time.sleep(0.3)

        if command.drop_joints is not None:
            target = np.deg2rad(command.drop_joints)
            logger.info("[triage] drop to joints %s", target)
            self._slew_to_joints(target, duration=4.0)
            time.sleep(0.5)
        elif command.drop_pose:
            self.move_to_pose(**command.drop_pose, duration=4.0)

        logger.info("[triage] release")
        self.gripper_open()

        logger.info("[triage] return home")
        self._slew_to_joints(np.zeros(6), duration=4.0)
        time.sleep(0.5)
        logger.info("[triage] done")
        return True

    def disable(self) -> None:
        self._arm.disable()

    def disconnect(self) -> None:
        if self._endpos:
            self._endpos.end()
            self._endpos = None
        elif self._arm:
            self._arm.disconnect()
        self._arm = None

    @property
    def is_connected(self) -> bool:
        return True
