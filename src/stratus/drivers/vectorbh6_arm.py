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
    feedback_id: int = 7
    model: str = "4310"
    open_pos: float = 0.6
    close_pos: float = 0.0
    vlim: float = 3.0
    settle_time: float = 0.8
    pos_kp: float = 50.0
    pos_ki: float = 1.0
    vel_kp: float = 0.0008
    vel_ki: float = 0.002


class VectorBH6ArmDriver:
    def __init__(self, config_path: str | None = None,
                 gripper: GripperConfig | None = None):
        self._arm = VBArm(config_path)
        self._endpos: ArmEndPos | None = None
        self._gripper_cfg = gripper
        self._gripper_motor = None

    def connect(self) -> None:
        self._arm.connect()
        self._arm.mode_pos_vel()
        if self._gripper_cfg is not None:
            self._init_gripper()
        else:
            self._arm.enable()
        self._endpos = ArmEndPos(self._arm)
        self._endpos.start()

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
            self._arm.enable()
            time.sleep(0.3)

            for attempt in range(30):
                ctrl.poll_feedback_once()
                mot.request_feedback()
                time.sleep(0.05)
                ctrl.poll_feedback_once()
                st = mot.get_state()
                if st is not None:
                    logger.info("Gripper attempt %d: pos=%.3f status=%d t_rot=%.1f",
                                attempt, st.pos, st.status_code, st.t_rotor)
                    if st.status_code == 12:
                        mot.write_register_f32(2, 120.0)
                        time.sleep(0.1)
                        mot.clear_error()
                        time.sleep(0.1)
                        self._arm.enable()
                        time.sleep(0.2)
                    if st.status_code == 1:
                        mot.write_register_f32(25, cfg.vel_kp)
                        mot.write_register_f32(26, cfg.vel_ki)
                        mot.write_register_f32(27, cfg.pos_kp)
                        mot.write_register_f32(28, cfg.pos_ki)
                        mot.ensure_mode(Mode.POS_VEL, 1000)
                        time.sleep(0.2)
                        self._gripper_motor = mot
                        logger.info("Gripper ID %d enabled in POS_VEL mode", cfg.motor_id)
                        return
                time.sleep(0.1)

            logger.warning("Gripper ID %d failed to enable (status != 1 after 30 attempts)", cfg.motor_id)
            self._gripper_motor = mot
        except Exception as e:
            logger.warning("Gripper init failed: %s", e)

    def _gripper_cmd(self, pos: float) -> None:
        if self._gripper_motor is None:
            return
        cfg = self._gripper_cfg
        try:
            self._gripper_motor.send_pos_vel(pos, cfg.vlim)
            time.sleep(cfg.settle_time)
        except Exception as e:
            logger.warning("gripper pos_vel failed: %s", e)

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
        self._arm.pos_vel(pos=positions)

    def move_to_pose(self, x: float, y: float, z: float,
                     roll: float = 0, pitch: float = 0, yaw: float = 0) -> bool:
        if self._endpos is None:
            return False
        ok = self._endpos.move_to_traj(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=2.0)
        if ok:
            deadline = time.monotonic() + 5.0
            while self._endpos._moving and time.monotonic() < deadline:
                time.sleep(0.05)
        return ok

    def execute_triage(self, command: TriageCommand) -> bool:
        if not command.pickup_pose:
            return False
        pu = command.pickup_pose
        approach_z = pu.get("z", 0) + 0.07

        self.gripper_open()

        if not self.move_to_pose(x=pu.get("x", 0), y=pu.get("y", 0), z=approach_z,
                                 roll=pu.get("roll", 0), pitch=pu.get("pitch", 0), yaw=pu.get("yaw", 0)):
            logger.warning("approach failed")
            return False
        time.sleep(0.3)

        self.move_to_pose(**pu)
        time.sleep(0.5)
        self.gripper_close()

        lift_z = pu.get("z", 0) + 0.07
        self.move_to_pose(x=pu.get("x", 0), y=pu.get("y", 0), z=lift_z,
                          roll=pu.get("roll", 0), pitch=pu.get("pitch", 0), yaw=pu.get("yaw", 0))
        time.sleep(0.3)

        if command.drop_joints is not None:
            target = np.deg2rad(command.drop_joints)
            self.send_joint_positions(target)
            time.sleep(1.5)
        elif command.drop_pose:
            self.move_to_pose(**command.drop_pose)
            time.sleep(0.3)
        self.gripper_open()

        if command.drop_joints is not None:
            self.send_joint_positions(np.deg2rad([0, 0, 0, 0, 0, 0]))
        else:
            self.send_joint_positions(np.zeros(6))
        time.sleep(1)
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
