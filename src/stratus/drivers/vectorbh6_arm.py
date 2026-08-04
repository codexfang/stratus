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
    open_pos: float = 4.0           # wide open — confirmed working range
    close_pos: float = -5.0         # fully closed (used after drop to reset)
    grip_pos: float = -2.0          # gentle hold — stops when object blocks movement
    mit_kp: float = 10.0
    mit_kd: float = 1.0
    settle_time: float = 2.0        # seconds to wait after sending gripper command
    grip_delta_threshold: float = 0.5  # min pos delta vs target to confirm object held


# Scan pose joint angles (radians).
# Joint layout: [base_rotation, shoulder, elbow, forearm_roll, wrist_pitch, wrist_roll]
#
# CRITICAL: Testing shows joint directions may be inverted from documentation
# - joint[0] = 0.0: base centered (MUST be zero)
# - joint[1] = -0.3: shoulder raised
# - joint[2] = -0.8: elbow extends forward
# - joint[3] = 0.0: no forearm roll
# - joint[4] = -1.4: wrist pitch NEGATIVE (testing shows negative = DOWN)
# - joint[5] = 0.0: no wrist roll
DEFAULT_SCAN_JOINTS = [0.0, -0.3, -0.8, 0.0, -1.4, 0.0]


class VectorBH6ArmDriver:
    def __init__(self, config_path: str | None = None,
                 gripper: GripperConfig | None = None):
        self._arm = VBArm(config_path)
        self._endpos: ArmEndPos | None = None
        self._gripper_cfg = gripper if gripper is not None else GripperConfig()
        self._gripper_motor = None
        # Scan pose used both at startup and when returning home after a pick.
        # Set via move_to_scan_pose(scan_joints=...) before connect() if needed,
        # or override with set_scan_joints() after construction.
        self._scan_joints: list[float] = list(DEFAULT_SCAN_JOINTS)

    def set_scan_joints(self, joints: list[float]) -> None:
        """Override the scan pose used at startup and after each pick."""
        if len(joints) != 6:
            raise ValueError(f"scan_joints must have 6 values, got {len(joints)}")
        self._scan_joints = list(joints)
        logger.info("Scan joints updated: %s", np.round(self._scan_joints, 3))

    def connect(self) -> None:
        from motorbridge import Mode
        self._arm.connect()
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
        logger.info("[connect] current joints on connect: %s", np.round(q_curr, 3))

        # FORCE joint[0] to exactly 0.0 — base MUST face forward, no tolerance
        if abs(q_curr[0]) > 0.01:
            logger.warning("[connect] joint[0] is %.3f (NOT zero) — forcing to 0.0", q_curr[0])
            q_zero = q_curr.copy()
            q_zero[0] = 0.0
            # Send command 3 times to ensure it takes
            for _ in range(3):
                self._arm.mit(pos=q_zero, kp=self._mit_kp, kd=self._mit_kd, request_feedback=False)
                time.sleep(0.3)
            q_curr, _, _ = self._arm.get_state()
            logger.info("[connect] joint[0] after force-reset: %.3f", q_curr[0])
        else:
            logger.info("[connect] joint[0] already at zero (%.3f)", q_curr[0])

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

            logger.warning("Gripper ID %d failed to enable after 30 attempts", cfg.motor_id)
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
                for _ in range(int(cfg.settle_time / 0.2)):
                    time.sleep(0.2)
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
            logger.info("[gripper] no motor — skip open")
            return False
        cfg = self._gripper_cfg
        self._gripper_hold_target = cfg.open_pos
        ok = self._gripper_cmd(cfg.open_pos)
        logger.info("[gripper] open -> %.2f %s", cfg.open_pos, "ok" if ok else "FAILED")
        return ok

    def gripper_close(self) -> None:
        if self._gripper_motor is None:
            logger.info("[gripper] no motor — skip close")
            return
        self._gripper_hold_target = self._gripper_cfg.close_pos
        self._gripper_cmd(self._gripper_cfg.close_pos)
        logger.info("[gripper] close -> %.2f", self._gripper_cfg.close_pos)

    def gripper_grip(self, suppress_open: bool = False) -> bool:
        """Close gripper onto object. Returns True if an object was detected in the grip."""
        if self._gripper_motor is None:
            logger.info("[gripper] no motor — skip grip")
            return False
        cfg = self._gripper_cfg
        ctrl = self._arm._ctrl_map.get("damiao")
        target = cfg.grip_pos

        self._gripper_hold_target = None
        ok = self._gripper_cmd(target)
        if not ok:
            logger.warning("[gripper] grip cmd failed")
            if not suppress_open:
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
            if not suppress_open:
                self.gripper_open()
            return False

        actual = st.pos
        delta = abs(actual - target)

        # delta > threshold means gripper was blocked by an object before reaching full close
        if delta > cfg.grip_delta_threshold:
            logger.info("[gripper] GRIPPED object: pos=%.3f (delta=%.3f threshold=%.3f)",
                        actual, delta, cfg.grip_delta_threshold)
            self._gripper_hold_target = actual  # hold at blocked position
            return True
        else:
            logger.info("[gripper] MISSED: pos=%.3f (delta=%.3f threshold=%.3f) — no object",
                        actual, delta, cfg.grip_delta_threshold)
            if not suppress_open:
                self.gripper_open()
            return False

    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._arm.get_state()

    def get_observation(self) -> ArmObservation:
        pos, vel, torq = self._arm.get_state()
        return ArmObservation(
            joint_positions=pos, joint_velocities=vel, joint_torques=torq,
        )

    def send_joint_positions(self, positions: npt.NDArray[np.float64]) -> None:
        self._arm.mit(pos=positions)

    def move_to_joints(self, q_target: np.ndarray, duration: float = 4.0,
                       frame_cb: callable = None) -> None:
        """Move to target joint positions using smooth cosine-interpolated slewing."""
        q_target = np.asarray(q_target, dtype=np.float64)
        self._arm.stop_control_loop()
        self._slew_mit(q_target, duration, frame_cb=frame_cb)
        self._endpos._q_target[:] = q_target.copy()
        self._arm.start_control_loop(self._endpos._loop_cb, rate=10)

    def move_to_scan_pose(self, scan_joints: list[float] | None = None,
                          duration: float = 4.0,
                          frame_cb: callable = None) -> None:
        """Raise arm to the scan pose so the top-mounted camera can see the
        full workspace below/in front of it.

        scan_joints: 6-element list of target joint angles in radians.
                     If provided, also saves them as the new default so
                     _safe_return_home uses the same pose.
        duration:    time to complete the move (seconds).
        """
        if scan_joints is not None:
            self.set_scan_joints(scan_joints)
        target = np.array(self._scan_joints, dtype=np.float64)
        logger.info("[scan] BEFORE move — current joints: %s", 
                    np.round(self._arm.get_state()[0], 3))
        logger.info("[scan] TARGET scan pose: %s (%.1fs)", np.round(target, 3), duration)
        logger.info("[scan] joint[0]=%+.4f (MUST be 0 for forward), joint[4]=%+.4f (negative=DOWN)",
                    target[0], target[4])
        self.move_to_joints(target, duration=duration, frame_cb=frame_cb)
        final_q, _, _ = self._arm.get_state()
        logger.info("[scan] AFTER move — final joints: %s", np.round(final_q, 3))
        logger.info("[scan] joint[0] error: %+.4f (should be ~0)", final_q[0])

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
        """Cosine-interpolated slew to target joint positions."""
        q_start, _, _ = self._arm.get_state()
        n = max(1, int(duration / 0.05))
        dt = duration / n
        for i in range(1, n + 1):
            t = i / n
            alpha = (1 - np.cos(t * np.pi)) / 2
            q = q_start + alpha * (target - q_start)
            # Enforce joint[0]=0 if target[0]=0 to prevent base rotation drift
            if abs(target[0]) < 0.001:
                q[0] = 0.0
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
        # Final target command with joint[0] enforcement
        target_final = target.copy()
        if abs(target[0]) < 0.001:
            target_final[0] = 0.0
        self._arm.mit(pos=target_final, kp=self._mit_kp, kd=self._mit_kd, request_feedback=False)
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
        """
        Full pick-and-place sequence.

        The engine calls this AFTER _approach_and_pick() has moved the arm to
        HOVER_Z above the object with gripper already open. Sequence:

            1. Descend to approach height (object + 8 cm)    — gripper still open
            2. Open gripper explicitly (confirm fully open near object)
            3. Nudge forward 3 cm into the object             — gripper wraps around it
            4. Close gripper firmly to hold
            5. Retry once lower if missed
            6. Lift to clearance height
            7. Transport to drop location
            8. Open gripper to release
            9. Return to scan pose (home)
        """
        if not command.pickup_pose:
            logger.warning("[triage] no pickup_pose — aborting")
            return False

        logger.info("[triage] start: %s", command.detected_labels[:3])

        pu = command.pickup_pose
        px = pu.get("x", 0.0)
        py = pu.get("y", 0.0)
        pz = pu.get("z", 0.10)      # object surface height
        pitch = pu.get("pitch", 0.4)

        # ── 1. Descend to just above the object ───────────────────────────
        approach_z = pz + 0.08      # 8 cm above object — gripper fingers clear the top
        logger.info("[triage] descend to approach z=%.3f", approach_z)
        if not self.move_to_pose(x=px, y=py, z=approach_z,
                                 roll=0, pitch=pitch, yaw=0,
                                 duration=3.0, frame_cb=frame_cb):
            logger.warning("[triage] approach IK failed — joint-space descent")
            q, _, _ = self._arm.get_state()
            q_down = q.copy()
            q_down[2] += 0.35
            self.move_to_joints(q_down, duration=2.5, frame_cb=frame_cb)

        # ── 2. Open gripper here — fingers spread around the object ───────
        if frame_cb:
            frame_cb()
        logger.info("[triage] opening gripper beside object")
        self.gripper_open()

        # ── 3. Descend to grip height ─────────────────────────────────────
        logger.info("[triage] descend to grip z=%.3f", pz)
        if not self.move_to_pose(x=px, y=py, z=pz,
                                 roll=0, pitch=pitch, yaw=0,
                                 duration=2.0, frame_cb=frame_cb):
            q, _, _ = self._arm.get_state()
            q_grip = q.copy()
            q_grip[2] += 0.22
            self.move_to_joints(q_grip, duration=1.8, frame_cb=frame_cb)

        # ── 4. Nudge forward 3 cm into the object ────────────────────────
        # This pushes the cup into the gripper fingers so the close has
        # something to bite on rather than closing on air beside the cup.
        nudge_x = px + 0.03
        logger.info("[triage] nudge forward to x=%.3f", nudge_x)
        if not self.move_to_pose(x=nudge_x, y=py, z=pz,
                                 roll=0, pitch=pitch, yaw=0,
                                 duration=1.2, frame_cb=frame_cb):
            # Joint-space nudge: increase shoulder extension slightly
            q, _, _ = self._arm.get_state()
            q_nudge = q.copy()
            q_nudge[1] += 0.06
            self.move_to_joints(q_nudge, duration=1.0, frame_cb=frame_cb)

        # ── 5. Close gripper firmly ───────────────────────────────────────
        if frame_cb:
            frame_cb()
        logger.info("[triage] closing gripper...")
        gripped = self.gripper_grip(suppress_open=False)

        if not gripped:
            # One retry: descend another 2 cm and nudge again
            logger.warning("[triage] grip missed — descending 2 cm and retrying")
            q, _, _ = self._arm.get_state()
            q_lower = q.copy()
            q_lower[2] += 0.10
            self.move_to_joints(q_lower, duration=0.8, frame_cb=frame_cb)
            if frame_cb:
                frame_cb()
            gripped = self.gripper_grip(suppress_open=False)
            if not gripped:
                logger.warning("[triage] second grip missed — aborting pick")
                self._safe_return_home(frame_cb)
                return False

        # ── 6. Lift to clearance height ───────────────────────────────────
        lift_z = max(pz + 0.30, 0.34)
        logger.info("[triage] lifting to z=%.3f", lift_z)
        if not self.move_to_pose(x=nudge_x, y=py, z=lift_z,
                                 roll=0, pitch=pitch, yaw=0,
                                 duration=4.0, frame_cb=frame_cb):
            q, _, _ = self._arm.get_state()
            q_lift = q.copy()
            q_lift[2] = min(q_lift[2] - 0.7, -0.5)  # straighten elbow = raise end-effector
            q_lift[1] -= 0.15
            self.move_to_joints(q_lift, duration=3.5, frame_cb=frame_cb)

        # ── 7. Transport to drop location ─────────────────────────────────
        if command.drop_joints is not None:
            target_q = np.deg2rad(np.array(command.drop_joints, dtype=np.float64))
            logger.info("[triage] transporting to drop joints %s", np.round(target_q, 3))
            self.move_to_joints(target_q, duration=5.0, frame_cb=frame_cb)
        elif command.drop_pose:
            logger.info("[triage] transporting to drop pose %s", command.drop_pose)
            self.move_to_pose(**command.drop_pose, duration=6.0, frame_cb=frame_cb)
        else:
            logger.warning("[triage] no drop target — dropping in place")

        # ── 8. Release ────────────────────────────────────────────────────
        if frame_cb:
            frame_cb()
        logger.info("[triage] releasing into bin")
        self.gripper_open()
        time.sleep(0.4)     # let object settle before arm swings away

        # Close gripper back to neutral so it's ready for next pick
        if frame_cb:
            frame_cb()
        logger.info("[triage] closing gripper to neutral")
        self.gripper_close()
        time.sleep(0.3)

        # ── 9. Return home / scan pose ────────────────────────────────────
        self._safe_return_home(frame_cb)
        logger.info("[triage] done")
        return True

    def _safe_return_home(self, frame_cb: callable = None) -> None:
        """Return arm to the scan pose via a safe clearance arc.
        We go to scan pose (not joint zeros) so the camera is immediately
        ready for the next detection cycle."""
        q, _, _ = self._arm.get_state()
        logger.info("[triage] returning to scan pose from %s", np.round(q, 3))

        self._arm.stop_control_loop()

        # First raise elbow to a safe clearance before rotating base/shoulder,
        # to avoid sweeping the arm through objects on the table.
        clearance = q.copy()
        clearance[2] = min(q[2], -0.4)   # straighten elbow (negative = up on this arm)
        if np.any(np.abs(clearance - q) > 0.02):
            logger.info("[triage] elbow clearance %s", np.round(clearance, 3))
            self._slew_mit(clearance, duration=2.5, frame_cb=frame_cb)

        # Slew to scan pose — use the instance's stored scan joints so
        # this always matches whatever was set at startup or via CLI.
        scan_q = np.array(self._scan_joints, dtype=np.float64)
        self._slew_mit(scan_q, duration=5.0, frame_cb=frame_cb)
        self._endpos._q_target[:] = scan_q
        self._arm.start_control_loop(self._endpos._loop_cb, rate=10)
        time.sleep(0.5)
        if frame_cb:
            frame_cb()
        logger.info("[triage] scan pose reached")

    def disable(self) -> None:
        self._arm.disable()

    def stop_control_loop(self) -> None:
        self._arm.stop_control_loop()

    def start_control_loop(self, callback, rate: int = 10) -> None:
        self._arm.start_control_loop(callback, rate=rate)

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
