from __future__ import annotations
import sys
import os
import re
import glob
import time
import subprocess
import logging
import threading
import cv2
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path.home() / "reBotArm_control_py"))
from reBotArm_control_py.actuator import RobotArm as VBArm
from reBotArm_control_py.controllers import ArmEndPos
from motorbridge import Mode

from stratus.core.arm_driver import ArmDriver, ArmObservation, TriageCommand

logger = logging.getLogger(__name__)


@dataclass
class GripperConfig:
    motor_id: int = 7
    feedback_id: int = 0x17
    model: str = "4310"
    open_pos: float = 9.0           # ~2.5x wider than 3.6: DM4310 MIT range is ±12.5 rad; 3.6 was only ~0.57 rev (a few mm of finger travel). 9.0 rad ≈ 1.4 rev — fingers sweep 2.5x further
    open_limit: float = 12.0        # hard clamp just inside the ±12.5 rad model limit — never over-limit-faults
    close_pos: float = 0.0          # neutral park after drop (safe middle, won't fault)
    grip_pos: float = -0.8          # gentle close within range (-2.5 over-limit FAULTS motor)
    mit_kp: float = 12.0            # stiffer than 8: the proven 4310 wrist joints run kp 18/kd 2; 12 drives the big travel without slamming
    mit_kd: float = 1.5
    settle_time: float = 0.4        # thread holds the position continuously — no need to idle 2s
    # ensure_mode uses the EXTENDED CAN protocol (register 10 write). This
    # gripper never acks it ("register 10 write ack not received") and the stray
    # frame may corrupt the MIT hold — so it defaults OFF for the gripper. The
    # DM4310 moves on MIT frames alone once enabled.
    # ensure_mode is OFF — proven harmful on this motor: it writes CTRL_MODE
    # (register 10) and the motor then ignores MIT frames entirely (no motion
    # at all), verified on hardware. Do NOT re-enable.
    use_ensure_mode: bool = False
    grip_delta_threshold: float = 0.5  # delta vs target to confirm object held


# Scan pose joint angles (radians).
# Physical joint limits confirmed from testing:
#   idx2 (elbow) cannot reach +0.4 when shoulder is raised — hits mechanical limit at ~0.08
#   idx2 negative = elbow extends arm FORWARD (away from base) — no limit issue
#   idx1=-0.5, idx2=-0.5 confirmed working: arm extends up and forward
#   idx1=-1.2 folds arm backwards (too far)
#   idx1=-0.6, idx2=+0.4 → elbow only reaches 0.08, grippers droop down
#
# Solution: moderate shoulder raise + negative elbow to extend FORWARD
# Camera on top of a forward-extended arm naturally looks slightly downward.
#   idx0 =  0.0: base faces FORWARD
#   idx1 = -0.5: shoulder raises arm (moderate, won't fold backwards)
#   idx2 = -0.5: elbow extends arm FORWARD (whole arm rises and reaches out)
#   idx3 =  0.0: no forearm roll
#   idx4 =  0.0: neutral
#   idx5 =  0.0: NO wrist roll
DEFAULT_SCAN_JOINTS = [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]

DEFAULT_ARM_YAML = str(Path.home()
                       / "reBotArm_control_py" / "config" / "arm.yaml")


def find_dm_serial_channels() -> list[str]:
    """List USB-to-CAN serial ports present on this machine."""
    found: list[str] = []
    for pat in ("/dev/cu.usbmodem*", "/dev/tty.usbmodem*",
                "/dev/cu.usbserial-*", "/dev/cu.usbserial*",
                "/dev/cu.usb*", "/dev/ttyACM*"):
        found.extend(glob.glob(pat))
    seen: set[str] = set()
    ordered: list[str] = []
    for c in found:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def resolve_arm_channel(preferred: str | None) -> str | None:
    """Pick a working serial channel for the arm.

    Order: 1) STRATUS_ARM_CHANNEL env var, 2) the configured/preferred port if
    it exists, 3) the first detected USB serial device. Returns None if none.
    """
    env = os.environ.get("STRATUS_ARM_CHANNEL")
    if env and (Path(env).exists() or env.startswith("/dev/tty")):
        return env
    if preferred and Path(preferred).exists():
        return preferred
    devs = find_dm_serial_channels()
    if devs:
        logger.info("Detected DM serial channel -> %s (%s)",
                    devs[0], devs)
        return devs[0]
    return None


def _usb_report() -> str:
    """Best-effort dump of what macOS believes is on the USB bus right now."""
    lines: list[str] = []
    try:
        devs = find_dm_serial_channels()
        lines.append(f"  /dev serial candidates found: {devs or 'NONE'}")
    except Exception:
        pass
    try:
        out = subprocess.run(["system_profiler", "SPUSBDataType"],
                             capture_output=True, text=True, timeout=20)
        txt = (out.stdout or "").strip()
        if txt:
            lines.append("  USB bus (system_profiler):")
            for ln in txt.splitlines()[:20]:
                lines.append(f"    {ln}")
        else:
            lines.append("  USB bus (system_profiler): (empty / no bus info)")
    except Exception as e:
        lines.append(f"  USB bus query failed: {e}")
    return "\n".join(lines)


def make_arm(config_path: str | None) -> VBArm:
    """Construct the arm, auto-resolving/overriding the serial channel so a
    changed or missing hardcoded port never blocks startup."""
    if config_path is not None:
        return VBArm(config_path)

    preferred = "/dev/ttyACM0"
    try:
        txt = Path(DEFAULT_ARM_YAML).read_text()
        m = re.search(r'(?m)^channel:\s*(\S+)', txt)
        if m:
            preferred = m.group(1).strip()
    except Exception:
        pass

    channel = resolve_arm_channel(preferred)
    if channel is None:
        raise RuntimeError(
            "[arm] NO USB-to-CAN serial device found. Plug in the dongle and "
            "re-run. stratus looks for /dev/cu.usbmodem*, /dev/tty.usbmodem*, "
            "/dev/cu.usbserial*, /dev/ttyACM*.\n"
            f"{_usb_report()}")
    if channel != preferred:
        patched = re.sub(r'(?m)^channel:\s*\S+', f'channel: {channel}', txt)
        tmp = Path("/tmp/") / "stratus_arm_autodetect.yaml"
        tmp.write_text(patched)
        logger.info("[arm] channel %s not present — using detected %s "
                    "(patched config %s)", preferred, channel, tmp)
        return VBArm(str(tmp))
    return VBArm(config_path)


class VectorBH6ArmDriver:
    def __init__(self, config_path: str | None = None,
                 gripper: GripperConfig | None = None,
                 prime_on_connect: bool = True):
        self._arm = make_arm(config_path)
        self._endpos: ArmEndPos | None = None
        self._gripper_cfg = gripper if gripper is not None else GripperConfig()
        self._gripper_motor = None
        self._gripper_thread = None
        self._gripper_stop = True
        self._gripper_limp_active = False
        self._prime_on_connect = prime_on_connect
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
        if self._gripper_motor is not None and self._gripper_cfg.use_ensure_mode:
            try:
                self._gripper_motor.ensure_mode(Mode.MIT, 1000)
            except Exception:
                pass
        self._arm._request_and_poll()

        # Check each joint and attempt fault recovery for any that failed to enable
        for jc in self._arm._joints:
            mot = self._arm._motor_map.get(jc.name)
            if mot is None:
                continue
            try:
                mot.ensure_mode(Mode.MIT, 1000)
            except Exception:
                pass
            st = mot.get_state()
            if st is not None:
                logger.info("Joint %s: status=%d pos=%.3f", jc.name, st.status_code, st.pos)
                # status_code=12 means fault — attempt clear+re-enable
                if st.status_code not in (0, 1):
                    logger.warning("[connect] %s faulted (status=%d) — clearing error",
                                   jc.name, st.status_code)
                    for _ in range(3):
                        try:
                            mot.clear_error()
                            time.sleep(0.3)
                            mot.enable()
                            time.sleep(0.3)
                            mot.ensure_mode(Mode.MIT, 1000)
                            time.sleep(0.2)
                        except Exception:
                            pass
                        mot.request_feedback()
                        time.sleep(0.05)
                        st2 = mot.get_state()
                        if st2 is not None and st2.status_code == 1:
                            logger.info("[connect] %s recovered (status=%d)", jc.name, st2.status_code)
                            break
                        logger.warning("[connect] %s still faulted (status=%s)",
                                       jc.name, st2.status_code if st2 else 'None')

        self._endpos = ArmEndPos(self._arm)
        self._mit_kp = np.array([100.0, 100.0, 100.0, 18.0, 18.0, 18.0], dtype=np.float64)
        self._mit_kd = np.array([8.0, 8.0, 8.0, 2.0, 2.0, 2.0], dtype=np.float64)
        self._gripper_hold_target = None
        q_curr, _, _ = self._arm.get_state()
        logger.info("[connect] current joints on connect: %s", np.round(q_curr, 3))

        self._endpos._q_target[:] = q_curr
        self._endpos._loop_cb = lambda ctrl, dt: self._arm_loop(ctrl, dt)
        self._arm.start_control_loop(self._endpos._loop_cb, rate=10)
        self._endpos._running = True

        # Dedicated 50Hz gripper stream so the gripper always has continuous
        # MIT frames regardless of what the 10Hz arm control loop is doing.
        self._start_gripper_stream()

        # Prime the gripper (open->grip->open a few times) so it is definitely
        # in MIT mode and moving before the first pick. Skip when disabled
        # (e.g. the loose-gripper utility must not yank the fingers around).
        if self._prime_on_connect:
            try:
                self.prime_gripper()
            except Exception as e:
                logger.warning("[connect] gripper prime failed (ignored): %s", e)

    def _init_gripper(self) -> None:
        cfg = self._gripper_cfg
        ctrl = self._arm._ctrl_map.get("damiao")
        if ctrl is None:
            logger.warning("No damiao controller for gripper")
            return
        try:
            from motorbridge import Mode
            mot = ctrl.add_damiao_motor(cfg.motor_id, cfg.feedback_id, cfg.model)

            # Clear errors / fault recovery
            try:
                mot.clear_error()
            except Exception:
                pass
            time.sleep(0.1)
            try:
                mot.enable()
                time.sleep(0.2)
            except Exception:
                pass
            if cfg.use_ensure_mode:
                try:
                    mot.ensure_mode(Mode.MIT, 1000)
                    time.sleep(0.2)
                except Exception:
                    pass

            # Quick best-effort feedback poll (logging only — do NOT block on it).
            try:
                mot.request_feedback()
                time.sleep(0.02)
                ctrl.poll_feedback_once()
                st = mot.get_state()
                if st is not None:
                    logger.info("Gripper ID %d feedback: pos=%.3f status=%d",
                                cfg.motor_id, st.pos, st.status_code)
                else:
                    logger.info("Gripper ID %d: no feedback yet — commands will still be sent (fire-and-forget)",
                                cfg.motor_id)
            except Exception:
                pass

            try:
                mot.set_can_timeout_ms(60000)
            except Exception:
                pass

            self._gripper_motor = mot
            self._widen_gripper_pmax()
            logger.info("Gripper ID %d ready in MIT mode (timeout=60s)", cfg.motor_id)
        except Exception as e:
            logger.warning("Gripper init failed: %s", e)

    def _arm_loop(self, ctrl, dt) -> None:
        # NOTE: the gripper is NOT driven here — a dedicated thread
        # (self._gripper_loop) streams its MIT frames at ~50Hz so this 10Hz
        # loop's feedback polling can never starve the gripper's continuous
        # position stream (a single send_mit only pulses a Damiao motor).
        self._arm.mit(self._endpos._q_target,
                      kp=self._mit_kp, kd=self._mit_kd, request_feedback=False)

    def _gripper_loop(self) -> None:
        """Continuous MIT stream for the gripper at ~50Hz — PURE SEND.

        Feedback on this motor NEVER arrives (it does not ack the register
        protocol), yet request_feedback()+poll_feedback_once() sit on the
        shared arm serial. pöll waits for a response that never comes and can
        block on a bus timeout, starving the MIT stream EXACTLY while the arm
        is slewing — which made fingers open "just a little" in the pipeline
        but fully when the arm was idle. So: no request, no poll — just a
        fixed-rate position stream. The motor moves on MIT frames alone.
        """
        cfg = self._gripper_cfg
        hard_error_streak = 0
        while not self._gripper_stop:
            mot = self._gripper_motor
            target = self._gripper_hold_target
            if mot is not None and not self._gripper_limp_active and target is not None:
                # Hard clamp (see _gripper_cmd): >open_limit over-limit-FAULTS
                # the motor, which then silently ignores every frame.
                target = min(max(target, -2.4), cfg.open_limit)
                try:
                    mot.send_mit(target, 0.0, cfg.mit_kp, cfg.mit_kd, 0.0)
                    hard_error_streak = 0
                except Exception as e:
                    if hard_error_streak >= 3:
                        try:
                            mot.enable()
                            time.sleep(0.1)
                            hard_error_streak = 0
                        except Exception:
                            pass
                    else:
                        hard_error_streak += 1
                    logger.warning("[gripper] loop send failed (%s) streak=%d",
                                   e, hard_error_streak)
                time.sleep(0.02)
            elif mot is not None and self._gripper_limp_active:
                # Zero-stiffness stream: motor stays powered but applies no
                # torque, so the fingers can be moved freely by hand.
                try:
                    mot.send_mit(0.0, 0.0, 0.0, 0.0, 0.0)
                except Exception:
                    pass
                time.sleep(0.02)
            else:
                time.sleep(0.02)

    def _start_gripper_stream(self) -> None:
        if self._gripper_stop is False:
            return
        self._gripper_stop = False
        self._gripper_thread = threading.Thread(
            target=self._gripper_loop, daemon=True, name="gripper-stream")
        self._gripper_thread.start()

    def _stop_gripper_stream(self) -> None:
        self._gripper_stop = True
        if self._gripper_thread is not None:
            self._gripper_thread.join(timeout=1.0)
            self._gripper_thread = None

    def _gripper_cmd(self, pos: float) -> bool:
        if self._gripper_motor is None:
            return False
        cfg = self._gripper_cfg

        # Hard safety clamp: commanding past the open-side limit (which sits
        # between 3.0 working and 4.0 faulting) over-limit-FAULTS the motor
        # and it then silently ignores EVERYTHING. Never allow it again.
        if pos > cfg.open_limit or pos < -2.4:
            logger.warning("[gripper] target %.2f outside safe range [-2.4, %.2f] "
                           "— clamping (prevents over-limit fault)", pos, cfg.open_limit)
            pos = float(np.clip(pos, -2.4, cfg.open_limit))

        # Self-healing fault recovery BEFORE every command. An over-limit fault
        # (e.g. from a bad earlier run commanding 4.0/6.0) makes the motor
        # silently ignore ALL MIT frames while send_mit() still returns ok.
        # clear_error+enable usually clears it without a power cycle.
        for _ in range(2):
            try:
                self._gripper_motor.clear_error()
                time.sleep(0.1)
                self._gripper_motor.enable()
                time.sleep(0.1)
            except Exception:
                pass

        # Send the MIT position command immediately — the dedicated 50Hz
        # streaming thread also holds this position continuously. Feedback on
        # this gripper is almost always absent, so a status-based recovery is
        # useless: just send, and re-enable only if the send itself throws.
        try:
            self._gripper_motor.send_mit(pos, 0.0, cfg.mit_kp, cfg.mit_kd, 0.0)
        except Exception as e:
            logger.warning("[gripper] send_mit to %.1f failed (%s) — retrying after enable", pos, e)
            try:
                self._gripper_motor.enable()
                time.sleep(0.2)
                self._gripper_motor.send_mit(pos, 0.0, cfg.mit_kp, cfg.mit_kd, 0.0)
            except Exception as e2:
                logger.warning("[gripper] resend to %.1f failed: %s", pos, e2)
                return False

        # Wait for settle — continuously re-send the MIT position so the
        # motor is never left without a frame during the settle window
        # (feedback is best-effort/logging only).
        for _ in range(int(cfg.settle_time / 0.2)):
            time.sleep(0.2)
            try:
                self._gripper_motor.send_mit(pos, 0.0, cfg.mit_kp, cfg.mit_kd, 0.0)
            except Exception:
                pass
        return True

    def _widen_gripper_pmax(self, retries: int = 3) -> None:
        """Raise the motor's stored MIT position mapping range (PMAX).

        Damiao MIT frames are scaled against the device's PMAX register (21).
        If the stored PMAX on THIS gripper is small (the observed symptom:
        fingers travel only a few mm no matter if we command 2.0, 3.6 or 9.0),
        every command beyond it is clamped in firmware. Writing the 4310
        catalog values (PMAX 12.5 / VMAX 30 / TMAX 10) widens the mapping.
        Write frames are fire-and-forget on the bus — Damiao executes them
        even though this motor never acks (only the read-verify fails).
        """
        mot = self._gripper_motor
        if mot is None:
            return
        for rid, val in ((21, 12.5), (22, 30.0), (23, 10.0)):
            for _ in range(retries):
                try:
                    mot.write_register_f32(rid, val)
                    time.sleep(0.12)
                    break
                except Exception as e:
                    logger.warning("[gripper] write reg %d=%.1f failed (%s) — retry", rid, val, e)
        logger.info("[gripper] PMAX/VMAX/TMAX widened to 12.5/30/10")

    def prime_gripper(self, reps: int = 4, wait: float = 0.45) -> bool:
        """Force the gripper through open->close->open at startup so it is
        unambiguously enabled, in MIT mode, and moving before a pick.

        The feedback pipe on this gripper is flaky (get_state() ~always None),
        so we cannot confirm motion from feedback — driving it for a couple
        open/close cycles is the reliable way to prime it into a known state.
        """
        if self._gripper_motor is None:
            logger.info("[gripper] no motor — skip prime")
            return False
        cfg = self._gripper_cfg
        logger.info("[gripper] priming: open %.2f -> grip %.2f x%d",
                    cfg.open_pos, cfg.grip_pos, reps)
        self._widen_gripper_pmax()
        try:
            # Clear any persistent over-limit fault BEFORE moving: a faulted
            # motor silently ignores every MIT command until cleared+re-enabled,
            # and this gripper is prone to faulting if it ever hit an out-of-range
            # close (e.g. the old -2.5 grip). Retry since acks are unreliable.
            for _ in range(3):
                try:
                    self._gripper_motor.clear_error()
                    time.sleep(0.15)
                    self._gripper_motor.enable()
                    time.sleep(0.2)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("[gripper] prime enable issue (ignored): %s", e)
        for i in range(reps):
            for _ in range(2):
                try:
                    self._gripper_motor.clear_error()
                    self._gripper_motor.enable()
                    time.sleep(0.15)
                except Exception:
                    pass
            logger.info("[gripper] prime cycle %d/%d: OPEN %.2f (%.1fs)", i + 1, reps,
                        cfg.open_pos, wait)
            self.gripper_drive(cfg.open_pos, wait)
            logger.info("[gripper] prime cycle %d/%d: GRIP %.2f (%.1fs)", i + 1, reps,
                        cfg.grip_pos, wait)
            self.gripper_drive(cfg.grip_pos, wait)
        self.gripper_drive(cfg.open_pos, 0.8)
        logger.info("[gripper] prime complete, holding open %.2f", cfg.open_pos)
        return True

    def gripper_drive(self, pos: float, duration: float,
                      poll_feedback: bool = False) -> None:
        """Command the gripper to `pos` and stream it for `duration` seconds.

        The dedicated gripper thread streams the MIT frames at ~50Hz from
        `_gripper_hold_target`, so this just sets the target and lets it run
        (a single send_mit only pulses a Damiao motor — actuation needs the
        continuous stream the thread provides). Feedback polling is OFF by
        default: this motor never acks, and the poll can block on the shared
        bus, starving the stream.
        """
        if self._gripper_motor is None:
            logger.info("[gripper] no motor — skip drive")
            return
        cfg = self._gripper_cfg
        self._gripper_limp_active = False
        pos = float(max(min(pos, cfg.open_limit), -2.4))   # hard safety clamp
        self._gripper_hold_target = pos
        if duration <= 0:
            return
        t_end = time.monotonic() + duration
        while time.monotonic() < t_end:
            time.sleep(0.02)

    def gripper_open(self) -> bool:
        if self._gripper_motor is None:
            logger.info("[gripper] no motor — skip open")
            return False
        cfg = self._gripper_cfg
        self._gripper_limp_active = False
        self._gripper_hold_target = cfg.open_pos
        ok = self._gripper_cmd(cfg.open_pos)
        logger.info("[gripper] open -> %.2f %s", cfg.open_pos, "ok" if ok else "FAILED")
        return ok

    def gripper_close(self) -> None:
        """Park the gripper in a LOOSE state after a drop.

        Zero stiffness (kp=kd=0) — the motor stays powered but the fingers
        move freely by hand. Called after releasing an object so the next
        pick starts from a free, predictable state.
        """
        if self._gripper_motor is None:
            logger.info("[gripper] no motor — skip close")
            return
        self._gripper_hold_target = None
        self._gripper_limp_active = True
        logger.info("[gripper] neutral/loose — fingers free")

    def gripper_limp(self, duration: float = 0.0) -> None:
        """Make the fingers free-moving (zero torque).

        Args:
            duration: seconds to stay limp; 0 (default) = until the next
                gripper command (open/grip/drive) cancels it.
        """
        if self._gripper_motor is None:
            logger.info("[gripper] no motor — skip limp")
            return
        self._gripper_hold_target = None
        self._gripper_limp_active = True
        if duration and duration > 0:
            time.sleep(duration)
            self._gripper_limp_active = False
        logger.info("[gripper] limp: fingers free%s",
                    f" for {duration}s" if duration else " until next command")

    def gripper_grip(self, suppress_open: bool = False) -> bool:
        """Close gripper onto object. Returns True if an object was detected in the grip."""
        if self._gripper_motor is None:
            logger.info("[gripper] no motor — skip grip")
            return False
        cfg = self._gripper_cfg
        target = cfg.grip_pos

        self._gripper_limp_active = False
        # Keep the streaming thread on the grip target for the whole command
        # window (never drop to idle during the settle), then correct the hold
        # point to wherever the motor actually stopped after we poll it.
        self._gripper_hold_target = target
        ok = self._gripper_cmd(target)
        if not ok:
            logger.warning("[gripper] grip cmd failed")
            if not suppress_open:
                self.gripper_open()
            return False

        # This motor never acks feedback (verified: get_state always None), so
        # polling here would only block on bus timeouts. Fire-and-forget: it
        # closed, and the streaming thread holds the grip target so the cup
        # stays pinched through lift and transport.
        logger.info("[gripper] no feedback — holding at grip target")
        self._gripper_hold_target = target
        return True

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
        logger.info("[scan] idx0(base)=%+.3f idx4(j5)=%+.3f idx5(j6/wrist)=%+.3f",
                    target[0], target[4], target[5])
        self.move_to_joints(target, duration=duration, frame_cb=frame_cb)
        final_q, _, _ = self._arm.get_state()
        logger.info("[scan] AFTER move — final joints: %s", np.round(final_q, 3))

        # Check if any joint failed to reach target (e.g. still faulted)
        errs = np.abs(final_q - target)
        bad = np.where(errs > 0.15)[0]
        if len(bad) > 0:
            logger.warning("[scan] joints %s did not reach target (errors: %s) — retrying",
                           bad, np.round(errs[bad], 3))
            # Give them a second attempt with more time
            self.move_to_joints(target, duration=duration + 2.0, frame_cb=frame_cb)
            final_q, _, _ = self._arm.get_state()
            logger.info("[scan] retry AFTER move — final joints: %s", np.round(final_q, 3))

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
        # Final target command
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
        """
        Full pick-and-place sequence.

        The engine calls this AFTER _approach_and_pick() has moved the arm to
        HOVER_Z above the object with gripper already open. Sequence:

            1. Continuous descend straight to grip height  — gripper stays OPEN
            2. Nudge forward into the object                — fingers wrap around it
            3. Close gripper firmly to hold
            4. Lift to clearance height
            5. Transport to drop location
            6. Open gripper to release
            7. Return to scan pose (home)
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

# ── 1. ONE continuous descent: hover -> grip height ──────────────
        # The gripper was already opened at hover and the streaming thread
        # holds it open the whole way down — no intermediate stop, no second
        # open call, just one fluid motion onto the cup.
        logger.info("[triage] continuous descend to grip z=%.3f", pz)
        if not self.move_to_pose(x=px, y=py, z=pz,
                                 roll=0, pitch=pitch, yaw=0,
                                 duration=3.5, frame_cb=frame_cb):
            logger.warning("[triage] descend IK failed — joint-space descent")
            q, _, _ = self._arm.get_state()
            q_down = q.copy()
            q_down[2] += 0.35
            self.move_to_joints(q_down, duration=2.5, frame_cb=frame_cb)

        # ── 2. Nudge forward into the object so the fingers wrap it ──────
        # Pushes the cup into the open gripper fingers so the close has
        # something to bite on. 10cm past center — reaches well past the cup.
        nudge_x = px + 0.10
        logger.info("[triage] nudge forward to x=%.3f", nudge_x)
        if not self.move_to_pose(x=nudge_x, y=py, z=pz,
                                 roll=0, pitch=pitch, yaw=0,
                                 duration=1.2, frame_cb=frame_cb):
            # Joint-space nudge: increase shoulder extension slightly
            q, _, _ = self._arm.get_state()
            q_nudge = q.copy()
            q_nudge[1] += 0.12
            self.move_to_joints(q_nudge, duration=1.0, frame_cb=frame_cb)

        # ── 3. Close gripper firmly ───────────────────────────────────────
        if frame_cb:
            frame_cb()
        logger.info("[triage] closing gripper...")
        gripped = self.gripper_grip()

        # Firm up the hold: wait a moment then re-close so the cup is secure.
        # gripper_grip never re-opens on its own, so this is safe.
        time.sleep(0.3)
        if frame_cb:
            frame_cb()
        self.gripper_grip()

        if not gripped:
            logger.warning("[triage] no object detected in grip — holding anyway and continuing")

        # ── 4. Lift to clearance height ───────────────────────────────────
        # Stiffen the wrist/forearm joints during the load-bearing phase:
        # 4310 wrists at kp18 sag and oscillate under the cup, which reads as
        # a staggered, choppy vertical lift. Restored right after transport.
        kp_saved = self._mit_kp.copy()
        kd_saved = self._mit_kd.copy()
        self._mit_kp[3:6] = 40.0
        self._mit_kd[3:6] = 5.0
        lift_z = max(pz + 0.30, 0.34)
        logger.info("[triage] lifting to z=%.3f", lift_z)
        if not self.move_to_pose(x=nudge_x, y=py, z=lift_z,
                                 roll=0, pitch=pitch, yaw=0,
                                 duration=3.5, frame_cb=frame_cb):
            q, _, _ = self._arm.get_state()
            q_lift = q.copy()
            q_lift[2] = min(q_lift[2] - 0.7, -0.5)  # straighten elbow = raise end-effector
            q_lift[1] -= 0.15
            self.move_to_joints(q_lift, duration=3.0, frame_cb=frame_cb)

        # ── 5. Transport to drop location ─────────────────────────────────
        if command.drop_joints is not None:
            target_q = np.deg2rad(np.array(command.drop_joints, dtype=np.float64))
            logger.info("[triage] transporting to drop joints %s", np.round(target_q, 3))
            self.move_to_joints(target_q, duration=5.0, frame_cb=frame_cb)
        elif command.drop_pose:
            logger.info("[triage] transporting to drop pose %s", command.drop_pose)
            self.move_to_pose(**command.drop_pose, duration=6.0, frame_cb=frame_cb)
        else:
            logger.warning("[triage] no drop target — dropping in place")

        # Load-bearing stiffness phase is over — restore the base wrist gains
        self._mit_kp[:] = kp_saved
        self._mit_kd[:] = kd_saved

        # ── 6. Release ────────────────────────────────────────────────────
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
        self._stop_gripper_stream()
        self._gripper_hold_target = None
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
