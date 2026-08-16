import copy
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    MotorChain,
    MotorInfo,
    PassiveEncoderInfo,
)
from i2rt.robots.robot import Robot
from i2rt.robots.utils import (
    GripperForceLimiter,
    GripperType,
    JointMapper,
    detect_gripper_limits,
    rezero_gripper_midstroke,
)
from i2rt.utils.mujoco_utils import MuJoCoKDL


@dataclass
class JointStates:
    names: List[str]
    pos: np.ndarray
    vel: np.ndarray
    eff: np.ndarray
    temp_mos: np.ndarray  # MOS temperature (float): Motor MOS temperature.
    temp_rotor: np.ndarray  # ROTOR temperature (float): Motor ROTOR temperature.
    timestamp: float

    def asdict(self) -> Dict[str, Any]:
        return {
            "names": self.names,
            "pos": self.pos.flatten().tolist(),
            "vel": self.vel.flatten().tolist(),
            "eff": self.eff.flatten().tolist(),
        }


@dataclass
class JointCommands:
    torques: np.ndarray

    pos: np.ndarray
    vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    indices: Optional[List[int]] = None

    @classmethod
    def init_all_zero(cls, n_joints: int) -> "JointCommands":
        return cls(
            torques=np.zeros(n_joints),
            pos=np.zeros(n_joints),
            vel=np.zeros(n_joints),
            kp=np.zeros(n_joints),
            kd=np.zeros(n_joints),
        )


class MotorChainRobot(Robot):
    """A generic Robot protocol."""

    def __init__(
        self,
        motor_chain: MotorChain,
        xml_path: Optional[str] = None,
        use_gravity_comp: bool = True,
        gravity: Optional[np.ndarray] = None,
        gravity_comp_factor: float = 1.0,  # New parameter with default value
        gripper_index: Optional[int] = None,  # Zero starting index: if you have a 6 dof arm and last one is gripper: 6
        kp: Union[float, List[float]] = 10.0,
        kd: Union[float, List[float]] = 1.0,
        joint_limits: Optional[np.ndarray] = None,  # if provided, override the mujoco xml joint limits
        gripper_limits: Optional[np.ndarray] = None,  # [closed, open]
        limit_gripper_force: float = -1,  # whether to limit the gripper effort when it is blocked. -1 means no limit.
        clip_motor_torque: float = np.inf,  # clip the offset motor torque, real motor torque can still still be larger than this setting depending on the motor onboard PID loop
        gripper_type: GripperType = GripperType.LINEAR_4310,
        temp_record_flag: bool = False,  # whether record the motor's temperature
        enable_gripper_calibration: bool = False,  # whether to auto-detect gripper limits
        zero_gravity_mode: bool = True,
        # below are calibration parameters
        test_torque: float = 0.5,  # test torque for gripper detection (Nm)
        test_duration: float = 2.0,  # max test duration for each direction (s)
        position_threshold: float = 0.01,  # minimum position change to consider motor still moving (rad)
        check_interval: float = 0.05,  # time interval between checks (s)
        gripper_close_margin: float = 0.0,  # push detected 'closed' deeper (rad) so the jaw fully seats
        gripper_open_span: Optional[float] = None,  # no open hardstop (worm-gear damiao): probe close stall only, open = closed - span (rad)
        gripper_max_track_torque: Optional[float] = None,  # hard cap (Nm) on gripper tracking torque: clamps cmd pos within tau/kp of current pos EVERY tick (force limiter only engages at stall)
        gripper_effort_window: float = 0.1,  # force-limiter effort averaging window (s); longer = slower clog latch/unlatch, steadier hold on slow worm-gear grippers
        gripper_torque_mode: bool = False,  # drive the gripper with torque cmds via a software position loop instead of onboard MIT position control. REQUIRED when the gripper's absolute position can exceed the ±12.5 rad p16 command encoding (long-stroke worm gear + multi-turn zero drift): position cmds get silently clamped at ±12.5, torque cmds don't.
        gripper_torque_mode_cap: float = 2.0,  # torque clip (Nm) for the software gripper loop
        gripper_rezero_midstroke: bool = False,  # boot: stall-probe closed, drive open span/2, save that as the motor's zero -> limits [+span/2, -span/2] always fit the ±12.5 p16 encoding regardless of multi-turn drift. Plain position control thereafter.
        clog_speed_threshold_scale: Optional[float] = None,
        hard_latch_effort_multiplier: Optional[float] = None,
        clog_grace_s: float = 0.0,  # latch suppression window at close-stroke onset -- see GripperForceLimiter  # per-rig hard-latch trip point as a multiple of the scaled soft threshold (default 2.5) -- see GripperForceLimiter  # ADDED 2026-08-15: per-rig multiplier on the soft-latch speed gate -- see GripperForceLimiter. Worm-gear rigs need >1 or the latch never fires on compliant objects.
        clog_force_threshold_scale: Optional[float] = None,  # ADDED 2026-07-22: per-rig multiplier on GripperType's shared clog_force_threshold/hard_latch trigger -- different physical grippers of the same nominal type can have different baseline mechanical friction, so free-space closing motion alone can cross the shared class-level threshold on one rig before it would on another (false "clogged" latch with nothing actually gripped, closing stops short). None = use the class default unscaled.

        pinned_cpu: int | None = None,
        joint_state_saver_factory: Optional[Callable[[], Any]] = None,
        set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        # Set up CPU pinning and real-time scheduling if requested
        if pinned_cpu is not None and set_realtime_and_pin_callback is not None:
            set_realtime_and_pin_callback(pinned_cpu)

        self._joint_state_saver_factory = joint_state_saver_factory
        self._set_realtime_and_pin_callback = set_realtime_and_pin_callback
        self.temp_record_flag = temp_record_flag
        if gripper_index is not None:
            assert gripper_index == len(motor_chain) - 1, (
                "Gripper index should be the last one, but got {gripper_index}"
            )

            # Auto-detect gripper limits if enabled and gripper_limits is None
            print(
                f"initializing motorchain robot, gripper_limits: {gripper_limits}, enable_gripper_calibration: {enable_gripper_calibration}"
            )
            if gripper_limits is None and enable_gripper_calibration:
                logger = logging.getLogger(__name__)
                if gripper_rezero_midstroke:
                    assert gripper_open_span, "gripper_rezero_midstroke requires gripper_open_span"
                    logger.info("Calibrating gripper via mid-stroke rezero...")
                    detected_limits = rezero_gripper_midstroke(
                        motor_chain=motor_chain,
                        gripper_index=gripper_index,
                        test_torque=test_torque,
                        open_span=gripper_open_span,
                        position_threshold=position_threshold,
                        check_interval=check_interval,
                        max_duration=test_duration,
                    )
                else:
                    logger.info("Auto-detecting gripper limits...")
                    detected_limits = detect_gripper_limits(
                        motor_chain=motor_chain,
                        gripper_index=gripper_index,
                        test_torque=test_torque,
                        max_duration=test_duration,
                        position_threshold=position_threshold,
                        check_interval=check_interval,
                        open_span=gripper_open_span,
                    )
                gripper_limits = np.array(detected_limits)
                logger.info(f"Gripper limits auto-detected: {gripper_limits}")
            elif gripper_limits is None:
                raise ValueError(
                    f"{self}: Gripper limits are required if gripper index is provided and auto-calibration is disabled."
                )
            else:
                # Use the provided gripper_limits
                logger = logging.getLogger(__name__)
                logger.info(f"Using provided gripper limits: {gripper_limits}")

            # Push the 'closed' limit deeper by a fixed margin so the jaw fully
            # seats. detect_gripper_limits stops at a gentle test torque, which can
            # be short of the true mechanical close (the left YAM gripper stalls
            # ~0.4 rad early). The GripperForceLimiter caps force at the hard stop,
            # so commanding slightly past it is safe. Applied in the close direction
            # (sign of closed-open) so it is polarity-agnostic. This lets BOTH arms
            # run pure auto-calibration (immune to multi-turn zero shifts on
            # power-cycle) instead of pinned limits that silently go stale.
            if gripper_close_margin and gripper_limits is not None:
                _closed, _opened = float(gripper_limits[0]), float(gripper_limits[1])
                gripper_limits = np.array(
                    [_closed + np.sign(_closed - _opened) * float(gripper_close_margin), _opened]
                )
                logging.getLogger(__name__).info(
                    f"Applied gripper_close_margin={gripper_close_margin}: limits -> {gripper_limits}"
                )

            # Env-gated dump of the (detected or provided) gripper limits so the
            # values can be pinned into the robot config for deterministic boots.
            if os.environ.get("GRIP_LIMITS_DUMP") and gripper_limits is not None:
                try:
                    tag = (
                        getattr(motor_chain, "channel", None)
                        or getattr(motor_chain, "motor_chain_name", None)
                        or f"pid{os.getpid()}"
                    )
                    with open(f"/tmp/gripper_limits_{tag}.txt", "w") as _f:
                        _f.write(f"{float(gripper_limits[0])},{float(gripper_limits[1])}\n")
                except Exception:
                    pass

        self._last_gripper_command_qpos = 1  # initialize as fully open
        assert clip_motor_torque >= 0.0
        self._clip_motor_torque = clip_motor_torque
        self.motor_chain = motor_chain
        self.use_gravity_comp = use_gravity_comp
        self.gravity_comp_factor = gravity_comp_factor  # Store the factor

        # variables for gripper effort limiting
        self._gripper_index = gripper_index
        self._gripper_max_track_torque = gripper_max_track_torque
        self._gripper_torque_mode = gripper_torque_mode
        self._gripper_torque_mode_cap = gripper_torque_mode_cap
        self.remapper = JointMapper({}, len(motor_chain))  # so it works without gripper
        self._gripper_limits = gripper_limits

        if self._gripper_index is not None:
            self._chain_name = (
                getattr(motor_chain, "motor_chain_name", None)
                or getattr(motor_chain, "channel", None)
                or getattr(motor_chain, "name", None)
            )
            self._gripper_force_limiter = GripperForceLimiter(
                max_force=limit_gripper_force,
                gripper_type=gripper_type,
                kp=kp[gripper_index],
                average_torque_window=gripper_effort_window,
                clog_force_threshold_scale=clog_force_threshold_scale,
                clog_speed_threshold_scale=clog_speed_threshold_scale,
                hard_latch_effort_multiplier=hard_latch_effort_multiplier,
                clog_grace_s=clog_grace_s,
                name=self._chain_name,
            )  # force in newton
            self._limit_gripper_force = limit_gripper_force

            self.remapper = JointMapper(
                index_range_map={gripper_index: gripper_limits},
                total_dofs=len(motor_chain),
            )

        # make sure kp, kd are float number not int
        self._kp = (
            np.array(
                [
                    kp,
                ]
                * len(motor_chain)
            )
            if isinstance(kp, float)
            else np.array(kp)
        )
        self._kd = (
            np.array(
                [
                    kd,
                ]
                * len(motor_chain)
            )
            if isinstance(kd, float)
            else np.array(kd)
        )

        self._joint_limits: Optional[np.ndarray] = None
        if xml_path is not None:
            self.xml_path = os.path.expanduser(xml_path)
            self.kdl = MuJoCoKDL(self.xml_path)
            if gravity is not None:
                self.kdl.set_gravity(gravity)
            # Load the joint limits from the xml file
            self._joint_limits = self.kdl.joint_limits
        else:
            assert use_gravity_comp is False, "Gravity compensation requires a valid XML path."

        # override the xml joint limits with the provided joint_limits
        if joint_limits is not None:
            joint_limits = np.array(joint_limits)
            assert np.all(joint_limits[:, 0] < joint_limits[:, 1]), (
                "Lower joint limits must be smaller than upper limits"
            )
            self._joint_limits = joint_limits
        # Initialize joint state saver if factory is provided
        if self._joint_state_saver_factory is not None:
            self._joint_state_saver = self._joint_state_saver_factory()
        else:
            self._joint_state_saver = None

        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._joint_state: Optional[JointStates] = None
        while self._joint_state is None:
            # wait to recive joint data
            time.sleep(0.05)
            self._joint_state = self._motor_state_to_joint_state(self.motor_chain.read_states())
        self._commands = JointCommands.init_all_zero(len(motor_chain))
        # For SWE-454, check if the current qpos is in the joint limits
        self._check_current_qpos_in_joint_limits()

        self._stop_event = threading.Event()  # Add a stop event
        self._server_thread = threading.Thread(target=self.start_server, name="robot_server")
        self._server_thread.start()

        if not zero_gravity_mode:
            # set current qpos as target pos with the default PD parameters
            self.command_joint_pos(self._joint_state.pos)

    def __repr__(self) -> str:
        return f"MotorChainRobot(motor_chain={self.motor_chain})"

    def _grip_trace(self, gs: dict, pre_target: float, post_cmd: float) -> None:
        """Env-gated follower gripper telemetry (set GRIP_FOLLOWER_TRACE=1).

        Writes one CSV row per control tick so we can see exactly when the
        GripperForceLimiter flips to 'clogged' during a slow close and how far
        it backs the command off (the source of the grating limit-cycle).
        """
        import os as _os
        if not _os.environ.get("GRIP_FOLLOWER_TRACE"):
            return
        try:
            if not hasattr(self, "_gtf"):
                name = getattr(self.motor_chain, "name", None) or getattr(
                    self.motor_chain, "motor_chain_name", None
                ) or f"pid{_os.getpid()}"
                self._gtf = open(f"/tmp/gfollow_{name}.csv", "w")
                self._gtf.write(
                    "t,target_qpos,current_qpos,qvel,eff,clogged,pre_target,post_cmd,adj\n"
                )
            clog = int(getattr(self._gripper_force_limiter, "_is_clogged", False))
            self._gtf.write(
                f"{time.time():.4f},{gs['target_qpos']:.5f},{gs['current_qpos']:.5f},"
                f"{gs['current_qvel']:.5f},{gs['current_eff']:.5f},{clog},"
                f"{pre_target:.5f},{post_cmd:.5f},{post_cmd - pre_target:.5f}\n"
            )
            # Flushing every tick (~300 Hz) issues ~300 fsyncs/s and throttles the
            # control loop itself, so buffer and flush periodically instead — the
            # trace must observe the loop, not slow it down.
            self._gtf_n = getattr(self, "_gtf_n", 0) + 1
            if self._gtf_n % 250 == 0:
                self._gtf.flush()
        except Exception:
            pass

    def _check_current_qpos_in_joint_limits(self, buffer_rad: float = 0.1) -> None:
        """Check if the self._joint_state is in the joint limits.
        If violated, raise an error.
        """
        if self._joint_state is None or self._joint_limits is None:
            raise RuntimeError(
                f"{self}: Joint limits:{self._joint_limits} or joint state:{self._joint_state} are not set."
            )

        current_pos = self._joint_state.pos

        # Check arm joints (exclude gripper if present)
        if self._gripper_index is not None:
            # Only check arm joints, not the gripper
            arm_pos = current_pos[: self._gripper_index]
            arm_limits = self._joint_limits
        else:
            # Check all joints
            arm_pos = current_pos
            arm_limits = self._joint_limits

        # Check if any joint is outside its limits
        lower_limits = arm_limits[:, 0] - buffer_rad
        upper_limits = arm_limits[:, 1] + buffer_rad

        # Find joints that violate lower limits
        lower_violations = arm_pos < lower_limits
        # Find joints that violate upper limits
        upper_violations = arm_pos > upper_limits

        if np.any(lower_violations) or np.any(upper_violations):
            violation_details = []

            for i, (pos, lower, upper) in enumerate(zip(arm_pos, lower_limits, upper_limits, strict=False)):
                if pos < lower:
                    violation_details.append(f"Joint {i}: {pos:.4f} < {lower:.4f} (lower limit)")
                elif pos > upper:
                    violation_details.append(f"Joint {i}: {pos:.4f} > {upper:.4f} (upper limit)")

            violation_msg = "; ".join(violation_details)
            # turn off the main motor control thread as well.
            self.motor_chain.running = False
            raise RuntimeError(
                f"{self}: Joint limit violation detected: {violation_msg}, the root reason should be zero position offset. possible solution: 1. move the arm to zero position and power cycle the robot. 2. Recalibrate the motor zero position."
            )

    def get_robot_info(self) -> Dict[str, Any]:
        """Get the robot information, such as kp, kd, joint limits, gripper limits, etc."""
        return {
            "kp": self._kp,
            "kd": self._kd,
            "joint_limits": self._joint_limits,
            "gripper_limits": self._gripper_limits,
            "gravity_comp_factor": self.gravity_comp_factor,
            "limit_gripper_effort": self._limit_gripper_force,
            "gripper_index": self._gripper_index,
        }

    def start_server(self) -> None:
        """Start the server."""
        last_time = time.time()
        iteration_count = 0
        self.update()

        logging.info("initializing, ....")

        while not self._stop_event.is_set():  # Check the stop event
            current_time = time.time()
            elapsed_time = current_time - last_time

            self.update()
            if not self.motor_chain.running:
                raise RuntimeError(f"{self}: motor_chain_robot's motor chain is not running, exiting the robot server")
            time.sleep(0.004)

            iteration_count += 1
            if elapsed_time >= 10.0:
                control_frequency = iteration_count / elapsed_time
                # Overwrite the current line with the new frequency information
                logging.info(f"{self}: Grav Comp Control Frequency: {control_frequency:.2f} Hz")
                if control_frequency < 100:
                    logging.warning(
                        f"{self}: Gravity compensation control loop is slow, current frequency: {control_frequency:.2f} Hz"
                    )
                # Reset the counter and timer
                last_time = current_time
                iteration_count = 0

    def update(self) -> None:
        """Update the robot.

        Send Torques and update the joint state.
        """
        with self._command_lock:
            joint_commands = copy.deepcopy(self._commands)
        with self._state_lock:
            g = self._compute_gravity_compensation(self._joint_state)
            motor_torques = joint_commands.torques + g * self.gravity_comp_factor
            motor_torques = np.clip(motor_torques, -self._clip_motor_torque, self._clip_motor_torque)

            if self._gripper_index is not None:
                _dbg_raw_target = float(joint_commands.pos[self._gripper_index])
                if self._limit_gripper_force > 0 and self._joint_state is not None:
                    # Get current gripper state in raw robot joint pos space
                    gripper_state = {
                        "target_qpos": joint_commands.pos[self._gripper_index],
                        "current_qpos": self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[
                            self._gripper_index
                        ],
                        "current_qvel": self._joint_state.vel[self._gripper_index],
                        "current_eff": self._joint_state.eff[self._gripper_index],
                        "current_normalized_qpos": self._joint_state.pos[self._gripper_index],
                        "target_normalized_qpos": self.remapper.to_command_joint_pos_space(joint_commands.pos)[
                            self._gripper_index
                        ],
                        "last_command_qpos": self._last_gripper_command_qpos,
                    }

                    _pre = float(gripper_state["target_qpos"])
                    joint_commands.pos[self._gripper_index] = self._gripper_force_limiter.update(gripper_state)
                    self._grip_trace(gripper_state, _pre, float(joint_commands.pos[self._gripper_index]))

                # Hard tracking-torque cap: onboard torque = kp*(cmd - pos), so
                # clamp cmd within tau/kp of the current pos every tick. Unlike
                # the force limiter (stall-only), this bounds torque in free
                # travel too — the gripper can never be driven harder than this.
                if self._gripper_max_track_torque is not None and self._joint_state is not None:
                    _kp_g = float(self._kp[self._gripper_index])
                    if _kp_g > 0:
                        _max_err = self._gripper_max_track_torque / _kp_g
                        _cur = self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[self._gripper_index]
                        joint_commands.pos[self._gripper_index] = np.clip(
                            joint_commands.pos[self._gripper_index], _cur - _max_err, _cur + _max_err
                        )

                # add final clip so the gripper won't be over-adjusted
                joint_commands.pos[self._gripper_index] = np.clip(
                    joint_commands.pos[self._gripper_index],
                    min(self._gripper_limits),
                    max(self._gripper_limits),
                )
                self._last_gripper_command_qpos = joint_commands.pos[self._gripper_index]

                # --- Gripper velocity feedforward (opt-in: GRIP_VEL_FF=<cap rad/s>) ---
                # v_des=0 makes the DM4310 kd term a brake (kd*(0 - v)); feeding a
                # velocity in the COMMAND direction turns kd into a driver, so the jaw
                # travels faster WITHOUT raising kp or stall force. Capped low and cut
                # out near contact/stall (see cutout below) and tapered over the
                # final approach so it can never slam the worm gear into the stop.
                # Off entirely unless GRIP_VEL_FF is set in the environment.
                if not hasattr(self, "_vff_cap"):
                    import os as _vff_os
                    try:
                        self._vff_cap = float(_vff_os.environ.get("GRIP_VEL_FF", "0") or 0)
                    except ValueError:
                        self._vff_cap = 0.0
                    self._vff_deadband = 0.1   # rad: no FF once within this of target (was 0.3, shrunk 2026-07-19 for speed -- last stretch is pure kp tracking, disproportionately slow relative to distance)
                    self._vff_taper = 0.4      # rad: linearly ramp FF over the last this-much (was 1.0, shrunk 2026-07-19 to match)
                    self._vff_stall_speed = 0.15  # rad/s: ADDED 2026-07-29. When the force limiter is bypassed (limit_gripper_force<=0), the jaw counts as stalled-on-contact once speed drops below this WITH the track-torque clamp saturated -> FF cut so kd*vff can't add torque past gripper_max_track_torque. Free travel is faster than this, so FF stays on there.
                self._dbg_vff = 0.0
                if self._vff_cap > 0 and self._joint_state is not None:
                    _vff_cur = self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[self._gripper_index]
                    # FIXED 2026-07-19: was joint_commands.pos[...] here, which by this
                    # point has already been squeezed to within
                    # gripper_max_track_torque/kp (~0.33 rad) of actual by the
                    # position-clamp above -- starving this error input so the
                    # deadband/taper ramp (0.3-1.0 rad) never saw enough distance to
                    # reach anywhere near _vff_cap, capping effective speed around
                    # ~vff_cap*0.33 regardless of how far the true target was.
                    # Use the true raw (pre-clamp) target instead so the taper reflects
                    # genuine remaining distance; the hard-latch + clog cutout below
                    # still bounds actual contact force independent of this.
                    _vff_err = _dbg_raw_target - _vff_cur
                    # Contact/stall cutout -- kill FF so the kd*(vff - vel) term can
                    # never add torque past the force ceiling on contact. Two regimes:
                    #  - force limiter active (limit_gripper_force > 0): use its clog flag.
                    #  - force limiter bypassed (<= 0, gem10 worm grippers): derive the
                    #    equivalent from the track-torque cap -- the clamp is saturated
                    #    (raw target pulls past cap/kp of travel) AND the jaw has stalled
                    #    (speed < _vff_stall_speed) => pushing on something at the torque
                    #    ceiling. During fast free travel the clamp is also saturated but
                    #    speed is high, so FF stays on there. Without this, a stalled jaw
                    #    (vel~0) leaves kd*(vff-vel)=kd*vff added on top of the cap.
                    if self._limit_gripper_force > 0:
                        _vff_clog = bool(getattr(self._gripper_force_limiter, "_is_clogged", False))
                    elif self._gripper_max_track_torque is not None:
                        _kp_g_vff = float(self._kp[self._gripper_index])
                        _vff_sat = _kp_g_vff > 0 and abs(_vff_err) > (self._gripper_max_track_torque / _kp_g_vff)
                        _vff_slow = abs(float(self._joint_state.vel[self._gripper_index])) < self._vff_stall_speed
                        _vff_clog = bool(_vff_sat and _vff_slow)
                    else:
                        _vff_clog = False
                    if (not _vff_clog) and abs(_vff_err) > self._vff_deadband:
                        _vff = np.sign(_vff_err) * self._vff_cap * min(1.0, abs(_vff_err) / self._vff_taper)
                        joint_commands.vel[self._gripper_index] = float(_vff)
                        self._dbg_vff = float(_vff)
                    else:
                        joint_commands.vel[self._gripper_index] = 0.0

                # Env-gated live print of the gripper's applied torque (opt-in:
                # GRIP_TORQUE_PRINT=<interval_s>, e.g. "0.1"; any non-numeric truthy
                # value, e.g. "1", defaults to a 0.2s cadence; unset/"" = off).
                # REPLACES a prior unconditional "TEMP DEBUG" print (2026-07-15) that
                # ran every 0.3s regardless of env, spamming stdout in every session.
                # "applied" is the motor's OWN measured effort (current_eff) -- the
                # real torque it's exerting right now, as read back from the drive --
                # vs "cmd_torque", the torque the onboard PD loop is about to command
                # given the final (post force-limiter/track-torque-cap/limits-clip)
                # position target: kp*(final_cmd - current). Comparing the two shows
                # whether the position loop is actually achieving its computed torque
                # or the gripper is stuck (applied plateaus while cmd keeps climbing).
                if not hasattr(self, "_grip_torque_print_interval"):
                    import os as _gtp_os
                    _gtp_raw = _gtp_os.environ.get("GRIP_TORQUE_PRINT", "")
                    if _gtp_raw:
                        try:
                            self._grip_torque_print_interval = max(0.0, float(_gtp_raw))
                        except ValueError:
                            self._grip_torque_print_interval = 0.2
                    else:
                        self._grip_torque_print_interval = None  # disabled
                if self._grip_torque_print_interval is not None:
                    _gtp_now = time.time()
                    if (
                        not hasattr(self, "_grip_torque_last_print")
                        or _gtp_now - self._grip_torque_last_print >= self._grip_torque_print_interval
                    ):
                        self._grip_torque_last_print = _gtp_now
                        _gtp_cur = float(
                            self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[self._gripper_index]
                        )
                        _gtp_final_cmd = float(joint_commands.pos[self._gripper_index])
                        _gtp_kp = float(self._kp[self._gripper_index])
                        _gtp_cmd_torque = _gtp_kp * (_gtp_final_cmd - _gtp_cur)
                        _gtp_applied = float(self._joint_state.eff[self._gripper_index])
                        _gtp_clogged = bool(getattr(self._gripper_force_limiter, "_is_clogged", False))
                        print(
                            f"[GRIP_TORQUE {getattr(self, '_chain_name', None) or 'gripper'}] "
                            f"applied={_gtp_applied:+.3f} Nm cmd_torque={_gtp_cmd_torque:+.3f} Nm "
                            f"cap={self._gripper_max_track_torque} raw_target={_dbg_raw_target:.3f} "
                            f"final_cmd={_gtp_final_cmd:.3f} current={_gtp_cur:.3f} "
                            f"vff={getattr(self, '_dbg_vff', 0.0):.3f} clogged={_gtp_clogged}",
                            flush=True,
                        )

                # Torque mode: run the position loop HERE (wrap-aware absolute
                # positions) and send pure torque. Onboard position control is
                # useless past ±12.5 rad — the p16 encoding clamps the command,
                # the motor thinks it's on target, and the jaw parks (found via
                # grip_diag: closed at +18.7 abs, cmd clamped to 12.5, eff ~0).
                if self._gripper_torque_mode and self._joint_state is not None:
                    _g = self._gripper_index
                    _cur = self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[_g]
                    _vel = self._joint_state.vel[_g]
                    _err = joint_commands.pos[_g] - _cur
                    _tau = float(
                        np.clip(
                            self._kp[_g] * _err - self._kd[_g] * _vel,
                            -self._gripper_torque_mode_cap,
                            self._gripper_torque_mode_cap,
                        )
                    )
                    motor_torques[_g] += _tau
                    joint_commands.pos[_g] = 0.0
                    joint_commands.vel[_g] = 0.0
                    joint_commands.kp[_g] = 0.0
                    joint_commands.kd[_g] = 0.0
            if not self.motor_chain.start_thread_flag:
                self.motor_chain.set_commands(
                    motor_torques,
                    pos=joint_commands.pos,
                    vel=joint_commands.vel,
                    kp=joint_commands.kp,
                    kd=joint_commands.kd,
                )
                self.motor_chain.start_thread()
                self.motor_chain.start_thread_flag = True
            self._update_joint_state(motor_torques, joint_commands)

    def _update_joint_state(
        self,
        motor_torques: np.ndarray,
        joint_commands: "JointCommands",
        encoder_infos: Optional[List[PassiveEncoderInfo]] = None,
    ) -> None:
        """Send commands to motor chain, update joint state, and optionally save to disk."""
        if (
            hasattr(self.motor_chain, "get_same_bus_device_states")
            and callable(self.motor_chain.get_same_bus_device_states)
            and self.motor_chain.same_bus_device_driver is not None
        ):
            has_gripper_encoder = True
            encoder_infos = self.motor_chain.get_same_bus_device_states()
            assert len(encoder_infos) == 1, "Only one encoder is supported"
            assert isinstance(encoder_infos[0], PassiveEncoderInfo), "Encoder info must be a PassiveEncoderInfo"
        else:
            has_gripper_encoder = False

        motor_state = self.motor_chain.set_commands(
            motor_torques,
            pos=joint_commands.pos,
            vel=joint_commands.vel,
            kp=joint_commands.kp,
            kd=joint_commands.kd,
        )
        self._joint_state = self._motor_state_to_joint_state(motor_state)

        # For SWE-454: keep monitoring qpos during runtime
        self._check_current_qpos_in_joint_limits()

        if self._joint_state_saver is not None:
            assert not (has_gripper_encoder and self._gripper_index is not None), (
                "Either has_gripper_encoder=True or self._gripper_index is not None"
            )
            ee_pos = ee_vel = ee_eff = None
            if has_gripper_encoder:
                ee_pos = np.array([info.position for info in encoder_infos])
                ee_vel = np.array([info.velocity for info in encoder_infos])
            elif self._gripper_index is not None:
                ee_pos = self._joint_state.pos[self._gripper_index]
                ee_vel = self._joint_state.vel[self._gripper_index]
                ee_eff = self._joint_state.eff[self._gripper_index]

            if self._gripper_index is None:
                pos = self._joint_state.pos
                vel = self._joint_state.vel
                eff = self._joint_state.eff
            else:
                pos = self._joint_state.pos[: self._gripper_index]
                vel = self._joint_state.vel[: self._gripper_index]
                eff = self._joint_state.eff[: self._gripper_index]

            self._joint_state_saver.add(
                timestamp=self._joint_state.timestamp,
                pos=pos,
                vel=vel,
                eff=eff,
                ee_pos=ee_pos,
                ee_vel=ee_vel,
                ee_eff=ee_eff,
            )

    def _motor_state_to_joint_state(self, motor_state: List[MotorInfo]) -> JointStates:
        """Convert motor state to joint state.

        Args:
            motor_state (List[Any]): The motor state.

        Returns:
            Dict[str, np.ndarray]: The joint state.
        """
        names = [str(i) for i in range(len(motor_state))]
        pos = np.array([motor.pos for motor in motor_state])
        pos = self.remapper.to_command_joint_pos_space(pos)
        vel = np.array([motor.vel for motor in motor_state])
        vel = self.remapper.to_command_joint_vel_space(vel)
        eff = np.array([motor.eff for motor in motor_state])
        temp_mos = np.array([motor.temp_mos for motor in motor_state])
        temp_rotor = np.array([motor.temp_rotor for motor in motor_state])
        timestamp = motor_state[0].timestamp
        return JointStates(
            names=names,
            pos=pos,
            vel=vel,
            eff=eff,
            temp_mos=temp_mos,
            temp_rotor=temp_rotor,
            timestamp=timestamp,
        )

    def _compute_gravity_compensation(self, joint_state: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        if joint_state is None or not self.use_gravity_comp:
            return np.zeros(len(self.motor_chain))
        elif self.use_gravity_comp:
            q = joint_state.pos[: self._gripper_index] if self._gripper_index is not None else joint_state.pos
            t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            # print gravity torque to 2f
            if np.max(np.abs(t)) > 25.0:
                print([f"{s:.2f}" for s in t])
                raise RuntimeError(f"{self}: too large torques")
            if self._gripper_index is None:
                return self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            else:
                t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
                return np.append(t, 0.0)

    # ----------------- Server Functions ----------------- #

    def num_dofs(self) -> int:
        """Get the number of joints of the robot, including the gripper.

        Returns:
            int: The number of joints of the robot.
        """
        return len(self.motor_chain)

    def get_joint_pos(self) -> np.ndarray:
        """Get the current state of the leader robot, including the gripper in radian.

        Returns:
            T: The current state of the leader robot.
        """
        with self._state_lock:
            return self._joint_state.pos

    def _clip_robot_joint_pos_command(self, pos: np.ndarray) -> np.ndarray:
        """Clip the robot joint pos command to the joint limits. Do not clip the gripper pos.
        Args:
            pos (np.ndarray): The joint pos command to clip.
        Returns:
            np.ndarray: The clipped joint pos command.
        """

        if self._joint_limits is not None:
            if self._gripper_index is not None:
                pos[: self._gripper_index] = np.clip(
                    pos[: self._gripper_index],
                    self._joint_limits[:, 0],
                    self._joint_limits[:, 1],
                )
            else:
                pos = np.clip(pos, self._joint_limits[:, 0], self._joint_limits[:, 1])
        return pos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_pos (np.ndarray): The state to command the leader robot to.
        """
        pos = self._clip_robot_joint_pos_command(joint_pos)
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.kp = self._kp
            self._commands.kd = self._kd

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (Dict[str, np.ndarray]): The state to command the leader robot to.
        """
        pos = self._clip_robot_joint_pos_command(joint_state["pos"])
        vel = joint_state["vel"]
        self._commands = JointCommands.init_all_zero(len(self.motor_chain))
        kp = joint_state.get("kp", self._kp)
        kd = joint_state.get("kd", self._kd)
        with self._command_lock:
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.vel = self.remapper.to_robot_joint_vel_space(vel)
            self._commands.kp = kp
            self._commands.kd = kd

    def zero_torque_mode(self) -> None:
        logging.info(f"Entering zero_torque_mode for {self}")
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._kp = np.zeros(len(self.motor_chain))
            self._kd = np.zeros(len(self.motor_chain))

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        with self._state_lock:
            if self._gripper_index is None:
                result = {
                    "joint_pos": self._joint_state.pos,
                    "joint_vel": self._joint_state.vel,
                    "joint_eff": self._joint_state.eff,
                }
            else:
                result = {
                    "joint_pos": self._joint_state.pos[: self._gripper_index],
                    "joint_vel": self._joint_state.vel[: self._gripper_index],
                    "joint_eff": self._joint_state.eff[: self._gripper_index],
                    "gripper_pos": np.array([self._joint_state.pos[self._gripper_index]]),
                    "gripper_vel": np.array([self._joint_state.vel[self._gripper_index]]),
                    "gripper_eff": np.array([self._joint_state.eff[self._gripper_index]]),
                }
            if self.temp_record_flag:
                result["temp_mos"] = self._joint_state.temp_mos
                result["temp_rotor"] = self._joint_state.temp_rotor
            return result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context related to this object."""
        self.close()

    def move_joints(self, target_joint_positions: np.ndarray, time_interval_s: float = 2.0) -> None:
        """Move the robot to a given joint positions."""
        with self._state_lock:
            current_pos = self._joint_state.pos
        assert len(current_pos) == len(target_joint_positions)
        steps = 50  # 50 steps over time_interval_s
        for i in range(steps + 1):
            alpha = i / steps  # Interpolation factor
            target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions  # Linear interpolation
            self.command_joint_pos(target_pos)
            time.sleep(time_interval_s / steps)

    def close(self) -> None:
        """Safely close the robot by setting all torques to zero."""
        # self.move_to_zero()
        self._stop_event.set()  # Signal the thread to stop
        self._server_thread.join()  # Wait for the thread to finish
        self.motor_chain.close()
        print("Robot closed with all torques set to zero.")

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        assert kp.shape == self._kp.shape == kd.shape
        self._kp = kp
        self._kd = kd

    def start_recording(self, save_dir: str) -> bool:
        """Start recording joint state data asynchronously."""
        if self._joint_state_saver is None:
            raise RuntimeError("Joint state saver factory not provided, recording not available")
        self._joint_state_saver.start_recording(save_dir)
        return True

    def stop_recording(self, prefix: str = "") -> Tuple[bool, str]:
        """Stop recording joint state data asynchronously."""
        if self._joint_state_saver is None:
            raise RuntimeError("Joint state saver not available")
        succ = self._joint_state_saver.stop_recording(prefix)
        if succ:
            return succ, "Recording stopped successfully"
        return succ, "Recording failed to stop"


if __name__ == "__main__":
    import argparse
    import time

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.utils.utils import override_log_level

    override_log_level(level=logging.INFO)

    args = argparse.ArgumentParser()
    args.add_argument("--gripper_type", type=str, default="linear_4310")
    args.add_argument("--channel", type=str, default="can0")
    args.add_argument("--operation_mode", type=str, default="gravity_comp")

    args = args.parse_args()

    gripper_type = GripperType.from_string_name(args.gripper_type)

    print(f"Initializing yam with gripper type: {gripper_type}")
    robot = get_yam_robot(args.channel, gripper_type=gripper_type)

    if args.operation_mode == "gravity_comp":
        while True:
            # print(robot.get_observations())
            time.sleep(1)
    elif args.operation_mode == "test_gripper":
        assert gripper_type != GripperType.YAM_TEACHING_HANDLE, (
            "test_gripper is not supported for YAM_TEACHING_HANDLE, teaching handle is a passive device"
        )
        for _ in range(30):
            for gripper_pos in [0.8, 0.0]:
                print(f"gripper_pos: {gripper_pos}")
                robot.command_joint_pos(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_pos]))
                time.sleep(4)
                print(robot.get_observations())
    elif args.operation_mode == "stay_current_qpos":
        current_qpos = robot.get_joint_pos()
        robot.command_joint_pos(current_qpos)
        while True:
            time.sleep(1)
