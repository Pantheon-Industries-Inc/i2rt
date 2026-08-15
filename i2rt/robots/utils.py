import enum
import logging
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from i2rt.motor_drivers.dm_driver import DMChainCanInterface

logger = logging.getLogger(__name__)

I2RT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Arm XML paths
ARM_YAM_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/arm/yam/yam.xml")
ARM_YAM_PRO_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/arm/yam_pro/yam_pro.xml")
ARM_YAM_ULTRA_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/arm/yam_ultra/yam_ultra.xml")
ARM_BIG_YAM_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/arm/big_yam/big_yam.xml")

# Gripper XML paths
GRIPPER_CRANK_4310_PATH = os.path.join(I2RT_ROOT, "robot_models/gripper/crank_4310/crank_4310.xml")
GRIPPER_LINEAR_3507_PATH = os.path.join(I2RT_ROOT, "robot_models/gripper/linear_3507/linear_3507.xml")
GRIPPER_LINEAR_4310_PATH = os.path.join(I2RT_ROOT, "robot_models/gripper/linear_4310/linear_4310.xml")
GRIPPER_TEACHING_HANDLE_PATH = os.path.join(
    I2RT_ROOT, "robot_models/gripper/yam_teaching_handle/yam_teaching_handle.xml"
)
GRIPPER_NO_GRIPPER_PATH = os.path.join(I2RT_ROOT, "robot_models/gripper/no_gripper/no_gripper.xml")


def combine_arm_and_gripper_xml(
    arm_path: str,
    gripper_path: str,
    ee_mass: Optional[float] = None,
    ee_inertia: Optional[np.ndarray] = None,
) -> str:
    """Combine arm and gripper XML files into a single XML string.

    Replaces the <body name="link_6"> subtree in the arm XML with the one from the
    gripper XML (if present). If ee_mass or ee_inertia are provided, update the
    inertial properties of the resulting link_6. Returns path to combined XML in /tmp/.

    Args:
        arm_path: Path to the arm MuJoCo XML file.
        gripper_path: Path to the gripper MuJoCo XML file. If falsy, the arm XML
            is used as-is (no gripper replacement).
        ee_mass: Optional end-effector mass (kg) to override in link_6's inertial.
        ee_inertia: Optional end-effector inertia array. Expected as a flat array of
            10 elements: [ipos(3), quat(4), diaginertia(3)].

    Returns:
        Path to the combined XML file written to /tmp/.
    """
    arm_tree = ET.parse(arm_path)
    arm_root = arm_tree.getroot()

    # Resolve arm mesh paths to absolute
    arm_dir = os.path.dirname(os.path.abspath(arm_path))
    arm_compiler = arm_root.find("compiler")
    arm_meshdir = arm_compiler.get("meshdir", "") if arm_compiler is not None else ""
    arm_asset = arm_root.find("asset")
    if arm_asset is not None:
        for child in arm_asset:
            if child.get("file") and not os.path.isabs(child.get("file")):
                abs_file = os.path.join(arm_dir, arm_meshdir, child.get("file"))
                child.set("file", os.path.abspath(abs_file))

    # Remove meshdir from compiler (all paths now absolute)
    if arm_compiler is not None and arm_compiler.get("meshdir"):
        del arm_compiler.attrib["meshdir"]

    # attempt to load gripper and replace link_6 if available
    if gripper_path:
        try:
            grip_tree = ET.parse(gripper_path)
            grip_root = grip_tree.getroot()
            grip_body = grip_root.find(".//body[@name='link_6']")
            if grip_body is None:
                grip_body = grip_root.find(".//body[@name='link6']")
        except Exception:
            grip_root = None
            grip_body = None

        # merge assets (avoid duplicates), resolving gripper mesh paths to absolute
        if grip_root is not None:
            grip_dir = os.path.dirname(os.path.abspath(gripper_path))
            grip_compiler = grip_root.find("compiler")
            grip_meshdir = grip_compiler.get("meshdir", "") if grip_compiler is not None else ""

            grip_asset = grip_root.find("asset")
            if grip_asset is not None:
                if arm_asset is None:
                    arm_asset = ET.Element("asset")
                    worldbody = arm_root.find("worldbody")
                    if worldbody is not None:
                        arm_root.insert(list(arm_root).index(worldbody), arm_asset)
                    else:
                        arm_root.append(arm_asset)
                existing = {(c.tag, c.get("name")) for c in arm_asset}
                for child in grip_asset:
                    key = (child.tag, child.get("name"))
                    if key not in existing:
                        elem = deepcopy(child)
                        if elem.get("file") and not os.path.isabs(elem.get("file")):
                            abs_file = os.path.join(grip_dir, grip_meshdir, elem.get("file"))
                            elem.set("file", os.path.abspath(abs_file))
                        arm_asset.append(elem)
                        existing.add(key)

        # replace arm's link_6 with gripper's if found
        if grip_body is not None:
            replaced = False
            for parent in arm_root.iter():
                children = list(parent)
                for idx, child in enumerate(children):
                    if child.tag == "body" and child.get("name") in ("link_6", "link6"):
                        parent.remove(child)
                        parent.insert(idx, deepcopy(grip_body))
                        replaced = True
                        break
                if replaced:
                    break

        # merge optional top-level sections (equality, contact) from gripper
        if grip_root is not None:
            for section_tag in ("equality", "contact"):
                grip_section = grip_root.find(section_tag)
                if grip_section is None:
                    continue
                arm_section = arm_root.find(section_tag)
                if arm_section is None:
                    arm_section = ET.SubElement(arm_root, section_tag)
                for child in grip_section:
                    arm_section.append(deepcopy(child))

    # find resulting link_6 and apply end-effector overrides (mass/inertia)
    if ee_mass is not None or ee_inertia is not None:
        res_body = arm_root.find(".//body[@name='link_6']")
        if res_body is None:
            res_body = arm_root.find(".//body[@name='link6']")
        if res_body is not None:
            inertial = res_body.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(res_body, "inertial")

            if ee_mass is not None:
                inertial.set("mass", str(float(ee_mass)))

            if ee_inertia is not None:
                arr = np.asarray(ee_inertia).ravel()
                ipos = " ".join(str(float(x)) for x in arr[:3])
                inertial.set("ipos", ipos)
                quat = " ".join(str(float(x)) for x in arr[3:7])
                inertial.set("quat", quat)
                diagin = " ".join(str(float(x)) for x in arr[-3:])
                inertial.set("diaginertia", diagin)

    # write combined xml to /tmp/ and return filepath
    out_path = tempfile.NamedTemporaryFile(suffix=".xml", prefix="i2rt_combined_", delete=False, dir="/tmp").name
    arm_tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


class ArmType(enum.Enum):
    YAM = "yam"
    YAM_PRO = "yam_pro"
    YAM_ULTRA = "yam_ultra"
    BIG_YAM = "big_yam"

    @classmethod
    def from_string_name(cls, name: str) -> "ArmType":
        try:
            return cls(name)
        except ValueError:
            raise ValueError(
                f"Unknown arm type: {name}, arm has to be one of the following: {ArmType.available_arms()}"
            ) from None

    @classmethod
    def available_arms(cls) -> List[str]:
        return [arm.value for arm in cls]

    def get_xml_path(self) -> str:
        _xml_map = {
            ArmType.YAM: ARM_YAM_XML_PATH,
            ArmType.YAM_PRO: ARM_YAM_PRO_XML_PATH,
            ArmType.YAM_ULTRA: ARM_YAM_ULTRA_XML_PATH,
            ArmType.BIG_YAM: ARM_BIG_YAM_XML_PATH,
        }
        if self not in _xml_map:
            raise ValueError(f"Unknown arm type: {self}")
        return _xml_map[self]


class GripperType(enum.Enum):
    CRANK_4310 = "crank_4310"  # a 4310 motor with a crank
    LINEAR_3507 = "linear_3507"  # a 3507 motor with a linear actuator
    LINEAR_4310 = "linear_4310"  # a 4310 motor with a linear actuator
    LINEAR_4310_FAST = "linear_4310_fast"  # LINEAR_4310 worm gearbox swapped for a smaller-driven-gear (lower reduction, faster) variant -- 2026-07-16, gem1 left arm

    # technically not a gripper
    YAM_TEACHING_HANDLE = "yam_teaching_handle"
    NO_GRIPPER = "no_gripper"

    @classmethod
    def from_string_name(cls, name: str) -> "GripperType":
        try:
            return cls(name)
        except ValueError:
            raise ValueError(
                f"Unknown gripper type: {name!r}, must be one of: {GripperType.available_grippers()}"
            ) from None

    @classmethod
    def available_grippers(cls) -> List[str]:
        return [gripper.value for gripper in GripperType]

    def get_gripper_limits(self) -> Optional[tuple[float, float]]:
        if self == GripperType.CRANK_4310:
            return 0.0, -2.7
        return None

    def get_gripper_needs_calibration(self) -> bool:
        return self in (GripperType.LINEAR_3507, GripperType.LINEAR_4310, GripperType.LINEAR_4310_FAST)

    def get_xml_path(self) -> str:
        _xml_map = {
            GripperType.CRANK_4310: GRIPPER_CRANK_4310_PATH,
            GripperType.LINEAR_3507: GRIPPER_LINEAR_3507_PATH,
            GripperType.LINEAR_4310: GRIPPER_LINEAR_4310_PATH,
            # same body/mesh as LINEAR_4310 -- only the internal worm gearbox ratio
            # differs, which isn't represented in the mujoco model. Revisit if the
            # new gearbox turns out to change external geometry too.
            GripperType.LINEAR_4310_FAST: GRIPPER_LINEAR_4310_PATH,
            GripperType.YAM_TEACHING_HANDLE: GRIPPER_TEACHING_HANDLE_PATH,
            GripperType.NO_GRIPPER: GRIPPER_NO_GRIPPER_PATH,
        }
        if self not in _xml_map:
            raise ValueError(f"Unknown gripper type: {self}")
        return _xml_map[self]

    def get_motor_kp_kd(self) -> tuple[float, float]:
        if self in (GripperType.CRANK_4310, GripperType.LINEAR_4310, GripperType.LINEAR_4310_FAST):
            return 20, 0.5
        elif self == GripperType.LINEAR_3507:
            return 10, 0.3
        elif self in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER):
            return -1.0, -1.0
        else:
            raise ValueError(f"Unknown gripper type: {self}")

    def get_motor_type(self) -> str:
        if self in (GripperType.CRANK_4310, GripperType.LINEAR_4310, GripperType.LINEAR_4310_FAST):
            return "DM4310"
        elif self == GripperType.LINEAR_3507:
            return "DM3507"
        elif self in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER):
            return ""
        else:
            raise ValueError(f"Unknown gripper type: {self}")

    def get_gripper_limiter_params(self) -> tuple[float, float, float, callable]:
        """
        clog_force_threshold: float,
        clog_speed_threshold: float,
        sign: float,
        gripper_force_torque_map: callable,
        """
        if self == GripperType.CRANK_4310:
            return (
                0.5,
                0.2,
                1.0,
                partial(
                    zero_linkage_crank_gripper_force_torque_map,
                    motor_reading_to_crank_angle=lambda x: -x + 0.174,
                    gripper_close_angle=8 / 180.0 * np.pi,
                    gripper_open_angle=170 / 180.0 * np.pi,
                    gripper_stroke=0.071,  # unit in meter
                ),
            )
        elif self == GripperType.LINEAR_3507:
            return (
                0.5,
                0.3,
                1.0,
                partial(
                    linear_gripper_force_torque_map,
                    motor_stroke=6.57,
                    gripper_stroke=0.096,
                ),
            )
        elif self == GripperType.LINEAR_4310:
            return (
                0.5,
                # clog_speed_threshold: bumped 0.05->0.08 (2026-07-15) -- real
                # teleop is fast pull/release, rarely holds still enough to latch
                # at 0.05. Old rationale: only treat as stalled when essentially
                # stationary; the old 0.3 rad/s value misclassified deliberate SLOW
                # closes as stalls, causing an audible limit-cycle ("grating")
                # close. 0.08 stays well below that 0.3 problem value while giving
                # fast real squeezes a realistic chance to latch on contact.
                0.08,
                1.0,
                partial(
                    linear_gripper_force_torque_map,
                    motor_stroke=6.57,
                    gripper_stroke=0.096,
                ),
            )
        elif self == GripperType.LINEAR_4310_FAST:
            return (
                0.75,  # RAISED from 0.5 (2026-07-22), per direct request -- grip harder before latching now that exact-position freeze (no backoff) makes the frozen hold safe to trust.
                0.08,
                1.0,
                partial(
                    linear_gripper_force_torque_map,
                    # 2026-07-16: approximated from the measured gripper_open_span (left,
                    # post-swap) via scripts/calibrate_gripper_open_span.py --side left --write
                    # (close-stall +9.2262 -> chosen open +0.0288 -> span 9.197 rad). This is
                    # motor-side rad from the close hardstop to an operator-chosen open point,
                    # not a from-scratch full-mechanical-travel measurement the way the old
                    # 6.57 constant nominally was -- so treat as a much better estimate, not a
                    # ground truth. force = torque * motor_stroke / gripper_stroke: if this is
                    # too large, required torque for a target force is understated (fails
                    # weak/under-grips, not strong) -- re-tighten while re-tuning
                    # limit_gripper_force under real load.
                    motor_stroke=9.197,
                    gripper_stroke=0.096,  # physical jaw travel (m) -- unchanged, same end effector
                ),
            )
        elif self in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER):
            return -1.0, -1.0, -1.0, None
        else:
            raise ValueError(f"Unknown gripper type: {self}")


class JointMapper:
    def __init__(self, index_range_map: Dict[int, Tuple[float, float]], total_dofs: int):
        """_summary_
        This class is used to map the joint positions from the command space to the robot joint space.

        Args:
            index_range_map (Dict[int, Tuple[float, float]]): 0 indexed
            total_dofs (int): num of joints in the robot including the gripper if the girpper is the second robot
        """
        self.empty = len(index_range_map) == 0
        if not self.empty:
            self.joints_one_hot = np.zeros(total_dofs).astype(bool)
            self.joint_limits = []
            for idx, (start, end) in index_range_map.items():
                self.joints_one_hot[idx] = True
                self.joint_limits.append((start, end))
            self.joint_limits = np.array(self.joint_limits)
            self.joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]

    def to_robot_joint_pos_space(self, command_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_pos
        command_joint_pos = np.asarray(command_joint_pos, order="C")
        result = command_joint_pos.copy()
        needs_remapping = command_joint_pos[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range + self.joint_limits[:, 0]
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_robot_joint_vel_space(self, command_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_vel
        result = command_joint_vel.copy()
        needs_remapping = command_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_vel_space(self, robot_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_vel
        result = robot_joint_vel.copy()
        needs_remapping = robot_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_pos_space(self, robot_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_pos
        result = robot_joint_pos.copy()
        needs_remapping = robot_joint_pos[self.joints_one_hot]
        needs_remapping = (needs_remapping - self.joint_limits[:, 0]) / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result


def linear_gripper_force_torque_map(
    motor_stroke: float, gripper_stroke: float, gripper_force: float, current_angle: float
) -> float:
    """Maps the motor stroke required to achieve a given gripper force.

    Args:
        motor_stroke (float): in rad
        gripper_stroke (float): in meter
        gripper_force (float): in newton
    """
    # force = torque * motor_stroke / gripper_stroke
    return gripper_force * gripper_stroke / motor_stroke


def zero_linkage_crank_gripper_force_torque_map(
    gripper_close_angle: float,
    gripper_open_angle: float,
    motor_reading_to_crank_angle: Callable[[float], float],
    gripper_stroke: float,
    current_angle: float,
    gripper_force: float,
) -> float:
    """Maps the motor crank torque required to achieve a given gripper force. For Yam style gripper (zero linkage crank)

    Args:
        gripper_close_angle (float): Angle of the crank in radians at the closed position.
        gripper_open_angle (float): Angle of the crank in radians at the open position.
        gripper_stroke (float): Linear displacement of the gripper in meters.
        current_angle (float): Current crank angle in radians (relative to the closed position).
        gripper_force (float): Required gripping force in Newtons (N).

    Returns:
        float: Required motor torque in Newton-meters (Nm).
    """
    current_angle = motor_reading_to_crank_angle(current_angle)
    # Compute crank radius based on the total stroke and angle change
    crank_radius = gripper_stroke / (2 * (np.cos(gripper_close_angle) - np.cos(gripper_open_angle)))
    # gripper_position = crank_radius * (np.cos(gripper_close_angle) - np.cos(current_angle))
    grad_gripper_position = crank_radius * np.sin(current_angle)

    # Compute the required torque
    target_torque = gripper_force * grad_gripper_position
    return target_torque


class LockFreeCircularBuffer:
    """
    Lock-free circular buffer.
    There is a ~microsecond level race condition for this, but we're only using it to tell if the gripper is clogged or not.
    So 1 stale reading out of 1000 is not a big deal (FOR THAT PARTICULAR USE CASE!!!).
    """

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.timestamps = np.zeros(maxsize)
        self.values = np.zeros(maxsize)
        self.write_idx = 0

    def put(self, timestamp: float, value: float) -> None:
        """Add a timestamped value to the buffer."""
        idx = self.write_idx % self.maxsize
        self.timestamps[idx] = timestamp
        self.values[idx] = value
        self.write_idx += 1

    def get_recent_values(self, time_window: float, current_time: Optional[float] = None) -> np.ndarray:
        """Get values within the specified time window."""
        if current_time is None:
            current_time = time.time()

        valid_mask = self.timestamps > (current_time - time_window)
        return self.values[valid_mask]


class GripperForceLimiter:
    def __init__(
        self,
        max_force: float,
        gripper_type: GripperType,
        kp: float,
        average_torque_window: float = 0.1,  # in seconds
        debug: bool = False,
        clog_force_threshold_scale: Optional[float] = None,  # ADDED 2026-07-22: per-rig multiplier -- see MotorChainRobot's clog_force_threshold_scale docstring.
        clog_speed_threshold_scale: Optional[float] = None,  # ADDED 2026-08-15: per-rig multiplier on the soft-latch SPEED gate. The class default (0.08 rad/s for LINEAR_4310) is linear-drive tuning; a worm-gear jaw crushing a compliant object still creeps at 0.1-0.3 rad/s, so the soft latch never fires and force winds to the track-torque cap. Scale up so the latch can fire while the jaw is still creeping under load; keep the scaled value BELOW free-travel speed (gem10 right: ~0.55-0.65 rad/s) or it will false-latch mid-close.
        name: Optional[str] = None,  # ADDED 2026-07-29: motor_chain_name (e.g. "yam_left"), used only to label clog log lines.
    ):
        self.max_force = max_force
        self._name = name or "gripper"
        self._clog_force_threshold_scale = clog_force_threshold_scale
        self._clog_speed_threshold_scale = clog_speed_threshold_scale
        self.gripper_type = gripper_type
        self._is_clogged = False
        self._gripper_adjusted_qpos = None
        self._kp = kp
        self._past_gripper_effort_buffer = LockFreeCircularBuffer(maxsize=1000)
        self.average_torque_window = average_torque_window
        self.debug = debug
        # FIXED 2026-07-19: unlatch debounce. The raw unlatch check (average_effort <
        # 0.2) could flip on a single noisy reading, causing _is_clogged to chatter
        # on/off while genuinely stalled against an object -- observed via grip_diag.py
        # as eff oscillating -1.09..+1.94 with the jaw not moving. Require the unlatch
        # condition to hold continuously for _unlatch_debounce_s before actually
        # unlatching; any violation resets the timer.
        self._clog_unlatch_candidate_since = None
        self._unlatch_debounce_s = 0.15
        # ADDED 2026-07-22: unlatch-on-reopen deadband. The position half of
        # unlatch_condition below (current more closed than target) used to
        # fire on ANY positive gap, so ordinary leader-hand jitter while
        # holding an object (bilateral kickback, tremor) nudged target_qpos
        # open by a hair and, after _unlatch_debounce_s, fully released real
        # grip force even though effort was still high. Require the gap to
        # exceed this margin (normalized 0-1 qpos units) before it counts as
        # "operator wants it open."
        self._unlatch_open_deadband = 0.02
        # FIXED 2026-07-19: hard-impact override multiplier -- see compute_target_gripper_torque.
        self.hard_latch_effort_multiplier = 2.5  # REVERTED 2026-07-22 (was briefly 1.5) -- gripper_max_track_torque (per-arm YAML, always-on every tick, independent of clog detection) is now the real safety net and already confirmed catching fast pulls without breaking anything. The lowered multiplier here was over-sensitive and made LEFT false-latch (freeze, wont close) on an ordinary fast pull. Back to 2.5.
        # RETUNED 2026-07-22 (round 2, per direct request): the hard-impact
        # override briefly used raw instantaneous current_eff (single tick) to
        # react faster to fast trigger pulls -- but a single tick is noisy
        # (same noise this file elsewhere averages away, e.g. the unlatch
        # debounce above) and was false-latching on noise spikes at the start
        # of a fast pull, with nothing actually gripped -- observed live as
        # "pull trigger, nothing happens" (can't unlatch via a close command,
        # only via a deliberate open). Average over a SHORT window instead --
        # enough ticks to reject single-sample noise, far shorter than the
        # main average_torque_window so it still reacts well before a slow
        # approach would.
        self.hard_latch_window_s = 0.03
        (self.clog_force_threshold, self.clog_speed_threshold, self.sign, _gripper_force_torque_map) = (
            self.gripper_type.get_gripper_limiter_params()
        )
        if self._clog_force_threshold_scale is not None:
            self.clog_force_threshold *= self._clog_force_threshold_scale
        if self._clog_speed_threshold_scale is not None:
            self.clog_speed_threshold *= self._clog_speed_threshold_scale
        self.gripper_force_torque_map = partial(
            _gripper_force_torque_map,
            gripper_force=self.max_force,
        )

    def compute_target_gripper_torque(self, gripper_state: Dict[str, float]) -> bool:
        current_speed = gripper_state["current_qvel"]
        relevant_history_effort = self._past_gripper_effort_buffer.get_recent_values(self.average_torque_window)
        if len(relevant_history_effort) > 0:
            average_effort = np.abs(np.mean(relevant_history_effort))
        else:
            average_effort = 0.0

        if self.debug:
            print(f"average_effort: {average_effort}")

        relevant_history_effort_short = self._past_gripper_effort_buffer.get_recent_values(self.hard_latch_window_s)
        if len(relevant_history_effort_short) > 0:
            hard_latch_effort = np.abs(np.mean(relevant_history_effort_short))
        else:
            hard_latch_effort = np.abs(gripper_state["current_eff"])

        self._just_latched = False
        if self._is_clogged:
            # SIMPLIFIED 2026-07-22 (round 2, per direct request): once latched,
            # freeze at the EXACT raw position where contact/overshoot was
            # detected -- no force-model extrapolation (round 1 computed a
            # position from gripper_force_torque_map/limit_gripper_force, which
            # could back the jaw OFF the true overshoot point to match that
            # model -- that backing-off was the remaining "drift backwards").
            # Only release on a real, deliberate open command from the leader.
            normalized_current_qpos = gripper_state["current_normalized_qpos"]
            normalized_target_qpos = gripper_state["target_normalized_qpos"]
            # 0 close 1 open
            if normalized_target_qpos > normalized_current_qpos + self._unlatch_open_deadband:
                self._is_clogged = False
                self._gripper_adjusted_qpos = None  # recompute fresh next time it latches
                logger.info(
                    "%s: gripper clog latch RELEASED (operator opened past deadband, "
                    "target=%.4f current=%.4f)",
                    self._name, normalized_target_qpos, normalized_current_qpos,
                )
        elif average_effort > self.clog_force_threshold and np.abs(current_speed) < self.clog_speed_threshold:
            self._is_clogged = True
            self._just_latched = True
            logger.warning(
                "%s: gripper CLOG detected (soft latch) — avg_effort=%.3f Nm > threshold=%.3f Nm "
                "while speed=%.4f rad/s < %.4f rad/s. Freezing position; see hard_latch_effort_multiplier "
                "for the fast-impact path.",
                self._name, average_effort, self.clog_force_threshold, current_speed, self.clog_speed_threshold,
            )
        elif hard_latch_effort > self.clog_force_threshold * self.hard_latch_effort_multiplier:
            # FIXED 2026-07-19: hard-impact override. The normal path above also
            # requires |current_speed| < clog_speed_threshold before latching --
            # fine after a SLOW approach (speed is already low), but after a FAST/
            # high-momentum impact, measured speed doesn't settle near-zero
            # instantly (bounce/ringing/sensor lag), so the latch (and therefore
            # any velocity-feedforward cutoff gated on it) can lag a hard fast
            # impact -- observed live as the gripper "ignoring force constraints"
            # on a fast trigger pull. This path latches immediately on
            # unambiguously-high effort alone, with no speed gate, closing that
            # race regardless of how fast the approach was.
            # RETUNED 2026-07-22: uses hard_latch_effort (short-window average,
            # see hard_latch_window_s) rather than raw instantaneous current_eff
            # -- a single tick was too noisy and false-latched on nothing.
            self._is_clogged = True
            self._just_latched = True
            logger.warning(
                "%s: gripper CLOG detected (hard-impact latch) — hard_latch_effort=%.3f Nm > "
                "threshold=%.3f Nm (%.1fx soft threshold). Freezing position.",
                self._name, hard_latch_effort, self.clog_force_threshold * self.hard_latch_effort_multiplier,
                self.hard_latch_effort_multiplier,
            )

        return self._is_clogged

    def update(self, gripper_state: Dict[str, float]) -> None:
        current_ts = time.time()
        self._past_gripper_effort_buffer.put(current_ts, gripper_state["current_eff"])
        is_clogged = self.compute_target_gripper_torque(gripper_state)

        if is_clogged:
            if self._just_latched or self._gripper_adjusted_qpos is None:
                # RETUNED 2026-07-22 (round 4, per direct request): round 3's
                # one-way rule (only push further closed, never back off) meant
                # a hard/fast impact (typically a bigger, more rigid object) kept
                # WHATEVER overshoot it built up with no ceiling, while a gentle
                # slow contact (typically a smaller object) got capped down to
                # the moderate target_eff baseline -- backwards from "consistent,
                # not stronger on big objects." Fix: snap to the angle-normalized
                # target_eff position in EITHER direction, but do it ONCE, right
                # here, then freeze permanently (see the `elif self._is_clogged`
                # branch below -- no further recompute ever). That one-time snap
                # is NOT the old "settle-then-ease-back" bug -- that was a
                # continuous multi-tick re-tracking loop from live noisy effort,
                # visibly drifting over time. This resolves within one tick and
                # never touches the position again.
                current_eff = gripper_state["current_eff"]
                target_eff = self.gripper_force_torque_map(current_angle=gripper_state["current_qpos"]) + 0.3
                command_sign = np.sign(gripper_state["target_qpos"] - gripper_state["current_qpos"]) * self.sign
                current_zero_eff_pos = (
                    gripper_state["last_command_qpos"] - command_sign * np.abs(current_eff) / self._kp
                )
                self._gripper_adjusted_qpos = current_zero_eff_pos + command_sign * np.abs(target_eff) / self._kp
                if self.debug:
                    print(f"clogged (latching): current_eff={current_eff} target_eff={target_eff} adj={self._gripper_adjusted_qpos}")
            elif self.debug:
                print("clogged (holding)", self._gripper_adjusted_qpos)
            return self._gripper_adjusted_qpos
        else:
            if self.debug:
                print("unclogged")
            self._gripper_adjusted_qpos = gripper_state["current_qpos"]
            return gripper_state["target_qpos"]

def rezero_gripper_midstroke(
    motor_chain: DMChainCanInterface,
    gripper_index: int = 6,
    test_torque: float = 1.0,
    open_span: float = 14.0,
    position_threshold: float = 0.005,
    check_interval: float = 0.05,
    max_duration: float = 30.0,
) -> List[float]:
    """Anchor the gripper's operating window mid-stroke so position commands
    always fit the ±12.5 rad p16 encoding.

    The DM4310's multi-turn zero drifts across power cycles; on boots where the
    close stop lands beyond ±12.5 rad absolute, position commands get silently
    clamped at the wire and the gripper parks short (verified via grip_diag).
    Sequence: stall-probe closed (torque, gentle) -> torque-drive open by
    span/2 (position-monitored from the verified stall, aborts on early jam so
    it can NEVER over-open) -> save current position as the motor's zero ->
    limits become [+span/2, -span/2].

    Returns [closed, open] in the new zeroed frame.
    """
    logger = logging.getLogger(__name__)
    closed_abs = detect_gripper_limits(
        motor_chain, gripper_index, test_torque, max_duration,
        position_threshold, check_interval, open_span=0.0,
    )[0]
    motor_direction = motor_chain.motor_direction[gripper_index]
    close_dir = 1 if motor_direction > 0 else -1
    num = len(motor_chain.motor_list)
    half = abs(open_span) / 2.0

    logger.info(f"Rezero: close stall at {closed_abs:.3f}, driving open {half:.2f} rad to mid-stroke...")
    torques = np.zeros(num)
    torques[gripper_index] = -close_dir * test_torque
    start = time.time()
    last_pos, stable = None, 0
    while True:
        if time.time() - start > max_duration:
            motor_chain.set_commands(torques=np.zeros(num))
            raise RuntimeError("gripper mid-stroke drive timed out")
        motor_chain.set_commands(torques=torques)
        time.sleep(check_interval)
        pos = motor_chain.read_states()[gripper_index].pos
        if abs(pos - closed_abs) >= half:
            break
        if last_pos is not None and abs(pos - last_pos) < position_threshold:
            stable += 1
            if stable >= 80:  # bumped from 20 (2026-07-15): 1s was too impatient, false-tripping on worm-gear static-friction catch points that manual jogging worked through fine on this exact mechanism. ~4s at check_interval=0.05s.
                motor_chain.set_commands(torques=np.zeros(num))
                raise RuntimeError(
                    f"gripper stalled while opening to mid-stroke at "
                    f"{abs(pos - closed_abs):.2f}/{half:.2f} rad — check the mechanism / open_span"
                )
        else:
            stable = 0
        last_pos = pos

    motor_chain.set_commands(torques=np.zeros(num))
    time.sleep(0.3)

    # Direct bus transaction: stop the chain thread first so frames don't race.
    motor_chain.running = False
    time.sleep(0.2)
    motor_id = motor_chain.motor_list[gripper_index][0]
    motor_chain.motor_interface.save_zero_position(motor_id)
    time.sleep(0.2)
    if motor_chain.absolute_positions is not None:
        motor_chain.absolute_positions[gripper_index] = 0.0
    motor_chain.start_thread()
    time.sleep(0.2)

    closed = close_dir * half
    opened = -close_dir * half
    logger.info(f"Rezero done: gripper zeroed mid-stroke, limits [{closed:.2f}, {opened:.2f}]")
    return [closed, opened]


def detect_gripper_limits(
    motor_chain: DMChainCanInterface,
    gripper_index: int = 6,
    test_torque: float = 0.2,
    max_duration: float = 2.0,
    position_threshold: float = 0.01,
    check_interval: float = 0.1,
    open_span: Optional[float] = None,
) -> List[float]:
    """
    Detect gripper limits by applying test torques and monitoring position changes.

    Args:
        motor_chain: Motor chain interface
        gripper_index: Index of gripper motor
        test_torque: Test torque for gripper detection (Nm)
        max_duration: Maximum test duration for each direction (s)
        position_threshold: Minimum position change to consider motor still moving (rad)
        check_interval: Time interval between checks (s)
        open_span: If set, the gripper has NO mechanical hardstop on the open side
            (e.g. damiao DM4310 on the worm-gear end effector) so only the CLOSE
            direction is stall-probed; 'open' is defined as closed minus this many
            rad of travel. Still immune to multi-turn zero shifts since it re-anchors
            on the close stall every boot. Find the value with
            scripts/calibrate_gripper_open_span.py.

    Returns:
        List of detected limits [closed, open]
    """
    logger = logging.getLogger(__name__)
    positions = []
    num_motors = len(motor_chain.motor_list)
    zero_torques = np.zeros(num_motors)

    # Get motor direction for the gripper
    motor_direction = motor_chain.motor_direction[gripper_index]

    # Record initial position
    initial_states = motor_chain.read_states()
    init_torque = np.array([state.eff for state in initial_states])
    initial_pos = initial_states[gripper_index].pos
    positions.append(initial_pos)
    logger.info(f"Gripper calibration starting from position: {initial_pos:.4f}")

    # Close direction in the raw motor frame: with motor_direction > 0 the
    # 'closed' limit is the max position (see ordering below), i.e. +torque.
    close_direction = 1 if motor_direction > 0 else -1

    # Test both directions — or close-only when open has no hardstop
    directions = [close_direction] if open_span is not None else [1, -1]
    for direction in directions:
        logger.info(f"Testing gripper direction: {direction}")
        test_torques = init_torque
        test_torques[gripper_index] = direction * test_torque

        start_time = time.time()
        last_pos = None
        position_stable_count = 0

        while time.time() - start_time < max_duration:
            motor_chain.set_commands(torques=test_torques)
            time.sleep(check_interval)

            states = motor_chain.read_states()
            current_pos = states[gripper_index].pos
            positions.append(current_pos)

            # Check if position has stopped changing (gripper hit limit)
            if last_pos is not None:
                pos_change = abs(current_pos - last_pos)
                if pos_change < position_threshold:
                    position_stable_count += 1
                else:
                    position_stable_count = 0

                # Check if gripper has hit limit (position stable)
                if position_stable_count >= 60:  # bumped from 3 (2026-07-15): ~150ms was far too impatient, false-tripping the CLOSE-direction stall probe on the same static-friction catch points as the open-direction issue, landing the whole calibrated range short of the true closed position
                    logger.info(f"Gripper limit detected: pos={current_pos:.4f}")
                    break

            last_pos = current_pos

        time.sleep(0.3)

    if open_span is not None:
        # Closed = where the close-direction probe stalled; open is software-defined
        # at open_span rad of travel back from closed (no hardstop to probe).
        closed = positions[-1]
        opened = closed - close_direction * abs(open_span)
        logger.info(
            f"Gripper close stall at {closed:.4f}, software open limit (span {open_span}): {opened:.4f}"
        )
        return [closed, opened]

    # Calculate detected limits
    min_pos = min(positions)
    max_pos = max(positions)

    # Order based on motor direction
    if motor_direction > 0:
        # Positive direction: [max, min]
        detected_limits = [max_pos, min_pos]
    else:
        # Negative direction: [min, max]
        detected_limits = [min_pos, max_pos]

    logger.info(f"Motor direction: {motor_direction}, detected limits: {detected_limits}")
    return detected_limits
