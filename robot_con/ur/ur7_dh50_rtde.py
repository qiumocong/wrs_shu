import math
import time
import numpy as np
from pathlib import Path

import toppra.solverwrapper.cy_seidel_solverwrapper

# Custom project libraries
import motion.trajectory.polynomial_wrsold as pwp
import motion.trajectory.piecewisepoly_toppra as pwp_toppra
from basis import robot_math as rm
import rtde_control
import rtde_receive
import drivers.urx.ur_robot as urrobot
import robot_con.ur.program_builder_dh as pb


# from drivers.devices.dh.ag95 import Ag95Driver  # Uncomment if Ag95 gripper used

class UR5Ag95X_RTDE(object):
    """
    A refactored version of the UR5 control class using the ur_rtde library.

    author: weiwei (original), Gemini (refactor)
    date: 20250704
    """

    def __init__(self, robot_ip='192.168.125.9', gp_port='com3'):
        """
        Initializes the robot arm and gripper.

        :param robot_ip: IP address of the UR5 robot.
        :param gp_port: COM port for Ag95 gripper.
        """
        # Arm setup
        self.rtde_frequency = 1000.0
        self._rtde_c = rtde_control.RTDEControlInterface(robot_ip, self.rtde_frequency)
        self._rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip, self.rtde_frequency)
        self._arm = urrobot.URRobot(robot_ip)

        # Robot Gripper script and config
        self.pb = pb.ProgramBuilder()
        dh_script_path = Path(r"F:\Study\point cloud\wrs-qiu\wrs-qiu\drivers\devices\dh\ac.script")
        self.dh_prog = self.pb.get_str_from_file(dh_script_path)

        # Set default payload (1.28kg at flange assumed center of gravity)
        self._rtde_c.setPayload(1.28, (0, 0, 0))

        # Gripper Setup (Ag95/Other) - Uncomment as required
        # self._hnd = Ag95Driver(port=gp_port)

        print(f"Successfully connected to UR5 at {robot_ip} and gripper at {gp_port}.")

    # --- Gripper control using DH SCRIPT ---
    def _replace_dh_script(self, speed, force, width=None):
        prog = self.dh_prog
        prog = prog.replace("program_replace_speed", f"dh_pgc_set_speed(1,{speed})")
        prog = prog.replace("program_replace_force", f"dh_pgc_set_speed(1,{force})")
        cmd_pos = 100 if width is None else width
        prog = prog.replace("program_replace_command", f'dh_pgc_set_position(1,{cmd_pos})')
        return prog

    def open_gripper_dh(self, speed=100, force=100):
        prog = self._replace_dh_script(speed, force, width=100)
        print(prog)
        self._arm.send_program(prog)
        self._rtde_c.disconnect()
        self._rtde_c.reconnect()

    def close_gripper_dh(self, speed=100, force=100):
        prog = self._replace_dh_script(speed, force, width=0)
        print(prog)
        self._arm.send_program(prog)
        self._rtde_c.disconnect()
        self._rtde_c.reconnect()

    def close_to_dh(self, width, speed=100, force=100):
        width = width/0.06 * 100
        prog = self._replace_dh_script(speed, force, width)
        print(prog)
        self._arm.send_program(prog)
        self._rtde_c.disconnect()
        self._rtde_c.reconnect()

    # ---- Robot RTDE properties ----
    @property
    def rtde_c(self):
        """Read-only RTDE control interface."""
        return self._rtde_c

    @property
    def rtde_r(self):
        """Read-only RTDE receive interface."""
        return self._rtde_r

    # @property
    # def hnd(self):
    #    """Read-only property for the gripper driver."""
    #    return self._hnd

    # ---- Robot Movement ----
    def move_jnts(self, jnt_values, vel=1.0, acc=1.0, wait=True):
        """
        Moves the robot to a target joint configuration.

        :param jnt_values: List of 6 joint angles (radians).
        :param vel: Joint velocity (rad/s).
        :param acc: Joint acceleration (rad/s^2).
        :param wait: Sync/wait for completion.
        """
        self._rtde_c.moveJ(jnt_values, vel, acc, not wait)

    def regulate_jnts_pmpi(self):
        """Moves all joints to their equivalent angle within the [-pi, pi] range."""
        current_jnts = self.get_jnt_values()
        regulated_jnts = rm.regulate_angle(-math.pi, math.pi, current_jnts)
        print("Regulating joints to [-pi, pi] range.")
        self.move_jnts(regulated_jnts)

    def move_jntspace_path(self, path, interval_time=1.0, control_frequency=0.002,
                           vel=0.5, acc=0.8,
                           speed_gain=300,
                           blend=0.0, toppra_vels=None, toppra_accs=None):
        """
        Executes a trajectory defined by a list of joint-space waypoints.

        :param path: List of Nx6 joint configurations (radians).
        :param interval_time: Interpolation interval.
        :param control_frequency: Interpolation frequency.
        :param vel: Point velocity.
        :param acc: Point acceleration.
        :param blend: Blend radius for corner smoothing.
        :param toppra_vels: Max velocity per joint.
        :param toppra_accs: Max acceleration per joint.
        """
        if toppra_vels is None:
            toppra_vels = [vel] * 6  # max=3.14
        if toppra_accs is None:
            toppra_accs = [acc] * 6

        tpply = pwp_toppra.PiecewisePolyTOPPRA()
        interpolated_path = tpply.interpolate_by_max_spdacc(
            path=path,
            control_frequency=control_frequency,
            max_vels=toppra_vels,
            max_accs=toppra_accs,
            toggle_debug=False
        )[1:]

        velocity = vel
        acceleration = acc
        dt = 1.0 / self.rtde_frequency  # e.g. 2ms
        lookahead_time = 0.1
        gain = speed_gain

        for q_joint in interpolated_path:
            t_start = self._rtde_c.initPeriod()
            self._rtde_c.servoJ(q_joint, velocity, acceleration, dt, lookahead_time, gain)
            self._rtde_c.waitPeriod(t_start)

    # ---- State Getters ----
    def get_jnt_values(self):
        """
        Returns current joint angles in radians.
        :return: Numpy array of 6 joint angles.
        """
        return np.asarray(self._rtde_r.getActualQ())

    def get_pose(self):
        """
        Returns current TCP pose (position and rotation matrix).
        :return: (pos: np.array(3,), rot: np.array(3,3))
        """
        tcp_pose_vec = self._rtde_r.getActualTCPPose()
        pos = np.asarray(tcp_pose_vec[:3])
        rot = rm.rotmat_from_euler(tcp_pose_vec[3], tcp_pose_vec[4], tcp_pose_vec[5])
        return pos, rot

    # ---- Session/Connection ----
    def disconnect(self):
        """Disconnects RTDE control interface."""
        self._rtde_c.disconnect()
        print("Disconnected from the robot.")


# ---- Example Usage (Main) ----
if __name__ == '__main__':
    import socket

    robot_ip = "192.168.125.30"
    u5rag95_x1 = UR5Ag95X_RTDE(robot_ip=robot_ip, gp_port='COM5')

    # Gripper operations
    u5rag95_x1.open_gripper_dh()
    u5rag95_x1.rtde_c.disconnect()
    u5rag95_x1.rtde_c.reconnect()
    u5rag95_x1.rtde_c.zeroFtSensor()

    # Force mode setup
    task_frame = [0, 0, 0, 0, 0, 0]
    selection_vector = [0, 0, 1, 1, 1, 0]
    wrench_down = [0, 0, -10, 0, 0, 0]
    force_type = 2
    limits = [2, 2, 2, 1, 1, 1]

    # Apply force until contact is reached (example)
    while u5rag95_x1.rtde_r.getActualTCPForce()[2] < 9.0:
        u5rag95_x1.rtde_c.forceMode(task_frame, selection_vector, wrench_down, force_type, limits)
        u5rag95_x1.rtde_c.waitPeriod(u5rag95_x1.rtde_c.initPeriod())

    u5rag95_x1.close_gripper_dh()
    u5rag95_x1.rtde_c.disconnect()
    u5rag95_x1.rtde_c.reconnect()
    u5rag95_x1.rtde_c.zeroFtSensor()
    u5rag95_x1.rtde_c.forceMode(task_frame, selection_vector, [0, 0, 10, 0, 0, 0], force_type, limits)
