import math
import time
import numpy as np

# Assuming these custom libraries from the original project are available
import motion.trajectory.polynomial_wrsold as pwp
from basis import robot_math as rm
# from drivers.devices.dh.ag95 import Ag95Driver
from robot_con.ag145.ag145 import ag145
# Import the new UR RTDE library components
import rtde_control
import rtde_receive


class UR5Ag95X_RTDE(object):
    """
    A refactored version of the UR5 control class using the ur_rtde library.

    author: weiwei (original), Gemini (refactor)
    date: 20250704
    """

    def __init__(self,
                 robot_ip='192.168.125.9',
                 gp_port='com3'):
        """
        Initializes the robot arm and gripper using the ur_rtde library.

        :param robot_ip: The IP address of the UR5 robot.
        :param gp_port: The COM port for the Ag95 gripper.
        """
        # --- Setup Arm using ur_rtde ---
        self.rtde_frequency = 1000.0
        self._rtde_c = rtde_control.RTDEControlInterface(robot_ip, self.rtde_frequency)
        self._rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip, self.rtde_frequency)

        # Set default TCP and payload
        # self._rtde_c.setTCP((0, 0, 0, 0, 0, 0))
        # Original payload was 1.28kg. Assuming Center of Gravity is at the flange.
        self._rtde_c.setPayload(1.28, (0, 0, 0))

        # --- Setup Hand ---
        # This part remains unchanged as it's for a separate device
        # self._hnd = Ag95Driver(port=gp_port)

        # --- Setup Trajectory Planner ---
        # This part remains unchanged
        self.trajt = pwp.TrajPoly(method='quintic')

        print(f"Successfully connected to UR5 at {robot_ip} and gripper at {gp_port}.")

    @property
    def rtde_c(self):
        """Read-only property for the RTDE control interface."""
        return self._rtde_c

    @property
    def rtde_r(self):
        """Read-only property for the RTDE receive interface."""
        return self._rtde_r

    @property
    def hnd(self):
        """Read-only property for the gripper driver."""
        return self._hnd

    def open_gripper(self):
        """Opens the Ag95 gripper."""
        self.hnd.open_g()
        print("Gripper opened.")

    def slow_open_gripper(self):
        self.hnd.set_speed(80)
        self.hnd.open_g()
        self.hnd.set_speed(100)

    def close_gripper(self):
        """Closes the Ag95 gripper."""
        self.hnd.close_g()
        print("Gripper closed.")

    def move_jnts(self, jnt_values, vel=1.0, acc=1.0, wait=True):
        """
        Moves the robot to a target joint configuration.

        :param jnt_values: A list of 6 joint angles in radians.
        :param vel: Joint velocity in rad/s.
        :param acc: Joint acceleration in rad/s^2.
        :param wait: If True, blocks until the movement is complete.
        """
        # The 'asynchronous' parameter in moveJ is the opposite of 'wait'
        self._rtde_c.moveJ(jnt_values, vel, acc, not wait)

    def regulate_jnts_pmpi(self):
        """
        Moves all joints to their equivalent angle within the [-pi, pi] range.
        Useful for resetting joint configurations after multiple rotations.
        """
        current_jnts = self.get_jnt_values()
        regulated_jnts = rm.regulate_angle(-math.pi, math.pi, current_jnts)
        print("Regulating joints to [-pi, pi] range.")
        self.move_jnts(regulated_jnts)

    def move_jntspace_path(self, path, interval_time=1.0, control_frequency=.008,
                           vel=0.5, acc=0.8,
                           speed_gain=300,
                           blend=0.0):
        """
        Executes a trajectory defined by a list of joint-space waypoints.

        :param path: A list of 1x6 joint configurations (in radians).
        :param interval_time: Time interval for trajectory interpolation.
        :param control_frequency: Control frequency for interpolation.
        :param vel: Velocity for each point in the trajectory.
        :param acc: Acceleration for each point in the trajectory.
        :param blend: Blend radius for smoothing corners between points.
        """
        # Interpolate the path using the provided trajectory planner
        interpolated_confs, _, _ = self.trajt.piecewise_interpolation(path, control_frequency, interval_time)

        # Format the trajectory for the ur_rtde moveJ command
        # Each point needs: [q1...q6,]
        rtde_path = interpolated_confs
        print(f"Executing trajectory with {len(rtde_path)} points...")
        # Send the entire trajectory to the robot. This call is synchronous and waits for completion.
        velocity = 0.5
        acceleration = 0.8
        dt = 1.0 / self.rtde_frequency  # 2ms
        lookahead_time = 0.1
        gain = speed_gain
        for q_joint in rtde_path:
            t_start = self._rtde_c.initPeriod()
            self._rtde_c.servoJ(q_joint, velocity, acceleration, dt, lookahead_time, gain)
            self._rtde_c.waitPeriod(t_start)
        self._rtde_c.servoStop()  # Stop the servo motion
        print("Trajectory execution complete.")

    def get_jnt_values(self):
        """
        Returns the current joint angles of the robot in radians.
        :return: A numpy array of 6 joint angles.
        """
        return np.asarray(self._rtde_r.getActualQ())

    def get_pose(self):
        """
        Returns the current TCP pose of the robot.

        :return: A tuple (pos, rot) where:
                 - pos is a 1x3 numpy array for [x, y, z].
                 - rot is a 3x3 numpy array representing the rotation matrix.
        """
        tcp_pose_vec = self._rtde_r.getActualTCPPose()
        pos = np.asarray(tcp_pose_vec[:3])
        # Convert the rotation vector to a 3x3 rotation matrix
        rot = rm.rotmat_from_euler(tcp_pose_vec[3], tcp_pose_vec[4], tcp_pose_vec[5])
        return pos, rot

    def disconnect(self):
        """Disconnects the RTDE control interface."""
        self._rtde_c.disconnect()
        print("Disconnected from the robot.")


if __name__ == '__main__':
    # This example requires the user's custom libraries to run.
    # The structure of the main block is preserved.
    # import visualization.panda.world as wd
    # base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
    u5rag95_x1 = UR5Ag95X_RTDE(robot_ip='192.168.125.50', gp_port='COM5')
    print(u5rag95_x1.get_jnt_values())
    # try:
    #     # Initialize the first robot and gripper
    #     u5rag95_x1 = UR5Ag95X_RTDE(robot_ip='10.2.0.50', gp_port='COM5')
    #     print(u5rag95_x1.get_jnt_values())
    #     u5rag95_x1.move_jnts(np.array(
    #         [-0.77652818, -1.29162676, 1.12792969, -0.3230756, -1.30920202, -0.693847
    #          ]
    #     ))
    #     exit(0)
    #     # Initialize the second robot and gripper
    #     u5rag95_x2 = UR5Ag95X_RTDE(robot_ip='192.168.1.10', gp_port='COM4')
    #
    #     # --- Example Usage ---
    #
    #     # Print initial joint values
    #     print("Robot 1 initial joints:", u5rag95_x1.get_jnt_values())
    #     print("Robot 2 initial joints:", u5rag95_x2.get_jnt_values())
    #
    #     j = u5rag95_x1.get_jnt_values()
    #     j[-1] = j[-1] + math.pi / 2  # Adjust the last joint angle
    #     u5rag95_x1.move_jnts(j, vel=0.1, acc=0.1, wait=True)
    #     print("WAIT")
    #     print(u5rag95_x1.get_pose())
    #     # Close both grippers
    #     u5rag95_x1.close_gripper()
    #     u5rag95_x2.close_gripper()
    #
    #     time.sleep(1)  # Wait for grippers
    #
    #     # Open both grippers
    #     u5rag95_x1.open_gripper()
    #     u5rag95_x2.open_gripper()
    #
    #     print("Finished script.")
    #
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #
    # finally:
    #     # It's good practice to disconnect
    #     if 'u5rag95_x1' in locals():
    #         u5rag95_x1.disconnect()
    #     if 'u5rag95_x2' in locals():
    #         u5rag95_x2.disconnect()

    # base.run()
