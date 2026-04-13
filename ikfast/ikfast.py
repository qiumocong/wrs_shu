"""
Author: <Mocong Qiu>
Created: <2025/9/26>

A wrapper for ikfast/ur_kinematics.
"""

import numpy as np
from ur_ikfast import ur_kinematics


class UrIkFast:
    def __init__(self, robot_name: str = "ur3e"):
        """
        Create a UR IKFast instance.

        :param robot_name: e.g. 'ur3', 'ur3e', 'ur5', 'ur5e', 'ur10', 'ur10e'
        """
        self.robot_name = robot_name
        self._solver = ur_kinematics.URKinematics(robot_name)
        self.dof = 6  # 通常UR系列机械臂为6自由度

    def fk(self, q: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Forward kinematics.

        :param q: 1x6 joint values (radians)
        :return: (position, rotation matrix)
        """
        q = np.asarray(q).reshape(-1)
        assert q.size == self.dof, f"q must be a 1x{self.dof} array, got {q.shape}"
        pose_matrix = self._solver.forward(q, 'matrix')  # 4x4
        pos = pose_matrix[:3, 3]
        rot = pose_matrix[:3, :3]
        return pos, rot

    def ik(self, tgt_pos: np.ndarray, tgt_rot: np.ndarray, seed_jnt_values: np.ndarray = None, all_solutions: bool = True) -> None or np.ndarray:
        """
        Solve IK.

        :param tgt_pos: 1x3 target position
        :param tgt_rot: 3x3 target rotation matrix
        :param seed_jnt_values: 1x6 seed joint values (for solution selection)
        :return: None if no solution, otherwise 1x6 joint values
        """
        # 构造4x4位姿矩阵
        pose_matrix = np.zeros([3, 4])
        pose_matrix[:3, :3] = tgt_rot
        pose_matrix[:3, 3] = tgt_pos

        # ur_kinematics 的 inverse 支持 q_guess 选最近解
        solutions = self._solver.inverse(pose_matrix, all_solutions=all_solutions, q_guess=seed_jnt_values)
        if not solutions or len(solutions) == 0:
            return None
        else:
            return solutions
