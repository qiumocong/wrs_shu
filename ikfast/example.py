from ur_ikfast import ur_kinematics
import numpy as np

ur3e_arm = ur_kinematics.URKinematics('ur5e')

# joint_angles = [-3.1, -1.6, 1.6, -1.6, -1.6, 0.]  # in radians
# joint_angles = [-1.2392502,  -3.17189683,  1.15300089, -2.69349299,  1.57079563, -2.81004731]
# joint_angles = [ 1.53371139, -0.31673156,  0.29575013, -1.54981431, -1.57079593,  3.10450771]
joint_angles = [ 1.52304391+0.0, -0.33097409+0.001,  0.32462612, -4.70604101, -4.71238875, -0.04775242]
# joint_angles = [0, -0.75*np.pi, 0.5*np.pi,-0.5*np.pi,-0.5*np.pi,0]
  # in radians

print("joint angles", joint_angles)

pose_quat = ur3e_arm.forward(joint_angles)
pose_matrix = ur3e_arm.forward(joint_angles, 'matrix')

print("forward() quaternion \n", pose_quat)
print("forward() matrix \n", pose_matrix)

# print("inverse() all", ur3e_arm.inverse(pose_quat, True))
print("inverse() one from quat", ur3e_arm.inverse(pose_quat, True, q_guess=joint_angles))

print("inverse() one from matrix", ur3e_arm.inverse(pose_matrix, True, q_guess=joint_angles))