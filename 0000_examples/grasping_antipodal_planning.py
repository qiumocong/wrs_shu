import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.grippers.yumi_gripper.yumi_gripper as yg

base = wd.World(cam_pos=[1, 1, 1],w=960,
                 h=540, lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object_tube = cm.CollisionModel("objects/test_long.stl")
object_tube.set_rgba([.9, .75, .35, 1])
object_tube.attach_to(base)

# hnd_s
gripper_s = yg.YumiGripper()
grasp_info_list = gpa.plan_grasps(gripper_s, object_tube,
                                  angle_between_contact_normals=math.radians(177),
                                  openning_direction='loc_x',
                                  max_samples=15, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.005)
gpa.write_pickle_file('test_long', grasp_info_list, './', 'yumi_tube_big.pickle')
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=(1,0,0,0.1)).attach_to(base)
base.run()