import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.collision_model as cm
import modeling.geometric_model as gm
import manipulation.pick_place_planner as ppp
import robot_sim.robots.xme3p.xme3p as xme3p
import robot_sim.end_effectors.gripper.dh50.dh50 as dh50
import grasping.planning.antipodal as gpa


if __name__ == "__main__":
    robots = xme3p.XME3P()
    gripper_s = dh50.Dh50()
    ppp = ppp.PickPlacePlanner(robots)
    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])

    box = cm.CollisionModel('./objects/box.stl')
    grasp_info_list = gpa.plan_grasps(gripper_s, box)

    start_pos = np.array([0, 0.5, 0.2])
    goal_pos = np.array([0.3, 0.5, 0.2])

    start_conf = robots.ik("arm", start_pos, rm.rotmat_from_axangle([0, 1, 0], np.pi),seed_jnt_values=robots.get_jnt_init("arm"))
    goal_conf = robots.ik("arm", goal_pos, rm.rotmat_from_axangle([0, 1, 0], np.pi), seed_jnt_values=robots.get_jnt_init("arm"))

    robots.fk("arm", start_conf)
    robots.gen_meshmodel().attach_to(base)
    print(f"start conf: {robots.is_collided()}")
    robots.fk("arm", goal_conf)
    robots.gen_meshmodel().attach_to(base)
    print(f"goal conf: {robots.is_collided()}")
    base.run()

    object_start_homomat = rm.homomat_from_posrot(start_pos, np.eye(3))
    object_goal_homomat = rm.homomat_from_posrot(goal_pos, np.eye(3))

    conf_list, jawwidth_list, objpose_list = ppp.gen_pick_and_place_motion(
                                            hnd_name='hnd',
                                              objcm=box,
                                              grasp_info_list=grasp_info_list,
                                              start_conf=start_conf,
                                              end_conf=start_conf,
                                              goal_homomat_list=[object_start_homomat, object_goal_homomat,
                                                                 object_start_homomat],
                                              approach_direction_list=[None, np.array([0, 0, -1]), None],
                                              approach_distance_list=[.05] * 3,
                                              depart_direction_list=[np.array([0, 0, 1]), None, None],
                                              depart_distance_list=[.05] * 3)

    def update(task):
        if len(conf_list) > 0:
            robots.fk('arm', conf_list.pop(0))
            robots.jaw_to(jawwidth_list.pop(0), 'hnd')
            if len(objpose_list) > 0:
                box.set_homomat(objpose_list.pop(0))
                box.attach_to(base)
        return task.cont
    base.taskMgr.add(update, "update")
    base.run()
