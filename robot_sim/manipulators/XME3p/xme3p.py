import os
import math
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi
import modeling.collision_model as cm

class XME3P(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='xme3p', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0.177])
        self.jlc.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(ai=0, aj=0, ak=0)
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[2]['loc_pos'] = np.array([0, .0155, 0.1645])
        self.jlc.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(ai=math.pi / 2, aj=0, ak=math.pi)
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[3]['loc_pos'] = np.array([0, 0.2295, -0.0155])
        self.jlc.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(ai=math.pi / 2, aj=0, ak=-math.pi)
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[4]['loc_pos'] = np.array([0, 0.0155, 0.1645])
        self.jlc.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(ai=math.pi / 2, aj=0, ak=math.pi)
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[5]['loc_pos'] = np.array([0, 0.2178, -0.0155])
        self.jlc.jnts[5]['loc_rotmat'] = rm.rotmat_from_euler(ai=math.pi / 2, aj=0, ak=-math.pi)
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[6]['loc_pos'] = np.array([0., 0.0118, 0.1482])
        self.jlc.jnts[6]['loc_rotmat'] = rm.rotmat_from_euler(ai=math.pi / 2, aj=0, ak=math.pi)
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[7]['loc_pos'] = np.array([0., 0.1933, -0.01192])
        self.jlc.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(ai=math.pi / 2, aj=0, ak=-math.pi)
        self.jlc.jnts[7]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[8]['loc_pos'] = np.array([0., 0., 0.0570])
        self.jlc.jnts[8]['loc_rotmat'] = rm.rotmat_from_euler(ai=0, aj=0, ak=0)
        self.jlc.jnts[8]['loc_motionax'] = np.array([0, 0, 1])

        # links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[1]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mass'] = 2.0
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "XME3p-R760-base.stl")
        self.jlc.lnks[0]['rgba'] = [.5, .5, .5, 1]

        self.jlc.lnks[1]['name'] = "shoulder"
        self.jlc.lnks[1]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[1]['com'] = np.array([.0, -.02, .0])
        self.jlc.lnks[1]['mass'] = 1.95
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "XME3p-R760-yao.stl")
        self.jlc.lnks[1]['rgba'] = [.5, .5, .5, 1]

        self.jlc.lnks[2]['name'] = "upperarm"
        self.jlc.lnks[2]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[2]['com'] = np.array([.13, 0, .1157])
        self.jlc.lnks[2]['mass'] = 3.42
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "XME3p-R760-dabiqian.stl")
        self.jlc.lnks[2]['rgba'] = [.5, .5, .5, 1]

        self.jlc.lnks[3]['name'] = "forearm"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['com'] = np.array([.05, .0, .0238])
        self.jlc.lnks[3]['mass'] = 1.437
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "XME3p-R760-dabihou.stl")
        self.jlc.lnks[3]['rgba'] = [.5, .5, .5, 1]

        self.jlc.lnks[4]['name'] = "wrist1"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[4]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[4]['mass'] = 0.871
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "XME3p-R760-xiaobiqian.stl")
        self.jlc.lnks[4]['rgba'] = [.5, .5, .5, 1]

        self.jlc.lnks[5]['name'] = "wrist2"
        self.jlc.lnks[5]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[5]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[5]['mass'] = 0.8
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "XME3p-R760-xiaobihou.stl")
        self.jlc.lnks[5]['rgba'] = [.5, .5, .5, 1]

        self.jlc.lnks[6]['name'] = "wrist3"
        self.jlc.lnks[6]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[6]['com'] = np.array([.0, .0, 0])
        self.jlc.lnks[6]['mass'] = 0.8
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "XME3p-R760-shouwan.stl")
        self.jlc.lnks[6]['rgba'] = [.5, .5, .5, 1]
        self.jlc.reinitialize()

        self.jlc.lnks[7]['name'] = "wrist4"
        self.jlc.lnks[7]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[7]['com'] = np.array([.0, .0, 0])
        self.jlc.lnks[7]['mass'] = 0.8
        self.jlc.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "XME3p-R760-bashou.stl")
        self.jlc.lnks[7]['rgba'] = [.5, .5, .5, 1]
        self.jlc.reinitialize()


        # collision checker
        if enable_cc:
            super().enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6],
                      self.jlc.lnks[7]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[0], self.jlc.lnks[1]]
        intolist = [self.jlc.lnks[3], self.jlc.lnks[5], self.jlc.lnks[6], self.jlc.lnks[7]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[2]]
        intolist = [self.jlc.lnks[4], self.jlc.lnks[5], self.jlc.lnks[6], self.jlc.lnks[7]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[3]]
        intolist = [self.jlc.lnks[6], self.jlc.lnks[7]]
        self.cc.set_cdpair(fromlist, intolist)

if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = XME3P(enable_cc=True)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    # manipulator_meshmodel.show_cdprimit()
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    base.run()