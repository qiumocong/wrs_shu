import math
import time
from basis import robot_math as rm
import drivers.urx.ur_robot as urrobot
import robot_con.ur.program_builder as pb
import threading
import socket
import struct
import os
import numpy as np
import motion.trajectory.polynomial_wrsold as pwp
# from drivers.devices.dh.ag95 import Ag95Driver
import robot_con.ag145.ag145 as ag145server
import visualization.panda.world as wd


class UR5Ag95X(object):
    """
    author: weiwei
    date: 20180131
    """

    def __init__(self,
                 robot_ip='10.2.0.50',
                 pc_ip='10.2.0.100',
                 gp_port='com3', ):
        """
        :param robot_ip:
        :param pc_ip:
        """
        # setup arm
        self._arm = urrobot.URRobot(robot_ip)
        self._arm.set_tcp((0, 0, 0, 0, 0, 0))
        self._arm.set_payload(1.28)
        # setup hand
        # self._hnd = Ag95Driver(port=gp_port)
        self._hnd = ag145server.Ag145driver(port=gp_port,baudrate = 115200, force = 100, speed = 100)
        # setup pc server
        self._pc_server_socket_addr = (pc_ip, 0)  # 0: the system finds an available port
        self._pc_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._pc_server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._pc_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._pc_server_socket.bind(self._pc_server_socket_addr)
        self._pc_server_socket.listen(5)
        self._jnts_scaler = 1e6
        self._pb = pb.ProgramBuilder()
        self._script_dir = os.path.dirname(__file__)
        self._pb.load_prog(os.path.join(self._script_dir, "urscripts_cbseries/moderndriver_cbseries.script"))
        self._modern_driver_urscript = self._pb.get_program_to_run()
        self._modern_driver_urscript = self._modern_driver_urscript.replace("parameter_ip",
                                                                            self._pc_server_socket_addr[0])
        self._modern_driver_urscript = self._modern_driver_urscript.replace("parameter_port",
                                                                            str(self._pc_server_socket.getsockname()[
                                                                                    1]))
        self._modern_driver_urscript = self._modern_driver_urscript.replace("parameter_jointscaler",
                                                                            str(self._jnts_scaler))
        self._ftsensor_thread = None
        self._ftsensor_values = []
        self.trajt = pwp.TrajPoly(method='quintic')

    @property
    def arm(self):
        # read-only property
        return self._arm

    @property
    def hnd(self):
        # read-only property
        return self._hnd

    @property
    def pc_server_socket_info(self):
        """
        :return: [ip, port]
        """
        return self._pc_server_socket.getsockname()

    @property
    def pc_server_socket(self):
        return self._pc_server_socket

    @property
    def jnts_scaler(self):
        return self._jnts_scaler

    def open_gripper(self, speedpercentange=70, forcepercentage=50, fingerdistance=85):
        """
        open the rtq85 hand on the arm specified by arm_name
        :param arm_name:
        :return:
        author: weiwei
        date: 20180220
        """
        self.hnd.open_g()

    def close_gripper(self, speedpercentange=80, forcepercentage=50):
        """
        close the rtq85 hand on the arm specified by arm_name
        :param arm_name:
        :return:
        author: weiwei
        date: 20180220
        """
        self.hnd.close_g()

    def move_jnts(self, jnt_values, wait=True):
        """
        :param jnt_values: a 1-by-6 list in degree
        :param arm_name:
        :return:
        author: weiwei
        date: 20170411
        """
        self._arm.movej(jnt_values, acc=.3, vel=.3, wait=wait)
        # targetarm.movejr(jointsrad, acc = 1, vel = 1, radius = radius, wait = False)

    def regulate_jnts_pmpi(self):
        """
        TODO allow settings for pmpi pm 2pi
        the function move all joints back to -360,360
        due to improper operations, some joints could be out of 360
        this function moves the outlier joints back
        :return:
        author: weiwei
        date: 20180202
        """
        jnt_values = self.get_jnt_values()
        regulated_jnt_values = rm.regulate_angle(-math.pi, math.pi, jnt_values)
        self.move_jnts(regulated_jnt_values)

    def move_jntspace_path(self, path, control_frequency=.008, interval_time=1.0, interpolation_method=None,
                           timeout=100.0):
        """
        move robot_s arm following a given jointspace path
        :param path: a list of 1x6 arrays
        :param control_frequency: the program will sample time_intervals/control_frequency confs, see motion.trajectory
        :param interval_time: equals to expandis/speed, speed = degree/second
                              by default, the value is 1.0 and the speed is expandis/second
        :param interpolation_method
        :return:
        author: weiwei
        date: 20210331
        """
        # if interpolation_method:
        #     self.trajt.change_method(interpolation_method)
        # interpolated_confs, _, _, _ = self.trajt.interpolate_by_time_interval(path, control_frequency, interval_time)
        interpolated_confs, _, _, = self.trajt.piecewise_interpolation(path, control_frequency, interval_time)
        # upload a urscript to connect to the pc server started by this class
        self._arm.send_program(self._modern_driver_urscript)
        # accept arm socket
        pc_server_socket, pc_server_socket_addr = self._pc_server_socket.accept()
        print("PC server onnected by ", pc_server_socket_addr)
        # send trajectory
        keepalive = 1
        buf = bytes()
        for id, conf in enumerate(interpolated_confs):
            if id == len(interpolated_confs) - 1:
                keepalive = 0
            jointsradint = [int(jnt_value * self._jnts_scaler) for jnt_value in conf]
            buf += struct.pack('!iiiiiii', jointsradint[0], jointsradint[1], jointsradint[2],
                               jointsradint[3], jointsradint[4], jointsradint[5], keepalive)
        pc_server_socket.send(buf)
        pc_server_socket.close()
        time.sleep(1)
        st = time.time()
        dist = np.linalg.norm(interpolated_confs[-1] - self.get_jnt_values())
        while dist > 0.01:
            dist = np.linalg.norm(interpolated_confs[-1] - self.get_jnt_values())
            print(
                f'waiting for the robot to reach the final conf:{dist}')
            if time.time() - st > timeout:
                print('timeout')
                raise RuntimeError('timeout in move_jntspace_path')
            time.sleep(0.5)

    def get_jnt_values(self):
        """
        get the joint angles in radian
        :param arm_name:
        :return:
        author: ochi, revised by weiwei
        date: 20180410
        """
        return np.asarray(self._arm.getj())

    def get_pose(self):
        p = self._arm.getl()
        print('pose is ', p)
        pos = np.asarray(p[:3])
        rot = rm.rotmat_from_euler(*p[3:6], )
        return pos, rot


if __name__ == '__main__':
    base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
    # u5rag95_x = UR5Ag95X(robot_ip='192.168.125.50', pc_ip='192.168.125.1', gp_port='COM5')
    # print(u5rag95_x.get_jnt_values())
    # u5rag95_x.move_jnts(np.array([ 1.8091223,  -1.47670562,  1.62137777, -3.39050355,  4.50955915 , 1.88655987]))
    u5rag95_x2 = UR5Ag95X(robot_ip='192.168.1.10', pc_ip='192.168.1.100', gp_port='COM3')
    # print(u5rag95_x.get_jnt_values())
    # print(u5rag95_x.get_pose())
    # u5rag95_x.close_gripper()
    u5rag95_x2.close_gripper()
    time.sleep(1)
    # u5rag95_x.open_gripper()
    # u5rag95_x2.open_gripper()
    print("Finished")
    base.run()
